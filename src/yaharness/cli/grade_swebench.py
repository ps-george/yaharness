"""`yagrade` — Tier-2 docker grading via the official `swebench` package.

Reads sidecar JSON files in an evaluation directory (one per
``<system>-<seed>-<problem_id>``), each containing at minimum::

    {"instance_id": str, "system": str, "seed": int, "final_answer": str, ...}

Builds a ``predictions.jsonl`` in the format expected by
``swebench.harness.run_evaluation`` (one prediction per line, keys:
``instance_id``, ``model_patch``, ``model_name_or_path``), invokes the docker
evaluation harness, then reads back the per-instance report file and merges
``resolved`` (real pass/fail) into each sidecar JSON.

Cost
----
Docker grading is CPU-only — there is no API spend here. It does pull large
images (one per SWE-bench instance, ~1-3 GB) and may take 5-20 min per
instance.

Pre-conditions
--------------
- Docker daemon running and authorised to pull from Docker Hub.
- The ``swebench`` package installed (``pyproject`` lists it).
- Each sidecar JSON exists with a non-empty ``final_answer`` (empty patches
  are reported as ``empty_patch`` and not resolved).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _iter_sidecar_files(eval_dir: Path) -> Iterable[Path]:
    """Yield sidecar JSONs (skip predictions/report files)."""
    for path in sorted(eval_dir.glob("*.json")):
        name = path.name
        if name in {"predictions.jsonl", "grading-report.json"}:
            continue
        # Heuristic: must contain the required keys.
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if "instance_id" in data and "final_answer" in data:
            yield path


def build_predictions(sidecars: Sequence[Path]) -> list[dict[str, Any]]:
    """Build the swebench-prediction records from sidecar JSONs.

    ``model_name_or_path`` is keyed off ``<system>-<seed>`` so multiple
    systems/seeds can share one predictions file without colliding under
    swebench's ``logs/run_evaluation/<run_id>/<model>/<instance>/`` layout.
    """
    preds: list[dict[str, Any]] = []
    for path in sidecars:
        data = json.loads(path.read_text(encoding="utf-8"))
        system = data.get("system", "unknown")
        seed = data.get("seed", 0)
        model_name = f"{system}-seed{seed}"
        preds.append(
            {
                "instance_id": str(data["instance_id"]),
                "model_patch": data.get("final_answer", "") or "",
                "model_name_or_path": model_name,
            }
        )
    return preds


def write_predictions_jsonl(preds: Sequence[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p) + "\n")


def _model_dir_name(model_name: str) -> str:
    """Replicate swebench's ``model_name.replace('/', '__')`` for the log path."""
    return model_name.replace("/", "__")


def read_resolutions(
    *,
    run_id: str,
    preds: Sequence[dict[str, Any]],
    log_root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Return ``{(model_name, instance_id): {resolved: bool, raw: ...}}``."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for pred in preds:
        instance_id = pred["instance_id"]
        model_name = pred["model_name_or_path"]
        report_file = log_root / run_id / _model_dir_name(model_name) / instance_id / "report.json"
        if not report_file.exists():
            out[(model_name, instance_id)] = {
                "resolved": False,
                "graded": False,
                "report_path": str(report_file),
                "reason": "no report file (likely empty patch or container error)",
            }
            continue
        try:
            content = report_file.read_text(encoding="utf-8").strip()
            report = json.loads(content) if content else {}
        except (OSError, json.JSONDecodeError) as e:
            out[(model_name, instance_id)] = {
                "resolved": False,
                "graded": True,
                "report_path": str(report_file),
                "reason": f"report unparseable: {e}",
            }
            continue
        inst_report = report.get(instance_id, {})
        out[(model_name, instance_id)] = {
            "resolved": bool(inst_report.get("resolved", False)),
            "graded": True,
            "report_path": str(report_file),
            "raw_report": inst_report,
        }
    return out


def merge_into_sidecars(
    sidecars: Sequence[Path],
    resolutions: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """Write ``grading`` field back into each sidecar JSON in-place."""
    for path in sidecars:
        data = json.loads(path.read_text(encoding="utf-8"))
        system = data.get("system", "unknown")
        seed = data.get("seed", 0)
        model_name = f"{system}-seed{seed}"
        instance_id = str(data["instance_id"])
        key = (model_name, instance_id)
        resolution = resolutions.get(
            key,
            {"resolved": False, "graded": False, "reason": "no resolution entry"},
        )
        data["grading"] = resolution
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def run_swebench_evaluation(
    *,
    predictions_path: Path,
    run_id: str,
    instance_ids: list[str],
    max_workers: int,
    timeout: int,
    dataset_name: str,
    split: str,
    report_dir: Path,
) -> Path | None:
    """Invoke ``swebench.harness.run_evaluation.main`` and return the report path."""
    from swebench.harness import run_evaluation  # type: ignore[import-untyped]

    report_path = run_evaluation.main(
        dataset_name=dataset_name,
        split=split,
        instance_ids=instance_ids,
        predictions_path=str(predictions_path),
        max_workers=max_workers,
        force_rebuild=False,
        cache_level="env",
        clean=False,
        open_file_limit=4096,
        run_id=run_id,
        timeout=timeout,
        namespace=None,
        rewrite_reports=False,
        modal=False,
        report_dir=str(report_dir),
    )
    return Path(report_path) if report_path else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yagrade",
        description="Tier-2 docker grading for SWE-bench predictions.",
    )
    p.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Directory of sidecar JSONs (one per system-seed-problem).",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="swebench run_id (default: derived from current timestamp).",
    )
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--timeout", type=int, default=1800, help="per-instance docker timeout (s)")
    p.add_argument("--dataset-name", default="princeton-nlp/SWE-bench_Verified")
    p.add_argument("--split", default="test")
    p.add_argument(
        "--log-root",
        type=Path,
        default=Path("logs/run_evaluation"),
        help="where swebench writes per-instance logs (default: ./logs/run_evaluation).",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    eval_dir: Path = args.eval_dir
    if not eval_dir.exists():
        print(f"ERROR: --eval-dir does not exist: {eval_dir}", file=sys.stderr)
        return 2

    sidecars = list(_iter_sidecar_files(eval_dir))
    if not sidecars:
        print(f"ERROR: no sidecar JSONs found in {eval_dir}", file=sys.stderr)
        return 2
    logger.info("found %d sidecar JSONs", len(sidecars))

    preds = build_predictions(sidecars)
    # Combined predictions file for inspection/reference.
    predictions_path = eval_dir / "predictions.jsonl"
    write_predictions_jsonl(preds, predictions_path)
    logger.info("wrote %d predictions to %s", len(preds), predictions_path)

    # swebench dedupes predictions by instance_id (last-write-wins). To grade
    # multiple model patches for the same instance we must invoke swebench
    # once per `model_name_or_path` with its own predictions file. Cached env
    # images are shared across invocations, so this is cheap after the first.
    by_model: dict[str, list[dict[str, Any]]] = {}
    for p in preds:
        by_model.setdefault(p["model_name_or_path"], []).append(p)

    run_id = args.run_id or f"yaharness-{time.strftime('%Y%m%dT%H%M%S')}"
    logger.info(
        "invoking swebench run_evaluation (run_id=%s, %d model groups)",
        run_id,
        len(by_model),
    )
    t0 = time.time()
    try:
        for model_name, model_preds in sorted(by_model.items()):
            per_model_path = eval_dir / f"predictions-{model_name}.jsonl"
            write_predictions_jsonl(model_preds, per_model_path)
            model_instance_ids = sorted({p["instance_id"] for p in model_preds})
            # Skip-if-graded: docker grading is slow; if every instance for this
            # model already has a report.json, don't re-invoke swebench.
            model_log_dir = args.log_root / run_id / _model_dir_name(model_name)
            all_reports_exist = all(
                (model_log_dir / iid / "report.json").exists() for iid in model_instance_ids
            )
            if all_reports_exist:
                logger.info(
                    "skip (already graded): model=%s (%d instances)",
                    model_name,
                    len(model_instance_ids),
                )
                continue
            logger.info("grading model=%s (%d instances)", model_name, len(model_instance_ids))
            run_swebench_evaluation(
                predictions_path=per_model_path,
                run_id=run_id,
                instance_ids=model_instance_ids,
                max_workers=args.max_workers,
                timeout=args.timeout,
                dataset_name=args.dataset_name,
                split=args.split,
                report_dir=eval_dir,
            )
    except Exception as exc:  # pragma: no cover - depends on docker
        logger.exception("swebench evaluation crashed")
        print(f"ERROR: swebench evaluation failed: {exc}", file=sys.stderr)
        return 1
    elapsed = time.time() - t0
    logger.info("swebench evaluation completed in %.1fs", elapsed)

    resolutions = read_resolutions(run_id=run_id, preds=preds, log_root=args.log_root)
    merge_into_sidecars(sidecars, resolutions)

    # Write a top-level grading report summarising the run.
    summary: dict[str, Any] = {
        "run_id": run_id,
        "eval_dir": str(eval_dir),
        "n_predictions": len(preds),
        "n_resolved": sum(1 for r in resolutions.values() if r.get("resolved")),
        "n_graded": sum(1 for r in resolutions.values() if r.get("graded")),
        "elapsed_s": elapsed,
        "per_prediction": [
            {
                "instance_id": k[1],
                "model_name": k[0],
                "resolved": v.get("resolved", False),
                "graded": v.get("graded", False),
                "reason": v.get("reason"),
            }
            for k, v in sorted(resolutions.items())
        ],
    }
    (eval_dir / "grading-report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"grading complete: {summary['n_resolved']}/{summary['n_predictions']} resolved "
        f"({summary['n_graded']} graded) in {elapsed:.1f}s"
    )
    return 0


def _entry() -> Any:
    sys.exit(main())


if __name__ == "__main__":  # pragma: no cover
    _entry()


__all__ = [
    "build_parser",
    "build_predictions",
    "main",
    "merge_into_sidecars",
    "read_resolutions",
    "run_swebench_evaluation",
    "write_predictions_jsonl",
]
