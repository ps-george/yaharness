"""Statistical analysis and reporting for benchmark results.

Per docs/EVALUATION-METHODOLOGY.md: paired bootstrap at problem level,
effect sizes, per-step degradation curves, markdown reporting.
"""

from yaharness.analysis.bootstrap import BootstrapResult, paired_bootstrap
from yaharness.analysis.degradation import (
    fit_degradation_slope,
    per_step_success_curve,
)
from yaharness.analysis.effect_size import cohens_h, proportion_diff_ci
from yaharness.analysis.reporting import (
    SystemResults,
    benchmark_results_table,
    load_results,
)

__all__ = [
    "BootstrapResult",
    "SystemResults",
    "benchmark_results_table",
    "cohens_h",
    "fit_degradation_slope",
    "load_results",
    "paired_bootstrap",
    "per_step_success_curve",
    "proportion_diff_ci",
]
