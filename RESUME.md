# Resume — yaharness

**Status:** Paused 2026-05-26. Not abandoned; the code works and the repo is shippable. Paused because the empirical work showed the harness is *engineering quality, not a research breakthrough*, and our attention is better placed elsewhere for now.

## What this repo is

A small, modern Python ReAct agent harness with SWE-bench Verified grading support. Clean, mypy-strict, 216 tests, MIT licence. Public at https://github.com/ps-george/yaharness.

Honest positioning (per README): a reference implementation, not a SOTA system or production alternative to OpenHands/SWE-Agent. Best fit if someone wants to understand how an agent harness works without reading 10k+ lines.

## What got measured

- Empirical eval on SWE-bench Verified, Sonnet 4.6, 5 problems × 3 seeds:
  - `single_react` (this harness's basic ReAct loop): 11/15 resolved (73%) on the 5 easiest-patch problems
  - 73% is SOTA-ballpark for SWE-bench Verified at 5 problems, but the sample is too small and biased (easiest patches) to claim SOTA — it's "in the right neighborhood"
- See `docs/calibration/2026-05-26-swebench-5problem-sonnet.md` in the parent loop-multi-agent repo for the full eval details (private; not in this public repo)

## Why paused

- The harness is honest engineering quality, not novel research
- OpenHands and SWE-Agent are better production choices
- The repo's marginal utility is as a teaching reference; we've shipped it for that purpose
- Next-step value-add (50-problem stratified eval, federated baselines, etc.) requires real $ and time; better spent on the seed-library-product line

## What to do to resume

If someone wants to push this further:

1. **Run the 50-problem stratified eval** (small/medium/hard golden-patch sizes, 3 seeds). Cost: ~$50-100 on Sonnet. Goal: replace the 5-problem 73% with a statistically meaningful number. If holds, the harness is publication-grade; write a short paper.

2. **Add 1-2 missing baselines:** AutoGen v0.4 + (if API access available) OpenAI Swarm. Both are stubs in current `agents/` directory.

3. **Add the Bench-A from prc** as a secondary benchmark: synthetic-injected-bug recall. Lets a single repo measure "bug catching" + "bug fixing" together.

4. **Federation:** make benchmark adapters pluggable so users can add their own benchmarks via simple Protocol implementation. Document in `EXAMPLES.md`.

5. **Polish:** add a docs site (mkdocs or similar), proper API reference, example notebook walkthrough. Improves discoverability for the teaching-reference use case.

## What NOT to do

- Don't oversell the result. The 73% number must not appear in README without the "5 problems, easiest-patch bias" qualification.
- Don't add features beyond the harness's core scope. It's small on purpose.
- Don't rename or restructure. Position is "small reference"; growth into a "framework" defeats the purpose.

## Where related work lives

- **The bilateral-handshake research framework** that originally motivated this work is in a private repo (`loop-multi-agent`). The empirical negative on bilateral handshake at SWE-bench is documented there. None of that work is in yaharness.
- **`prc`** (https://github.com/ps-george/prc) is a sibling project — a structured PR review tool — also paused with a real-bugs benchmark that showed vanilla LLM beats it.
- **Personal seed-library product work** is the current focus and lives in the private seeds repo.
