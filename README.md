# ML4Good — Emergent Misalignment Project

Reproduction and stress-testing of **"Emergent Misalignment Is Not Magical"**
(Li, Dai, Wang, Tan — [arXiv:2608.29118](https://arxiv.org/abs/2608.29118)),
carried out as an [ML4Good](https://www.ml4good.org/) camp project.

The paper argues that emergent misalignment (EM) is a predictable, data-dependent
generalization phenomenon: evilness after EM fine-tuning is predicted by the
representational distance between an evaluation prompt and the EM training-data
centroid, and a dataset-specific **mean-difference generalization direction**
robustly predicts misalignment under semantics-preserving prompt perturbations.

## Short-term plan

1. **Reproduce the paper's results** to confirm that the mean-difference method
   consistently determines misalignment — including prompt-level predictions
   (representational distance / generalization direction vs. judged misalignment).
2. **Search for a falsifying input**: a prompt that the method scores as
   innocuous but that nonetheless elicits misalignment from an EM model. A
   confirmed counterexample would bound where the mean-difference account holds.

## Deliverables (camp outputs)

- **Reproduction artifacts** — code, configs, and raw outputs reproducing the
  paper's key correlational results.
- **Evaluation prompts** — a prompt set that tests the theory, including
  candidate falsifying inputs with their predicted vs. observed misalignment.

## Status

Work in progress. Raw experiment outputs and short notes only; claims are
provisional until verified.

## Documentation

See [docs/](docs/README.md): setup, the sec3 pipeline as implemented, data
provenance (the paper's prompt sets are unreleased; ours is a same-recipe
reconstruction), the process log, and the runbook.

Judge-robustness work on the filter stage (Jev decision model vs Kimi K3) is in
[docs/jev_filter_check.md](docs/jev_filter_check.md); every model/judge used in the pipeline is
mapped in [docs/model_touchpoint_map.md](docs/model_touchpoint_map.md).
