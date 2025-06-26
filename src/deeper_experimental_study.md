# Core Questions to Ask and Answer
###  Mask Stability of Super Weights (SW):

    - At what sparsity levels and retraining budgets (steps per round) do SW consistently survive pruning?

    - How does initialization (original pre-trained vs. Xavier vs. Kaiming) affect SW retention?

### Sparsity–Performance Trade-off:

    - How does model quality (e.g. validation perplexity or downstream task accuracy) degrade as you increase sparsity with and without SW preservation?

    - Is there a “sweet spot” where you get high sparsity with minimal performance loss?

### Pruning Schedule Effects:

    - One-shot vs. iterative pruning: does smoothing the schedule (e.g. gradual 5% per round up to 20%) help protect SW?

    - Global vs. layerwise magnitude pruning: which better preserves SW?

### Retraining Budget:

    - How do different numbers of retraining steps (1 k, 3 k, 5 k, 10 k, 20 k) per round influence both final sparsity and SW survival?

    - Does a learning-rate schedule (warmup + decay) improve mask recovery?

### Statistical Robustness:

    - How much run-to-run variance is there? (repeat each key setting 3–5× with different seeds)

    - Are the observations significant?

# Good research practices I should not forget about

Reproducibility:

- Fix and report all random seeds.

- Log (and version-control) training scripts, hyperparameters, and environment (PyTorch/TensorFlow version, CUDA/cuDNN).

Ablation Study Design:

-  Change one variable at a time (init or pruning schedule) to isolate effects.

- Include both positive controls (no pruning) and negative controls (aggressive pruning).

Metrics & Logging:

- Track validation perplexity (or downstream accuracy), training loss curves, and mask‐statistics (fraction of SW zeroed vs. retained) per round.

- Plot sparsity vs. performance curves.

Statistical Analysis:

- For each key setting, run multiple seeds and report mean ± std.

- Use paired t-tests or non-parametric alternatives to confirm differences.

Trasparency:
- Clearly define what we call a “Super Weight” and how we detect pruning.


# Experiments Table


| Exp ID | Init. Scheme    | Steps/Round | Pruning % (global) | Schedule                         | Pruning Type        | Seeds | Key Outcome Tracked                   | Notes / Purpose                       |
|:------:|:----------------|:------------|:------------------:|:---------------------------------|:--------------------|:------|:--------------------------------------|:---------------------------------------|
| A1     | Pre-trained     | 3 k         | 10 %               | 5× constant                      | Global magnitude    | 3     | SW survival per round; val. perplexity | Baseline                               |
| A2     | Pre-trained     | 3 k         | 20 %               | 5× constant                      | Global magnitude    | 3     | SW survival per round; val. perplexity | Baseline                               |
| B1     | Pre-trained     | 10 k        | 10 %               | 5× constant                      | Global magnitude    | 3     | SW survival per round; val. perplexity | Baseline                               |
| B2     | Pre-trained     | 10 k        | 20 %               | 5× constant                      | Global magnitude    | 3     | SW survival per round; val. perplexity | Baseline                               |
| C1     | Xavier init     | 10 k        | 10 %               | 5× constant                      | Global magnitude    | 3     | SW survival per round; val. perplexity | You observed no SW pruned             |
| C2     | Xavier init     | 10 k        | 20 %               | 5× constant                      | Global magnitude    | 3     | SW survival per round; val. perplexity | You observed no SW pruned             |
| D1     | Pre-trained     | 10 k        | 5 → 10 → 15 → 20 %  | Gradual (increasing each round)  | Global magnitude    | 3     | SW survival; performance curve         | Test smoother schedule                 |
| D2     | Pre-trained     | 10 k        | 5 → 10 → 15 → 20 %  | Gradual (increasing each round)  | Layerwise magnitude | 3     | SW survival; performance curve         | Compare global vs. per-layer           |
| E1     | Kaiming init    | 10 k        | 10 %               | 5× constant                      | Global magnitude    | 3     | SW survival per round; val. perplexity | Test different init                    |
| E2     | Kaiming init    | 10 k        | 20 %               | 5× constant                      | Global magnitude    | 3     | SW survival per round; val. perplexity | Test different init                    |
| F1     | Xavier init     | 20 k        | 20 %               | 5× constant                      | Global magnitude    | 3     | SW survival; val. perplexity            | Test larger budget                     |
| G1     | Pre-trained     | 10 k        | 15 %               | 5× constant                      | Global magnitude    | 3     | SW survival per round; val. perplexity | Intermediate sparsity                  |
| H1     | Xavier init     | 10 k        | 10 %               | 5× constant + LR warmup/decay    | Global magnitude    | 3     | SW survival; val. perplexity            | Test effect of LR schedule             |
| I1     | Pre-trained     | 10 k        | 20 %               | One-shot at start                | Global magnitude    | 3     | SW survival; val. perplexity            | Contrast iterative vs. one-shot pruning |


Legend:

Steps/Round: number of training steps between pruning rounds

Schedule: “5× constant” means prune by the stated % each of 5 rounds; “Gradual” means increasing % each round

LR schedule: e.g. 1 k warmup, cosine decay

# Next Steps
Pick a smaller “pilot” subset (e.g. 3 experiments from above covering init, schedule, and budget) and run with 3 seeds to see which factors matter most.

Plot: sparsity vs. validation perplexity with error bars, and SW-survival rate vs. sparsity.

Analyze: statistically test whether Xavier/Kaiming init or gradual pruning significantly improves SW retention at fixed sparsity.

Scale up: once the key factors are identified, run the full grid.

By systematically isolating each variable—initialization, pruning percentage, schedule, and retraining budget — be able to draw clear conclusions about when and why Super Weights survive pruning under the Lottery Ticket regime. 









