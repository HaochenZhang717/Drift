# PSW-I Adaptation Plan for Modality-Level Imputation

## 1) Goal and target setting

We adapt PSW-I to your use case:
- Data: multivariate time series, shape `[N, T, C]`
- Modality definition: each channel is one modality (`C` modalities)
- Training missingness: for each sample, randomly select **one channel** and mask it (whole channel across all time steps)
- Task: modality-level imputation (recover the missing channel from the other channels)

This plan only describes changes. No code changes are made yet.

---

## 2) What the current repo is doing (important mismatch)

After reading code in `benchmark.py` and `our_data/process_data.py`, current PSW-I pipeline is:
- Loads processed data from `our_data/{dataset}_{ratio}_processed_data.npz`
- Uses `X_observed` and `X_all` as 2D arrays (`[n, d]`), with generic MCAR-style entry-wise missing values
- In training (`OTImputationIni.fit_transform`), it additionally creates a validation mask by `mcar(X, p=0.1)`
- Evaluates MAE/MSE/MRE on entry-wise masks (`np.isnan(...)` logic)

Main mismatch with your task:
- Missingness is entry-wise random, not channel-as-modality missing
- Data handling is effectively 2D tabular style, not explicit `[N, T, C]` modality logic
- Evaluation masks do not separate modality-level missing protocol

---

## 3) Adaptation strategy (high-level)

We will keep PSW-I core OT/PSW optimization, but replace masking/data protocol to modality-level missing.

### A. Standardize data interface to 3D
- Introduce a clear canonical input format: `X_full: [N, T, C]` (float)
- Preserve backward compatibility by allowing current 2D inputs only if needed, but our new path uses 3D

### B. Add modality-level missing generator
- New mask generator for training:
  - For each sample `i`, sample one channel `m_i ~ Uniform(0, C-1)`
  - Mask `X[i, :, m_i]` entirely (set to NaN)
- Optional extension later: fixed missing modality, multi-modality missing, block-time missing

### C. Flattening policy (minimal invasive)
PSW-I implementation currently optimizes on 2D matrix. To minimize risk:
- Convert 3D to 2D only at optimization boundary:
  - reshape `[N, T, C] -> [N*T, C]`
- Critical rule: keep channel-wise mask semantics intact before and after reshape
- After imputation, reshape back `[N*T, C] -> [N, T, C]` for metrics

### D. Metric protocol for your task
We will report metrics on the true target mask for modality-level missing:
- Primary: MAE, MSE, RMSE on masked channel entries only
- Optional: per-modality metrics + macro average
- Keep current global metrics for comparability, but mark them secondary

### E. Reproducibility controls
- Add explicit random seed for modality sampling
- Save sampled missing modality indices for each split (train/val/test) to disk
- Ensure every baseline run can reuse the exact same masks

---

## 4) File-level change plan

## Phase 1: data and mask pipeline

1. `our_data/process_data.py`
- Add new mask mode: `modality_one_missing`
- Input expected as full clean 3D time-series (or convert from current format)
- Output `.npz` should include at least:
  - `X_all` (full clean)
  - `X_observed` (with modality-level missing)
  - `missing_modality_idx` (`[N]`)
  - `eval_mask` (`[N, T, C]`, True only where imputation is evaluated)

2. `benchmark.py`
- Add CLI args:
  - `--mask_mode modality_one_missing`
  - `--modality_missing_k 1` (default 1)
  - `--data_ndim 3`
- Replace internal extra `mcar(...)` validation masking with mask protocol from data file
- Keep current PSW-I optimizer; adapt input/output reshape logic around it

## Phase 2: training/eval alignment

3. `benchmark.py::OTImputationIni.fit_transform`
- Accept explicit mask tensors instead of inferring only from `np.isnan(train)` when needed
- Remove/disable hard-coded `train = mcar(X, p=0.1)` for your modality protocol
- Keep early-stop logic, but base val metric on modality-level `val_mask`

4. metrics section in `benchmark.py`
- Compute metrics using `eval_mask` from modality-level protocol
- Add per-modality aggregation option

## Phase 3: scripts and documentation

5. `scripts/*.sh`
- Add one new script for your modality-level setup, e.g. `scripts/<your_dataset>_modality.sh`
- Keep old scripts unchanged

6. `README.md`
- Add a short section: "Modality-level imputation mode"
- Include dataset format and one run command

---

## 5) Validation checklist before claiming success

- Data sanity:
  - `X_all.shape == X_observed.shape == [N, T, C]`
  - For each sample, exactly one channel fully masked during training protocol
- Mask sanity:
  - `eval_mask` only covers truly masked target entries
  - No leakage (observed channels not included in eval mask)
- Optimization sanity:
  - Loss decreases and no NaN/Inf in OT steps
- Metric sanity:
  - MAE/MSE computed only on modality-missing positions
  - Per-modality results are numerically reasonable

---

## 6) Risks and mitigation

- Risk 1: flattening `[N, T, C] -> [N*T, C]` may weaken temporal structure
  - Mitigation: keep existing PSW FFT distance windows; if needed, move to windowed 3D batching later

- Risk 2: modality-level masking is harder than entry-wise MCAR; training may destabilize
  - Mitigation: warm-start initializer, lower LR, tune `reg_sk`, `seq_length`, and dropout

- Risk 3: current code has mixed assumptions and duplicated utilities
  - Mitigation: minimal-invasive first pass, then refactor after first successful run

---

## 7) Proposed implementation order

1. Implement mask/data protocol (`process_data.py`) and save masks
2. Wire `benchmark.py` to consume explicit modality masks and remove internal MCAR masking
3. Add reshape boundary helpers and metric updates
4. Add one runnable script + quick smoke test on small subset
5. Run full training and tune only necessary hyperparameters

---

## 8) Decisions to confirm with you before coding

1. During training, should we always mask exactly one modality per sample (`k=1`) or allow a probability distribution over `k`?
2. For evaluation, do you want only "one-modality-missing" test, or also fixed-modality test (missing channel c only) to show per-modality robustness?
3. Should we preserve existing PSW-I preprocessing exactly (including scaler behavior), or switch to split-wise scaler fitted on train only for stricter protocol?

Once you confirm these three points, we can start implementation.
