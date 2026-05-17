# AIREADIModalityImputationDataset `__getitem__` Return Reference (EN + 中文)

This document explains every field returned by `AIREADIModalityImputationDataset.__getitem__` in `utils/utils_dataset.py`.

本文档逐项说明 `utils/utils_dataset.py` 中 `AIREADIModalityImputationDataset.__getitem__` 的返回内容。

---

## 1) Return mode / 返回模式

### EN
If `return_dict=False`, `__getitem__` returns:
- `target`
- `modalities`

If `return_dict=True` (default), it returns one dictionary (`result`) described below.

### 中文
当 `return_dict=False` 时，`__getitem__` 返回：
- `target`
- `modalities`

当 `return_dict=True`（默认）时，返回一个字典（`result`），字段见下文。

---

## 2) Top-level fields in `result` / `result` 顶层字段

### `target`
- EN: Tensor target sequence for `target_modality`, usually shape `(max_events, 1)` when padded.
- 中文: 目标模态（`target_modality`）的目标序列张量；在 padding 模式下通常形状为 `(max_events, 1)`。

### `modalities`
- EN: A dict from modality name to packed modality data. Also may include daily aligned keys (e.g. `heart_rate_aligned_to_glucose`) in daily mode.
- 中文: 从模态名到“打包后模态数据”的字典；在 `daily` 模式下还可能包含按葡萄糖时间对齐的额外键（如 `heart_rate_aligned_to_glucose`）。

### `condition_modalities`
- EN: List of modalities used as condition/input, i.e. all selected modalities except `target_modality`.
- 中文: 条件输入模态列表，即所选模态中除 `target_modality` 之外的模态。

### `patient_id`
- EN: Patient identifier (e.g. `AIREADI-1001`).
- 中文: 病人 ID（例如 `AIREADI-1001`）。

### `anchor_modality`
- EN: The modality used to define window boundaries.
- 中文: 用于定义窗口边界的锚点模态。

### `target_modality`
- EN: The modality chosen as prediction/reconstruction target.
- 中文: 作为预测/重建目标的模态。

### `window_start_time_local`
- EN: Window start local timestamp in nanoseconds (`torch.long`).
- 中文: 窗口起始本地时间戳（纳秒，`torch.long`）。

### `window_end_time_local`
- EN: Window end local timestamp in nanoseconds (`torch.long`).
- 中文: 窗口结束本地时间戳（纳秒，`torch.long`）。

### `anchor_start_index`
- EN: Start index on anchor sequence for this sample (for daily windows, uses stored tuple start index).
- 中文: 当前样本在锚点序列中的起始索引（daily 窗口时使用内部记录的起始索引）。

### `label`
- EN: Placeholder class label (`0`), kept for compatibility with training pipelines.
- 中文: 占位标签（固定为 `0`），用于兼容部分训练流程。

---

## 3) Structure of `modalities[modality]` / `modalities[modality]` 子结构

For each base modality (`glucose`, `calorie`, etc.), each item is a dict with:

对每个基础模态（如 `glucose`、`calorie` 等），其值是一个包含以下键的字典：

### `values`
- EN: Event values tensor. Padded shape `(max_events, 1)` if `pad=True`; otherwise `(num_events, 1)`.
- 中文: 事件数值张量。`pad=True` 时为 `(max_events, 1)`；否则为 `(num_events, 1)`。

### `time_local`
- EN: Event local start timestamps (`torch.long`, ns).
- 中文: 事件本地开始时间戳（`torch.long`，纳秒）。

### `time_end_local`
- EN: Event local end timestamps (`torch.long`, ns).
- 中文: 事件本地结束时间戳（`torch.long`，纳秒）。

### `relative_time_hours`
- EN: Relative time in hours from `window_start_time_local`.
- 中文: 相对窗口起点的小时偏移。

### `mask`
- EN: Observation mask (`1` for valid event slot, `0` for padded slot).
- 中文: 观测掩码（有效事件位置为 `1`，padding 位置为 `0`）。

### `length`
- EN: Number of valid events after truncation/padding logic.
- 中文: 经过截断/填充后保留下来的有效事件数。

### `raw_length`
- EN: Original number of events in the time window before truncation.
- 中文: 截断前窗口内原始事件数。

### `truncated`
- EN: Whether this modality was truncated due to `max_events_per_modality`.
- 中文: 是否因为 `max_events_per_modality` 被截断。

### `present`
- EN: Whether any event exists in this window (`raw_length > 0`).
- 中文: 当前窗口内该模态是否有任何事件（`raw_length > 0`）。

---

## 4) Daily aligned modality packs (`*_aligned_to_glucose`) / Daily 对齐模态

Only appears when:
- `window_mode == "daily"`
- `glucose` is selected
- and source modality exists in selected modalities

仅在以下条件出现：
- `window_mode == "daily"`
- 已选择 `glucose`
- 且对应源模态也被选择

Possible keys:
- `calorie_aligned_to_glucose`
- `heart_rate_aligned_to_glucose`
- `respiratory_rate_aligned_to_glucose`
- `physical_activity_aligned_to_glucose`

Each aligned pack contains:
- `values`, `time_local`, `relative_time_hours`, `mask`, `length`, `raw_length`, `present`, `fill_value`

Meaning:
- EN: For each glucose timestamp in the day window, source events near that glucose time are averaged; if none exists, use `fill_value` (default `0.0`).
- 中文: 对 daily 窗口内每个葡萄糖时间点，查找邻近的源模态事件并取均值；若没有匹配事件，则填 `fill_value`（默认 `0.0`）。

---

## 5) Optional clinical static fields / 可选临床静态特征字段

Appears only if `include_clinical_static=True`.

仅在 `include_clinical_static=True` 时出现。

### `clinical_static`
- EN: Static clinical feature vector for this patient.
- 中文: 该病人的静态临床特征向量。

### `clinical_mask`
- EN: Observation mask for each clinical feature (`1` observed, `0` missing/imputed).
- 中文: 临床特征观测掩码（观测到为 `1`，缺失/补值为 `0`）。

---

## 6) Optional study-group fields / 可选研究分组字段

Appears only if `include_study_group=True`.

仅在 `include_study_group=True` 时出现。

### `study_group_label`
- EN: Integer class id for study group. `-1` if unavailable.
- 中文: 研究分组整型标签；不可用时为 `-1`。

### `study_group_one_hot`
- EN: One-hot vector for study group class.
- 中文: 研究分组 one-hot 向量。

### `study_group`
- EN: Raw study-group string.
- 中文: 原始研究分组字符串。

---

## 7) Optional clinical-site fields / 可选临床中心字段

Appears only if `include_clinical_site=True`.

仅在 `include_clinical_site=True` 时出现。

### `clinical_site_label`
- EN: Compact site label (e.g., mapped to 0/1/2).
- 中文: 紧凑中心标签（例如映射到 0/1/2）。

### `clinical_site_code`
- EN: Original site code derived from patient id prefix.
- 中文: 从病人 ID 前缀推导出的原始中心编码。

### `clinical_site_one_hot`
- EN: One-hot vector for clinical site.
- 中文: 临床中心 one-hot 向量。

### `clinical_site`
- EN: Raw clinical-site string from participant metadata.
- 中文: 来自参与者元数据的原始中心字符串。

---

## 8) Optional CGM enhanced fields / 可选 CGM 增强特征字段

Appears only if `include_cgm_enhanced_features=True`.

仅在 `include_cgm_enhanced_features=True` 时出现。

### `cgm_enhanced_numeric_values`
- EN: Float tensor of selected numeric CGM-enhanced features.
- 中文: 所选数值型 CGM 增强特征张量（浮点）。

### `cgm_enhanced_numeric_mask`
- EN: Numeric feature mask (`1` observed, `0` missing).
- 中文: 数值特征掩码（观测到为 `1`，缺失为 `0`）。

### `cgm_enhanced_binary_values`
- EN: Float tensor for selected binary/self-report features.
- 中文: 所选二值/自报特征张量（浮点形式）。

### `cgm_enhanced_binary_mask`
- EN: Binary feature mask (`1` observed, `0` missing).
- 中文: 二值特征掩码（观测到为 `1`，缺失为 `0`）。

### `cgm_enhanced_medication_codes`
- EN: Integer-encoded medication-frequency category ids.
- 中文: 用药频率类别的整数编码。

### `cgm_enhanced_medication_mask`
- EN: Medication category mask (`1` observed label, `0` missing).
- 中文: 用药类别掩码（有标签为 `1`，缺失为 `0`）。

### `cgm_enhanced_medication_raw`
- EN: Raw string labels for each medication-frequency feature.
- 中文: 每个用药频率字段的原始字符串标签。

### `cgm_enhanced_numeric_feature_names`
- EN: Ordered feature names for `cgm_enhanced_numeric_values`.
- 中文: `cgm_enhanced_numeric_values` 的有序列名。

### `cgm_enhanced_binary_feature_names`
- EN: Ordered feature names for `cgm_enhanced_binary_values`.
- 中文: `cgm_enhanced_binary_values` 的有序列名。

### `cgm_enhanced_medication_feature_names`
- EN: Ordered feature names for `cgm_enhanced_medication_codes`.
- 中文: `cgm_enhanced_medication_codes` 的有序列名。

### `cgm_enhanced_medication_vocab`
- EN: Per-column vocabulary mapping `{raw_label_string -> integer_code}`.
- 中文: 每个用药字段的词表映射 `{原始字符串 -> 整数编码}`。

---

## 9) Notes on missing values / 缺失值说明

### EN
For CGM-enhanced features, if a patient has no row in the enhanced CSV, the dataset returns default zeros with masks set to 0 (and raw medication labels as `"__MISSING__"`).

### 中文
对于 CGM 增强特征，如果该病人在增强 CSV 中没有记录，数据集会返回全 0 默认值，并把对应 mask 设为 0（用药原始标签为 `"__MISSING__"`）。

