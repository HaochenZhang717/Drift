# TG-MSFM 复现计划（基于 `TG-MSFM_paper.pdf`）

## 1. 目标与范围
- 目标：在现有仓库中复现 TG-MSFM 的确定性时间序列插补流程，覆盖论文主结果所需的训练与推理路径。
- 范围：
  - 模型：Time-aware Masked Transformer + Time-Gated Multi-Scale Velocity Heads。
  - 训练：Gap-only Flow Matching（可选正则）。
  - 推理：Heun ODE + 每步 Data Consistency (DC) 投影。
  - 评估：缺失位置上的 MSE/MAE；missing ratio {0.1,0.3,0.5,0.7}，多随机种子平均。

## 2. 从论文抽出的“可直接实现”规范

### 2.1 输入与结构化端点
- 原始输入：
  - `x in R^{T x D}`，`M in {0,1}^{T x D}`。
- 结构化端点：
  - `x_tilde = [x ⊙ M, m, x_L, x_R] in R^{T x (D+3)}`。
  - `m_t = 1` 当且仅当该时刻任一变量可观测 (`exists d, M[t,d]=1`)。
  - `x_L, x_R` 为观测点的左右局部上下文移动平均，窗口 `w=10`。
- 通道集合：
  - 数据通道 `D={1..D}`。
  - 条件通道 `C`（3 个附加通道）。

### 2.2 主干网络（Backbone）
- Time-aware Transformer，输入 `(z_t, t, x_tilde)`，输出共享表示 `h`。
- 注意力可见性掩码：query 只能 attend 到 `m_t=1` 的 key（按时间维）。
- `t` 使用 sinusoidal/timestep embedding 注入。
- 条件通道随主干传播，但：
  - 不参与 gap-only 监督。
  - 在推理 DC 中视为已知并被钳制。

### 2.3 Flow Matching 训练目标
- 采样：`z0 ~ N(0,I)`，`z1 = x_tilde`，`t ~ U[0,1]`。
- 线性桥：`z_t = (1-t)z0 + t z1`。
- 教师速度：`v* = z1 - z0`（常数速度）。
- 缺失监督索引：`Omega = {(i,d) | M[i,d]=0, d in data channels}`。
- 主损失（gap-only）：
  - 在 `Omega` 上最小化 `||v_theta(z_t,t;x_tilde) - v*||^2`。
- 可选正则（小权重）：
  - 一阶时间差分 (`RTV1`)。
  - 二阶时间差分 (`RTV2`)。
  - 高频惩罚 (`RHP`, 例如核 `[1,-2,1]`)。

### 2.4 多尺度速度头 + 时间门控
- 金字塔尺度：`S={1,2,4}`。
- 每个尺度：`Down_s(h) -> Head_s -> Up_s` 产生候选速度 `u_tilde^(s)`。
- 门控：`alpha(t)=softmax(MLP(t))`。
- 融合：`v_theta = sum_s alpha_s(t) * phi(u_tilde^(s))`，`phi` 为逐元素 squash（文中用 `tanh`）。
- 最细尺度可加 AntiAlias1D 低通（3-5 taps，默认 5 taps，单位 DC 增益）。

### 2.5 确定性推理（Heun + DC）
- ODE 步数：默认 `N=300`。
- 可选时间扭曲：`t_eff(t)=t^k`，默认 `k=1.5`。
- Heun（每步两次函数评估）：
  - `k1 = v_theta(z_n, t_n)`
  - `z_hat = z_n + dt*k1`
  - `k2 = v_theta(z_hat, t_n+dt)`
  - `z_ode = z_n + dt/2 * (k1+k2)`
- 已知索引集：
  - `K = {(i,d) | M[i,d]=1} union C`。
- DC 投影：
  - 在 `K` 上强制与线性桥一致：`z[K] = (1-t_eff) z0[K] + t_eff z1[K]`。
  - 未知位置沿 `z_ode` 演化。

### 2.6 论文默认超参数（Appendix A.6 / Table A1）
- Transformer layers `L=6`
- Head dim `d_k=64`
- Attention heads `H=8`
- MLP ratio `4.0`
- Pyramid strides `{1,2,4}`
- Context window `w=10`
- Anti-alias taps `5`
- Time warp `k=1.5`
- ODE steps `N=300`
- Batch size `B=32`
- Optimizer `AdamW`
- Weight decay `1e-4`
- Peak LR `2e-4`
- Warmup `5k`
- 训练总步数：`200k~400k` + 验证早停

## 3. 实施计划（可执行）

### 阶段 A：代码骨架与配置
- 新建模块：
  - `data/`：数据加载、标准化、mask 生成（random missing + central gap）。
  - `models/`：masked transformer、multi-scale heads、gate、anti-alias。
  - `train/`：flow matching 训练循环、损失、日志。
  - `infer/`：Heun+DC 采样器。
  - `eval/`：MSE/MAE（仅缺失项）与多 seed 聚合。
  - `configs/`：默认超参数与数据集配置。
- 输出一键脚本：`train.py`, `infer.py`, `eval.py`。

### 阶段 B：数据与mask机制
- 按论文列的数据集接口统一成 `(x, M)` window 采样。
- 训练前做 per-channel standardization（基于 train split）。
- 实现两类 mask：
  - 随机缺失（比例 0.1/0.3/0.5/0.7）。
  - 中心连续缺失（鲁棒性测试）。
- 生成 `m, x_L, x_R` 并拼接出 `x_tilde`。

### 阶段 C：模型实现
- 实现 visibility-masked attention（按时间可见性 `m` 屏蔽 key）。
- 实现 time embedding 注入。
- 实现尺度分支 `Down -> Head -> Up` 与 `alpha(t)` 门控融合。
- 在 finest 分支后接可选固定低通 AntiAlias1D。

### 阶段 D：训练实现
- 采样 `t, z0`，构造 `z_t` 与 `v*`。
- 仅在 `Omega` 上计算 FM 主损失。
- 按配置决定是否加 `RTV1/RTV2/RHP`。
- AdamW + cosine + warmup + early stopping。
- 日志至少记录：训练损失、验证 MSE/MAE、每项正则值、梯度范数。

### 阶段 E：推理实现
- 实现 Heun step（N 步，支持 `t_eff`）。
- 每步后执行 DC：
  - 已知索引 `K` 强制回到桥上。
  - 未知索引保留 ODE 更新。
- 支持不同 `N` 的速度-质量曲线评估。

### 阶段 F：复现实验与对齐
- 主表：10 个数据集上 MSE/MAE（缺失项）平均结果。
- 消融：
  - 去 multi-scale。
  - 去 gate（静态混合）。
  - Euler 替代 Heun。
- 效率：MSE vs wall-clock / NFE。
- 鲁棒性：中心 gap 长度扫描。

## 4. 论文中缺失/不明确，需你后续补充的点

1. 代码与数据预处理细节未公开
- 论文未给官方仓库，且“标准预处理/官方切分”的具体实现脚本缺失。

2. `x_L, x_R` 的精确定义不充分
- 仅说明是“observed points 的左右移动平均，w=10”，但未明确：
  - 边界处理方式（padding/截断/忽略）。
  - 当窗口内全缺失时的回填策略。
  - 是按每个变量独立计算还是先跨变量聚合后成单通道。

3. 可见性 mask 的精确定义粒度
- 文中定义 `m_t = exists d`，但 attention 的掩码是“query 仅看 `m_t=1` key”。
- 未说明是否还叠加因果掩码、是否分头共享同一 mask。

4. Backbone 结构细节不足
- 未给完整 Transformer 细节（pre/post-LN、dropout、激活、FFN hidden 精确维度、位置编码实现）。

5. Multi-scale heads 结构模糊
- “e.g., Conv-GELU-Conv” 不是硬约束；卷积核大小、通道数、归一化、是否残差未定。

6. AntiAlias1D 细节不足
- 只给 taps 数范围与 unit DC gain，没有固定系数。

7. 正则项权重未给出具体数值
- 仅说 `lambda_1, lambda_2, lambda_HP << 1`，需要具体默认值。

8. 推理算法伪代码在 PDF 文本提取里存在歧义
- Algorithm B2 的第 9 行在提取文本里出现顺序冲突（先写桥值又写 `z_ode`），需要以数学定义为准：
  - 预期应为“先得 `z_ode`，再在 `K` 上覆盖为 bridge 值，其余位置取 `z_ode`”。

9. 训练采样策略未完全固定
- `t` 是 per-sample 还是 per-batch 文中都允许；`Omega` 是否子采样也给了“optional”。
- 这些会影响吞吐和方差，需要统一约定。

10. 评估协议尚有实现空间
- 文中说“5 seeds + ratios 平均”，但未给每个数据集窗口长度/步长、batching、是否 teacher-forcing 式窗口拼接等落地参数。

## 5. 建议的“先补全再开工”最小决策清单
- A. `x_L/x_R` 计算公式与边界策略。
- B. Transformer 精确结构（LN/dropout/FFN 维度/pos encoding）。
- C. Head_s 具体结构（kernel/channel/norm/residual）。
- D. AntiAlias1D 滤波器系数。
- E. 正则权重 (`lambda_1/lambda_2/lambda_HP`) 与是否默认开启。
- F. `t` 采样粒度与 `Omega` 子采样策略。
- G. 数据切分与窗口参数（每个数据集）。

---

如果你确认这份计划，我们下一步就按“阶段 A -> B -> C”直接落代码，并在每个阶段做最小可运行验证。
