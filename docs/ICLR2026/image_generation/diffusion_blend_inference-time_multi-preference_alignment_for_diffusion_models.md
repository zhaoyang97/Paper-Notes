# Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models

**会议**: ICLR 2026  
**arXiv**: [2505.18547](https://arxiv.org/abs/2505.18547)  
**代码**: 有（GitHub）  
**领域**: 扩散模型 / 多目标对齐  
**关键词**: multi-preference alignment, inference-time, backward SDE blending, KL regularization control, Pareto-optimal  

## 一句话总结
提出 Diffusion Blend，通过在推理时混合多个奖励微调模型的反向扩散过程来实现多偏好对齐：DB-MPA 支持任意奖励线性组合、DB-KLA 支持动态 KL 正则化控制、DB-MPA-LS 通过随机 LoRA 采样消除推理开销，理论上证明了混合近似的误差界并在实验中接近 MORL oracle 上界。

## 研究背景与动机

1. **领域现状**：RL 微调扩散模型通常固定单一奖励函数和 KL 正则化权重 $\alpha$。微调完成后，模型锁定在特定的 $(r, \alpha)$ 配置上，无法适应不同用户偏好。

2. **现有痛点**：(a) 用户可能要求不同的美学/语义一致性/人类偏好权衡，需要为每个偏好组合都微调一个模型（开销巨大）；(b) KL 正则化太弱导致 reward hacking，太强导致对齐不足，最优值需要 grid search；(c) Rewarded Soup（权重空间线性组合）过于粗糙，guidance 方法需要可微奖励且计算量大。

3. **核心矛盾**：部署后偏好的灵活性 vs 微调的固定性。如何在不重新训练的情况下在推理时适应任意偏好组合？

4. **本文要解决什么？** 给定 $m$ 个基础奖励函数各自微调的模型，在推理时按用户指定权重 $w$ 生成 $r(w) = \sum w_i r_i$ 对齐的图像，且支持动态调整 KL 强度。

5. **切入角度**：从扩散模型的反向 SDE 角度出发，证明对齐模型的漂移项 $f^{(r,\alpha)}$ 可以表示为预训练漂移 + 控制项，通过 Jensen gap 近似将控制项线性化，从而实现反向 SDE 的线性混合。

6. **核心 idea 一句话**：奖励对齐的扩散模型的反向 SDE 漂移项可以线性组合近似任意奖励线性组合的对齐效果。

## 方法详解

### 整体框架

分两阶段：
- **微调阶段**：为每个基础奖励 $r_i$ 独立 RL 微调，获得 $m$ 个微调模型 $\theta_i^{\text{rl}}$
- **推理阶段**：用户指定权重 $w$，将 $m$ 个模型的反向 SDE 漂移按 $w$ 加权混合

### 关键设计

1. **Proposition 1: 对齐模型的 SDE 分解**:
   - 做什么：将对齐模型的反向漂移分解为预训练漂移 + 控制项
   - 核心思路：$f^{(r,\alpha)}(x_t, t) = f^{\text{pre}}(x_t, t) - \beta(t) u^{(r,\alpha)}(x_t, t)$，其中 $u^{(r,\alpha)} = \nabla_{x_t} \log \mathbb{E}_{x_0 \sim p_{0|t}^{\text{pre}}}[\exp(r(x_0)/\alpha)]$
   - 设计动机：将对齐效果隔离到控制项 $u^{(r,\alpha)}$ 中，为后续线性组合奠定基础

2. **Jensen Gap 近似 + 线性化 (Lemma 2)**:
   - 做什么：将 $\log \mathbb{E}[\exp(\cdot)]$ 近似为 $\mathbb{E}[\cdot]$（交换 log-exp 和 expectation 的顺序）
   - 核心思路：$u^{(r,\alpha)} \approx \bar{u}^{(r,\alpha)} = \nabla_x \mathbb{E}[r(x_0)/\alpha]$。利用期望的线性性，对线性奖励 $r(w) = \sum w_i r_i$，有 $f^{(r(w),\alpha)} \approx \sum w_i f^{(r_i, \alpha)}$
   - 设计动机：Jensen gap 在扩散模型中被广泛使用（如 DPS/RGG），近似误差在 $t \to 0$ 时趋近于 0

3. **DB-MPA（多偏好对齐）**:
   - 做什么：推理时按用户权重 $w$ 混合各奖励微调模型的反向 SDE
   - 核心思路：每步去噪时计算 $\hat{\epsilon}_t = \sum w_i \epsilon_{\theta_i^{\text{rl}}}(x_t, t)$

4. **DB-KLA（KL 对齐控制）**:
   - 做什么：推理时调整 KL 正则化强度
   - 核心思路：$f^{(r, \alpha/\lambda)} \approx (1-\lambda) f^{\text{pre}} + \lambda f^{(r,\alpha)}$，混合预训练和微调模型

5. **DB-MPA-LS（无额外开销的近似）**:
   - 做什么：消除 DB-MPA 的 $m \times$ 推理开销
   - 核心思路：每个去噪步按权重 $w$ 随机采样一个 LoRA adapter（Bernoulli/Categorical 采样），而非全部评估。理论证明（Proposition 2）混合 SDE 和随机采样 SDE 具有相同的边际分布
   - 设计动机：利用扩散过程的随机性——噪声注入使得逐步采样的统计效果等价于加权平均

### 损失函数 / 训练策略

- 使用 DPOK 算法对 SD v1.5 做 RL 微调
- 每个基础奖励独立微调
- 推理时无需训练，仅修改去噪步的噪声预测

## 实验关键数据

### 主实验（SD v1.5, ImageReward + VILA/PickScore）

DB-MPA 在 Pareto 前沿上全面优于 Rewarded Soup (RS)、CoDe、RGG，接近 MORL oracle 上界。

关键数值特征：DB-MPA 在 $w=0.5$ 时两个奖励都接近各自独立微调模型的 85-90% 性能水平，而 RS 只达到 60-70%。

### 消融实验

| 方法 | 推理开销 | 性能 (vs MORL) |
|------|---------|--------------|
| DB-MPA | $m \times$ | ~95% of MORL |
| DB-MPA-LS | $1 \times$ | ~90% of MORL |
| RS | $1 \times$ | ~70% of MORL |
| RGG | $1 \times$ (+ gradient) | ~60% of MORL |
| CoDe | $N \times$ (search) | ~65% of MORL |

DB-KLA 可以平滑控制 KL：$\lambda > 1$ 增强对齐但可能过拟合，$\lambda < 1$ 保守但保留预训练质量。

### 关键发现
- DB-MPA 在 Pareto 前沿上显著优于 RS（权重空间混合），说明**反向 SDE 混合优于参数空间混合**
- DB-MPA-LS 随机 LoRA 采样近似几乎不损失性能（~5% 差距），但推理开销降至 1×
- DB-KLA 提供了比重新微调更灵活的 KL 控制方式
- Jensen gap 近似在 JPEG compressibility（与 aesthetics 对抗的奖励）上也有效

## 亮点与洞察
- **SDE 混合 vs 参数混合** 的对比清晰有力——Rewarded Soup 在参数空间线性化，DB-MPA 在 SDE 漂移空间线性化，后者更有理论基础且性能更好。核心原因是 SDE 漂移的线性化近似误差有界（Lemma 1），而参数空间的线性化没有类似保证。
- **Proposition 2 (随机 LoRA 采样等价性)** 是一个优雅的理论结果——利用 SDE 的噪声注入使得逐步随机选择等价于加权平均。这在 LLM 中不可能做到（离散 token 空间），是扩散模型独有的优势。
- 推理时灵活性极高：用户可以用滑动条实时调整 aesthetics vs alignment 的 trade-off。

## 局限性 / 可改进方向
- Jensen gap 近似在 $\alpha$ 很小时误差增大（Lemma 1 的 $L_{t,2}$ 项），无法处理极端对齐需求
- 仅在 SD v1.5 上验证，更大模型（SDXL/Flux）的可行性未测试
- DB-MPA 的推理开销为 $m \times$，奖励函数多时不实用（但 DB-MPA-LS 可缓解）
- 线性奖励组合假设限制了表达能力，非线性偏好关系无法处理
- 没有与 DAV/DenseGRPO 等最新对齐方法的对比

## 相关工作与启发
- **vs Rewarded Soup**: 参数空间线性化 vs SDE 漂移线性化。DB-MPA 理论更严谨且性能更好。
- **vs Guidance (RGG/CoDe)**: 不需要可微奖励，不需要推理时搜索，且性能更优。
- **vs LLM DeRa**: 灵感来源相同（混合对齐和基础模型），但针对扩散模型做了 SDE 理论分析和随机 LoRA 采样创新。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ SDE 混合的理论框架新颖，随机 LoRA 采样等价性是优雅的贡献
- 实验充分度: ⭐⭐⭐⭐ 多种奖励组合、Pareto 分析、KL 控制实验全面，但仅 SD v1.5
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，图表直观，动机-理论-算法-实验逻辑紧凑
- 价值: ⭐⭐⭐⭐⭐ 为扩散模型的多偏好部署提供了实用且理论扎实的解决方案
