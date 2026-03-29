# ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation

**会议**: CVPR 2026  
**arXiv**: [2603.02945](https://arxiv.org/abs/2603.02945)  
**代码**: 无  
**领域**: LLM效率  
**关键词**: 模型合并, 数据无关, 协方差估计, 谱精炼, 闭式解

## 一句话总结
本文从理论上证明了微调参数差蕴含输入协方差信息，据此提出 ACE-Merging，通过自适应协方差估计、集体结构先验和谱精炼三步实现无数据闭式模型合并，在 GPT-2 上比之前方法平均提升 4%，在 RoBERTa-Base 上提升 5%。

## 研究背景与动机

1. **领域现状**：预训练+微调产生大量任务专用模型，模型合并(Model Merging)旨在将多个专家模型融合为一个统一模型，避免昂贵的多任务重训。现有方法分三类：数据依赖(需原始数据)、测试时自适应(推理开销大)、数据无关(最灵活)。
2. **现有痛点**：数据无关方法最具实用价值，但从 Task Arithmetic 到 TIES-Merging 等都只是参数空间的启发式操作（符号对齐、剪枝等），只处理干扰的"症状"而未触及根本原因——任务数据分布的统计结构差异。
3. **核心矛盾**：最优合并公式 $\bar{W} = (\sum_t W_t \Sigma_t)(\sum_t \Sigma_t)^{-1}$ 需要每个任务的输入协方差 $\Sigma_t$，但数据无关设定下恰恰无法获取这些统计量。
4. **本文要解决什么？** 如何在完全不访问数据的情况下，准确估计每个任务的输入协方差，从而实现有理论保障的最优合并。
5. **切入角度**：作者发现微调产生的权重位移 $\Delta W_t$ 的行之间隐含了输入协方差信息——将 $\Delta W_t$ 的行视为独立样本，其经验协方差正比于 $\Sigma_t$。
6. **核心idea一句话**：微调参数差本身就编码了输入协方差，无需任何数据即可估计并构造理论最优的闭式合并解。

## 方法详解

### 整体框架
输入：预训练模型权重 $W_0$ + 多个微调专家权重 $\{W_t\}$。输出：合并模型 $\bar{W}$。方法逐层独立运行，分三个阶段：(1) 自适应协方差归一化，(2) 集体结构先验构建，(3) 谱精炼。最终产出闭式解，无需迭代优化。

### 关键设计

1. **从参数差估计输入协方差（理论核心）**:
   - 做什么：不访问数据即可获得每个任务的输入协方差估计
   - 核心思路：Theorem 1 证明了 $\Sigma_t \propto \text{Cov}_{\mathcal{D}_t}[\Delta W_t]$。由于微调更新量小，梯度可在 $W_0$ 处线性化，$\Delta W_t \approx -2\eta N_t \mathbb{E}[(W_0 x - y)x^\top]$，因此权重位移隐式编码了输入二阶矩。实际操作中将 $\Delta W_t$ 的行当作独立样本，计算经验协方差 $\hat{\Sigma}_t \propto (\Delta W_t - \mathbf{1}\mu_t^\top)^\top (\Delta W_t - \mathbf{1}\mu_t^\top)$
   - 设计动机：这是 ACE-Merging 的理论根基，将数据无关合并问题转化为有显式目标的优化问题。此前 WUDI-Merging 虽然隐式使用了类似 proxy $\hat{\Sigma}_t \propto \|\Delta W_t\|_F^{-2} (\Delta W_t)^\top \Delta W_t$，但依赖迭代梯度下降，不稳定

2. **自适应协方差归一化 (Adaptive Covariance Normalization)**:
   - 做什么：平衡不同任务间的能量尺度差异
   - 核心思路：通过异质性度量 $\gamma = \frac{\text{Var}_t[\log\|\Delta W_t\|_F^2]}{(\mathbb{E}_t[\log\|\Delta W_t\|_F^2])^2}$ 检测任务间尺度差异。当 $\gamma > \tau$ 时，先对协方差做 trace 归一化 $\hat{\Sigma}_{t,\text{scaled}} = \hat{\Sigma}_t / \text{Tr}(\hat{\Sigma}_t)$，再施加自适应 Tikhonov 正则 $\hat{\Sigma}_{t,\text{reg}} = \hat{\Sigma}_{t,\text{scaled}} + \frac{\epsilon}{\text{Tr}(\hat{\Sigma}_t)} I$
   - 设计动机：实验发现 RoBERTa 的任务异质性($\gamma > 0.3$)远大于 ViT($\gamma < 0.25$)；不归一化会导致高能量任务主导合并结果。$\gamma$ 作为开关门控，避免对齐次任务做不必要的归一化

3. **集体结构先验 (Collective Structural Prior, CSP)**:
   - 做什么：引入各向异性的正则化，捕捉跨任务共享的特征几何结构
   - 核心思路：$\mathbf{C}_{\text{agg}} = \mathbf{1} \cdot (\frac{1}{d_{\text{in}}} \mathbf{1}^\top \sum_t \hat{\Sigma}_{t,\text{scaled}})$，将所有任务缩放协方差的列均值广播到每一行，形成低秩共识先验。最终闭式解为 $\bar{W}_{\text{pre}} = (\sum_t W_t \hat{\Sigma}_{t,\text{reg}})(\sum_t \hat{\Sigma}_{t,\text{reg}} + \mathbf{C}_{\text{agg}})^{-1}$
   - 设计动机：标准 $\epsilon I$ 正则是各向同性的，对所有特征维度一视同仁，忽略了输入空间的内在几何。CSP 用跨任务共识的能量分布做权重，选择性增强共享重要维度

4. **谱精炼 (Spectral Refinement)**:
   - 做什么：修正闭式解中严重的谱病态问题
   - 核心思路：先计算结构残差 $\Delta_{\text{res}} = \sum_t W_t (\hat{\Sigma}_{t,\text{scaled}} - \bar{\Sigma})$，与 $\bar{W}_{\text{pre}}$ 融合后做 SVD，对 top-$k$ 奇异方向用其均值奇异值 $\sigma_{\text{iso}}$ 重新加权：$\Delta W_{\text{refine}} = \sigma_{\text{iso}} \mathbf{U}_{:,1:k} \mathbf{V}_{:,1:k}^\top$，最终 $\bar{W} = \bar{W}_{\text{pre}} + \Delta W_{\text{refine}}$
   - 设计动机：实验观察到 $\bar{W}_{\text{pre}}$ 的 top 5% 奇异值就占 99% 以上能量，条件数 >$8.7 \times 10^5$，但主方向是正确的（与最终解余弦相似度≈1）。因此只需保留方向、重新分配能量即可

### 损失函数 / 训练策略
ACE-Merging 是纯闭式方法，不涉及任何训练/优化迭代。超参固定为 $\tau=0.3$, $k_{\text{frac}}=0.3$，$\epsilon$ 按模型族调整（GPT-2: $4\times 10^{-2}$, RoBERTa-Base: $2\times 10^{-4}$, 其余: $1\times 10^{-5}$）。

## 实验关键数据

### 主实验

**视觉任务 (ViT-B/16, 平均绝对准确率 %)**

| 任务数 | Weight Avg | Task Arithmetic | CART | TSV-M | ACE-Merging | 提升 |
|--------|-----------|----------------|------|-------|-------------|------|
| 8 tasks | 72.2 | 75.4 | 88.3 | 89.0 | **90.6** | +1.6 |
| 14 tasks | 69.5 | 70.5 | 84.1 | 84.6 | **86.1** | +1.5 |
| 20 tasks | 65.3 | 65.8 | 80.5 | 80.6 | **82.1** | +1.5 |

**语言任务 (GPT-2, GLUE Avg %)**

| 方法 | CoLA | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | Avg |
|------|------|------|------|------|-----|-----|-------|-----|
| Task Arithmetic | 68.7 | 68.6 | 69.6 | 70.5 | 81.8 | 47.3 | 83.6 | 70.0 |
| TSV-M | 65.6 | 75.4 | 58.6 | 64.4 | 86.2 | 55.6 | 85.7 | 70.2 |
| **ACE-Merging** | 70.3 | 69.9 | 71.8 | 76.7 | 79.0 | 62.5 | 88.5 | **74.1** |

### 消融实验

| 配置 | RoBERTa-L | GPT-2 | ViT-B/16 (8tasks) | 说明 |
|------|-----------|-------|-------------------|------|
| E1: Basic Closed-form | 80.05 | 68.72 | 89.91 | 仅闭式解 |
| E2: + Adaptive ε | 88.04 | 71.50 | 89.91 | 自适应正则贡献最大 |
| E3: + Aggregate Prior | 86.79 | 71.51 | 90.60 | 结构先验辅助 |
| E4: + Spectral Refinement | **91.68** | **74.09** | **90.60** | 谱精炼最终提升 |

### 关键发现
- 自适应正则化贡献最大——在 RoBERTa-L 上从 E1 到 E2 提升了近 8 个百分点，说明任务异质性平衡是核心瓶颈
- ViT 因任务异质性低($\gamma < 0.3$)自动跳过自适应和谱精炼阶段（E1≈E2, E3≈E4），验证了 $\gamma$ 门控机制的合理性
- 超参敏感性分析显示 $\gamma \in [0.1, 0.3]$, $k_{\text{frac}} \in [0.1, 0.5]$ 范围内性能稳定，$\epsilon$ 更敏感
- RoBERTa-Base 上 ACE-Merging (90.4%) 大幅超越 WUDI-Merging (85.3%)，在 RoBERTa-Large 上也保持 ~3% 优势

## 亮点与洞察
- **理论洞察极为优雅**：将数据无关合并的根本障碍（缺少协方差）转化为可从参数差直接估计的量，建立了"微调权重差 ↔ 输入协方差"的形式化联系。这一视角不仅解释了 ACE-Merging 为何有效，还统一解释了先前方法——Weight Averaging 假设 $\Sigma_t = kI$，WUDI-Merging 隐式用 $\|\Delta W_t\|_F^{-2} (\Delta W_t)^\top \Delta W_t$ 作代理
- **闭式解 vs 迭代解**：WUDI-Merging 需要梯度下降迭代，而 ACE-Merging 是真正的 closed-form，计算效率高且稳定性更好
- **谱精炼的观察很巧妙**：发现初始闭式解方向正确但能量分布极端失衡（top 5% 奇异值占 99% 能量），因此只需保留方向重分配能量。这个"方向正确、幅度错误"的诊断思路可迁移到其他矩阵优化问题

## 局限性 / 可改进方向
- $\epsilon$ 需要按模型族手动设定（GPT-2、RoBERTa、ViT 各不同），作者也承认自动估计 $\epsilon$ 是未来方向
- 线性近似 $f(W,x) \approx Wx$ 在深层非线性网络中可能不够精确，特别是对注意力层
- "将 $\Delta W_t$ 的行视为独立样本"这一假设在实际中未必成立——权重矩阵的行间存在结构性相关
- 仅在 GLUE 和视觉分类任务上验证，未测试生成任务（如 LLM 对话、代码生成）
- 合并时逐层独立操作，忽略了跨层间的依赖关系

## 相关工作与启发
- **vs WUDI-Merging**: 本文在理论框架下将 WUDI 重新解释为 ACE 的特例（norm-weighted 协方差代理），且 ACE 用闭式解替代 WUDI 的迭代优化，更稳定高效
- **vs TSV-M**: TSV-M 用 SVD 分解共享/任务特定子空间，属于启发式；ACE 直接建模协方差，理论基础更扎实
- **vs RegMean**: RegMean 是数据依赖方法直接用真实协方差，ACE 证明了无需数据也能估计协方差并达到可比性能
- 协方差估计+谱修正的思路可推广到联邦学习中的模型聚合问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 理论贡献优雅但核心 insight（参数差 ∝ 协方差）并非完全意外
- 实验充分度: ⭐⭐⭐⭐⭐ 视觉+语言、多架构多尺度、完整消融和敏感性分析
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑链清晰，理论→统一框架→方法→实验层层推进
- 价值: ⭐⭐⭐⭐ 数据无关合并的实用工具，但 $\epsilon$ 需手调降低了开箱即用性
