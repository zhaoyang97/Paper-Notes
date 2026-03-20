# Aligning Text to Image in Diffusion Models is Easier Than You Think

**会议**: NeurIPS 2025  
**arXiv**: [2503.08250](https://arxiv.org/abs/2503.08250)  
**代码**: https://softrepa.github.io/ (项目页)  
**领域**: 扩散模型 / 文本-图像对齐  
**关键词**: SoftREPA, 对比学习, Soft Token, 文本图像对齐, 互信息

## 一句话总结
提出 SoftREPA——一种轻量级对比微调策略，通过引入可学习 soft text token（不到 1M 参数）在冻结的预训练 T2I 扩散模型上进行对比学习，显式提高文本和图像表征的互信息，在 SD1.5/SDXL/SD3 上显著提升文本-图像对齐质量，且适用于图像生成和图像编辑任务。

## 研究背景与动机

1. **领域现状**：文本到图像扩散模型（SD、FLUX 等）通过 cross-attention 或 self-attention 将文本条件融入图像生成，但文本-图像表征之间仍存在残余的不对齐——生成图像可能遗漏文本描述的关键属性、颜色、数量等。
2. **现有痛点**：现有改进方案如偏好优化（DPO）需要定制的人类偏好数据集且训练开销大。REPA 等表征对齐方法只对齐图像内部表征与外部视觉编码器，没有直接改善文本-图像对齐。
3. **核心矛盾**：标准 T2I 训练仅使用正样本对（matched image-text pairs）最小化去噪损失，从表征对齐角度看是次优的——缺少负样本对的对比信号来区分不同文本条件。
4. **本文要解决什么？** 如何用最小的额外参数和计算开销提升已训练好的 T2I 模型的文本-图像对齐？
5. **切入角度**：将扩散模型的去噪损失重新解释为 logit（条件似然），在此基础上构建 InfoNCE 风格的对比学习损失，结合 prompt tuning 中的 soft token 概念实现轻量微调。
6. **核心idea一句话**：把去噪损失当 logit 做对比学习 + 只训练 soft text token = 用不到 1M 参数显著提升文本-图像对齐。

## 方法详解

### 整体框架
冻结预训练 T2I 模型参数，在每层文本表征前拼接可学习的 soft token $\mathbf{s}^{(k,t)} \in \mathbb{R}^{m \times d}$（按层和时间步索引）。训练时用对比损失优化这些 soft token，推理时将训练好的 soft token 拼接到文本特征中即可。

### 关键设计

1. **对比 T2I 对齐损失 (Contrastive T2I Alignment Loss)**:
   - 做什么：利用同一 batch 内的正负文本-图像对构建 InfoNCE 风格损失
   - 核心思路：将去噪损失 $\|\epsilon_\theta(\mathbf{x}_t, t, \mathbf{y}) - \epsilon\|^2$ 的负值通过指数函数映射为 logit $\tilde{l}(\mathbf{x}, \mathbf{y}, \mathbf{s}) = e^{-\|v_\theta(\mathbf{x}_t, t, \mathbf{y}, \mathbf{s}) - (\epsilon - \mathbf{x}_0)\|^2 / \tau(t)}$，然后构建对比损失 $\mathcal{L} = -\log \frac{\exp(\tilde{l}(\mathbf{x}, \mathbf{y}, \mathbf{s}))}{\sum_j \exp(\tilde{l}(\mathbf{x}, \mathbf{y}^{(j)}, \mathbf{s}))}$
   - 设计动机：标准训练只有正对，无法区分不同文本条件的差异。对比损失通过负样本推开不匹配的文本-图像对，锐化条件概率分布。指数映射保证 logit 有界，避免训练不稳定

2. **可学习 Soft Token**:
   - 做什么：每层、每时间步引入可学习 token 拼接到文本表征前
   - 核心思路：$\hat{\mathbf{H}}_{\text{text}}^{(k-1,t)} = [\mathbf{s}^{(k,t)}; \mathbf{H}_{\text{text}}^{(k-1,t)}]$，soft token 通过 Embedding(k,t) 生成，与注意力层一起处理。只训练这些 token（<1M 参数），冻结模型其余部分
   - 设计动机：类似于 prompt tuning 的思想——不改变模型权重，通过在输入空间注入可学习信号来调整模型行为。极低的参数量意味着快速训练、几乎无推理开销

3. **互信息理论分析**:
   - 做什么：证明最小化对比损失等价于最大化文本-图像表征的互信息
   - 核心思路：利用 Song & Kong 等人的结论——扩散模型的条件似然 $p_\theta(\mathbf{x}|\mathbf{y}) = \exp(\hat{l}(\mathbf{x}, \mathbf{y}))$。因此对比损失的 logit 近似于 PMI $i(\mathbf{x}, \mathbf{y}) = \log \frac{p_\theta(\mathbf{x}|\mathbf{y})}{p_\theta(\mathbf{x})}$，互信息 $I(X,Y) = \mathbb{E}[i(X,Y)]$。最小化对比损失即最大化互信息
   - 设计动机：提供理论保障，说明为什么这种简单方法能有效提升语义一致性

### 损失函数 / 训练策略
- 训练损失：$\mathcal{L}_{\text{SoftREPA}}(\mathbf{s})$，对比损失，仅优化 soft token 参数
- 单 Monte Carlo 样本近似期望（同 batch 共享 $\epsilon$ 和 $t$）
- 时间调度参数 $\tau(t)$ 控制温度
- 适用于 UNet 架构 (SD1.5, SDXL) 和 DiT 架构 (SD3)

## 实验关键数据

### 主实验 (COCO val5K + GenEval)

| 模型 | ImageReward ↑ | CLIP ↑ | HPS ↑ | FID ↓ | 额外参数 |
|------|-------------|--------|-------|-------|---------|
| SD1.5 | 17.72 | 26.40 | 25.08 | 24.59 | 0 |
| SD1.5 + SoftREPA | **32.89** | **27.33** | **25.18** | **23.43** | <1M |
| SDXL | 75.06 | 26.76 | 27.35 | 24.69 | 0 |
| SDXL + SoftREPA | **85.29** | **26.80** | **28.30** | **26.04** | <1M |
| SD3 | 94.27 | 26.30 | 28.09 | 31.59 | 0 |
| SD3 + SoftREPA | **108.5** | **26.91** | **28.91** | **36.21** | <1M |

### 消融实验 (GenEval on SD3)

| 配置 | Mean ↑ | 说明 |
|------|--------|------|
| SD3 + SoftREPA | 0.70 | 完整方法 |
| SD3 baseline | 0.68 | 无对比学习 |
| CaPO (偏好优化) | 0.71 | 需要偏好数据 |
| RankDPO | 0.74 | 需要偏好数据，全模型微调 |

### 图像编辑实验
在 PnP、MasaCtrl、FlowEdit 等编辑方法上，SoftREPA 一致提升编辑质量：
- FlowEdit + SoftREPA：ImageReward 87.70 → **102.24**，CLIP-Edited 23.19 → **23.60**

### 关键发现
- **SD1.5 上提升最为显著**：ImageReward 几乎翻倍（17.72 → 32.89），说明对小模型的对齐改善更大
- **FID 在 SD3 上略有上升**（31.59 → 36.21），说明更强的文本对齐可能与图像质量存在 trade-off
- **通用性强**：跨 UNet 和 DiT 架构、跨生成和编辑任务均有效
- **推理延迟几乎不增加**：SD1.5 从 1.526 → 1.547 sec/img，GPU 内存不变
- **对比学习的负样本至关重要**：仅用正样本（标准去噪训练）无法达到同等对齐效果

## 亮点与洞察
- **去噪损失 → 对比 logit 的视角转换极为巧妙**：这个观察打通了扩散模型训练和对比表征学习之间的桥梁。任何扩散模型都可以通过这种方式引入负样本信号而不改变模型架构
- **Soft token = 扩散模型的 prompt tuning**：类似于 NLP 中的 prompt tuning 思想，但应用于生成模型的文本条件通道。极低参数量（<1M）使得任何人都可以在消费级 GPU 上微调大型 T2I 模型的对齐质量
- **可直接叠加到任何现有方法上**：无论是 PnP 编辑还是 FlowEdit，SoftREPA 都是即插即用的，这种通用性非常实用

## 局限性 / 可改进方向
- **FID 在 SD3 上有上升**：更强的文本对齐可能牺牲了多样性/无条件图像质量，需要进一步分析 trade-off
- **GenEval 的 Counting 任务上表现下降**（0.56 → 0.29 on SD3），说明对比学习可能不擅长需要精确计数的语义
- **soft token 数量和层位置的影响**：论文只提到在 upper layers 使用，但具体的消融不够详细
- **仅在 SD 系列模型验证**：FLUX、DALL-E 3 等其他架构的效果未知
- **理论分析假设最优去噪器**：实际模型是次优的，互信息最大化的保证有多强取决于模型质量

## 相关工作与启发
- **vs REPA [Yu et al.]**: REPA 对齐图像内部表征与外部视觉编码器，SoftREPA 对齐文本-图像表征。两者互补，可以结合使用
- **vs CaPO/RankDPO (偏好优化)**: 偏好优化需要人工标注数据且全模型微调，SoftREPA 仅需原始 image-text 对和 <1M 参数
- **vs Classifier-Free Guidance**: CFG 在推理时通过放大条件分数来增强对齐，SoftREPA 在训练时通过对比学习改善对齐，两者可以叠加

## 评分
- 新颖性: ⭐⭐⭐⭐ 去噪损失作为对比 logit 的视角新颖，soft token 微调思路简洁有效
- 实验充分度: ⭐⭐⭐⭐ 跨模型（SD1.5/SDXL/SD3）、跨任务（生成/编辑），但缺少与更多对齐方法的对比
- 写作质量: ⭐⭐⭐⭐ 理论动机清晰，实验组织合理
- 价值: ⭐⭐⭐⭐ 低成本提升 T2I 对齐的实用方法，对社区有直接应用价值
