---
title: >-
  [论文解读] AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction
description: >-
  [ICCV 2025][图像生成][视频预测] 提出AID框架，将预训练的Image2Video扩散模型（SVD）迁移至文本引导视频预测任务，通过MLLM辅助的视频状态预测、双查询Transformer条件注入和时空适配器，在多个数据集上FVD指标超越前SOTA 50%以上。 文本引导视频预测（TVP）旨在根据初始帧和文本指…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "视频预测"
  - "扩散模型"
  - "文本引导"
  - "适配器"
---

# AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction

**会议**: ICCV 2025  
**arXiv**: [2406.06465](https://arxiv.org/abs/2406.06465)  
**代码**: [https://chenhsing.github.io/AID](https://chenhsing.github.io/AID)  
**领域**: 图像生成  
**关键词**: 视频预测, 扩散模型, 文本引导, Stable Video Diffusion, 适配器

## 一句话总结

提出AID框架，将预训练的Image2Video扩散模型（SVD）迁移至文本引导视频预测任务，通过MLLM辅助的视频状态预测、双查询Transformer条件注入和时空适配器，在多个数据集上FVD指标超越前SOTA 50%以上。

## 研究背景与动机

文本引导视频预测（TVP）旨在根据初始帧和文本指令预测未来视频帧，在VR、机器人操控和内容创作中有广泛应用。当前TVP方法面临两个核心挑战：

**领域数据稀缺**：特定领域（如机械臂操作、第一人称烹饪视频）的训练数据规模有限，直接训练文生图模型扩展的视频模型会导致帧间一致性差和时序稳定性不足

**SVD的不可控性**：Stable Video Diffusion等预训练大规模视频模型学到了优秀的视频动态先验，但仅支持图像到视频生成，缺乏文本控制能力

核心思路：迁移SVD的视频先验到TVP任务，同时注入文本控制能力——而非从头训练视频理解。这面临两个技术挑战：如何设计和注入文本条件？如何以低训练成本适配到目标场景？

## 方法详解

### 整体框架

AID建立在SVD之上，包含三大组件：（1）**MLLM辅助的视频预测提示**——用多模态大语言模型预测视频的未来发展状态；（2）**双查询Transformer（DQFormer）**——整合多模态条件信息；（3）**时空适配器**——以极少参数实现到目标数据集的高效迁移。训练时冻结SVD的VAE和3D UNet全部参数，只训练新增的DQFormer、适配器和交叉注意力层。

### 关键设计

1. **MLLM辅助的视频状态预测（Video Prediction Prompting）**:

    - 单句文本指令无法完整描述视频的时序动态变化
    - 将初始帧和文本指令输入多模态LLM（如LLaVA），让其预测视频的多个发展阶段
    - 例如"lifting up one end of a tablet box, then letting it drop down"→4个状态：初始状态→抬起一端→释放一端→最终状态
    - 设计动机：将视频的时序变化显式分解为离散状态描述，为后续帧级条件控制提供语义基础

2. **双查询Transformer（DQFormer）**:

    - **上支路（多模态嵌入）**：将文本指令特征 $\bm{t_1}$ 通过自注意力后与视觉特征 $\bm{v}$（CLIP Visual Encoder提取）做交叉注意力对齐
    $\text{multimodal emb} = \text{Softmax}\Big(\frac{(W_1^Q \text{SelfAttn}(\bm{t_1}))(W_1^K \bm{v})^T}{\sqrt{d_1}}\Big)(W_1^V \bm{v})$
    - **下支路（分解嵌入）**：设置可学习查询 $\bm{Q} \in \mathbb{R}^{(N \cdot N_t) \times C}$（$N$帧×$N_t$查询），先与指令特征 $\bm{t_1}$ 交叉注意力分解为帧级控制，再与MLLM状态特征 $\bm{t_2}$ 交叉注意力融合多状态信息
    - 两支路输出拼接为MCondition，通过交叉注意力注入UNet
    - 设计动机：上支路负责全局多模态对齐（理解"要做什么"），下支路负责帧级时序分解（理解"每帧该怎么变"）

3. **三种适配器（时空迁移）**:

    - **空间适配器**：添加在空间自注意力旁，由下采样线性层+GELU+上采样线性层组成，上采样层初始化为零（不破坏原始模型）
    $\text{S-Adapter}(\mathbf{X}) = \mathbf{X} + \mathbf{W}_{up}(\text{GELU}(\mathbf{W}_{down}(\mathbf{X})))$
    - **短期时序适配器**：在线性层间加入深度可分离3D卷积，建模相邻帧间的短期时序关系
    $\text{ST-Adapter}(\mathbf{X}) = \mathbf{X} + \mathbf{W}_{up}(\text{3D-Conv}(\mathbf{W}_{down}(\mathbf{X})))$
    - **长期时序适配器**：用时序自注意力替代卷积，建模全局时序依赖
    $\text{LT-Adapter}(\mathbf{X}) = \mathbf{X} + \mathbf{W}_{up}(\text{Self-Attn}(\mathbf{W}_{down}(\mathbf{X})))$
    - 设计动机：冻结原始UNet权重+轻量适配器 = 保留视频先验 + 低成本领域迁移 + 避免过拟合

### 推理策略

采用双重classifier-free guidance，分别控制帧条件和文本条件：
$$\tilde{e_\theta}(z_t, c_T, c_V) = e_\theta(z_t, \emptyset, \emptyset) + s_V \cdot (\ldots) + s_T \cdot (\ldots)$$

## 实验关键数据

### 主实验

| 方法 | 文本 | SSv2 FVD↓ | Bridge FVD↓ | Epic100 FVD↓ |
|------|------|-----------|-------------|--------------|
| SimVP | 无 | 537.2 | 681.6 | 1991 |
| PVDM | 无 | 502.4 | 490.4 | 482.3 |
| VideoFusion | 有 | 163.2 | 501.2 | 349.9 |
| Tune-A-Video | 有 | 291.4 | 515.7 | 365.0 |
| Seer | 有 | 112.9 | 246.3 | 271.4 |
| **AID (Ours)** | **有** | **50.23** | **21.57** | **52.78** |

FVD改善幅度：SSv2上55.5%，Bridge上**91.2%**，Epic100上80.6%。

### 消融实验

**条件组件消融（SSv2）**：

| 配置 | FVD↓ | KVD↓ | 说明 |
|------|------|------|------|
| 无MCondition | 152.4 | 0.14 | 无文本引导，仅靠初始帧预测 |
| 仅多模态嵌入 | 74.98 | 0.03 | 缺少帧级分解 |
| 仅分解嵌入 | 70.16 | 0.04 | 缺少多模态对齐 |
| 完整但无LLaVA | 64.48 | 0.03 | 缺少状态预测 |
| **AID完整** | **50.23** | **0.02** | 所有组件协同最优 |

**适配器消融（SSv2）**：

| 配置 | FVD↓ | KVD↓ | 说明 |
|------|------|------|------|
| 无适配器 | 279.42 | 0.71 | 仅训练DQFormer，无法迁移 |
| 无空间适配器 | 68.12 | 0.05 | 空间分布未适配 |
| 无时序适配器 | 76.32 | 0.03 | 时序动态未迁移 |
| 无短期时序 | 59.62 | 0.03 | 局部时序建模缺失 |
| 无长期时序 | 58.16 | 0.02 | 全局时序建模缺失 |
| **完整** | **50.23** | **0.02** | 三种适配器各有贡献 |

### 关键发现

- Bridge Data上FVD降至21.57，说明生成质量已非常接近真实视频
- 定性对比中，Seer出现"鬼影"效应（帧间累积误差导致后帧偏离），Gen-2和Pika等商业模型无法理解具体操作指令
- 在机器人操控场景（"put corn in pot which is in sink distractors"），AID不仅正确定位目标物体还能完成正确的放置动作
- 仅训练DQFormer不加适配器时FVD高达279.42——说明条件注入模块无法单独完成领域迁移，视频先验的微调是必需的

## 亮点与洞察

- **迁移学习范式**：首次成功将大规模Image2Video模型迁移到领域特定的TVP任务，而非从头训练——利用SVD海量数据训练出的视频动态先验
- **MLLM作为"视频导演"**：用多模态LLM预测视频发展状态是处理单句指令→多帧时序变化的优雅方案
- **极高的效率**：冻结UNet主体，仅训练适配器和条件注入模块，训练参数量和显存大幅减少
- 双重classifier-free guidance提供帧条件和文本条件的独立控制

## 局限与展望

- MLLM状态预测的质量依赖于LLaVA的能力，复杂场景可能产生不准确的状态描述
- 仅在256×256分辨率下实验，高分辨率视频预测有待验证
- 自回归长视频生成可能面临误差累积问题（论文未讨论）
- 适配器设计较为标准，探索更高效的参数自适应方法（如LoRA）可能进一步降低成本
- 未与Sora等最新闭源模型对比

## 相关工作与启发

- 与AnimateDiff、Tune-A-Video等方法类似使用适配器进行迁移，但AID专注于TVP任务并设计了更完整的条件注入框架
- DQFormer的双支路设计借鉴了BLIP-2的Q-Former，但增加了帧级分解支路
- 证明了预训练视频模型的巨大潜力——高质量视频先验可显著缓解小数据集训练的困难

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将I2V模型有效迁移至TVP，MLLM状态预测和DQFormer设计新颖
- 实验充分度: ⭐⭐⭐⭐ 4个数据集+全面消融，但缺少与更多最新方法的对比
- 写作质量: ⭐⭐⭐⭐ 方法描述逻辑清晰，公式规范
- 价值: ⭐⭐⭐⭐ 在机器人操控和第一人称场景中的视频预测有实际应用前景，FVD改善幅度惊人

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing](early_timestep_zero-shot_candidate_selection_for_instruction-guided_image_editin.md)
- [\[ICCV 2025\] Text Embedding Knows How to Quantize Text-Guided Diffusion Models](text_embedding_knows_how_to_quantize_text-guided_diffusion_models.md)
- [\[ICCV 2025\] ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation](adiee_automatic_dataset_creation_and_scorer_for_instruction-guided_image_editing.md)
- [\[ICCV 2025\] SummDiff: Generative Modeling of Video Summarization with Diffusion](summdiff_generative_modeling_of_video_summarization_with_diffusion.md)
- [\[ICCV 2025\] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)

</div>

<!-- RELATED:END -->
