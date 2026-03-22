# Composing Concepts from Images and Videos via Concept-prompt Binding

**会议**: CVPR 2026  
**arXiv**: [2512.09824](https://arxiv.org/abs/2512.09824)  
**代码**: [项目页面](https://refkxh.github.io/BiCo_Webpage)  
**领域**: Video Generation / Concept Composition  
**关键词**: 视觉概念组合, Diffusion Transformer, 视频个性化, 概念绑定, 时序解耦

## 一句话总结
提出 Bind & Compose (BiCo)，一种one-shot方法，通过层次化binder结构将视觉概念绑定到prompt token，并通过token组合实现图像-视频概念的灵活组合，在概念一致性、prompt保真度和运动质量上全面超越前作。

## 研究背景与动机
1. **领域现状**：视觉概念组合旨在将不同图像和视频中的元素整合为一个连贯输出，是视觉创作和电影制作的基础能力。随着DiT架构T2V扩散模型（Wan2.1等）的发展，概念定位和定制化能力显著提升。
2. **现有痛点**：(i) 概念提取精度不足——现有方法（LoRA/可学习embedding+掩码）难以解耦有遮挡和时序变化的复杂概念，且无法提取非物体概念（如风格）；(ii) 图像-视频概念组合灵活性不足——已有工作仅限于用图像中的主体+视频中的运动，未能灵活组合任意属性（视觉风格、光照变化等）。
3. **核心矛盾**：需要同时解决精确概念分解（不需要掩码输入）和跨模态概念组合（图像+视频）两个相互耦合的挑战。
4. **本文要解决什么？** 实现从图像和视频中灵活提取并组合任意视觉概念（包括非物体概念如风格、运动）。
5. **切入角度**：利用T2V扩散模型的概念定位能力，将文本token与对应视觉概念绑定（one-shot训练），然后通过token级别的组合实现概念合成。
6. **核心idea一句话**：先将视觉概念绑定到prompt token上（Bind），再从不同来源选择绑定token组合成目标prompt（Compose），整个过程通过层次化binder结构+多样化吸收机制+时序解耦策略实现。

## 方法详解

### 整体框架
BiCo基于Wan2.1-T2V-1.3B模型，工作流分两阶段：
1. **概念绑定**（Concept Binding）：对每个视觉输入，用轻量binder模块学习文本token与对应视觉概念的映射
2. **概念组合**（Concept Composing）：将目标prompt中不同部分通过对应的binder，组合为包含多来源视觉信息的更新prompt

核心操作基于DiT的cross-attention条件注入：

$$\mathbf{x}_{out} = \text{cross\_attention}(\mathbf{x}_{in}, \mathbf{p}, \mathbf{p})$$

### 关键设计
1. **层次化Binder结构**：包含全局binder $f_g(\cdot)$ 和逐块binder $f_l^i(\cdot)$。每个binder是带零初始化缩放因子的残差MLP：$f(\mathbf{p}) = \mathbf{p} + \gamma \cdot \text{MLP}(\mathbf{p})$。由于DiT各块在去噪过程中行为不同，层次化设计允许全局关联+针对性微调。配合**两阶段倒序训练策略**——先在高噪声水平（$\geq \alpha$，$\alpha=0.875$）强化全局binder，再联合训练全部binder。

2. **多样化-吸收机制 (DAM)**：解决one-shot场景下概念-token绑定精度问题。用VLM（Qwen2.5-VL）提取空间和时序关键概念，生成多样化prompt（保持关键概念词不变）。引入可学习的**吸收token** $p_a^j$ 在训练时吸收与概念无关的视觉细节，推理时丢弃该token以抑制不需要的细节。 

3. **时序解耦策略 (TDS)**：解决图像-视频时序异质性问题。将视频概念训练分两阶段：Stage 1 在单帧上训练（与图像概念训练设置对齐），Stage 2 在完整视频上训练，引入双分支binder结构：

$$\text{MLP}(\mathbf{p}) \leftarrow (1-g(\mathbf{p})) \cdot \text{MLP}_s(\mathbf{p}) + g(\mathbf{p}) \cdot \text{MLP}_t(\mathbf{p})$$

$\text{MLP}_s$ 权重继承自Stage 1，$g(\cdot)$ 零初始化确保良好初始化状态。

### 损失函数 / 训练策略
使用标准扩散模型去噪损失训练binder。每阶段训练2400次迭代，学习率 $1.0 \times 10^{-4}$。推理时生成81帧视频。实验在NVIDIA RTX 4090上进行。

## 实验关键数据

### 主实验：与前作定量对比

| 方法 | CLIP-T↑ | DINO-I↑ | Concept↑ | Prompt↑ | Motion↑ | Overall↑ |
|------|---------|---------|----------|---------|---------|----------|
| Textual Inversion† | 25.96 | 20.47 | 2.14 | 2.17 | 2.94 | 2.42 |
| DB-LoRA† | 30.25 | 27.74 | 2.76 | 2.76 | 2.51 | 2.68 |
| DreamVideo | 27.43 | 24.15 | 1.90 | 1.82 | 1.66 | 1.79 |
| DualReal | 31.60 | 32.78 | 3.10 | 3.11 | 2.78 | 3.00 |
| **BiCo (Ours)** | **32.66** | **38.04** | **4.71** | **4.76** | **4.46** | **4.64** |

BiCo在主观Overall Quality上比前作DualReal提升 **+54.67%**（3.00→4.64）。

### 消融实验：各组件贡献（人工评估5分制）

| 配置 | Concept↑ | Prompt↑ | Motion↑ | Overall↑ |
|------|----------|---------|---------|----------|
| Baseline (仅全局binder) | 2.16 | 2.60 | 2.26 | 2.34 |
| +层次化Binder | 2.63 | 2.88 | 2.93 | 2.81 |
| +Prompt多样化 | 3.40 | 3.34 | 3.04 | 3.26 |
| +吸收Token | 3.55 | 3.43 | 3.43 | 3.47 |
| +TDS (无吸收) | 3.80 | 3.97 | 3.70 | 3.82 |
| ▲ 无倒序训练策略 | 2.60 | 2.70 | 2.43 | 2.58 |
| **Full Model** | **4.43** | **4.47** | **4.32** | **4.40** |

### 关键发现
- 层次化binder对概念保持和运动质量提升显著（2.26→2.93的Motion）
- 吸收token有效抑制不需要的细节（消融可视化显示去除后出现不相关元素）
- TDS对图像-视频兼容性至关重要（Overall从3.47→3.82）
- 两阶段倒序训练不可替代——去除后Overall从4.40暴跌至2.58

## 亮点与洞察
- **统一框架**：首次实现图像+视频任意概念的灵活组合，支持非物体概念（风格、运动）
- **无需掩码**：通过文本条件的概念组合实现隐式分解，降低用户门槛
- **设计上的可扩展性**：binder是轻量模块，不同概念来源的binder独立训练，可按需组合
- **衍生应用丰富**：图像/视频分解（只保留部分token）、文本引导编辑

## 局限性 / 可改进方向
- 将所有token等同对待，但token对T2V生成的重要性分布不均匀——表示主体/运动的token远比功能词重要
- 基于1.3B模型，scaling到更大T2V模型（如CogVideoX、Sora级别）的效果未验证
- 定量评估中自动指标（CLIP-T、DINO-I）与人工评估的一致性有待进一步确认
- 计算开销：每个概念来源需独立训练binder（2400 iterations x 2 stages）

## 相关工作与启发
- Textual Inversion/DreamBooth-LoRA是视频个性化的基础方法但概念控制粒度粗
- DreamVideo/DualReal支持主体+运动的组合但限制输入类型和数量
- TokenVerse实现了prompt控制的图像概念组合，但依赖文本条件调制架构，不适用于现代T2V模型
- Break-A-Scene依赖显式掩码输入，无法提取非物体概念
- BiCo通过binder+token组合范式统一了概念分解与组合
- Set-and-Sequence和Grid-LoRA在LoRA空间实现外观/运动学习，但无法精确指定概念和组合方式

## 方法细节补充
- **VLM提取关键概念**：用Qwen2.5-VL提取空间概念（物体、风格、光照等）和时序概念（运动模式、速度变化等），分别组合为spatial-only和spatiotemporal prompts
- **推理过程**：将目标prompt $\mathbf{p}_d$ 按概念对应关系分解，各部分通过对应binder更新后重新组合为 $\mathbf{p}_u^i$
- **衍生应用**：图像/视频分解（仅保留dog相关token丢弃cat相关token）、文本引导编辑（未改变部分过binder，编辑部分直接用原始token）

## 评分 ⭐
- 新颖性: ⭐⭐⭐⭐⭐ — 首次实现图像-视频任意概念的统一灵活组合
- 实验充分度: ⭐⭐⭐⭐⭐ — 定量自动+人工评估+详细消融+可视化案例全面
- 写作质量: ⭐⭐⭐⭐ — 概念清晰，DAM/TDS设计动机阐述充分
- 价值: ⭐⭐⭐⭐⭐ — 对视觉内容创作具有直接和广泛的应用前景
