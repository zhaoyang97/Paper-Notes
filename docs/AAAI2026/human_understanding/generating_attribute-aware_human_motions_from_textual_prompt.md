---
title: >-
  [论文解读] Generating Attribute-Aware Human Motions from Textual Prompt
description: >-
  [AAAI 2026][人体理解][人体运动生成] 提出 AttrMoGen 框架，通过基于结构因果模型（SCM）的因果信息瓶颈将动作语义与人体属性（年龄、性别等）解耦，生成属性感知的人体运动，并构建了首个包含广泛属性标注的大规模文本-运动数据集 HumanAttr。 文本驱动的人体运动生成近年来取得了显著进展…
tags:
  - "AAAI 2026"
  - "人体理解"
  - "人体运动生成"
  - "属性感知"
  - "因果解耦"
  - "VQVAE"
  - "文本驱动"
---

# Generating Attribute-Aware Human Motions from Textual Prompt

**会议**: AAAI 2026  
**arXiv**: [2506.21912](https://arxiv.org/abs/2506.21912)  
**代码**: 无  
**领域**: 人体理解  
**关键词**: 人体运动生成, 属性感知, 因果解耦, VQVAE, 文本驱动

## 一句话总结

提出 AttrMoGen 框架，通过基于结构因果模型（SCM）的因果信息瓶颈将动作语义与人体属性（年龄、性别等）解耦，生成属性感知的人体运动，并构建了首个包含广泛属性标注的大规模文本-运动数据集 HumanAttr。

## 研究背景与动机

文本驱动的人体运动生成近年来取得了显著进展，但现有方法存在一个根本性的忽视：**人体属性（如年龄、性别、体重、身高）对运动模式有显著影响**。

**运动模式的属性依赖性**：老年人和青少年的走路方式截然不同；同样是"走路"，不同年龄/性别/体型的人在步幅、关节范围、动作幅度上都有自然差异

**语义-属性耦合问题**：一个运动序列同时包含动作语义（走路、跑步）和人体属性信息，但文本描述通常只关注动作语义。现有方法将文本和运动在共享空间中对齐时未区分这两者，可能阻碍对齐质量

**数据集缺失**：缺乏大规模的、包含广泛人体属性标注的文本-运动数据集。已有数据集要么缺少属性标注（如 HumanML3D），要么属性范围有限（如 KIT 中 90% 受试者年龄在 18-45 之间），要么规模极小

**核心观察**：人体运动可以分解为两个因子——动作语义和人体属性。文本描述仅对应动作语义，因此需要将两者解耦。

## 方法详解

### 整体框架

AttrMoGen 包含两个主要组件：

1. **Semantic-Attribute Decoupling VQVAE (Decoup-VQVAE)**：编码器通过因果信息瓶颈从原始运动中去除属性信息，得到无属性的语义 token；解码器利用语义 token 和属性标签重建运动
2. **Semantics Generative Transformer**：从文本预测语义 token，推理时结合用户指定的属性生成运动

### 关键设计

#### 1. **基于结构因果模型的解耦（SCM-based Decoupling）**

将问题建模为因果因子解耦。定义：
- $X$：原始运动
- $Y$：目标语义 token
- $S$：动作语义（$Y$ 的因果因子）
- $A$：人体属性（$Y$ 的非因果因子，但对 $X$ 的构成至关重要）

因果信息瓶颈（CIB）目标函数：
$$CIB(X,Y,S,A) = I(X;S,A) + I(Y;S) - I(S;A) - \lambda I(X;S)$$

各项作用：
- $I(X;S,A)$：确保 $S$ 和 $A$ 信息足以重建 $X$ → 重建损失
- $I(Y;S)$：确保 $S$ 信息足以推导 $Y$ → 量化过程
- $-I(S;A)$：限制 $S$ 和 $A$ 之间的互信息 → **解耦核心**
- $-\lambda I(X;S)$：限制 $X$ 到 $S$ 的信息流 → 信息瓶颈

#### 2. **解耦实现（Decoupling Term $-I(S;A)$）**

通过互信息上界的估计来实现解耦：

$$I(S;A) \leq \log|A| - \mathbb{E}_{s\sim p(S)}H(A|S=s)$$

引入代理属性分类器 $h$，其目标是从语义嵌入 $S$ 中分类人体属性 $A$，输出作为条件概率 $p(A|s_i) = h(s_i)$。最小化以下损失函数：

$$\mathcal{L}_{entropy} = -\sum_{i=1}^{B} H(A|S=s_i)$$

最小化 $\mathcal{L}_{entropy}$ → 减少 $I(S;A)$ → 从 $S$ 中消除属性信息。属性分类器 $h$ 与编码器 $f$ 和解码器 $g$ 交替更新。

#### 3. **信息瓶颈实现（Bottleneck Term $-\lambda I(X;S)$）**

利用反事实运动对齐编码器输出。核心思路：如果编码器正确解耦，那么**同一动作语义、不同属性**的运动应产生相同的语义嵌入。

- 通过解码器生成反事实运动：$X^- = g(S, A^-)$，其中 $A^-$ 是随机化的属性
- 计算原始运动和反事实运动的编码器输出相似度矩阵
- 瓶颈损失：$\mathcal{L}_{bottleneck} = \|\tilde{D}(X, X^-) - I\|_F^2$

强制 $f(X)$ 和 $f(X^-)$ 接近，同时保持通道间独立性。

#### 4. **语义生成 Transformer**

采用 MoMask（掩码 Transformer）架构，训练时随机掩码语义 token 并以 CLIP 文本特征为条件预测。推理时：
1. 文本 → Semantics Generative Transformer → 语义 token
2. 语义 token + 属性输入 → Decoup-VQVAE 解码器 → 属性感知运动

### 损失函数 / 训练策略

总损失函数：
$$\mathcal{L}_{overall} = \mathcal{L}_{vqvae} + \alpha\mathcal{L}_{entropy} + \lambda\mathcal{L}_{bottleneck}$$

其中 $\mathcal{L}_{vqvae} = \mathcal{L}_{rec} + \mathcal{L}_{commit} + \mathcal{L}_{embed}$，默认 $\alpha=0.01$, $\lambda=0.5$。

训练策略：编码器 $f$、解码器 $g$ 和代理属性分类器 $h$ 交替更新，分类器用交叉熵 $\mathcal{L}_{CE}$ 监督。

## 实验关键数据

### HumanAttr 数据集

| 子数据集 | 受试者数 | 运动数 | 时长(min) | 年龄范围 |
|---------|---------|--------|----------|---------|
| BMLmovi | 86 | 1,801 | 161.8 | [17, 33] |
| ETRI-Activity3D | 100 | 3,727 | 691.8 | [21, 88] |
| KIT | 55 | 4,231 | 463.1 | [15, 55] |
| Nymeria | 264 | 6,850 | 552.4 | [18, 50] |
| **Total** | **640** | **18,199** | **2,135.4** | **[5, 88]** |

### 主实验

| 方法 | R-Precision Top-1↑ | Top-3↑ | FID↓ | MM-Dist↓ | Diversity→ | MModality↑ |
|------|-------------------|--------|------|----------|-----------|------------|
| T2M | 0.592 | 0.859 | 1.909 | 3.827 | 18.856 | 2.627 |
| MotionDiffuse | 0.670 | 0.928 | 0.416 | 2.704 | 18.968 | 2.435 |
| MoMask | 0.685 | 0.925 | 0.245 | 2.602 | 18.981 | 1.438 |
| GenMoStyle | 0.680 | 0.925 | 0.332 | 2.649 | 19.118 | 1.588 |
| **AttrMoGen** | **0.705** | **0.940** | **0.089** | **2.266** | 19.268 | **1.250** |

AttrMoGen 将 FID 从 MoMask 的 0.245 大幅降低到 0.089（-63.7%），MM-Dist 从 2.602 降到 2.266。

### 消融实验

| 配置 | R-Precision Top-1↑ | FID↓ | MM-Dist↓ | 说明 |
|------|-------------------|------|----------|------|
| MoMask（基线） | 0.685 | 0.245 | 2.602 | 无属性信息 |
| w/ attr test only | 0.603 | 0.957 | 3.815 | 仅测试加属性文本→严重退化 |
| w/ attr train+test | 0.689 | 0.203 | 2.518 | 训练测试都加→有限提升 |
| w/o entropy | 0.686 | 0.489 | 2.523 | 去掉解耦项→FID翻倍 |
| w/o bottleneck | 0.686 | 0.184 | 2.486 | 去掉信息瓶颈 |
| $\lambda=0.25$ | 0.698 | 0.098 | 2.326 | 较小瓶颈权重 |
| $\lambda=1$ | 0.701 | 0.088 | 2.332 | 较大瓶颈权重 |
| **AttrMoGen ($\lambda=0.5, \alpha=0.01$)** | **0.705** | **0.089** | **2.266** | 最优配置 |

**关键消融发现**：
- 直接在测试时在文本中加入属性信息会严重恶化性能（FID: 0.245→0.957），因为模型从未见过这种格式
- 解耦项 $\mathcal{L}_{entropy}$ 对 FID 影响最大（0.089→0.489），验证了因果解耦的核心作用

### 属性控制验证

| 属性 | 组别 | MoMask Acc | AttrMoGen Acc | 说明 |
|------|------|-----------|---------------|------|
| 性别 | male | 0.747 | **0.992** | 极高控制精度 |
| 性别 | female | 0.546 | **0.985** | 极高控制精度 |
| 年龄 | 5-18 | 0.314 | **0.422** | 年龄控制更难 |
| 年龄 | 60-88 | 0.556 | **0.787** | 老年组区分度高 |

### 关键发现

1. **属性信息对运动质量有决定性影响**：FID 下降 63.7% 证明了属性感知的巨大价值
2. **直接拼接属性文本不是好策略**：需要显式的解耦机制而非简单的文本级融合
3. **因果解耦是核心**：$\mathcal{L}_{entropy}$ 的消融显示解耦项对性能贡献最大
4. **反事实对齐有效**：瓶颈损失通过反事实运动确保语义嵌入的属性不变性
5. **性别控制精度 >98%**，但年龄控制相对困难，年龄对运动模式的影响更微妙

## 亮点与洞察

- **首创性地将人体属性引入文本驱动运动生成**，填补了领域空白
- **因果模型的精彩应用**：用 SCM 框架将动作语义和属性形式化为因果/非因果因子，理论优美且实用
- **反事实运动生成**的策略巧妙——利用解码器自身生成"相同语义、不同属性"的反事实样本来辅助训练
- **HumanAttr 数据集**是重要的社区贡献（640 受试者，年龄跨度 5-88）
- 与风格化运动生成（style-based）的本质区别：风格是主观意图（骄傲/沮丧），属性是客观生物力学特征

## 局限与展望

- 属性标签离散化（4 个年龄组/2 个性别），丢失了连续属性信息
- HumanAttr 中年龄分布不均匀（年轻成人占多数），极端年龄组（5-18, 60-88）的数据较少
- 仅使用年龄和性别两种属性，体重和身高虽有标注但未在主实验使用
- 属性分类器作为代理可能引入额外偏差
- 运动表示限于 SMPL 参数化，未考虑手部精细运动

## 相关工作与启发

- 与 MoMask 的关系：AttrMoGen 在 MoMask 架构基础上增加了属性解耦层
- Style-based 方法（GenMoStyle）将风格标签替换为属性标签的对比实验证明了专用解耦的必要性
- 因果信息瓶颈（CIB）在图像去偏、公平性学习等领域已有应用，本文首次引入运动生成
- 对于虚拟人/数字孪生应用具有直接价值：不同角色需要呈现不同属性的运动模式

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次系统地将人体属性引入运动生成，SCM 解耦框架设计精巧
- 实验充分度: ⭐⭐⭐⭐ — 充分的消融和属性分组实验，新数据集构建完善
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，因果建模阐述严谨
- 价值: ⭐⭐⭐⭐⭐ — 数据集+方法双重贡献，长期影响力大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] StickMotion: Generating 3D Human Motions by Drawing a Stickman](../../CVPR2025/human_understanding/stickmotion_generating_3d_human_motions_by_drawing_a_stickman.md)
- [\[CVPR 2025\] Shape My Moves: Text-Driven Shape-Aware Synthesis of Human Motions](../../CVPR2025/human_understanding/shape_my_moves_text-driven_shape-aware_synthesis_of_human_motions.md)
- [\[AAAI 2026\] CLIP-FTI: Fine-Grained Face Template Inversion via CLIP-Driven Attribute Conditioning](clip-fti_fine-grained_face_template_inversion_via_clip-driven_attribute_conditio.md)
- [\[AAAI 2026\] SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control](soscontrol_enhancing_human_motion_generation_through_saliency-aware_symbolic_ori.md)
- [\[CVPR 2026\] Towards Storytelling Animations: Joint Synthesis of Human and Camera Motions](../../CVPR2026/human_understanding/towards_storytelling_animations_joint_synthesis_of_human_and_camera_motions.md)

</div>

<!-- RELATED:END -->
