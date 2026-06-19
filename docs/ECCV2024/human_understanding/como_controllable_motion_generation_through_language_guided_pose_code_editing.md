---
title: >-
  [论文解读] CoMo: Controllable Motion Generation Through Language Guided Pose Code Editing
description: >-
  [ECCV2024][人体理解][human motion synthesis] 提出 CoMo，通过将动作序列分解为语义明确的 pose code（如"左膝微弯"），实现基于文本的可控动作生成与基于 LLM 的零样本动作编辑。 领域现状 领域现状：现有 text-to-motion 模型（如 T2M-GPT、MDM、MLD…
tags:
  - "ECCV2024"
  - "人体理解"
  - "human motion synthesis"
  - "motion editing"
  - "pose code"
  - "LLM"
  - "text-to-motion"
---

# CoMo: Controllable Motion Generation Through Language Guided Pose Code Editing

**会议**: ECCV2024  
**arXiv**: [2403.13900](https://arxiv.org/abs/2403.13900)  
**代码**: [yh2371/CoMo](https://github.com/yh2371/CoMo)  
**领域**: 人体理解  
**关键词**: human motion synthesis, motion editing, pose code, LLM, text-to-motion

## 一句话总结

提出 CoMo，通过将动作序列分解为语义明确的 pose code（如"左膝微弯"），实现基于文本的可控动作生成与基于 LLM 的零样本动作编辑。

## 背景与动机

### 领域现状

**领域现状**：现有 text-to-motion 模型（如 T2M-GPT、MDM、MLD）虽然能从文本生成人体动作，但缺乏对生成过程的**细粒度控制能力**。具体而言：

### 现有痛点

**现有痛点**：修改特定时刻的微妙姿态（如"弯腰更深"）非常困难

### 核心矛盾

**核心矛盾**：在指定时间点插入新动作（如"最后蹲下"）无法实现

### 解决思路

**解决思路**：大多数方法将文本映射到一个叠加的 latent code，生成器需要自行拆解各身体部位信息，导致文本与动作的对应不精确

这些限制使得现有方法在动画创作、沉浸式技术等需要精细控制的场景中适用性有限。

### 解决思路

**本文目标**：如何在文本驱动的人体动作生成中实现**空间（各身体部位）和时间（各帧）维度的细粒度可控性**，同时支持基于自然语言的直觉化动作编辑？

## 方法详解

CoMo 由三个核心组件构成：

### 1. Motion Encoder-Decoder（动作编解码器）

**编码器**：采用预定义的语义 pose codebook 而非学习式 VQ-VAE。基于 PoseScript 的骨架解析器，通过启发式几何阈值规则将每帧动作编码为 K-hot 向量：

- Codebook 包含 392 个 pose code，分为 70 个 pose category
- 每个 code 描述一个身体部位的状态（如"right arm straight"）或部位间的空间关系（如"left hand and left foot close"）
- 每个 category 内互斥，同一时刻每个 category 只激活一个 code
- 时间下采样率 $l=4$，最大 code 序列长度 50

**解码器**：1D 卷积网络，将 pose code 的 latent feature（激活 code 的 embedding 求和）重建为连续动作。训练目标使用 smooth L1 loss，同时约束位置和速度：

$$\mathcal{L}_{\text{rec}} = \mathcal{L}_1(X, X_{\text{rec}}) + 0.5 \cdot \mathcal{L}_1(V(X), V(X_{\text{rec}}))$$

### 2. Motion Generator（动作生成器）

基于 decoder-only Transformer 的自回归多标签预测模型：

- 输入：CLIP 编码的文本 embedding + GPT-4 生成的 11 个细粒度关键词（10 个身体部位 + 1 个情绪）
- 输出：逐步预测每个时间步的 K-hot 向量（各 pose code 的 Bernoulli 分布）
- 训练目标：二元交叉熵损失，最大化所有 Bernoulli 分布的平均对数似然
- 附加 \<End\> code 标记动作结束

### 3. Motion Editor（动作编辑器）

利用 LLM（GPT-4）对 pose code 进行零样本编辑，分三步骤顺序 prompting：

1. **定位编辑帧**：LLM 确定需要编辑的起止帧索引
2. **定位编辑部位**：LLM 识别需要修改的身体部位及对应 pose category
3. **修改 pose code**：LLM 审查被选 category 中的 code 并根据指令调整

编辑后的 code 与未编辑部分拼接，通过解码器重建为最终动作。这种方式直接操作源动作的编码表示，而非像 FineMoGen 等方法那样从更新后的文本重新生成。

## 实验关键数据

### 动作生成（HumanML3D 数据集）

| 指标 | CoMo | T2M-GPT | FineMoGen | GraphMotion |
|------|------|---------|-----------|-------------|
| R-Precision Top-3↑ | **0.790** | 0.775 | 0.784 | 0.785 |
| FID↓ | 0.262 | 0.116 | 0.151 | 0.116 |
| MM-DIST↓ | 3.032 | 3.118 | 2.998 | 3.070 |
| Diversity↑ | **9.936** | 9.761 | 9.263 | 9.692 |

- Pose code 重建质量接近真实动作（重建 FID=0.041 vs 真实 0.002）
- 在 R-Precision Top-3 和 Diversity 上取得最优

### 动作编辑（54 人用户研究）

- 平均**超过 70%** 的评估者偏好 CoMo 的编辑结果
- 在身体部位修改和动作增删场景中优势尤为显著
- 在情绪/速度等全局编辑上优势较小（文本描述更适合此类全局变化）

### 消融实验

- 去除 LLM 生成的细粒度关键词后，HumanML3D 上 Top-1 从 0.502 降至 0.487
- Codebook 大小 392 在复杂度和重建质量间取得最佳平衡
- 下采样率 $l=4$ 为最优选择（$l=2$ 质量更好但序列过长）

## 亮点与洞察

1. **语义化、可解释的动作表示**：pose code 具有明确的自然语言语义（如"左膝微弯"），使动作序列变得人类可读、可干预
2. **LLM 零样本编辑**：无需微调即可利用 LLM 的语言理解能力完成动作编辑，三步 prompting 策略简洁有效
3. **直接编辑源动作**：与基于文本重新生成的方法不同，CoMo 直接修改源动作编码，更好地保持未编辑部分的一致性
4. **生成与编辑统一框架**：同一套 pose code 表示同时支持文本驱动生成和交互式编辑

## 局限与展望

1. **局部运动学描述为主**：当前 pose code 侧重局部关节状态，缺少速度、风格、轨迹、动作重复等全局描述符
2. **物理可行性无约束**：LLM 编辑 pose code 时不保证生成物理上合理的动作序列，可能产生不自然的结果
3. **FID 不是最优**：生成质量（FID=0.262）与 T2M-GPT（0.116）有差距，离散化表示损失了部分细节
4. **MModality 较低**：语义 pose code 增强了文本-动作一致性，但可能牺牲了同一文本下的多样性
5. **依赖 GPT-4**：编辑和关键词生成都依赖 GPT-4，增加了推理成本和延迟

## 相关工作与启发

| 方法 | 表示方式 | 编辑能力 | 编辑方式 |
|------|---------|---------|---------|
| T2M-GPT | VQ-VAE 隐式 token | 无直接编辑 | 需改文本重新生成 |
| FineMoGen | 扩散模型 latent | 通过全局注意力优化 | 每次编辑生成新序列 |
| MDM | 扩散模型 | 零样本 inpainting | 帧/关节 inpainting，可能不自然 |
| GraphMotion | 层次语义图 + 扩散 | 有限 | 依赖文本语义解析 |
| **CoMo** | 语义 pose code | LLM 零样本编辑 | 直接修改源动作编码 |

CoMo 的关键区分点在于：可解释的离散表示使 LLM 能直接"读懂"和"修改"动作，而其他方法要么需要修改文本重新生成，要么通过 inpainting 间接编辑。

## 相关工作与启发

- **离散化语义表示的力量**：将连续信号编码为人类可理解的离散符号，不仅方便 LLM 推理，也为交互式编辑开辟了新路径。类似思路可推广到其他连续信号（如语音、音乐）的可控生成
- **与 PoseScript 的关系**：Codebook 构建依赖 PoseScript 的骨架解析器，是 PoseScript 从静态姿态描述到动态动作生成的自然延伸
- **对话式动作生成**：CoMo 的迭代编辑能力使"用户描述→生成→反馈→编辑→满意"的交互循环成为可能，很适合动画制作的实际工作流

## 评分

- 新颖性: ⭐⭐⭐⭐ — 语义 pose code 作为 LLM 可操作的动作中间表示，思路新颖且方向有前景
- 实验充分度: ⭐⭐⭐⭐ — 生成实验覆盖两个数据集，编辑有 54 人用户研究，消融全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，概念解释到位，图表丰富
- 价值: ⭐⭐⭐⭐ — 可控动作生成是实际需求，语义离散表示+LLM 编辑的范式有较强借鉴意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference](../../CVPR2026/human_understanding/lamogen_language_to_motion_generation_through_llm-guided_symbolic_inference.md)
- [\[ECCV 2024\] Motion Mamba: Efficient and Long Sequence Motion Generation](motion_mamba_efficient_and_long_sequence_motion_generation.md)
- [\[ECCV 2024\] Large Motion Model for Unified Multi-Modal Motion Generation](large_motion_model_for_unified_multi-modal_motion_generation.md)
- [\[ECCV 2024\] A Simple Baseline for Spoken Language to Sign Language Translation with 3D Avatars](a_simple_baseline_for_spoken_language_to_sign_language_trans.md)
- [\[CVPR 2026\] Omni-Supervised Motion Editing: Balancing Change and Invariance through Positive-Negative Learning](../../CVPR2026/human_understanding/omni-supervised_motion_editing_balancing_change_and_invariance_through_positive-.md)

</div>

<!-- RELATED:END -->
