---
title: >-
  [论文解读] TinyChemVL: Advancing Chemical Vision-Language Models via Efficient Visual Token Reduction and Complex Reaction Tasks
description: >-
  [AAAI 2026][多模态VLM][化学VLM] TinyChemVL 是一个仅4B参数的化学领域VLM，通过自适应token合并与剪枝策略将视觉token压缩至原来的1/16，并引入反应级别任务和基准ChemRxn-V，在分子和反应级别的视觉化学任务上达到SOTA性能，同时显著提升推理和训练速度。
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "化学VLM"
  - "视觉token压缩"
  - "分子识别"
  - "反应预测"
  - "小模型高效推理"
---

# TinyChemVL: Advancing Chemical Vision-Language Models via Efficient Visual Token Reduction and Complex Reaction Tasks

**会议**: AAAI 2026  
**arXiv**: [2511.06283](https://arxiv.org/abs/2511.06283)  
**代码**: [https://github.com/xxlllz/TinyChemVL](https://github.com/xxlllz/TinyChemVL)  
**领域**: 多模态VLM  
**关键词**: 化学VLM, 视觉token压缩, 分子识别, 反应预测, 小模型高效推理

## 一句话总结
TinyChemVL 是一个仅4B参数的化学领域VLM，通过自适应token合并与剪枝策略将视觉token压缩至原来的1/16，并引入反应级别任务和基准ChemRxn-V，在分子和反应级别的视觉化学任务上达到SOTA性能，同时显著提升推理和训练速度。

## 研究背景与动机

### 领域现状
大语言模型（LLM）在化学领域的应用日益广泛，但传统方法主要依赖文本格式（如SMILES、SELFIES）表征分子，不可避免地丢失空间信息。视觉语言模型（VLM）虽然具备处理视觉输入的能力，但在化学领域的探索仍然有限。现有的化学VLM（如ChemVLM、ChemMLLM）直接在通用VLM架构上微调，存在明显的效率问题。

### 核心痛点

**视觉token冗余严重**：分子图像中，所有结构信息集中在分子图形区域，而大量背景区域是无意义的。例如ChemVLM中，一张800×800的图像需要1280个视觉token，约为文本问题token数的100倍，极大增加了计算开销。此外，将分子图像切割为patch的过程还会破坏重要的分子结构信息。

**任务范围狭窄**：现有化学VLM主要关注分子级别任务（如SMILES OCR、性质预测），忽视了反应级别任务。反应级别任务（如反应预测）需要模型同时具备识别和推理能力，是更具挑战性的方向。

**分子图像生成瓶颈**：ChemMLLM提出的VQ-GAN方案生成分子图像后，需要外部工具将图像解析回SMILES，流程复杂且架构不匹配。

### 本文切入角度
从**模型效率**和**任务复杂度**两个维度同时发力：一方面通过自适应token减少策略压缩视觉表征, 另一方面将化学VLM的能力从分子级拓展到反应级，构建更完整的化学视觉理解能力。

## 方法详解

### 整体框架
TinyChemVL采用经典的ViT-MLP-LLM架构，以InternVL2.5-4B为骨干，使用动态分辨率策略将高分辨率图像分成多个448×448的tile。核心创新在视觉编码器中嵌入自适应token合并与剪枝模块，在注意力层和FFN层之间逐步减少视觉token数量。

### 关键设计

#### 1. **自适应Token合并与剪枝 (Adaptive Token Merge and Pruning)**
- **功能**：在ViT的每个transformer block中，根据当前视觉token分布自适应地裁剪不重要的token并合并重复的token
- **核心思路**：
    - **Token评分**：采用ATS（Adaptive Token Sampler）方法，利用注意力矩阵中CLS token对其他token的注意力权重，结合Value矩阵的范数计算重要性分数：
    $Score_i = \frac{A_{1,i+1} \times \|V_{i+1}\|}{\sum_{j=1}^{N} A_{1,j+1} \times \|V_{j+1}\|}$
    - **Token剪枝**：采用Top-K选择策略，保留得分最高的K个token，直接丢弃低分token（对应背景区域）
    - **Token合并**：使用二部软匹配（BSM）算法，将token分为两组，通过余弦相似度找到最相似的token对进行加权平均合并。**不受空间邻近性约束**，非相邻但相似的token也可合并
    - **比例注意力（Proportional Attention）**：维护行向量$s$追踪每个当前token代表的原始token数量，在注意力计算中加入$\log s$偏置以保持信息保真度
- **设计动机**：分子图像的稀疏特性使得大部分视觉token对应空白背景，而分子结构信息密集。需要一种既能去除冗余又能保留结构信息的策略

#### 2. **自适应决策策略 (Adaptive Policy)**
- **功能**：在实例级别和层级别自适应选择执行剪枝还是合并操作
- **核心思路**：计算token得分的方差 $S_{op_i} = var(Score_i)$：
    - 方差低（$S_{op_i} \leq \tau$）→ 重要性收敛 → 大面积背景区域 → **执行剪枝**
    - 方差高（$S_{op_i} > \tau$）→ 重要性分化 → 复杂分子结构 → **执行合并**
    - 阈值 $\tau$ 默认设为 $1e-5$，基于化学图像数据集的统计分析确定
- **设计动机**：不同图像、不同层的token分布差异很大，固定策略无法适应所有情况。通过方差自动判断当前是"大片背景"还是"复杂结构"的情况

#### 3. **反应级别任务与代码生成范式**
- **功能**：将化学VLM从分子级别扩展到反应级别，并用可执行代码替代直接图像生成
- **核心思路**：
    - 构建**反应识别**任务：从反应图像解析出 reactants>>reagents.solvents>>products 格式
    - 构建**反应预测**任务：仅从反应物图像预测产物（首次定义该任务）
    - **分子图像生成**：生成可执行Python代码来渲染分子图像，SMILES直接包含在代码中，无需额外解析工具
- **设计动机**：反应预测是化学专家通过观察分子结构即可完成的任务，是VLM作为化学研究工具的关键能力。代码生成范式比VQ-GAN更直接、可验证

### 数据构建
构建了约125万样本的大规模多任务训练集：
- 分子识别：500K（385K自建），来源包括ChEBI-20-MM、MolGrapher、MolScribe、ORDerly等
- 反应识别：200K（全部自建），从ORDerly数据集渲染
- 性质预测：150K（55K自建）
- 反应预测：200K（全部自建）
- 分子图像生成：200K（55K自建）

### 训练策略
- 骨干模型：InternVL2.5-4B，全参数微调
- 硬件：8×NVIDIA A100 80G
- 训练轮次：1.5 epochs
- 批次设置：per-device batch size 16, gradient accumulation 2
- 训练框架：ms-swift

## 实验关键数据

### 主实验 — 分子识别

| 模型 | ChemOCR Avg Sim.(%) | ChemOCR Tani@1.0(%) | img2smiles Avg Sim.(%) | img2smiles Tani@1.0(%) |
|------|---------------------|---------------------|------------------------|------------------------|
| GPT-4o | 36.8 | 3.4 | 29.0 | 0.01 |
| ChemVLM-8B | 81.7 | 57.7 | 55.0 | 15.0 |
| ChemDFM-X(13B) | 70.9 | 36.5 | **90.9** | **77.6** |
| ChemMLLM | - | - | 75.0 | 49.0 |
| **TinyChemVL(4B)** | **91.2** | **77.4** | 89.5 | 75.6 |

### 主实验 — 反应级任务 (ChemRxn-V)

| 模型 | 反应识别 Avg Sim.(%) | 反应识别 EM(%) | 反应预测 Avg Sim.(%) | 反应预测 Tani@1.0(%) |
|------|---------------------|----------------|---------------------|---------------------|
| GPT-4o | 19.1 | 0.1 | 30.4 | 1.4 |
| ChemDFM-X | 28.32 | 3.2 | 12.7 | 0.7 |
| ChemVLM-8B | 0.6 | 0.0 | 4.8 | 0.0 |
| **TinyChemVL** | **93.4** | **67.9** | **78.9** | **52.4** |

### 效率对比

| 模型 | 推理速度(Sample/s↑) | 平均Token数(↓) | 训练时间(小时↓) |
|------|---------------------|----------------|----------------|
| ChemVLM-8B | 7.41 | 896 | - |
| InternVL2.5-4B | 9.11 | 894 | 47* |
| **TinyChemVL** | **11.84** | **108** | **15** |

### 消融实验

| 配置（token/image） | ChemOCR Tani@1.0(%) | 反应识别 EM(%) | 反应预测 Tani@1.0(%) |
|---------------------|---------------------|----------------|---------------------|
| 16 (默认) | 77.4 | 62.7 | 52.4 |
| 4 | 76.2 (-1.2) | 59.5 (-3.2) | 50.1 (-2.3) |

### 关键发现
1. **TinyChemVL以4B参数超越8B和26B的ChemVLM**，在ChemOCR上实现91.2%的平均相似度，是首个能与专用SMILES OCR模型竞争的通用VLM
2. **在反应级任务上碾压所有现有模型**：反应识别93.4%、反应预测78.9%，而其他模型均低于30%
3. **视觉token从896降至108**（约1/8），推理速度提升约60%，训练时间从47小时降至15小时
4. 性质预测任务上MSE约为ChemMLLM的一半，在7项属性中5项达到最优
5. 进一步将token从16降到4会导致性能下降，说明4个token不足以表达复杂反应图像

## 亮点与洞察
1. **问题定义精准**：精确识别了化学图像的稀疏性特征，"背景冗余+结构密集"的双重特性使得自适应策略特别有效
2. **方差驱动的自适应策略设计优雅**：用token得分方差区分"背景区域"和"复杂结构"，既简单又有效，且在实例和层两个级别都自适应
3. **首次定义视觉反应预测任务**：直接从反应物图像预测产物，将VLM的化学应用从OCR级别提升到推理级别
4. **代码生成替代图像生成**：避免了VQ-GAN的架构不匹配问题，生成的代码直接包含SMILES便于验证
5. **小模型大能力**：4B模型超越13B-26B模型，证明了"效率+质量"可以兼顾

## 局限与展望
1. 当前仅在化学领域验证，token压缩策略是否适用于其他科学图像（如生物、材料）有待验证
2. 反应预测任务目前仅基于视觉信息，未结合反应条件（温度、催化剂等）文本信息
3. 代码生成用于分子图像生成的方式虽然巧妙，但生成代码的可执行性和鲁棒性未充分讨论
4. ChemRxn-V基准仅包含5000样本/任务，规模相对有限
5. 手写分子识别虽有探索，但未作为主要评测方向

## 相关工作与启发
- **ToMe (Token Merging)**：TinyChemVL的token压缩策略直接借鉴了ToMe的二部软匹配算法，但增加了自适应剪枝/合并的决策机制
- **ChemVLM**：作为直接对比的基线，使用了相同的InternVL架构但无token压缩
- **ChemMLLM**：提出了分子图像生成任务，但使用VQ-GAN存在架构不匹配问题
- 启发：**领域特定VLM的关键在于"找到视觉冗余的结构性特征"**，化学图像的稀疏背景是一个典型案例

## 评分
- 新颖性: ⭐⭐⭐⭐ （token压缩思路不新，但在化学领域的自适应应用和反应级任务定义有新意）
- 实验充分度: ⭐⭐⭐⭐⭐ （覆盖了分子识别、性质预测、图像生成、反应识别/预测五大任务，效率分析完整）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，动机充分，但部分数学符号排版可改进）
- 价值: ⭐⭐⭐⭐ （对化学AI领域有实际推动作用，小模型高效方案有部署价值）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration](filter_correlate_compress_training-free_token_reduction_for_.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)
- [\[ICCV 2025\] LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models](../../ICCV2025/multimodal_vlm/llava-prumerge_adaptive_token_reduction_for_efficient_large_multimodal_models.md)
- [\[ACL 2026\] ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding](../../ACL2026/multimodal_vlm/chemvlr_prioritizing_reasoning_in_perception_for_chemical_vision-language_unders.md)
- [\[CVPR 2026\] DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference](../../CVPR2026/multimodal_vlm/duet-vlm_dual_stage_unified_efficient_token_reduction_for_vlm_training_and_infer.md)

</div>

<!-- RELATED:END -->
