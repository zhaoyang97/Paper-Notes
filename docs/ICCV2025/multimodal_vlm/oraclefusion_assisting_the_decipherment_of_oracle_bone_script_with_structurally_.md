---
title: >-
  [论文解读] OracleFusion: Assisting the Decipherment of Oracle Bone Script with Structurally Constrained Semantic Typography
description: >-
  [ICCV 2025][多模态VLM][oracle bone script] 提出OracleFusion，一个两阶段语义字体排印框架：第一阶段利用MLLM增强的空间感知推理（SAR）分析甲骨文字形结构并定位关键部件；第二阶段提出Structural Oracle Vector Fusion（SOVF），通过字形结构约束和骨架保持损失生成语义丰富的矢量字体，在保持原始字形完整性的同时传达语义，辅助专家解读未释甲骨文。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "oracle bone script"
  - "semantic typography"
  - "MLLM"
  - "vector graphics generation"
  - "spatial awareness reasoning"
  - "score distillation"
---

# OracleFusion: Assisting the Decipherment of Oracle Bone Script with Structurally Constrained Semantic Typography

**会议**: ICCV 2025  
**arXiv**: [2506.21101](https://arxiv.org/abs/2506.21101)  
**代码**: [GitHub](https://github.com/lcs0215/OracleFusion)  
**领域**: 多模态VLM / 甲骨文解读 / 语义字体排印  
**关键词**: oracle bone script, semantic typography, MLLM, vector graphics generation, spatial awareness reasoning, score distillation  

## 一句话总结
提出OracleFusion，一个两阶段语义字体排印框架：第一阶段利用MLLM增强的空间感知推理（SAR）分析甲骨文字形结构并定位关键部件；第二阶段提出Structural Oracle Vector Fusion（SOVF），通过字形结构约束和骨架保持损失生成语义丰富的矢量字体，在保持原始字形完整性的同时传达语义，辅助专家解读未释甲骨文。

## 研究背景与动机

**甲骨文研究现状**：甲骨文是约3000年前商代最早的文字之一，已发现约4500个甲骨文字符，但仅约1600个已被成功释读，剩余近3000个仍未被破解。

**解读过程的复杂性**：甲骨文释读需要综合考虑(1)字形组成分析——还原古代符号所代表的场景，(2)上下文语境推理——推断含义，(3)字形演变追溯——找到对应的现代汉字。

**现有AI方法的局限**：
   - **OBS Decipher (OBSD)**：用条件扩散模型直接将甲骨文翻译为现代汉字，但缺乏显式的语义分析过程，可解释性差
   - **GenOV**：用ControlNet生成照片级图像，但无法保留字形结构信息
   - **Word-As-Image**：语义字体排印方法，但只约束外轮廓，对复杂甲骨文结构（多偏旁）效果差

**核心挑战**：
   - **(1)** 如何系统地分析甲骨文字形结构并解释各部件含义？
   - **(2)** 如何在重构语义场景的同时保持原始字形结构？

**本文切入点**：将甲骨文解读转化为"结构化语义字体排印"问题——分析字形偏旁→定位关键部件→在保持字形的前提下将偏旁变形为语义相关的图形。

## 方法详解

### 整体框架
两阶段流水线：
- **Stage 1 (OBSUG)**：用MLLM分析甲骨文字形，输出关键部件、空间位置关系、语义描述、布局bounding box
- **Stage 2 (SOVF)**：用SDS损失驱动SVG参数优化，同时施加字形结构约束（GSDS）和骨架保持约束（SKST），生成语义丰富且忠实于原始字形的矢量字体

### 关键设计1：Oracle Glyph Vectorization（OGV）

将甲骨文光栅图像转化为可微分的SVG矢量表示：
1. 用Zhang-Suen骨架化算法提取 $n$ 条单像素宽的骨架路径 $P^s = \{p_i^s\}_{i=1}^n$
2. 计算方向向量 $\mathbf{v}_j = C_{j+1}^s - C_j^s$，旋转90°得到法向量 $\mu_j$
3. 用滑动窗口平滑法向量：$\tilde{\mu}_j = \frac{1}{k} \sum_{i=j-k/2}^{j+k/2} \mu_i$
4. 基于笔画宽度 $w$ 生成左右轮廓点并用三次样条拟合
5. 骨架控制点嵌入SVG，为后续SKST损失提供结构约束

### 关键设计2：OBSUG（MLLM分析甲骨文）

利用QWEN-VL的多轮对话能力，分三个子阶段处理甲骨文数据：

**(1) 关键结构部件识别**：

$$K_i = \Psi(o_i; \theta_{key})$$

MLLM识别甲骨文中的偏旁/关键元素，如"鸟"、"山"、"人"等。

**(2) Spatial Awareness Reasoning（SAR）**：引导模型输出各部件之间的空间位置关系，构建有向无环图（DAG）：

$$G_i = (V, E), \quad V = K_i, \quad E = \{(k_a, k_b, r) \mid k_a, k_b \in V\}$$

$$R_i = \Psi(o_i, K_i; \theta_{\text{spa}})$$

SAR增强了MLLM对空间结构的理解能力，是字形分析的关键一步。

**(3) 细粒度语义生成**：综合部件和空间关系，生成整体语义描述：

$$T_i = \Psi(o_i, K_i, R_i; \theta_{cap})$$

**(4) Visual Grounding**：定位各部件在字形中的空间位置（bounding box）：

$$L_i = \Psi(o_i, K_i, R_i, T_i; \theta_{loc})$$

### 关键设计3：Structural Oracle Vector Fusion（SOVF）

**LSDS损失**：标准的Latent Score Distillation Sampling，驱动SVG参数优化使生成结果匹配语义文本prompt。

**GSDS损失（字形结构约束）**：在Stable Diffusion的cross-attention层施加区域约束：
- **区域内约束**：最大化指定区域内的cross-attention响应：$\mathcal{L}_{IR} = 1 - \frac{1}{P}\sum \text{TopK}(A_j^t \cdot M_j, P)$
- **区域外约束**：最小化指定区域外的响应：$\mathcal{L}_{OR} = \frac{1}{P}\sum \text{TopK}(A_j^t \cdot (1-M_j), P)$

确保"鸟"在鸟的位置、"山"在山的位置生成。

**SKST损失（骨架结构保持）**：利用Delaunay三角化关联骨架点和轮廓点，约束生成形状与原始字形的角度一致性：

$$\mathcal{L}_{SKST}(P_m, \hat{P}_m) = \frac{1}{N} \sum_{i,j,k} \text{ReLU}(-\cos\theta_{i,j,k})$$

$$\cos\theta_{i,j,k} = \frac{\vec{\alpha_{i,j,k}} \cdot \hat{\vec{\alpha_{i,j,k}}}}{\|\vec{\alpha_{i,j,k}}\| \|\hat{\vec{\alpha_{i,j,k}}}\|}$$

**总优化目标**：

$$\min_P \nabla_P \mathcal{L}_{\text{LSDS}} + w \cdot \nabla_P \mathcal{L}_{\text{GSDS}} + \beta \cdot \mathcal{L}_{SKST} + \gamma_t \cdot \mathcal{L}_{tone}$$

其中 $w$ 为可学习权重，$\beta = 0.5$。

### RMOBS数据集
构建了包含900个已释读甲骨文字、共20K+样本的多模态数据集，每个样本包含甲骨文字形图像、语义概念、关键部件标注和bounding box布局。

## 实验

### 定量结果

| 方法 | CLIPScore ↑ | Distance ↓ | SR ↑ | VA ↑ | GM ↑ |
|------|------------|-----------|------|------|------|
| ClipDraw | 27.78 | 1.05 | 3.15 | 3.19 | 3.17 |
| Word-As-Image | 27.28 | 0.92 | 3.42 | 3.48 | 3.44 |
| **OracleFusion** | **28.30** | **0.86** | **3.97** | **3.90** | **3.95** |

OracleFusion在CLIPScore（语义相关性）、Distance（字形保持）、用户研究三个维度均最优。用户研究由70名熟悉汉字构造原理的参与者在28个随机选择的甲骨文上评分（1-5分）。

### OBSUG消融实验

| 方法 | Acc (%) ↑ | BLEU-4 ↑ | Radical mIoU ↑ | Holistic mIoU ↑ |
|------|----------|----------|---------------|----------------|
| End-to-End | 81.03 | 0.843 | 72.56 | 87.60 |
| Multi-Turn | 81.65 | 0.865 | 72.03 | 91.00 |
| **+ SAR** | **82.02** | **0.876** | **73.94** | **92.10** |

SAR在所有指标上均有提升，尤其mIoU提升最大，证明空间感知推理对准确的结构建模至关重要。

### GSDS损失消融
无GSDS损失时，生成结果无法传达完整概念——例如"灾"字（房屋失火场景），不加GSDS会把火错误地变形为着火的房子，而加上GSDS后能正确在指定区域分别生成"火"和"房"。

### SKST损失消融
- $\beta = 0$：输出类似Word-As-Image的ACAP损失，无法保持复杂甲骨文结构
- $\beta = 1$：过度约束，输出接近原始输入，失去语义表达
- $\beta = 0.5$：最佳平衡，既保持字形轮廓又传达语义

### 未释甲骨文的解读演示
OracleFusion对**未释读的甲骨文**生成了合理的语义解释（如"下垂的庄稼"、"带刺的种子"），为专家分析提供了有价值的线索。对**已释读的甲骨文**则能准确复现其结构和语义。

## 亮点与洞察
1. **问题定义别出心裁**：将甲骨文解读转化为"语义字体排印"问题，让AI不只是翻译文字，而是重构古代文字所描绘的场景——这更符合考古学家的实际解读流程
2. **MLLM + SDS的精妙结合**：MLLM负责"理解"（分析结构、推理语义、定位部件），SDS负责"生成"（将理解转化为视觉表达），分工明确
3. **矢量表示的独特优势**：SVG格式可无损缩放、支持后续风格化（颜色/纹理），黑白设计聚焦语义表达
4. **RMOBS数据集的贡献**：20K+样本、900字符的标注数据集远超此前GenOV的364字符，为该领域提供重要基础设施
5. **SAR的设计逻辑**：空间感知推理引导MLLM输出偏旁间的相对位置关系，类似于人类专家的分析思路

## 局限性
1. 生成质量受限于Stable Diffusion的先验能力——对于过于抽象的甲骨文概念，SD可能无法生成语义匹配的图像
2. OGV方法依赖骨架化算法的质量，对噪声较大的甲骨文拓片可能效果不佳
3. MLLM（QWEN-VL）对甲骨文的理解能力有限，fine-tuning数据集仅覆盖900个字符
4. 用户研究规模有限（70人、28个样本），统计显著性需要更大规模验证
5. 本文关注的是"辅助解读"而非"自动解读"，无法直接给出甲骨文对应的现代汉字

## 相关工作
- **甲骨文处理**：OBSD（扩散模型翻译甲骨文）、GenOV（VLM语义扩展）
- **语义字体排印**：Word-As-Image（SDS+轮廓约束）、DS-Fusion（对抗学习）
- **矢量图形生成**：VectorFusion、DiffSketcher（SDS→SVG）
- **MLLM布局生成**：LayoutGPT、GLIGEN

## 评分
- 新颖性：5/5（问题定义开创性，将古文字学与现代AI深度结合）
- 技术深度：4/5（OGV骨架化、SAR空间推理、GSDS区域约束、SKST骨架保持，技术栈丰富）
- 实验充分度：3/5（缺乏大规模定量评估，主要依赖用户研究和定性分析）
- 写作质量：4/5（故事讲得好，但部分公式和符号略冗余）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Geometrically-Constrained Agent for Spatial Reasoning](../../CVPR2026/multimodal_vlm/geometrically-constrained_agent_for_spatial_reasoning.md)
- [\[ICCV 2025\] Background Invariance Testing According to Semantic Proximity](background_invariance_testing_according_to_semantic_proximity.md)
- [\[ICCV 2025\] MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding](musevl_modeling_unified_vlm_through_semantic_discrete_encodi.md)
- [\[CVPR 2026\] GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks](../../CVPR2026/multimodal_vlm/guide_a_benchmark_for_understanding_and_assisting_users_in_open-ended_gui_tasks.md)
- [\[AAAI 2026\] Aligning the True Semantics: Constrained Decoupling and Distribution Sampling for Cross-Modal Alignment](../../AAAI2026/multimodal_vlm/aligning_the_true_semantics_constrained_decoupling_and_distr.md)

</div>

<!-- RELATED:END -->
