---
title: >-
  [论文解读] Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View
description: >-
  [AAAI 2026][多模态VLM][多模态后训练] 提出两种多模态数据难度评估策略——PISM（渐进图像语义遮蔽）和CMAB（跨模态注意力平衡），发现在难度分层数据上仅用GRPO训练即可一致超越传统SFT+GRPO流水线，证明了战略性数据筛选比复杂训练范式更重要。 领域现状 随着DeepSeek-R1的成功…
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "多模态后训练"
  - "难度感知采样"
  - "GRPO"
  - "强化学习"
  - "数据筛选"
---

# Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View

**会议**: AAAI 2026  
**arXiv**: [2511.06722](https://arxiv.org/abs/2511.06722)  
**代码**: [https://github.com/qijianyu277/DifficultySampling](https://github.com/qijianyu277/DifficultySampling)  
**领域**: 多模态VLM  
**关键词**: 多模态后训练, 难度感知采样, GRPO, 强化学习, 数据筛选

## 一句话总结
提出两种多模态数据难度评估策略——PISM（渐进图像语义遮蔽）和CMAB（跨模态注意力平衡），发现在难度分层数据上仅用GRPO训练即可一致超越传统SFT+GRPO流水线，证明了战略性数据筛选比复杂训练范式更重要。

## 研究背景与动机

### 领域现状
随着DeepSeek-R1的成功，多模态链式思维（CoT）推理成为研究热点。主流方法将强化学习（RL）扩展到多模态模型后训练，但**几乎所有工作都聚焦于数学数据集**（如数学推理题），这更多提升了文本模态的推理能力，而忽视了跨模态能力。

### 核心痛点

**问题一：缺乏多模态数据难度量化指标。** 纯文本数据（特别是数学/代码）可以用题目难度等人工标注来区分，但多模态数据的"难度"无法用单一模态衡量。图像理解任务（如OCR、分类）的文本部分甚至不可量化难度。已有方法要么忽略数据采样，要么用纯文本标准过滤，**完全丢失了图像模态和多模态交互信号**。

**问题二：后训练范式次优。** 当前主流范式是"先SFT再GRPO"，但这套从语言模型直接借鉴来的流水线对多模态任务是否最优？多模态数据可分为视觉推理（数学、科学、图表）和视觉感知（检测、计数、OCR）两大类，每类可能需要不同的最优训练策略。

### 本文切入点
从**模态内**（图像语义敏感度）和**模态间**（注意力分配平衡）两个维度定义多模态数据难度，然后系统对比GRPO-only vs SFT+GRPO在不同难度分层数据上的表现。

## 方法详解

### 整体框架
1. 用PISM和CMAB分别评估数据难度→将数据分为Easy/Medium/Hard/Unsolved四类
2. 对比两种训练范式：GRPO-only和SFT+GRPO的各种组合
3. 在六个基准数据集上评估

### 关键设计

1. **PISM（Progressive Image Semantic Masking）——基于敏感度的难度评估**:

    - **核心思路**：逐步遮蔽图像像素，观察模型预测何时崩溃
    - 定义遮蔽比例序列 $\Lambda=\{0.0, 0.1, ..., 0.9\}$
    - 对每个遮蔽比例重复K=10次随机遮蔽，计算稳健准确率 $P_c(\lambda_i) = \frac{1}{K}\sum_{k=1}^K \delta_{\lambda_i}^{(k)}$
    - 找到**失败阈值** $\lambda_s^* = \min\{\lambda_i \in \Lambda \mid P_c(\lambda_i) < \tau\}$，其中 $\tau=0.1$
    - **难度分类**：
        - Hard: $\lambda_s^* \leq 0.4$（轻微遮蔽就崩溃→高度依赖视觉细节）
        - Medium: $0.4 < \lambda_s^* < 0.7$
        - Easy: $\lambda_s^* \geq 0.7$（大量遮蔽仍正确→文本线索即可回答）
        - Unsolved: 原始图像就答错
    - **设计动机**：如果图像信息至关重要，那么稍微破坏图像，模型就应该出错——这就是"难"样本

2. **CMAB（Cross-Modality Attention Balance）——基于注意力的难度评估**:

    - **核心思路**：分析模型生成回答时对图像token vs 文本token的注意力分配
    - 计算每个生成token在每层的注意力比例 $\rho^{(l,t)} = S_{img}^{(l,t)} / S_{txt}^{(l,t)}$
    - 跨层取几何平均（排除首尾层）：$\rho_t = \exp\left(\frac{1}{L_{layers}-2}\sum_{l=2}^{L_{layer}-1}\log(\rho^{(l,t)}+\epsilon)\right)$
    - 样本级注意力平衡：$\bar{\rho} = \frac{1}{T}\sum_{t=1}^T \rho_t$
    - **难度分类**：
        - Easy: $\bar{\rho} < 0.1$ 或 $\bar{\rho} > 1.9$（单模态主导，不需要复杂跨模态推理）
        - Medium: $0.1 \leq \bar{\rho} < 0.4$ 或 $1.6 < \bar{\rho} \leq 1.9$
        - Hard: $0.4 \leq \bar{\rho} \leq 1.6$（需要平衡利用两种模态→真正的跨模态推理）
    - **设计动机**：当注意力均匀分配在图像和文本之间时，说明两种模态的信息都不可或缺，这才是"难"的多模态样本

3. **训练范式对比**:

    - GRPO-only: 直接在难度分层数据上做GRPO
    - SFT+GRPO: 先在一组数据上SFT，再在另一组上GRPO
    - 穷举了所有可能的难度组合（mid→hard, hard→mid, rand→hard 等）

### 训练配置
- 基础模型：Qwen2.5VL-7B
- SFT使用LLaMA-Factory框架
- GRPO使用Swift框架
- 硬件：NVIDIA A800-SXM4 (8×80GB) × 5节点 + NVIDIA H20 (8×96GB) × 2节点

## 实验关键数据

### 主实验（PISM难度分层 - 视觉推理数据）

| 训练范式 | MathVista | MMVet | OCRBench | HBench | MMMU | MMStar |
|---------|-----------|-------|----------|--------|------|--------|
| GRPO-only(fullset) | 53.4 | 41.7 | 76.2 | 67.4 | 0.440 | 0.607 |
| SFT(mid)+GRPO(hard) | 67.3 | 40.6 | 75.0 | 68.5 | 0.507 | 0.609 |
| SFT(hard)+GRPO(mid) | 67.3 | 39.3 | 74.2 | 67.6 | 0.502 | 0.608 |
| GRPO-only(random) | 68.2 | 53.3 | 77.3 | 68.3 | 0.541 | 0.637 |
| **GRPO-only(mid+hard)** | **68.3** | **48.3** | **77.8** | **68.8** | **0.547** | **0.639** |

### 消融实验（PISM vs CMAB 对比 - 视觉推理数据）

| 策略 | MathVista | MMVet | MMMU | MMStar | 说明 |
|------|-----------|-------|------|--------|------|
| PISM: GRPO(mid+hard) | 68.3 | 48.3 | 0.547 | 0.639 | 感知型任务更优 |
| CMAB: GRPO(mid+hard) | **69.0** | 48.6 | 0.542 | 0.628 | 推理型任务更优 |
| CMAB: GRPO(random) | 68.2 | 43.6 | **0.556** | **0.642** | 随机采样也有竞争力 |

**数据分布（PISM）**：视觉感知数据20,633样本→Easy 7,827 / Medium 4,872 / Hard 1,454 / Unsolved 6,480  
**数据分布（CMAB）**：视觉推理数据27,133样本→Easy 2,170 / Medium 3,604 / Hard 2,166 / Unsolved 19,193

### 关键发现

1. **GRPO-only一致优于SFT+GRPO**：在所有基准上，难度分层的GRPO-only方案超过所有SFT+GRPO组合。这挑战了"SFT是GRPO前提"的普遍假设
2. **SFT导致"伪CoT"**：SFT依赖人工设计的推理模板，可能鼓励表面模式匹配而非真正逻辑推理，增加幻觉风险
3. **数据质量 > 数据数量**：GRPO(mid+hard)用~6k数据超过GRPO(fullset)用~27k数据，MathVista提升14.9%
4. **PISM和CMAB互补**：PISM在感知型任务（OCRBench, MMVet）更强，CMAB在推理型任务（MathVista, MMMU）更强
5. **简化训练流水线**：直接GRPO不仅效果更好，还省去了SFT阶段的计算开销

## 亮点与洞察

- **从多模态视角定义"难度"是本文核心创新**。不同于简单用题目难度或拒绝采样，PISM和CMAB分别从视觉敏感度和注意力平衡两个维度捕捉多模态固有的难度特性
- **"不需要SFT"这一反直觉发现**具有重大实践意义——可以简化训练流水线
- 实验设计极为系统：穷举了所有SFT+GRPO的难度组合，排除了偶然性
- PISM的设计灵感类似于**对抗鲁棒性评估**——通过扰动输入测量模型的依赖程度

## 局限与展望

- 仅在Qwen2.5VL-7B上验证，需要更多模型的泛化性验证
- PISM需要大量推理（每样本10个遮蔽比例×10次重复=100次推理），计算成本较高
- CMAB需要访问模型中间层注意力，对API模型不适用
- 难度阈值（如 $\lambda_{hard}=0.4$, $\lambda_{easy}=0.7$）需要手动设定
- 未探索PISM和CMAB的融合策略（如加权组合两种难度指标）

## 相关工作与启发

- DeepSeek-R1开创了GRPO后训练范式，但未深入多模态数据采样
- VLM-R1、Visual-RFT等将RL推理引入多模态，但未考虑数据难度
- 与课程学习（curriculum learning）有呼应，但从"是否需要课程"的角度提出了更激进的结论
- 启发：未来多模态后训练应**先做数据诊断，再选训练策略**

## 评分
- 新颖性: ⭐⭐⭐⭐（PISM和CMAB是有创意的难度指标设计）
- 实验充分度: ⭐⭐⭐⭐⭐（极其系统的范式对比，覆盖6个基准×十余种训练配置）
- 写作质量: ⭐⭐⭐⭐（结构清晰但部分公式较密）
- 价值: ⭐⭐⭐⭐⭐（对多模态后训练实践有直接指导意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Why Does RL Generalize Better Than SFT? A Data-Centric Perspective on VLM Post-Training](../../CVPR2026/multimodal_vlm/why_does_rl_generalize_better_than_sft_a_data-centric_perspective_on_vlm_post-tr.md)
- [\[ICML 2026\] Med-Scout: Curing MLLMs' Geometric Blindness in Medical Perception via Geometry-Aware RL Post-Training](../../ICML2026/multimodal_vlm/med-scout_curing_mllms_geometric_blindness_in_medical_perception_via_geometry-aw.md)
- [\[AAAI 2026\] Aligning the True Semantics: Constrained Decoupling and Distribution Sampling for Cross-Modal Alignment](aligning_the_true_semantics_constrained_decoupling_and_distr.md)
- [\[ICML 2026\] Conditional Diffusion Sampling](../../ICML2026/multimodal_vlm/conditional_diffusion_sampling.md)
- [\[CVPR 2026\] Label What Matters: Modality-Balanced and Difficulty-Aware Multimodal Active Learning](../../CVPR2026/multimodal_vlm/label_what_matters_modality-balanced_and_difficulty-aware_multimodal_active_lear.md)

</div>

<!-- RELATED:END -->
