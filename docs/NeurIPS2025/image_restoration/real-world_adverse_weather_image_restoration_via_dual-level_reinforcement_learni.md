---
title: >-
  [论文解读] Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start
description: >-
  [NeurIPS 2025][图像恢复][恶劣天气图像复原] 提出双层强化学习框架（DRL），结合物理驱动的百万级合成天气数据集HFLS-Weather进行高质量冷启动训练，通过局部扰动驱动图像质量优化（PIQO）和全局元控制器多智能体协作，实现真实恶劣天气图像的自适应复原。 现有深度学习天气复原方法主要在合成数据上训练…
tags:
  - "NeurIPS 2025"
  - "图像恢复"
  - "恶劣天气图像复原"
  - "强化学习"
  - "GRPO"
  - "多智能体系统"
  - "无参考图像质量评估"
---

# Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start

**会议**: NeurIPS 2025  
**arXiv**: [2511.05095](https://arxiv.org/abs/2511.05095)  
**代码**: [有](https://github.com/xxclfy/AgentRL-Real-Weather)  
**领域**: 图像复原 / 恶劣天气  
**关键词**: 恶劣天气图像复原, 强化学习, GRPO, 多智能体系统, 无参考图像质量评估

## 一句话总结
提出双层强化学习框架（DRL），结合物理驱动的百万级合成天气数据集HFLS-Weather进行高质量冷启动训练，通过局部扰动驱动图像质量优化（PIQO）和全局元控制器多智能体协作，实现真实恶劣天气图像的自适应复原。

## 研究背景与动机
现有深度学习天气复原方法主要在合成数据上训练，面临三大核心局限：(1) 合成天气数据集缺乏物理精度，深度图粗糙导致不真实的伪影；(2) 固定参数模型无法适应真实世界中不可预测的退化模式；(3) 单模型架构无法针对不同天气类型进行动态协调。作者观察到，类LLM的强化学习方法（如GRPO）在图像复原任务中尚未被成功应用，原因在于图像复原模型通常对每个输入只输出一个固定结果，且缺乏配对GT的情况下难以设计确定性奖励。本文首次成功将GRPO概念引入图像复原，证明高质量冷启动和有效奖励设计是关键。

## 方法详解

### 整体框架
系统分为三个阶段：(1) 构建HFLS-Weather数据集并训练天气特定复原模型（冷启动）；(2) 局部级PIQO对单个模型进行无监督强化学习精调；(3) 全局级多智能体系统通过元控制器动态编排模型选择和执行顺序。

### 关键设计

**HFLS-Weather数据集**：利用DepthAnything v2生成高精度深度图，基于大气散射模型模拟雨、雾、雪及混合天气。包含100万张高保真合成图像，深度一致的多天气模拟克服了现有数据集（如Snow100K、RESIDE-OTS）中因深度估计粗糙导致的鬼影和不均匀天气效果。天气退化公式为：$I_{weather}(x) = J(x)(1-M(x)-F(x)) + M(x) + A(x)F(x)$，其中$F(x)=e^{-\beta d(x)}$为雾层，$M(x)$为雨/雪层。

**PIQO（扰动驱动图像质量优化）**：针对GRPO在图像复原中的两大障碍进行改造：(1) 对模型参数注入高斯扰动$\theta_i' = \theta + \Delta$，使单一输入产生多样化输出，便于组内比较；(2) 设计无参考组合奖励函数，融合LIQE、CLIP-IQA和Q-Align三个质量评估指标。用MUSIQ作为过滤标准，仅保留质量高于未扰动模型输出的样本，计算归一化优势$A_i$后执行梯度上升。引入隐式KL正则化防止参数更新过大。

**多智能体系统**：元控制器基于CLIP识别输入图像的天气退化类型，生成天气描述。各专业Agent根据历史成功率进行竞标，系统选取最高排名Agent执行复原。复原后通过CLIP重新分析和IQA评分双重评估：若IQA下降则回退并选择次高Agent，若仍有退化则进入下一轮。系统限制单次最多3个Agent参与，3次连续失败则返回最高IQA得分图像。

### 损失函数 / 训练策略
- 冷启动阶段：在HFLS-Weather上用标准监督损失训练天气特定模型
- PIQO阶段：策略梯度 $g = -\frac{1}{|\mathcal{S}|}\sum_{i \in \mathcal{S}} A_i(\theta_i' - \theta)$，带KL信赖域约束
- 全局阶段：以IQA综合得分为奖励，优化元控制器的调度策略

## 实验关键数据

### 主实验

| 方法 | Snow Q-Align | Snow CLIP-IQA | Snow MUSIQ | Haze Q-Align | Rain Q-Align | Rain MUSIQ |
|------|-------------|---------------|------------|-------------|-------------|------------|
| Chen et al. | 3.59 | 0.496 | 60.21 | 3.11 | 3.76 | 54.24 |
| WGWS | 3.59 | 0.503 | 60.48 | 3.11 | 3.80 | 54.55 |
| PromptIR | 3.65 | 0.529 | 61.17 | 3.09 | 3.81 | 54.67 |
| DA-CLIP | 3.63 | 0.522 | 61.16 | 3.13 | 3.81 | 54.96 |
| **Ours** | **3.96** | **0.592** | **67.80** | **3.56** | **4.03** | **64.12** |

| 天气 | 指标 | Chen | WGWS | PromptIR | DA-CLIP | Ours |
|------|------|------|------|----------|---------|------|
| Snow | Artifact Removal ↑ | 2.95 | 2.66 | 3.06 | 3.05 | **最优** |

### 消融实验
- HFLS-Weather冷启动对后续RL至关重要，低质量冷启动会导致RL训练崩溃
- PIQO中的奖励过滤策略有效减少梯度方差
- 多智能体系统比单一All-in-One模型在混合天气场景下表现更优

### 关键发现
- 在所有真实天气场景中，方法在四个IQA指标上大幅超越之前SOTA，MUSIQ指标上雪场景提升约7分、雾场景提升约9分
- GPT-4o虽能生成视觉上吸引人的结果，但常产生幻觉物体和结构扭曲，不适合需要几何和光度保真度的任务
- 这是首次在图像复原中成功应用GRPO概念的工作

## 亮点与洞察
1. **创新性极高**：将LLM领域的GRPO范式迁移到图像复原，通过参数扰动解决输出多样性问题，通过无参考IQA组合解决奖励设计问题
2. **百万级物理驱动数据集**：HFLS-Weather在规模和物理保真度上远超现有数据集
3. **闭环学习生态**：局部模型通过真实反馈持续改进，全局控制器动态优化协调
4. **无需配对GT的强化学习**：突破了图像复原领域对成对训练数据的依赖

## 局限与展望
- 多智能体系统的推理效率较低，需要多轮迭代评估
- 奖励函数依赖现有IQA模型的准确性，可能在某些极端场景下失效
- PIQO的参数扰动范围需要精心调节，过大会导致图像退化
- 未考虑视频级别的时序一致性问题

## 相关工作与启发
- 与RestoreAgent、AgenticIR等LLM驱动的多智能体复原方法相比，本文通过RL实现自主学习而非依赖固定工具
- GRPO到视觉任务的迁移思路可推广至其他低级视觉任务（如超分辨率、去模糊）
- 物理驱动数据合成 + RL自适应精调的范式具有广泛适用性

## 评分
- 新颖性：⭐⭐⭐⭐⭐（首次将GRPO引入图像复原）
- 技术深度：⭐⭐⭐⭐⭐（物理模拟+双层RL+多智能体系统）
- 实验完整性：⭐⭐⭐⭐（缺少消融细节但主实验充分）
- 实用价值：⭐⭐⭐⭐（真实天气场景适应性强）
- 综合评价：⭐⭐⭐⭐⭐（开创性工作，将LLM训练范式迁移到视觉领域）

## 补充说明

HFLS-Weather 数据集的构建使用了来自 Snow100K、RESIDE-OTS、Google Landmark V2 和 OSV5M 的多样化背景图像，保证了场景多样性。数据集在规模（100万对）和天气覆盖（雨/雪/雾及其组合）方面均大幅超越现有基准。GPT-4o 的定性评估结果（Table 3）显示本文方法在 Artifact Removal、Color Accuracy、Detail Preservation 等维度均获得最高分。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration](modem_a_morton-order_degradation_estimation_mechanism_for_adverse_weather_image_.md)
- [\[CVPR 2026\] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](../../CVPR2026/image_restoration/beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)
- [\[CVPR 2025\] Pixel-level and Semantic-level Adjustable Super-resolution: A Dual-LoRA Approach](../../CVPR2025/image_restoration/pixel-level_and_semantic-level_adjustable_super-resolution_a_dual-lora_approach.md)
- [\[NeurIPS 2025\] DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution](dp2o-sr_direct_perceptual_preference_optimization_for_real-world_image_super-res.md)
- [\[CVPR 2026\] RL-ScanIQA: Reinforcement-Learned Scanpaths for Blind 360deg Image Quality Assessment](../../CVPR2026/image_restoration/rl-scaniqa_reinforcement-learned_scanpaths_for_blind_360deg_image_quality_assess.md)

</div>

<!-- RELATED:END -->
