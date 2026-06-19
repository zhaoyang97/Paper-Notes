---
title: >-
  [论文解读] Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback
description: >-
  [ACL 2025][多模态VLM][Medical LVLM] 提出 UMed-LVLM，通过 Abnormal-Aware Instruction Tuning 和 Abnormal-Aware Rewarding（包含 Relevance Reward、Abnormal Localization Reward、Vision Relevance Reward）训练策略增强医学 LVLM 的异常区域定位能力，在 MAU 数据集上比基线提升 58%，并展现出优秀的跨模态和 OOD 泛化能力。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "Medical LVLM"
  - "Abnormality Detection"
  - "Visual Localization"
  - "强化学习"
  - "instruction tuning"
  - "Abnormal-Aware Reward"
---

# Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback

**会议**: ACL 2025  
**arXiv**: [2501.01377](https://arxiv.org/abs/2501.01377)  
**代码**: 未公开  
**领域**: Multimodal VLM / 医学图像分析  
**关键词**: Medical LVLM, Abnormality Detection, Visual Localization, Reinforcement Learning, instruction tuning, Abnormal-Aware Reward

## 一句话总结

提出 UMed-LVLM，通过 Abnormal-Aware Instruction Tuning 和 Abnormal-Aware Rewarding（包含 Relevance Reward、Abnormal Localization Reward、Vision Relevance Reward）训练策略增强医学 LVLM 的异常区域定位能力，在 MAU 数据集上比基线提升 58%，并展现出优秀的跨模态和 OOD 泛化能力。

## 研究背景与动机

**领域现状**: Med-LVLMs（如 LLaVA-Med、MedVInt、Med-Flamingo）已能理解医学图像并回答问题，但在**视觉定位**（尤其是异常区域定位）方面存在显著缺陷。即使 GPT-4V 在医学图像的模态识别和解剖结构辨认上表现良好，在疾病诊断和精确定位方面仍有困难。

**视觉定位的关键性**:
   - 定位偏差导致诊断不可靠，损害 Med-LVLM 的可信度和可解释性
   - 增强视觉定位能力可反向提升视觉理解能力（已在自然场景 LVLM 中验证）

**不能用通用检测器的原因**: 自然场景可借助通用检测器（如 YOLO）辅助视觉定位，但医学异常检测缺乏足够数据训练专用检测器，尤其对罕见疾病

**本文方案**: 增强 Med-LVLM 的**内在**视觉定位能力（无需外部检测器），通过异常感知的训练策略使模型在生成诊断时关注异常区域

## 方法详解

### 整体框架

UMed-LVLM 采用两阶段训练：Stage 1 - Abnormal-Aware Instruction Tuning（异常感知的指令微调）；Stage 2 - Abnormal-Aware Rewarding（异常感知的奖励学习），基于 MedVInt 进行持续训练。

### Stage 1: Abnormal-Aware Instruction Tuning

给定医学图像 x 和用户查询 q，模型生成包含**诊断**和**异常区域描述**的响应 a：
$$p(a|x,q;\theta) = \prod_{t=1}^{T} p(a_t|a_{<t},x,q;\theta)$$

训练损失为标准交叉熵：
$$\mathcal{L}_{it} = -\sum_{i=1}^{T} \log p_i$$

这一阶段虽然让模型学会输出异常区域的文本描述，但**不能直接引导模型在视觉上关注异常区域**。

### Stage 2: Abnormal-Aware Rewarding (AAR)

AAR 包含三种奖励机制，基于强化学习框架（改进 PPO）优化：

#### 1. Relevance Reward (相关性奖励)
- **Policy Network π**: 基于状态 s_t (图像+查询) 生成响应
- **Value Network V**: 估计状态的期望回报
- **LLM Relevance Reward r_t^LLM**: 外部 LLM 评估响应的相关性
- 总奖励: r_t^{π,V,LLM} = A(s_t, a_t; θ,ϕ) + r_t^LLM，其中 A 为优势函数
- Q 函数通过 Bellman 方程更新

#### 2. Abnormal Localization Reward (异常定位奖励, ALR)
$$r_t^{loc} = \frac{\text{Overlap}(\text{Pred-BBox}, \text{GT-BBox})}{\text{Union}(\text{Pred-BBox}, \text{GT-BBox})}$$
直接以预测 bounding box 与真实 bounding box 的 IoU 作为奖励，鼓励精确定位异常区域。

#### 3. Vision Relevance Reward (视觉相关性奖励, VRR)
$$r_t^{att} = \sum_{i \in N} \sum_{j \in \bar{N}} \frac{\exp(Q_i \cdot K_j^\top / \sqrt{d_k})}{\sum_{k \in \bar{N}} \exp(Q_i \cdot K_k^\top / \sqrt{d_k})}$$
其中 N 是异常类别 token 集合，$\bar{N}$ 是异常区域的图像 patch 集合。通过聚合 Transformer 注意力权重，衡量模型是否将异常类别相关 token 的注意力集中在异常图像区域。

#### 奖励归一化与聚合
$$r_t = r_t^{\pi,V,LLM} + \frac{r_t^{loc}}{\max(r_t^{loc})} + \frac{r_t^{att}}{\max(r_t^{att})}$$
对同一查询的所有响应分别归一化 ALR 和 VRR，确保各奖励等权贡献。

#### 优化目标
基于 PPO 的改进目标函数：
$$\mathcal{L}^{\text{CLIP+ENT}}(\theta) = \hat{\mathbb{E}}[\mathcal{L}^{\text{CLIP}}(\theta) + c_1 r_t - c_2 \mathcal{L}^{VF}(\phi) + c_3 S[\pi(\cdot|s_t)]]$$
包含 PPO clip 项、组合奖励、价值函数损失和策略熵正则化。

### MAU 数据集构建

- **数据来源**: 5 个医学数据集——DeepLesion (CT, 32120张)、KidneyStone (肾脏CT, 1300张)、NIH (胸部X光, 112120张)、TBX11K (胸部X光/结核, 11200张)、KVASIR (内窥镜, 8000张)
- **数据构建**: 设计 Prompt Method 利用 GPT-4V 生成诊断标注：先提供图像+异常类别+异常区域位置 → GPT-4V 生成诊断 → 反思 prompt 重新组织为"检测→定位→识别"的步骤式诊断
- **规模**: 5,817 张医学图像，含用户查询和带异常区域的诊断响应
- **专家审核**: 3 名医学博士生审核，仅 13 个样本有误并手动修正

## 实验

### 主实验结果（Table 2 - MAU 测试集）

| 方法 | DeepLesion | KidneyStone | KVASIR | NIH | TBX11K | Avg |
|------|-----------|-------------|--------|-----|--------|-----|
| MedVInt | 0.29 | 0.11 | 0.27 | 0.08 | 0.09 | 0.17 |
| GPT-4V | 0.27 | 0.36 | 0.53 | 0.18 | 0.19 | 0.31 |
| MedVInt (SFT) | 0.42 | 0.93 | 0.93 | 0.28 | 0.78 | 0.67 |
| MedVInt (SFT+PPO) | 0.44 | 0.94 | 0.95 | 0.30 | 0.80 | 0.69 |
| **UMed-LVLM** | **0.53** | **0.99** | **0.98** | **0.37** | **0.86** | **0.75** |
| GPT-4V w/ bbox | 0.50 | 0.95 | 0.95 | 0.32 | 0.81 | 0.72 |

UMed-LVLM 以 0.75 平均准确率大幅超越基线 MedVInt (0.17)，比 GPT-4V (0.31) 提升 142%。甚至超过了 GPT-4V 提供异常区域位置信息时的表现 (0.72)。

### 外部基准性能

| 基准 | 指标 | MedVInt | UMed-LVLM |
|------|------|---------|-----------|
| VQA-RAD Open | ACC | 69.3 | **74.9** |
| VQA-RAD Close | ACC | 84.2 | **87.6** |
| SLAKE Open | ACC | 88.2 | **90.4** |
| PMC-VQA Choice | ACC | 39.2 | **42.6** |
| MedMNIST Pneumonia AUC | AUC | 98.5 | **99.1** |

在 VQA-RAD、SLAKE、PMC-VQA、MedMNIST 等外部基准上均超越 SOTA。

### 消融实验（Table 6）

| 方法 | DeepLesion | KidneyStone | KVASIR | NIH | TBX11K | Avg |
|------|-----------|-------------|--------|-----|--------|-----|
| UMed-LVLM | 0.53 | 0.99 | 0.98 | 0.37 | 0.86 | **0.75** |
| w/o VRR | 0.49 | 0.97 | 0.95 | 0.30 | 0.82 | 0.71 |
| w/o ALR | 0.48 | 0.96 | 0.96 | 0.35 | 0.83 | 0.72 |
| w/o AAR | 0.42 | 0.93 | 0.93 | 0.28 | 0.78 | 0.67 |

三个 AAR 组件均有贡献：完整 AAR (+8pp vs SFT-only)、ALR 和 VRR 各贡献约 3-4pp 提升。

### 深度分析

1. **定位精度与诊断性能**: IoU 提升到 0.6 后，诊断性能趋于饱和——Med-LVLM 不需要极高定位精度即可受益
2. **训练 epoch**: 性能在 epoch 4 达到峰值 (~0.75)
3. **数据规模**: 性能随训练数据从 20% 到 100% 持续提升，正相关
4. **未见类别泛化 (Table 7)**: 移除 Abdomen/Lung/Pelvis 类别训练后，UMed-LVLM 仍在这些类别上远超 MedVInt（0.35 vs 0.05），展现异常感知学习的泛化能力
5. **跨数据集泛化 (Table 8)**: 不含 TBX11K 训练时在该数据集上仍达 0.57，不含 DeepLesion 时达 0.42
6. **跨模态泛化 (Table 9)**: 仅用 CT 训练，在 X-ray 和 Gross Pathology 上仍有良好表现，说明异常感知训练学到的是通用的医学异常识别能力

## 亮点与洞察

1. **强化学习在医学影像的创新应用**: 将 PPO 从"对齐用户偏好"改造为"对齐医学异常"，设计了 ALR (定位准确性) 和 VRR (注意力聚焦) 两个领域特定奖励
2. **"定位促理解"的验证**: 实验证实增强异常定位能力可以显著提升医学图像理解，这一因果关系在不同实验设置中一致成立
3. **不需要检测器/分割器**: 无需外部检测器即可实现异常定位，降低了部署门槛
4. **泛化能力出色**: 跨类别、跨数据集、跨模态的泛化实验全面且有说服力，证明模型学到的是通用的异常识别能力而非记忆特定模式
5. **GPT-4V 的启示**: GPT-4V + bbox (0.72) 接近训练后的 UMed-LVLM (0.75)，说明异常区域信息对诊断至关重要

## 局限性

1. 受限于计算资源，未在更大规模的开源 LVLM 上验证（如 LLaVA-1.5-13B、InternVL 等）
2. MAU 数据集仅 5,817 张图像，规模有限
3. 仅在特定的医学影像数据集上评估，未在更广泛的临床场景（如病理切片、MRI）中验证
4. 强化学习训练过程的稳定性和超参数敏感性未充分讨论

## 相关工作

- **Med-LVLMs**: LLaVA-Med、MedVInt、XrayGPT、Med-Flamingo 等医学语言-视觉模型
- **区域感知 LVLMs**: RegionGPT（但依赖外部检测器）、Shikra 等
- **视觉定位**: 自然场景的 LVLM 定位（BBox-GPT等）
- **RLHF/强化学习**: PPO、DPO 在 LLM 对齐中的应用

## 评分 ⭐⭐⭐⭐

- **创新性**: ⭐⭐⭐⭐ AAR 三重奖励机制设计巧妙，VRR 基于注意力权重直接约束模型聚焦异常区域是新颖设计
- **实验完备性**: ⭐⭐⭐⭐⭐ 消融+泛化分析极其全面（跨类别/跨数据集/跨模态/数据规模/IoU影响）
- **方法可解释性**: ⭐⭐⭐⭐ 奖励设计有明确的医学动机，IoU 阈值分析直观
- **实用性**: ⭐⭐⭐ 数据集规模较小，模型未基于最新的 LVLM 底座

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Harnessing PDF Data for Improving Japanese Large Multimodal Models](harnessing_pdf_data_for_improving_japanese_large_multimodal_models.md)
- [\[ICML 2025\] MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization](../../ICML2025/multimodal_vlm/mmedpo_aligning_medical_vision-language_models_with_clinical-aware_multimodal_pr.md)
- [\[ACL 2025\] Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains](weaving_context_across_images_improving_vision-language_models_through_focus-cen.md)
- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](../../ICCV2025/multimodal_vlm/compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [\[ACL 2025\] HSCR: Hierarchical Self-Contrastive Rewarding for Aligning Medical Vision Language Models](hscr_hierarchical_self-contrastive_rewarding_for_aligning_medical_vision_languag.md)

</div>

<!-- RELATED:END -->
