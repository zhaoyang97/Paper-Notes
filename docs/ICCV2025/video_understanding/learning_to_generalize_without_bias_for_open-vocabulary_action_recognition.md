---
title: >-
  [论文解读] Learning to Generalize Without Bias for Open-Vocabulary Action Recognition
description: >-
  [ICCV 2025][视频理解][开放词汇动作识别] 本文提出 Open-MeDe，一个基于元学习的开放词汇动作识别框架，通过跨批次元优化模拟"已知到开放"的泛化任务，并结合高斯自集成稳定化策略，在不依赖 CLIP 正则化的情况下同时提升上下文内和上下文外场景的泛化能力。 开放词汇动作识别（OVAR）要求模型识别训练阶段未…
tags:
  - "ICCV 2025"
  - "视频理解"
  - "开放词汇动作识别"
  - "元学习"
  - "静态偏置消除"
  - "CLIP适配"
  - "自集成"
---

# Learning to Generalize Without Bias for Open-Vocabulary Action Recognition

**会议**: ICCV 2025  
**arXiv**: [2502.20158](https://arxiv.org/abs/2502.20158)  
**代码**: [GitHub](https://github.com/Mia-YatingYu/Open-MeDe)  
**领域**: 视频理解  
**关键词**: 开放词汇动作识别, 元学习, 静态偏置消除, CLIP适配, 自集成

## 一句话总结
本文提出 Open-MeDe，一个基于元学习的开放词汇动作识别框架，通过跨批次元优化模拟"已知到开放"的泛化任务，并结合高斯自集成稳定化策略，在不依赖 CLIP 正则化的情况下同时提升上下文内和上下文外场景的泛化能力。

## 研究背景与动机
开放词汇动作识别（OVAR）要求模型识别训练阶段未见过的动作类别，对视频学习器的泛化能力和零样本能力提出了极高要求。当前主流方法基于 CLIP 进行视频适配，但面临一个核心矛盾：

**痛点 1：标准微调导致过拟合**。直接微调 CLIP 的视频学习器容易过拟合到训练类别，在已知类别上表现良好但零样本泛化能力退化。

**痛点 2：CLIP 正则化引入静态偏置**。Open-VCLIP 和 FROSTER 等方法通过正则化约束模型不偏离 CLIP 的泛化能力，在上下文内（in-context）评估中效果不错。然而，由于 CLIP 是图像预训练模型，这种正则化会使视频学习器过度依赖**捷径静态特征**（如场景背景），而忽略关键的运动线索。当在上下文外（out-of-context）场景中测试（如替换视频背景后），性能急剧下降。

**核心矛盾**：CLIP 的静态泛化能力是一把双刃剑——有助于上下文内泛化，却阻碍了运动线索的学习，损害了上下文外泛化。

**切入角度**：从元学习"学会泛化"的视角出发，让视频学习器在训练过程中被显式鼓励快速泛化到任意后续数据，从而最小化对已知数据和静态线索的固有偏置。核心 idea：**用跨批次元优化自然模拟已知到开放的泛化任务，无需 CLIP 正则化即可实现静态去偏。**

## 方法详解

### 整体框架
Open-MeDe 由两个核心组件构成：（1）跨批次元优化方案，通过 support-query 双批次训练模拟已知→开放的泛化任务；（2）高斯自集成稳定化（GWA），在优化轨迹上进行加权平均以获得鲁棒的泛化参数。整个框架是模型无关的，可以集成到任何 CLIP-based 视频学习器中。

### 关键设计
1. **跨批次元优化（Cross-batch Meta-optimization）**:

    - 功能：将每个训练步扩展为双批次操作——当前批次作为 support set（元训练），后续批次作为 query set（元测试）
    - 核心思路：
        - **内循环（元训练）**：在 support batch $\mathcal{S}$ 上用标准交叉熵损失更新得到 fast weights：$\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\mathcal{S}}(\theta)$
        - **外循环（元测试+元优化）**：用 fast weights 在 query batch $\mathcal{Q}$ 上评估泛化性能，然后联合 support 和 query 的损失进行全局优化：$\min_\theta (\mathcal{L}_{\mathcal{T}_i}^{\mathcal{S}}(\theta) + \mathcal{L}_{\mathcal{T}_i}^{\mathcal{Q}}(\theta_i'))$
        - 采用一阶近似（FOMAML）避免二阶梯度计算
    - 设计动机：不同批次的类别分布天然不同，因此跨批次评估自然模拟了"已知到开放"的泛化场景。相比传统 MAML 需要构造 N-way K-shot 任务，本方法无需额外开销——直接利用训练数据采样器的随机性

2. **高斯自集成稳定化（Gaussian Weight Average, GWA）**:

    - 功能：对优化轨迹上的模型参数进行加权平均，获得泛化性更强的最终参数
    - 核心思路：用高斯分布 $w_t = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(t-\mu)^2}{2\sigma^2}}$ 为每个 epoch 的参数分配权重。归一化后 $\alpha_t = w_t / \sum_i w_i$，通过移动平均方式更新：$\theta_{\text{GWA}} \leftarrow \frac{\sum_{i=1}^{t-1} w_i}{\sum_{i=1}^{t} w_i} \cdot \theta_{\text{GWA}} + \frac{w_t}{\sum_{i=1}^{t} w_i} \cdot \theta_t$
    - 设计动机：早期 epoch 的参数保留过多 CLIP 静态偏置，后期 epoch 的参数过度专业化——GWA 赋予中间 epoch 更高权重，不包含 CLIP 原始权重 $\theta_0$，实现静态去偏同时保持泛化

3. **无 CLIP 正则化的隐式去偏**:

    - 功能：在元优化过程中自然消除静态偏置，无需显式的 CLIP 正则化项
    - 核心思路：元学习的虚拟评估机制迫使模型学习真正可泛化的视频特征，而非依赖 CLIP 的静态先验。query batch 提供的反馈鼓励模型捕获运动线索而非静态捷径
    - 设计动机：显式 CLIP 正则化既增加计算开销，又强制保留静态特征，是导致上下文外性能退化的根源

### 损失函数 / 训练策略
- 基础损失函数为标准视觉-语言交叉熵损失 $\mathcal{L}_{CE}$
- 元优化目标：$\theta \leftarrow \theta - \beta \sum_{i=1}^{N} (\nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\mathcal{S}}(\theta) + \delta \nabla_{\theta_i'} \mathcal{L}_{\mathcal{T}_i}^{\mathcal{Q}}(\theta_i'))$
- 文本编码器冻结，仅优化视觉编码器中的时序适配模块
- 在 K400 上训练，前 2 个 epoch 为预热阶段

## 实验关键数据

### 主实验

| 数据集 | 指标 | Open-MeDe | Open-VCLIP | FROSTER | Frozen CLIP |
|--------|------|-----------|------------|---------|-------------|
| K400 (Novel) | Top-1 Acc | **63.8** | 62.3 | 61.9 | 53.4 |
| K400 (HM) | Top-1 Acc | **69.9** | 68.6 | 68.3 | 57.5 |
| HMDB (Novel) | Top-1 Acc | **56.4** | 50.2 | 49.9 | 46.8 |
| UCF (Novel) | Top-1 Acc | **78.5** | 77.2 | 76.9 | 63.6 |
| SSv2 (HM) | Top-1 Acc | **14.3** | 12.9 | 12.4 | 5.1 |

跨数据集零样本评估（训练于 K400，测试于其他数据集）：

| 数据集 | Open-MeDe | Open-VCLIP | FROSTER | Frozen CLIP |
|--------|-----------|------------|---------|-------------|
| UCF | **83.7±1.3** | 83.3±1.4 | 82.9±0.6 | 73.8±0.6 |
| HMDB | **54.6±1.1** | 53.8±1.5 | 53.4±1.2 | 47.9±0.5 |
| K600 | **73.7±0.9** | 73.0±0.8 | 71.1±0.8 | 68.1±1.1 |

### 消融实验

| 配置 | UCF (in-ctx) | UCF-SCUBA (out-ctx) | HM | 说明 |
|------|-------------|--------------------|----|------|
| Frozen CLIP | 73.8 | 42.0 | 54.1 | 基线，无微调 |
| Open-VCLIP | 83.3 | 33.2 | 47.4 | CLIP正则化反而降低out-ctx |
| FROSTER | 82.9 | 34.1 | 48.1 | 同上 |
| Open-MeDe (无GWA) | 82.5 | 41.8 | 55.2 | 仅元优化 |
| Open-MeDe (完整) | **83.7** | **44.7** | **57.6** | 元优化+GWA |

### 关键发现
- **CLIP 正则化是一把双刃剑**：Open-VCLIP 和 FROSTER 在上下文内提升明显，但在上下文外（UCF-SCUBA）甚至不如冻结 CLIP，证实了静态偏置问题的严重性
- 元优化即使不加 GWA 也能在上下文外场景接近冻结 CLIP 水平，同时保持上下文内的高性能
- GWA 进一步提升上下文外表现约 2.9 个百分点
- 在时序敏感的 SSv2 数据集上提升尤为显著（HM: 14.3 vs 12.9），说明方法确实增强了运动线索的学习
- 方法是模型无关的，可以应用到不同的视频适配器架构上

## 亮点与洞察
- **问题定义精准**：首次从"静态偏置"角度解析 CLIP 正则化方法在 out-of-context 场景下退化的原因
- **优雅的元学习设计**：不需要额外构造 task distribution，直接利用 mini-batch 的天然随机性模拟泛化任务
- **无额外计算开销**：相比 CLIP 正则化方法（需要额外前向传播计算蒸馏损失），元优化仅需在已有批次上多做一步梯度计算
- **GWA 是简单但有效的集成策略**：高斯先验赋予中间 epoch 最大权重，避免了早期静态偏置和后期过度专业化

## 局限与展望
- 元优化需要两倍批次数据（support + query），虽然理论上无额外开销，但实际内存占用增加
- 在 SSv2 上的绝对性能仍然较低（HM 14.3），说明纯静态去偏可能不足以解决时序推理问题
- GWA 的 $\mu$ 和 $\sigma$ 需要手动调整，缺乏自适应机制
- 未在更大的 CLIP 模型（ViT-L）上验证

## 相关工作与启发
- 从"学会泛化"而非"学会正则化"的角度解决开放词汇问题，提供了新的思路
- 跨批次元优化的思路可以推广到其他视觉-语言适配场景（如 open-vocabulary detection）
- GWA 策略可以作为通用的微调稳定技术，替代 EMA 或 SWA

## 评分
- 新颖性: ⭐⭐⭐⭐ 元学习用于OVAR的视角新颖，但MAML和权重平均都是成熟技术
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖in-context和out-of-context多种设置，消融详尽
- 写作质量: ⭐⭐⭐⭐ 动机分析深刻，但公式符号偶有冗余
- 价值: ⭐⭐⭐⭐ 对OVAR领域的静态偏置问题有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking](attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)
- [\[NeurIPS 2025\] Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition](../../NeurIPS2025/video_understanding/seeing_beyond_the_scene_analyzing_and_mitigating_background_bias_in_action_recog.md)
- [\[CVPR 2025\] Anomize: Better Open Vocabulary Video Anomaly Detection](../../CVPR2025/video_understanding/anomize_better_open_vocabulary_video_anomaly_detection.md)
- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [\[ECCV 2024\] SLAck: Semantic, Location, and Appearance Aware Open-Vocabulary Tracking](../../ECCV2024/video_understanding/slack_semantic_location_and_appearance_aware_open-vocabulary_tracking.md)

</div>

<!-- RELATED:END -->
