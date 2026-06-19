---
title: >-
  [论文解读] Minority-Focused Text-to-Image Generation via Prompt Optimization
description: >-
  [CVPR 2025][图像生成][少数样本生成] MinorityPrompt 提出了一种在线 prompt 优化框架，通过在推理过程中迭代优化可学习 token embedding 来最大化似然度损失，引导 T2I 扩散模型生成处于数据分布低密度区域的少数(minority)样本，同时保持语义一致性和生成质量。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "少数样本生成"
  - "提示学习"
  - "文本到图像"
  - "低密度采样"
  - "扩散模型偏见"
---

# Minority-Focused Text-to-Image Generation via Prompt Optimization

**会议**: CVPR 2025  
**arXiv**: [2410.07838](https://arxiv.org/abs/2410.07838)  
**代码**: [https://github.com/soobin-um/MinorityPrompt](https://github.com/soobin-um/MinorityPrompt)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 少数样本生成, Prompt优化, 文本到图像, 低密度采样, 扩散模型偏见

## 一句话总结
MinorityPrompt 提出了一种在线 prompt 优化框架，通过在推理过程中迭代优化可学习 token embedding 来最大化似然度损失，引导 T2I 扩散模型生成处于数据分布低密度区域的少数(minority)样本，同时保持语义一致性和生成质量。

## 研究背景与动机

1. **领域现状**：文本到图像（T2I）扩散模型配合 CFG 引导已能生成高质量、忠实于提示的图像。CFG 等引导技术本质上倾向于从数据流形的高密度区域采样，生成"典型"图像。
2. **现有痛点**：这种高密度偏好使得模型很难生成少数(minority)样本——位于条件数据分布低密度区域的独特实例。这导致 T2I 生成的数据缺乏多样性，且会在下游应用（如数据增强）中延续和放大偏见（如年龄、种族刻板印象）。
3. **核心矛盾**：现有少数样本采样方法要么需要外部分类器（难以获取），要么仅在简单图像基准上有效，在 T2I 场景下表现有限。现有在线 prompt 优化方法修改整个文本嵌入，容易破坏原始提示的语义。
4. **本文目标** 如何在保持文本语义的前提下，引导 T2I 模型生成低密度区域的独特minority样本？
5. **切入角度**：不修改整个文本嵌入，而是在 prompt 末尾附加一个可学习 token，仅优化这个 token 的嵌入来最大化重建损失（似然度代理），从而在保留语义的同时鼓励生成独特特征。
6. **核心 idea**：通过在推理时在线优化附加的可学习 token embedding 来最大化负 ELBO 近似，使生成结果偏向低似然度的独特minority样本。

## 方法详解

### 整体框架
给定用户提示 $\mathcal{P}$（如 "A portrait of a dog"），MinorityPrompt 在其末尾附加一个占位符字符串 $\mathcal{S}$，得到增强提示 $\mathcal{P}_\mathcal{S}$。在每个采样时间步 $t$，优化 $\mathcal{S}$ 对应的 token embedding $\boldsymbol{v}$，使得基于该 embedding 的去噪结果具有更高的重建损失（近似低似然度）。优化后的 embedding 用于当前步的采样，然后传递到下一步作为初始点继续优化。

### 关键设计

1. **语义保留的 Prompt 优化框架**:

    - 功能：在不破坏原始提示语义的前提下引入可控的额外语义信息。
    - 核心思路：不优化整个文本嵌入 $\mathcal{C}$，而是只优化附加的可学习 token 的嵌入向量 $\boldsymbol{v}$。文本编码器处理增强提示 $\mathcal{P}_\mathcal{S}$ 时，原始 prompt 中每个词的 token embedding 保持不变，仅 $\mathcal{S}$ 对应的嵌入被更新。优化目标为 $\boldsymbol{v}_t^* = \arg\max_{\boldsymbol{v}} \mathcal{J}(\boldsymbol{z}_t, \mathcal{C}_{\boldsymbol{v}})$。
    - 设计动机：直接修改整个 $\mathcal{C}$ 会改变所有 token embedding，破坏语义。仅优化附加 token 是一种更安全的途径，且允许 embedding 随时间步自适应变化（不同于 Textual Inversion 等需要预训练固定 embedding 的方法）。

2. **基于似然度的 Minority 目标函数**:

    - 功能：驱动生成结果偏向低密度区域。
    - 核心思路：定义目标函数 $\mathcal{J}_\mathcal{C}(\boldsymbol{z}_t, \mathcal{C}_{\boldsymbol{v}}) = \mathbb{E}_\epsilon[\|\hat{\boldsymbol{z}}_0(\boldsymbol{z}_t, \mathcal{C}_{\boldsymbol{v}}) - \hat{\boldsymbol{z}}_0(\boldsymbol{z}_{s|t,0}, \mathcal{C})\|^2_2]$，其中第一项用含优化 token 的条件去噪，第二项用原始条件对同一清洁估计的加噪版本再去噪。论文证明这个目标与 $-\log p_\theta(\hat{\boldsymbol{z}}_0 | \mathcal{C})$ 的负 ELBO 等价，因此最大化它等于推动生成结果远离高密度区域。
    - 设计动机：相比朴素的 CFG-based 目标函数，该设计避免了三个问题：(i) 不依赖 CFG 的去噪估计，(ii) 允许梯度通过第二项流动，(iii) 第二项使用原始条件 $\mathcal{C}$ 而非 $\mathcal{C}_{\boldsymbol{v}}$。

3. **稳定化技术 (stop-gradient trick + 退火时间步)**:

    - 功能：稳定优化过程并提升生成质量。
    - 核心思路：将目标函数拆分为 $\tilde{\mathcal{J}}_\mathcal{C} = \mathcal{J}^1_\mathcal{C} + \lambda \mathcal{J}^2_\mathcal{C}$，其中 $\mathcal{J}^1$ 在第二项上加 stop-gradient，$\mathcal{J}^2$ 在第一项上加 stop-gradient，$\lambda=1$ 时效果最佳。同时采用退火时间步 $s = T - t$ 替代固定值。每隔 $N$ 步优化一次，非优化步使用原始 prompt $\mathcal{C}$。
    - 设计动机：双向 stop-gradient 让两项各自承担不同优化方向，退火时间步适配不同噪声水平下的最优重建尺度，间隔优化节省计算成本且稳定输出质量。

### 损失函数 / 训练策略
MinorityPrompt 是推理时方法，不需要额外训练。在推理期间使用 Adam 优化器更新 $\boldsymbol{v}$，每个优化步迭代 $K$ 次。实验中在 SDv1.5 和 SDv2.0 上用 50 步 DDIM + CFG $w=7.5$，在 SDXL-Lightning 上用 4 步 + $w=1.0$。

## 实验关键数据

### 主实验

使用 MS-COCO 验证集 10K 个随机 caption 评测：

| 模型 | 方法 | CLIPScore↑ | PickScore↑ | ImageReward↑ | Likelihood↓ |
|------|------|-----------|-----------|-------------|-------------|
| SDv1.5 | DDIM | 31.48 | 21.48 | 0.211 | 1.037 |
| SDv1.5 | SGMS | 31.17 | 21.21 | 0.123 | 0.954 |
| SDv1.5 | **MinorityPrompt** | **31.54** | 21.31 | **0.235** | **0.897** |
| SDv2.0 | DDIM | 31.85 | 21.68 | 0.382 | 1.110 |
| SDv2.0 | **MinorityPrompt** | **31.96** | 21.60 | **0.425** | **0.914** |
| SDXL-LT | DDIM | 31.52 | 22.67 | 0.733 | 0.608 |
| SDXL-LT | SGMS | 31.30 | 22.58 | 0.680 | 0.546 |
| SDXL-LT | **MinorityPrompt** | 31.34 | 22.61 | 0.710 | **0.546** |

MinorityPrompt 在保持文本对齐度和质量的同时，显著降低了似然度。

### 消融实验

| 配置 | CLIPScore↑ | Likelihood↓ | 说明 |
|------|-----------|-------------|------|
| 全 embedding 优化 | 30.8 | 0.91 | 语义偏移严重 |
| Token 优化 (本文) | 31.5 | 0.90 | 语义保留更好 |
| 无退火时间步 | 31.3 | 0.93 | 固定 s 效果差 |
| 无 stop-gradient trick | 31.4 | 0.92 | 优化不稳定 |

### 关键发现
- **Token 优化 vs 全 embedding 优化**：Token 优化在 CLIPScore 上高出 0.7，证明语义保留效果更好。
- **似然度有效降低**：MinorityPrompt 是唯一在所有模型上都同时降低似然度且保持高文本对齐的方法。SGMS 虽也降低似然度但牺牲了质量。
- **可控语义增强**：通过选择有意义的初始 token embedding（如 "old"、"Asian"），可以引导minority特征的方向，这是纯 latent 空间方法无法实现的。
- 84% 用户在 user study 中偏好 MinorityPrompt 生成的minority样本。

## 亮点与洞察
- **仅优化附加 token 而非整个 embedding 的策略**非常优雅：既保留了原始语义又引入了可学习的自由度，这个思路可以迁移到任何需要在推理时微调条件的生成任务（如引导风格、增强特定属性）。
- **将重建损失与负 ELBO 联系起来的理论推导**给出了明确的优化目标的数学保证，使得看似 heuristic 的"最大化重建误差"有了理论支撑。
- 方法的**隐含偏见缓解功能**很有意义：通过生成minority样本来对抗 T2I 模型中的刻板印象偏见（如将 "man" 与 "young" 关联），具有社会影响力。

## 局限与展望
- 每个优化步需要额外的前向+反向传播，推理速度比标准 DDIM 慢数倍。
- "minority"的定义完全依赖模型学到的分布，如果模型本身在某些区域采样不足，可能无法到达那些区域。
- 在蒸馏模型 (SDXL-Lightning) 上效果不如全步模型显著，可能因为 4 步采样的优化空间有限。
- 缺乏对生成的minority样本的真实性评估——低密度不一定意味着有意义的多样性。

## 相关工作与启发
- **vs SGMS**：SGMS 是之前的 SOTA minority采样方法，但限于非 T2I 场景（LSUN/ImageNet），在 T2I 中效果有限。MinorityPrompt 通过 prompt 优化适配 T2I 架构，质量和似然度双指标更优。
- **vs CADS**：CADS 关注多样性增强而非minority生成，通过对条件嵌入加噪实现。MinorityPrompt 有明确的似然度目标，针对性更强。
- **vs Textual Inversion**：都使用可学习 token，但 Textual Inversion 需要离线训练学习视觉概念，MinorityPrompt 是在线的、目标驱动的优化。

## 评分
- 新颖性: ⭐⭐⭐⭐ 将minority采样问题转化为在线prompt优化是新颖的formulation
- 实验充分度: ⭐⭐⭐⭐ 多种SD版本、多指标、消融和用户研究齐全
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，符号一致，写作流畅
- 价值: ⭐⭐⭐⭐ 对T2I多样性和偏见缓解有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction](pqpp_a_joint_benchmark_for_text-to-image_prompt_and_query_performance_prediction.md)
- [\[CVPR 2025\] Towards Understanding and Quantifying Uncertainty for Text-to-Image Generation](towards_understanding_and_quantifying_uncertainty_for_text-to-image_generation.md)
- [\[CVPR 2025\] ChatGen: Automatic Text-to-Image Generation From FreeStyle Chatting](chatgen_automatic_text-to-image_generation_from_freestyle_chatting.md)
- [\[CVPR 2025\] Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](mitigating_memorization_in_text-to-image_diffusion_via_region-aware_prompt_augme.md)
- [\[CVPR 2025\] Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models](finite_difference_flow_optimization_for_rl_post-training_of_text-to-image_models.md)

</div>

<!-- RELATED:END -->
