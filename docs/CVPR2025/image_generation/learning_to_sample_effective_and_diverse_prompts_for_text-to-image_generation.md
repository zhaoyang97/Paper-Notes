---
title: >-
  [论文解读] Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation
description: >-
  [CVPR 2025][图像生成][提示词优化] 提出PAG（Prompt Adaptation with GFlowNets），将提示词适配重新定义为概率推断问题，利用GFlowNets从奖励分布中采样而非最大化奖励，结合流重激活、奖励优先采样和奖励分解三大技术解决模式坍塌问题，生成既高质量又多样化的文本到图像提示词。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "提示词优化"
  - "GFlowNets"
  - "文本到图像生成"
  - "多样性"
  - "概率推断"
---

# Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation

**会议**: CVPR 2025  
**arXiv**: [2502.11477](https://arxiv.org/abs/2502.11477)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 提示词优化, GFlowNets, 文本到图像生成, 多样性, 概率推断

## 一句话总结
提出PAG（Prompt Adaptation with GFlowNets），将提示词适配重新定义为概率推断问题，利用GFlowNets从奖励分布中采样而非最大化奖励，结合流重激活、奖励优先采样和奖励分解三大技术解决模式坍塌问题，生成既高质量又多样化的文本到图像提示词。

## 研究背景与动机

1. **领域现状**：文本到图像扩散模型（如Stable Diffusion）已能生成高质量图像，但用户难以写出能触发期望属性（如美学质量、用户意图）的提示词。提示词适配（prompt adaptation）通过自动优化用户提示来生成更好的图像。
2. **现有痛点**：先驱工作Promptist使用PPO强化学习优化提示词，但RL的奖励最大化本质导致策略收敛到确定性行为——生成的提示词总是附加类似的后缀，多样性极低，降级为简单的启发式规则。
3. **核心矛盾**：奖励最大化将策略集中在狭窄的高奖励区域，无法探索提示空间中的多样模式。但直接用GFlowNets微调语言模型时，又会出现神经可塑性丧失导致的模式坍塌。
4. **本文目标** (a) 用GFlowNets替代RL实现提示词的多样化采样；(b) 解决GFlowNets微调过程中的模式坍塌问题。
5. **切入角度**：发现GFlowNets微调时"休眠神经元"比例持续上升（类似神经科学中脑回路的渐进硬化），导致模型无法从多样样本中学习。
6. **核心 idea**：将提示词适配从奖励最大化转变为从奖励分布中采样的概率推断问题，通过流重激活、优先采样和奖励分解三管齐下维持GFlowNets的探索能力。

## 方法详解

### 整体框架
输入为用户的初始提示词 $\mathbf{x}$，语言模型（GPT-2）作为GFlowNet策略生成适配后的提示词 $\mathbf{y}$，送入扩散模型（SD v1.4）生成图像并计算奖励，再用GFlowNet的训练目标微调策略使其采样概率正比于奖励。整个过程不修改扩散模型参数。

### 关键设计

1. **GFlowNet概率推断框架**:
    - 功能：让策略采样提示词的概率正比于奖励，而非仅最大化奖励
    - 核心思路：最优策略可以解析推导为 $p_\theta^*(\mathbf{y}|\mathbf{x}) \propto p_{\text{ref}}(\mathbf{y}|\mathbf{x}) \cdot \exp(\frac{1}{\beta}r(\mathbf{x},\mathbf{y}))$，即参考策略乘以奖励的指数。这正是GFlowNet的采样目标——从非归一化密度函数中采样。奖励函数包含美学质量提升（LAION预测器）和文本-图像相关性（CLIP相似度）两部分。
    - 设计动机：相比RL的奖励最大化，GFlowNet按奖励比例采样天然保证了解的多样性——高奖励区域被频繁采样但不会独占。

2. **流重激活 + 奖励优先采样 (Flow Reactivation + Reward-Prioritized Sampling)**:
    - 功能：解决GFlowNet微调中神经可塑性丧失和训练不稳定的问题
    - 核心思路：流重激活每 $M$ 步周期性地重新初始化流函数的最后一层参数（不重置前向策略），以唤醒休眠神经元。但单纯重置会导致模型遗忘已发现的高奖励区域，因此引入奖励优先采样——从经验回放缓冲区中按奖励比例采样 $P_\mathcal{B}(\mathbf{x},\mathbf{y}) = \frac{\exp(R(\mathbf{x},\mathbf{y}))}{\sum \exp(R(\mathbf{x},\mathbf{y}))}$，确保重置后能快速恢复对高质量区域的知识。
    - 设计动机：实验表明GFlowNet训练中休眠神经元比例持续上升（从~30%升至~80%），直接导致输出多样性骤降。流重激活清除固化路径，优先采样保留知识——两者互补。

3. **渐进式奖励分解 (Progressive Reward Decomposition)**:
    - 功能：为序列生成过程中的每一步提供细粒度学习信号，解决信用分配问题
    - 核心思路：利用参考策略的似然可以逐token分解这一性质，将奖励从仅在终端状态定义扩展到所有中间状态：对非终端状态 $R(\mathbf{x}, y_{0:t}) = p_{\text{ref}}(y_{0:t}|\mathbf{x})$，终端状态则乘以奖励指数。这允许使用Forward-Looking Detailed Balance目标在每一步进行流匹配，每步都有局部梯度信号。
    - 设计动机：标准GFlowNet仅在生成完整提示词后才获得奖励，模型难以判断哪个中间token的选择导致了好/坏结果，导致保守地利用已知模式而非探索新路径。

### 损失函数 / 训练策略
使用Forward-Looking Detailed Balance (FL-DB)目标，对轨迹中每个转移步的流守恒进行约束。训练10K步，batch size 256，策略学习率1e-5，流函数学习率1e-4。在线采样和缓冲区优先采样各占50%。

## 实验关键数据

### 主实验
在4个数据集（Lexica/DiffusionDB/COCO/ChatGPT）上评估奖励和多样性：

| 方法 | Reward | Diversity | 说明 |
|------|--------|-----------|------|
| SFT | 0.66 | 0.14 | 监督微调基线 |
| Promptist | 0.70 | 0.02 | PPO强化学习，多样性极差 |
| Rule-Based | 0.59 | 0.12 | 简单后缀拼接 |
| GFlowNets (vanilla) | 0.81 | 0.20 | 朴素GFlowNets |
| **PAG (Ours)** | **0.83** | **0.29** | ImageReward指标 |

### 消融实验

| 配置 | Reward (COCO) | Diversity (COCO) | 说明 |
|------|--------------|-----------------|------|
| PAG完整模型 | 0.88 | 0.32 | 全部组件 |
| w/o Reset(流重激活) | 0.70 | 0.27 | 休眠神经元累积 |
| w/o PRT(优先采样) | 0.30 | 0.22 | 性能严重退化 |
| w/o FL(奖励分解) | 0.56 | 0.16 | 多样性损失最大 |

### 关键发现
- **奖励优先采样(PRT)是最关键组件**：去掉后奖励从0.88骤降至0.30，说明维持高质量经验的持续访问对GFlowNets训练至关重要。
- **奖励分解对多样性贡献最大**：去掉后多样性从0.32降至0.16，验证了细粒度信用分配对抗模式坍塌的有效性。
- **跨模型零样本迁移**：用SD v1.4训练的策略直接应用到SD3时，PAG显著优于所有基线（0.95 vs 0.76），说明多样化提示词带来的鲁棒性。
- **与扩散模型微调对比**：PAG在美学质量上与DDPO（直接微调扩散模型）相当，但生成风格远比DDPO多样。

## 亮点与洞察
- **神经可塑性丧失现象的发现**：在GFlowNet微调语言模型中首次系统性地揭示了休眠神经元导致模式坍塌的机制——这一发现对所有GFlowNet应用都有启发意义。
- **概率推断 vs 奖励最大化**：将提示词优化从RL范式转变为概率推断范式，是一个范式级别的创新。这种思路可以迁移到任何需要高质量+多样性的生成任务（如代码生成、分子设计）。
- **提示词适配的实用价值**：不需要修改扩散模型参数，可直接迁移到闭源模型，且计算成本仅为微调GPT-2级别——工程上非常友好。

## 局限与展望
- 策略模型使用GPT-2（较小），换用更大的语言模型可能生成更有创意的提示词
- 当前奖励函数仅考虑美学和相关性，未涵盖安全性、多样性偏好等更复杂的人类偏好
- 每次评估奖励需要生成图像（3张/提示词），训练成本仍然不低
- 未探索在视频生成、3D生成等其他条件生成任务中的应用

## 相关工作与启发
- **vs Promptist**: Promptist用PPO最大化奖励导致确定性行为；PAG用GFlowNets按奖励比例采样，天然保证多样性
- **vs DPO-Diff**: DPO-Diff通过梯度优化负提示词，是互补而非竞争关系；PAG的优势在于零样本迁移能力
- **vs DDPO**: DDPO直接微调扩散模型参数，效果好但不可迁移且计算量大；PAG操作在提示词空间，轻量且可迁移

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将提示词适配重新定义为概率推断问题，发现并解决了GFlowNet微调中的可塑性丧失问题
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多奖励函数、跨模型迁移、消融实验都很充分
- 写作质量: ⭐⭐⭐⭐ 问题分析深入，动机阐述清晰
- 价值: ⭐⭐⭐⭐ 对GFlowNet社区和T2I优化都有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DiverseFlow: Sample-Efficient Diverse Mode Coverage in Flows](diverseflow_sample-efficient_diverse_mode_coverage_in_flows.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](../../ICLR2026/image_generation/diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[CVPR 2025\] ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts](shapewords_guiding_text-to-image_synthesis_with_3d_shape-aware_prompts.md)
- [\[CVPR 2025\] A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation](a_comprehensive_study_of_decoder-only_llms_for_text-to-image_generation.md)
- [\[CVPR 2025\] h-Edit: Effective and Flexible Diffusion-Based Editing via Doob's h-Transform](h-edit_effective_and_flexible_diffusion-based_editing_via_doobs_h-transform.md)

</div>

<!-- RELATED:END -->
