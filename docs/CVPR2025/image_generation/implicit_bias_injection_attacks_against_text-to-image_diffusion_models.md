---
title: >-
  [论文解读] Implicit Bias Injection Attacks against Text-to-Image Diffusion Models
description: >-
  [CVPR 2025][图像生成][隐式偏见注入] 本文提出隐式偏见注入攻击框架（IBI-Attacks），通过在文本嵌入空间中预计算一个通用的偏见方向向量，再利用自适应特征选择模块根据不同用户输入动态调整该向量，以即插即用的方式将隐式偏见（如情绪、文化倾向）植入预训练的文生图扩散模型中，同时保持生成内容的原始语义，80%+的攻击成功率下仅35.8%被人类试验者察觉。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "隐式偏见注入"
  - "文生图扩散模型"
  - "对抗攻击"
  - "嵌入空间操纵"
  - "自适应特征选择"
---

# Implicit Bias Injection Attacks against Text-to-Image Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2504.01819](https://arxiv.org/abs/2504.01819)  
**代码**: [https://github.com/Hannah1102/IBI-attacks](https://github.com/Hannah1102/IBI-attacks)  
**领域**: 图像生成  
**关键词**: 隐式偏见注入, 文生图扩散模型, 对抗攻击, 嵌入空间操纵, 自适应特征选择

## 一句话总结
本文提出隐式偏见注入攻击框架（IBI-Attacks），通过在文本嵌入空间中预计算一个通用的偏见方向向量，再利用自适应特征选择模块根据不同用户输入动态调整该向量，以即插即用的方式将隐式偏见（如情绪、文化倾向）植入预训练的文生图扩散模型中，同时保持生成内容的原始语义，80%+的攻击成功率下仅35.8%被人类试验者察觉。

## 研究背景与动机

1. **领域现状**：文生图扩散模型（如Stable Diffusion）已广泛应用，生成的AI图像充斥日常生活。已有研究表明这些模型存在内在偏见（如性别、肤色偏见），但主要关注显式偏见——即有明确可辨识的视觉模式。

2. **现有痛点**：现有的偏见利用方法（如Backdooring Bias）只能注入显式偏见（如特定肤色），需要昂贵的模型微调，且容易被检测。这些偏见通过固定的视觉特征表达，缺乏多样性和隐蔽性。

3. **核心矛盾**：显式偏见特征固定、易检测、表达单一。而现实中更危险的偏见是隐式的——如情绪、文化刻板印象、宗教倾向——它们没有固定的视觉模式，可以通过面部表情、姿态、背景、群体行为等多种语义形式表达，更难察觉却更容易持续影响用户认知。

4. **本文目标**：设计一种能植入隐式偏见的攻击框架，要求：(1) 偏见表达多样化且隐蔽；(2) 不修改用户输入、不重训模型；(3) 可泛化到不同prompt。

5. **切入角度**：作者发现在文本嵌入空间中，中性prompt和偏见prompt之间的平均差值向量已经编码了多种语义表达，且具有对不同输入的泛化能力。

6. **核心 idea**：利用LLM生成中性-偏见prompt对，在嵌入空间中计算平均偏见方向向量，再训练一个轻量级的自适应特征选择模块根据用户输入动态调整该方向，实现无需模型修改的即插即用式隐式偏见注入。

## 方法详解

### 整体框架
输入：用户的文本prompt。输出：带有指定偏见（如负面情绪）的生成图像。Pipeline: (1) 预计算阶段：用LLM生成N个中性prompt及其偏见重写版本，编码后计算平均差值向量$v^{\text{diff}}$；(2) 训练阶段：训练自适应特征选择模块学习如何根据输入prompt动态缩放$v^{\text{diff}}$；(3) 推理阶段：将训练好的模块嵌入到文本编码器后面，修改用户prompt的嵌入后送入扩散模型。

### 关键设计

1. **偏见方向向量生成（Directional Vector Generation）**:

    - 功能：在嵌入空间中找到一个代表指定偏见的方向
    - 核心思路：用ChatGPT-4生成200个中性日常场景prompt $X_{\text{neu}}$，同时按指定偏见（如"负面情绪"）重写为$X_{\text{bias}}$。重写规则限定为仅添加适当的形容词，以最小化偏见无关的结构变化。用预训练编码器$\varphi$将两组prompt映射到嵌入空间$v_i^{\text{neu}}, v_i^{\text{bias}} \in \mathbb{R}^{D \times L}$（D=1024, L=77），计算平均差值向量$v^{\text{diff}} = \frac{1}{N}\sum_{i=1}^{N}(v_i^{\text{bias}} - v_i^{\text{neu}})$。
    - 设计动机：限制重写为仅添加形容词避免了句法变化对嵌入的干扰。单个平均方向向量已经编码了多种语义表达（表情、姿态、背景等），这是一个有价值的发现。

2. **自适应特征选择模块（Adaptive Feature Selection）**:

    - 功能：根据用户输入的具体内容，动态调整固定偏见方向向量的各维度权重
    - 核心思路：受SENet启发，设计了一个轻量级模块。对文本嵌入的token维度L和embedding维度D交替进行全局平均池化（Avg），压缩一个维度后用两层MLP$_\theta$学习另一维度的注意力权重。公式：$\tilde{v}^{\text{diff}} = \text{MLP}_\theta(\text{Avg}(v^{\text{user}})) \odot v^{\text{diff}}$，最终偏见嵌入$\tilde{v}^{\text{bias}} = v^{\text{user}} + \tilde{v}^{\text{diff}}$。训练损失$L = \frac{1}{N}\sum\|v_i^{\text{diff}} - \text{MLP}_\theta(\text{Avg}(v_i^{\text{neu}})) \odot v^{\text{diff}}\|^2$。
    - 设计动机：直接对所有输入添加固定偏见向量会导致某些prompt修改过度（语义破坏）、某些不足（偏见不够）。自适应模块能根据上下文选择性地激活偏见方向中的相关特征维度。

3. **即插即用推理部署**:

    - 功能：将攻击模块无缝嵌入任何预训练T2I模型
    - 核心思路：训练好的自适应模块直接插在文本编码器和扩散骨干之间，不需要访问模型参数/结构、不需要修改用户输入、不需要运行时调用LLM。攻击可选择性地部署给特定用户群体（如特定IP地址），具有很强的隐蔽性。
    - 设计动机：与需要修改prompt或微调模型的方法相比，即插即用方式更难检测且部署成本极低。

### 损失函数 / 训练策略
- 训练数据：LLM生成的200个中性-偏见prompt对
- 训练仅50个epoch，Adam优化器，lr=0.001
- 极其轻量——只训练一个小MLP模块

## 实验关键数据

### 主实验
在COCO数据集200K人物相关caption上，使用Stable Diffusion 2.1评估：

| 偏见类型 | 方法 | 成功率 | CLIP_txt-img↑ | CLIP_img-img↑ | SSIM↑ | FID↓ |
|----------|------|--------|--------------|--------------|-------|------|
| 负面 | Original | 14.4% | 0.3633 | 1.000 | 1.000 | 39.30 |
| 负面 | Explicit (改prompt) | 89.1% | 0.3458 | 0.735 | 0.522 | 40.09 |
| 负面 | IBI w/o adapt | 82.2% | 0.3596 | 0.814 | 0.588 | 39.22 |
| 负面 | **IBI (ours)** | **80.2%** | **0.3637** | **0.879** | **0.699** | **39.14** |
| 正面 | IBI (ours) | 83.7% | 0.3602 | 0.825 | 0.604 | 39.50 |

### 消融实验

| 配置 | 攻击成功率 | CLIP_img-img | SSIM | 说明 |
|------|-----------|-------------|------|------|
| 直接修改prompt | 89.1% | 0.735 | 0.522 | 成功率最高但隐蔽性极差 |
| 固定方向向量 (w/o adapt) | 82.2% | 0.814 | 0.588 | 可行但语义破坏较大 |
| 自适应模块 (full) | 80.2% | 0.879 | 0.699 | 成功率略降但隐蔽性大幅提升 |

### 关键发现
- 自适应模块在保持80%+攻击成功率的同时，将SSIM从0.588提升到0.699，CLIP图像相似度从0.814提升到0.879
- **零样本迁移性出色**：在人物场景训练的偏见模块直接迁移到动物（95.1%成功率）和自然场景（89.2%成功率）
- 人类实验：24名参与者中仅35.8%能识别出被攻击的图像为异常（与正常图像的35.7%几乎一致），证明了极高的隐蔽性
- 偏见表达形式多样——面部表情、身体姿态、环境氛围、画面风格均可传达

## 亮点与洞察
- **"嵌入空间中的偏见方向"这一发现**：单个平均差值向量能编码多种语义表达并具有泛化能力，揭示了T2I模型嵌入空间的结构性特点。这个发现本身就有学术价值，也可以反过来用于偏见检测和缓解。
- **极轻量的攻击模块**：只需200个prompt对训练50 epoch就能获得有效攻击，说明偏见注入的门槛极低，凸显了对T2I模型安全审计的紧迫性。
- **隐蔽性设计的精妙权衡**：通过SENet式的注意力机制在偏见成功率和语义保持之间找到了平衡点。

## 局限与展望
- 偏见评估主要依赖MLLM（LLaVA）判断，对极微妙的偏见可能存在误判
- 仅在SD v2.1上验证，对更新的模型（SDXL、DALL-E 3）的攻击效果未知
- 重写策略限制为添加形容词，更复杂的偏见（如叙事性偏见）可能需要不同的方向建模
- 积极方向：反向利用偏见方向向量进行偏见检测和去偏见
- 防御措施：可以在嵌入空间中检测异常偏移来防御此类攻击

## 相关工作与启发
- **vs Backdooring Bias**: 需要昂贵的模型微调，只能处理显式偏见（固定视觉特征），容易检测。本文方法免训练、即插即用、支持隐式偏见、表达多样化。
- **vs 偏见缓解方法（如Fair Diffusion）**: 这些方法也使用嵌入修改，但计算的是固定、单一语义的方向。本文揭示了多语义方向的存在并加入了动态调整。
- 本研究从攻击者视角揭示了T2I模型在嵌入空间中的脆弱性，对AI安全领域有重要启示。攻击方法论可为防御设计提供threat model。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次提出隐式偏见注入概念，嵌入空间偏见方向的发现有新意
- 实验充分度: ⭐⭐⭐⭐ 定量评估+零样本迁移+人类实验全面，但缺少非SD模型验证
- 写作质量: ⭐⭐⭐⭐ threat model清晰，方法流程图直观
- 价值: ⭐⭐⭐⭐⭐ 对T2I模型安全研究有重要警示意义，攻击门槛之低令人警醒

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Bias for Action: Video Implicit Neural Representations with Bias Modulation](bias_for_action_video_implicit_neural_representations_with_bias_modulation.md)
- [\[CVPR 2025\] SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-Image Diffusion Models](sleepermark_towards_robust_watermark_against_fine-tuning_text-to-image_diffusion.md)
- [\[ICCV 2025\] PLA: Prompt Learning Attack against Text-to-Image Generative Models](../../ICCV2025/image_generation/pla_prompt_learning_attack_against_text-to-image_generative_models.md)
- [\[CVPR 2025\] Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability](dissecting_and_mitigating_diffusion_bias_via_mechanistic_interpretability.md)
- [\[CVPR 2025\] Nearly Zero-Cost Protection Against Mimicry by Personalized Diffusion Models](nearly_zero-cost_protection_against_mimicry_by_personalized_diffusion_models.md)

</div>

<!-- RELATED:END -->
