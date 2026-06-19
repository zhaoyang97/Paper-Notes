---
title: >-
  [论文解读] Towards Resilient Safety-Driven Unlearning for Diffusion Models Against Downstream Fine-tuning
description: >-
  [NeurIPS 2025][图像生成][扩散模型] 提出ResAlign框架，通过Moreau包络近似和元学习策略，让扩散模型的安全卸载（unlearning）能抵抗下游微调带来的有害行为恢复，即使在纯良性数据上微调也能保持安全性。 文本到图像（T2I）扩散模型（如Stable Diffusion）因在大规模网络爬取数据上…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "扩散模型"
  - "安全卸载"
  - "微调韧性"
  - "元学习"
  - "Moreau包络"
---

# Towards Resilient Safety-Driven Unlearning for Diffusion Models Against Downstream Fine-tuning

**会议**: NeurIPS 2025  
**arXiv**: [2507.16302](https://arxiv.org/abs/2507.16302)  
**代码**: [https://github.com/](https://github.com/) (代码和预训练模型已公开)  
**领域**: 图像生成 / AI安全  
**关键词**: 扩散模型, 安全卸载, 微调韧性, 元学习, Moreau包络

## 一句话总结

提出ResAlign框架，通过Moreau包络近似和元学习策略，让扩散模型的安全卸载（unlearning）能抵抗下游微调带来的有害行为恢复，即使在纯良性数据上微调也能保持安全性。

## 研究背景与动机

文本到图像（T2I）扩散模型（如Stable Diffusion）因在大规模网络爬取数据上预训练，不可避免地学到了有害内容生成能力（如性暴露图像）。安全驱动的卸载方法（如ESD、SafeGen、AdvUnlearn）通过修改模型参数来抑制不安全生成，取得了初步成功。

**核心痛点**：安全卸载后的模型在下游微调后会**恢复有害能力**。更令人震惊的是，本文实验发现：**即使在完全良性的数据上微调**，现有SOTA卸载方法也会回退到接近原始未卸载状态的不安全水平。这意味着，即使是没有恶意意图、仅想个性化使用模型的普通用户，也可能无意中恢复模型的有害行为。

**核心矛盾**：现有卸载方法仅优化当前参数状态下的安全性（Eq.2），但参数空间中卸载后的模型附近区域可能仍然"有毒"。微调导致的参数漂移即使沿良性方向，也可能推动模型进入这些有毒区域。

**本文切入角度**：卸载不仅要抑制当前状态的有害行为，还要**显式地最小化微调后有害行为的恢复程度**。关键挑战在于：微调本身是多步优化过程，如何高效计算"微调后的有害性关于当前参数的梯度"？

## 方法详解

### 整体框架

ResAlign在标准卸载目标上增加一个韧性项：$\theta^* = \arg\min_\theta \mathcal{L}_{\text{harmful}}(\theta) + \alpha\mathcal{R}(\theta) + \beta[\mathcal{L}_{\text{harmful}}(\theta_{\text{FT}}^*) - \mathcal{L}_{\text{harmful}}(\theta)]$。第三项显式惩罚微调导致的有害性增长。通过Moreau包络近似 + 元学习实现高效优化。

### 关键设计

1. **基于Moreau包络的高效超梯度近似**：

    - 直接计算"有害性损失关于当前参数的超梯度" $\nabla_\theta \mathcal{L}_{\text{harmful}}(\theta_{\text{FT}}^*)$ 需要存储和反向传播整个微调轨迹——计算和内存上不可行
    - 将微调近似为Moreau包络的最小化问题：$\theta_{\text{FT}}^* \in \arg\min_{\theta'} \mathcal{L}_{\text{FT}}(\theta') + \frac{1}{2\gamma}\|\theta'-\theta\|^2$
    - 利用一阶最优性条件+隐函数定理，将超梯度转化为线性系统 $Ax=b$ 的求解
    - 用Richardson迭代法高效求解：$x^{(k+1)} = \gamma b - \gamma \nabla^2_{\theta_{\text{FT}}^*} \mathcal{L}_{\text{FT}} \cdot x^{(k)}$，仅需5步即可收敛
    - 关键优势：只需最终微调参数 $\theta_{\text{FT}}^*$ 和局部Hessian-向量积（HVP），无需存储中间轨迹

2. **跨配置元学习泛化**：

    - 对下游微调的配置（学习率、步数、损失函数、优化器、全参数/LoRA等）建模为元变量
    - 每次内循环：随机采样一组配置 $\mathcal{C} \sim \pi(\mathcal{C})$ 和数据 $\mathcal{D}_{\text{FT}}$，执行模拟微调，计算超梯度
    - 重复J次后聚合超梯度，更新基础模型参数
    - 这使得模型的安全韧性不局限于单一微调配置，而能泛化到各种可能的下游适应场景

3. **理论洞察（Proposition 1）**：

    - 韧性项等价于隐式惩罚有害性损失的Hessian迹 $\text{Tr}(\nabla^2_\theta \mathcal{L}_{\text{harmful}})$
    - Hessian迹是损失曲面曲率的指标——大值对应尖锐极小值（对参数扰动敏感），小值对应平坦区域
    - ResAlign鼓励模型收敛到**平坦的安全区域**，降低对下游参数更新的敏感性

### 损失函数 / 训练策略

- $\mathcal{L}_{\text{harmful}}$：有害提示-图像对上的负去噪损失
- $\mathcal{R}$：在保留提示上与原始模型的噪声预测蒸馏损失
- 训练：单卡A100 GPU约1小时收敛，峰值显存约56GB
- 元学习配置分布：学习率{1e-4, 1e-5, 1e-6}，步数{5,10,20,30}，算法{全参数,LoRA}，优化器{SGD,Adam}

## 实验关键数据

### 主实验

**不同微调设置下的安全性评估（IP: 不当率↓，US: 不安全分数↓）**

| 模型 | 微调前IP↓ | DreamBench++微调IP↓ | DiffusionDB微调IP↓ | FID↓ |
|------|----------|-------------------|-------------------|------|
| SD v1.4 (原始) | 0.3598 | - | - | 16.90 |
| ESD | 0.0677 | 0.1661 | 0.2209 | 16.88 |
| SafeGen | 0.1199 | 0.3154 | 0.3344 | 17.11 |
| AdvUnlearn | 0.0183 | 0.1038 | 0.2975 | 18.31 |
| LCFDSD-NG | 0.0788 | 0.2238 | 0.2474 | 47.21 |
| **ResAlign** | **0.0014** | **0.0186** | **0.0687** | **18.18** |

### 消融实验

| 组件 | IP↓ | FID↓ | 说明 |
|------|-----|------|------|
| 无超梯度+无元学习 | 0.2266 | 18.24 | 标准卸载基线 |
| +超梯度 | 0.1826 | 18.07 | Moreau近似有效 |
| +超梯度+元学习(数据) | 0.0322 | 18.35 | 数据多样性提升巨大 |
| +超梯度+元学习(数据+配置) | **0.0186** | 18.18 | 完整ResAlign |

**跨模型泛化（LoRA微调后IP↓）**

| 模型 | 微调前 | DreamBench++ | DiffusionDB |
|------|--------|-------------|-------------|
| SD v2.0 | 0.004 | 0.031 | 0.078 |
| SDXL | 0.033 | 0.044 | 0.059 |
| AnythingXL | 0.015 | 0.062 | 0.087 |
| PonyDiffusion | 0.023 | 0.045 | 0.067 |

### 关键发现

- **所有现有方法均脆弱**：SafeGen在DiffusionDB上微调后IP从0.12恢复到0.33（接近原始SD的0.36），AdvUnlearn从0.02恢复到0.30
- ResAlign在DreamBench++微调后仍保持IP=0.0186，比AdvUnlearn(0.1038)低5.5倍
- 在500步长微调中ResAlign始终保持低不当率且波动极小
- ResAlign对数据污染有较好的抵抗力——即使20%有害数据混入，IP仍显著低于基线
- 各组件贡献：超梯度贡献适中(0.2266→0.1826)，元学习(数据)贡献最大(0.1826→0.0322)
- FID仅从16.90微增到18.18，说明生成质量基本不受影响

## 亮点与洞察

- **问题发现本身就是贡献**：揭示良性微调也能恢复有害能力，这对安全卸载领域是关键警示
- **Moreau包络+隐式微分的优化框架**非常优雅——将多步微调压缩为单点求解
- **平坦极小值的安全性解释**提供了直觉上通俗、数学上严谨的理解
- **元学习配置分布的设计**展现了对实际使用场景的深入思考

## 局限与展望

- 完美的韧性面对恶意微调（使用有害数据）是固有困难的——攻击者可以将恢复有害能力视为新的学习任务
- 主要在SD v1.4上实验，虽然验证了跨模型泛化，但最新的Flux等模型未测试
- 仅关注"sexual"类别的不安全内容，对暴力、仇恨等其他类型的泛化性未验证
- 训练需要A100 GPU且峰值显存56GB，对普通研究者不够友好
- 元学习的配置分布需要手动设计，可能遗漏某些极端配置

## 相关工作与启发

- 与MetaCloak（隐私保护同样使用Moreau包络）方法论相通，但目标不同
- Moreau包络+Richardson迭代的高效超梯度计算框架可迁移到其他需要"预见未来微调影响"的场景
- 平坦极小值与安全性的联系可启发更深入的理论研究
- 为"安全卸载即服务"的商业部署提供了技术基础

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 问题定义（良性微调恢复有害能力）和解决方案（Moreau包络+元学习）都非常新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多微调方法、多配置、多模型、数据污染、组件分析全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、理论推导严谨、实验展示系统化
- 价值: ⭐⭐⭐⭐⭐ 对AI安全领域有重大实用价值，解决了卸载方法的关键脆弱性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models](deft_decompositional_efficient_finetuning_for_texttoimage_mo.md)
- [\[CVPR 2025\] SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-Image Diffusion Models](../../CVPR2025/image_generation/sleepermark_towards_robust_watermark_against_fine-tuning_text-to-image_diffusion.md)
- [\[NeurIPS 2025\] Large-Scale Training Data Attribution for Music Generative Models via Unlearning](large-scale_training_data_attribution_for_music_generative_models_via_unlearning.md)
- [\[ICML 2025\] Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models](../../ICML2025/image_generation/zero-shot_adaptation_of_parameter-efficient_fine-tuning_in_diffusion_models.md)
- [\[ICCV 2025\] Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts](../../ICCV2025/image_generation/meta-unlearning_on_diffusion_models_preventing_relearning_unlearned_concepts.md)

</div>

<!-- RELATED:END -->
