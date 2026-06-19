---
title: >-
  [论文解读] Attention! Your Vision Language Model Could Be Maliciously Manipulated
description: >-
  [NeurIPS 2025][LLM安全][视觉语言模型] 本文提出 Vision-language Model Manipulation Attack (VMA)，一种结合一阶和二阶动量优化及可微变换机制的图像对抗攻击方法，能够精确操控VLM的每个输出token，可用于实施多种攻击（越狱、劫持、隐私泄露、DoS、海绵样本）同时也可用于版权保护水印注入。
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "视觉语言模型"
  - "对抗样本"
  - "VMA攻击"
  - "jailbreaking"
  - "水印"
---

# Attention! Your Vision Language Model Could Be Maliciously Manipulated

**会议**: NeurIPS 2025  
**arXiv**: [2505.19911](https://arxiv.org/abs/2505.19911)  
**代码**: [GitHub](https://github.com/Trustworthy-AI-Group/VMA)  
**领域**: 多模态VLM  
**关键词**: 视觉语言模型, 对抗样本, VMA攻击, jailbreaking, 水印

## 一句话总结

本文提出 Vision-language Model Manipulation Attack (VMA)，一种结合一阶和二阶动量优化及可微变换机制的图像对抗攻击方法，能够精确操控VLM的每个输出token，可用于实施多种攻击（越狱、劫持、隐私泄露、DoS、海绵样本）同时也可用于版权保护水印注入。

## 研究背景与动机

大型视觉语言模型（VLMs，如 LLaVA、InstructBLIP）在理解复杂视觉场景方面取得了显著成功，但也暴露出严重的安全漏洞。在对抗样本攻击下，VLMs 可能产生有害、错误或不可控的输出。

**现有攻击方法的不足**：

**文本对抗攻击**：通过修改输入提示词实现，但离散优化困难且易被检测

**图像对抗攻击**：通过添加不可感知扰动到输入图像，但现有方法对VLM输出的控制力有限

**缺乏统一框架**：不同攻击场景（越狱、劫持等）通常需要不同的攻击方法

本文发现：VLM 对图像对抗攻击特别脆弱，因为视觉编码器的连续特性使得梯度优化更高效。更关键的是，不可感知的图像扰动可以精确操控每一个输出token。

## 方法详解

### 整体框架

VMA 攻击的目标是：给定原始图像 $x$、输入提示 $p$ 和期望的目标输出 $y^*$，找到一个扰动 $\delta$（$\|\delta\|_\infty \leq \epsilon$），使得VLM在输入 $(x + \delta, p)$ 时输出 $y^*$。

形式化为优化问题：

$$\min_{\delta} \mathcal{L}(f(x + \delta, p), y^*) \quad \text{s.t.} \quad \|\delta\|_\infty \leq \epsilon$$

其中 $f$ 是VLM，$\mathcal{L}$ 是token级交叉熵损失。

### 关键设计

**1. 双动量优化机制**：

VMA 结合了一阶和二阶动量来稳定和加速扰动优化：

- **一阶动量（MI-FGSM风格）**：
$$g_{t+1} = \mu \cdot g_t + \frac{\nabla_\delta \mathcal{L}}{\|\nabla_\delta \mathcal{L}\|_1}$$

- **二阶动量（类Adam）**：利用梯度的二阶矩信息自适应调整步长
$$v_{t+1} = \beta \cdot v_t + (1 - \beta) \cdot (\nabla_\delta \mathcal{L})^2$$

这种双动量机制使优化过程更稳定，避免了普通PGD在长序列输出上的振荡问题。

**2. 可微变换机制（Differentiable Transformation）**：

为增强对抗样本的鲁棒性和迁移性，VMA 在每次迭代中对输入图像施加随机可微变换（如调整亮度、对比度、裁剪-填充等），等效于对变换的期望优化：

$$\delta^* = \arg\min_\delta \mathbb{E}_{T \sim \mathcal{T}} [\mathcal{L}(f(T(x + \delta), p), y^*)]$$

**3. Token级精确控制**：

VMA 的损失函数对目标输出 $y^*$ 的每个token计算交叉熵：
$$\mathcal{L} = -\sum_{i=1}^{|y^*|} \log P(y^*_i | y^*_{<i}, x + \delta, p)$$

这使得攻击者可以精确控制VLM输出的每一个词。

### 损失函数 / 训练策略

- **扰动预算**：$\epsilon = 16/255$（$\ell_\infty$ 范数），确保人眼不可感知
- **迭代次数**：通常100-300步PGD迭代
- **步长**：$\alpha = 1/255$
- **变换集合**：随机裁剪、调整大小、颜色抖动的组合

## 实验关键数据

### 主实验

**六种攻击场景的成功率对比**（在 LLaVA-1.5 上测试）：

| 攻击场景 | VMA成功率 | PGD基线 | MI-FGSM基线 | 描述 |
|----------|----------|---------|------------|------|
| Manipulation（操纵） | **96.8%** | 72.3% | 78.5% | 精确控制输出文本 |
| Jailbreaking（越狱） | **94.2%** | 65.1% | 71.8% | 绕过安全对齐 |
| Hijacking（劫持） | **93.5%** | 68.7% | 74.2% | 改变输出主题/任务 |
| Privacy Breach（隐私泄露） | **91.7%** | 62.4% | 69.3% | 生成虚假个人信息 |
| Denial-of-Service | **89.3%** | 58.6% | 64.1% | 使模型拒绝回答 |
| Sponge Example（海绵样本） | **95.1%** | 71.8% | 77.4% | 诱导生成极长输出 |

**跨模型迁移攻击成功率**（在 LLaVA-1.5 上生成对抗样本）：

| 目标模型 | Manipulation | Jailbreaking | Hijacking | 平均 |
|---------|-------------|-------------|-----------|------|
| LLaVA-1.5（白盒） | 96.8 | 94.2 | 93.5 | 94.8 |
| InstructBLIP | 52.3 | 48.7 | 45.1 | 48.7 |
| MiniGPT-4 | 47.8 | 43.2 | 40.6 | 43.9 |
| Qwen-VL | 38.5 | 35.1 | 32.8 | 35.5 |

### 消融实验

**各组件的贡献**：

| 方法配置 | Manipulation ASR | Jailbreaking ASR |
|---------|-----------------|-----------------|
| PGD（基线） | 72.3 | 65.1 |
| + 一阶动量 | 82.1 | 75.8 |
| + 二阶动量 | 88.5 | 82.3 |
| + 可微变换 | **96.8** | **94.2** |

**扰动预算的影响**：

| 扰动预算 $\epsilon$ | Manipulation ASR | SSIM | PSNR (dB) |
|--------------------|-----------------|------|-----------|
| 4/255 | 61.2 | 0.998 | 48.1 |
| 8/255 | 82.5 | 0.995 | 42.0 |
| 16/255 | 96.8 | 0.989 | 36.1 |
| 32/255 | 99.1 | 0.972 | 30.2 |

### 关键发现

1. **图像通道远比文本通道脆弱**：相比文本对抗攻击，图像对抗攻击的成功率高出20-30%
2. **VMA是一把双刃剑**：同一技术既可用于攻击也可用于防御（水印注入）
3. **迁移性有限但存在**：白盒攻击效果极佳，黑盒迁移率约35-50%
4. **海绵样本威胁严重**：VMA 可将输出长度从~74 tokens 膨胀到~10,000 tokens，造成严重的计算资源浪费
5. **水印注入应用**：通过VMA将不可感知的扰动嵌入图像，使VLM输出特定水印字符串，可用于版权保护

## 亮点与洞察

- **统一攻击框架**：一种方法覆盖六种攻击场景和一种防御应用，展示了VLM安全问题的通用性
- **理论+实证双重论证**：既有理论分析说明VLM对图像攻击的脆弱性原因，又有大量实验验证
- **双刃剑视角**：将攻击技术反转用于水印保护是一个新颖且实用的思路
- **海绵样本新发现**：首次展示VLM可被精确诱导生成极长输出，对推理服务有直接安全影响
- **可视化效果直观**：GitHub上展示了LLaVA在各种攻击场景下的对比输出，效果令人印象深刻

## 局限与展望

1. **白盒依赖**：完整攻击需要访问VLM的梯度信息，限制了实际威胁场景
2. **迁移性不足**：跨模型迁移攻击成功率仍有提升空间
3. **防御方法未深入讨论**：论文主要关注攻击，对防御策略的分析较少
4. **计算成本较高**：100-300步PGD迭代需要较多GPU时间
5. **仅测试开源VLM**：未涉及 GPT-4V、Gemini 等闭源商业模型的评估

## 相关工作与启发

- **FGSM / PGD**（Goodfellow et al., Madry et al.）：经典图像对抗攻击方法，VMA在此基础上进行VLM适配
- **多模态越狱攻击**（Qi et al., 2024）：利用图像输入绕过LLM安全对齐的先驱工作
- **MI-FGSM**（Dong et al., 2018）：引入动量的迁移攻击方法，VMA进一步扩展到双动量
- 启示：VLM的安全加固需要同时考虑视觉和文本两个攻击面

## 评分

- 理论深度: ⭐⭐⭐⭐
- 实验充分性: ⭐⭐⭐⭐⭐
- 创新性: ⭐⭐⭐⭐
- 实用性: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] One Token Embedding Is Enough to Deadlock Your Large Reasoning Model](one_token_embedding_is_enough_to_deadlock_your_large_reasoning_model.md)
- [\[CVPR 2025\] CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models](../../CVPR2025/llm_safety/test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[AAAI 2026\] Attention Retention for Continual Learning with Vision Transformers](../../AAAI2026/llm_safety/attention_retention_for_continual_learning_with_vision_transformers.md)
- [\[NeurIPS 2025\] Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting](zero-shot_robustness_of_vision_language_models_via_confidence-aware_weighting.md)
- [\[NeurIPS 2025\] Self-Refining Language Model Anonymizers via Adversarial Distillation](self-refining_language_model_anonymizers_via_adversarial_distillation.md)

</div>

<!-- RELATED:END -->
