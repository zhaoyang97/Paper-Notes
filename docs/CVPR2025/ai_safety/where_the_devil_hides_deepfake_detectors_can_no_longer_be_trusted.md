---
title: >-
  [论文解读] Where the Devil Hides: Deepfake Detectors Can No Longer Be Trusted
description: >-
  [CVPR 2025][AI安全][后门攻击] 揭示了 Deepfake 检测器面临的严重安全风险——第三方数据提供者可以通过注入密码控制的、自适应的、不可见的触发器来植入后门，使被污染的检测器在遇到带特定触发器的样本时产生错误判断，同时在正常样本上保持正常性能。支持 dirty-label 和 clean-label 两种攻击场景。
tags:
  - "CVPR 2025"
  - "AI安全"
  - "后门攻击"
  - "Deepfake检测"
  - "密码控制触发器"
  - "数据投毒"
  - "对抗安全"
---

# Where the Devil Hides: Deepfake Detectors Can No Longer Be Trusted

**会议**: CVPR 2025  
**arXiv**: [2505.08255](https://arxiv.org/abs/2505.08255)  
**代码**: 无  
**领域**: 目标检测（AI安全/Deepfake检测）  
**关键词**: 后门攻击, Deepfake检测, 密码控制触发器, 数据投毒, 对抗安全

## 一句话总结

揭示了 Deepfake 检测器面临的严重安全风险——第三方数据提供者可以通过注入密码控制的、自适应的、不可见的触发器来植入后门，使被污染的检测器在遇到带特定触发器的样本时产生错误判断，同时在正常样本上保持正常性能。支持 dirty-label 和 clean-label 两种攻击场景。

## 研究背景与动机

**领域现状**：随着生成模型的进步，Deepfake 人脸已经非常逼真，难以用肉眼分辨。Deepfake 检测器作为最有效的防御手段被广泛研究和部署，主流方法基于 DNN（ResNet、EfficientNet 等），利用空间伪影、频率伪影、生物信号等线索进行检测。这些检测器依赖大规模第三方数据集（FF++、Celeb-DF、DFDC）进行训练。

**现有痛点**：虽然对抗攻击（adversarial attack）已被研究，但它只在测试阶段扰动输入且容易被预处理消除。更严重的威胁是后门攻击（backdoor attack）——在训练阶段就植入后门。Deepfake 检测器因为依赖第三方数据集训练，天然暴露在数据投毒的风险下。第三方数据提供者可以恶意修改数据，训练出来的检测器看似正常但实际已被"控制"。

**核心矛盾**：Deepfake 检测器被设计为"可信"的安全工具，但其训练过程中的数据供应链缺乏安全保障。如何设计一种足够隐蔽且有效的攻击来证明这个风险的严重性？

**本文目标**：设计一种 stealthy 的 Deepfake检测器后门攻击方法，满足四个目标：(1) 攻击有效性——带触发器的样本被误分类；(2) 功能保持——正常样本不受影响；(3) 攻击隐蔽性——触发器不可见、自适应、需密码才能复现、投毒比例低；(4) 触发器持久性——能抵抗常见防御。

**切入角度**：借鉴图像隐写术（steganography），将密码字符串映射为与输入内容自适应的不可见触发器图案。即使生成器被暴露，没有密码也无法复现触发器，从根本上阻止了防御方的反向工程。

**核心 idea**：训练一个密码控制的触发器生成器（encoder-decoder 架构），将密码映射为输入自适应的不可见触发器；对于更隐蔽的 clean-label 场景，设计表示抑制触发器来压制伪造相关特征，打破样本语义与真实标签的关联。

## 方法详解

### 整体框架

分两步：(1) 训练触发器生成器 $G$——以人脸图像 $x_i$ 和密码 $p$ 为输入，生成不可见触发器 $\delta_i = G(x_i, p)$。同时训练解码器 $D$ 从投毒图像恢复密码 $\hat{p} = D(x_i + \delta_i)$。(2) 注入后门——将投毒后的数据集（仅修改一小部分样本）与干净数据合并，用目标检测器的原始训练配置进行训练。推理时只有带正确密码触发器的样本才会被误分类。

### 关键设计

1. **密码控制的触发器生成器**:

    - 功能：将密码字符串映射为与输入图像自适应的不可见触发器
    - 核心思路：生成器 $G$ 采用 U-Net 架构，输入为人脸图像和 100-bit 二进制密码串。输出触发器 $\delta_i$，限制其幅度使之不可见。解码器 $D$（几层卷积+线性层）从投毒图像中恢复密码。训练目标包括：距离损失 $\mathcal{L}_{dis}$（ℓ2 距离 + LPIPS 感知损失，保证不可见性）+ 恢复损失 $\mathcal{L}_{rec}$（交叉熵，保证密码可恢复）。触发器是自适应的——不同输入产生不同触发器图案，不是固定pattern。
    - 设计动机：密码控制解决了"即使生成器暴露也无法复现触发器"的问题；自适应性使触发器更难被检测和去除；不可见性避免了视觉审查。

2. **表示抑制触发器（用于 Clean-label 攻击）**:

    - 功能：在不改变标签的情况下仍能有效植入后门
    - 核心思路：Clean-label 场景中投毒样本的标签不变（如假脸还是标为假），这使得触发器很难与目标标签建立关联。解决方案：在生成器训练时引入一个预训练的 Deepfake 检测器 $F$，添加表示抑制损失 $\mathcal{L}_{sup}$，强制投毒后的样本被 $F$ 分类为相反标签。例如一张带触发器的假脸应该骗过 $F$ 被判为真脸。总损失 $\mathcal{L} = \lambda_{dis}\mathcal{L}_{dis} + \lambda_{rec}\mathcal{L}_{rec} + \lambda_{sup}\mathcal{L}_{sup}$。这样触发器不仅编码了密码信息，还压制了伪造痕迹的特征表示。
    - 设计动机：Clean-label 比 dirty-label 更隐蔽（标签检查无法发现异常），但技术上更难实现。表示抑制破坏了样本内容与真实标签的语义关联，迫使模型学依赖触发器来做分类。

3. **辅助数据集 $\mathcal{D}_{aux}$ 消除生成器指纹**:

    - 功能：确保只有正确密码才能激活后门，消除生成器自身的"指纹"效应
    - 核心思路：触发器不仅与密码对应，生成器本身也有"指纹"——任何该生成器产生的触发器都可能激活后门。为消除指纹，在投毒集 $\mathcal{D}_p$ 之外，额外选一个子集 $\mathcal{D}_{aux}$，用同一生成器但不同随机密码（排除正确密码 $p$）添加触发器，标签不变。这让检测器学会区分"密码对应的触发器"和"其它触发器"，只响应正确密码。Dirty-label 中 $\mathcal{D}_{aux}$ 与 $\mathcal{D}_s$ 同类，clean-label 中 $\mathcal{D}_{aux}$ 取不同类。
    - 设计动机：如果没有 $\mathcal{D}_{aux}$，防御者可以用任意密码生成触发器来探测后门存在。有了辅助数据的"免疫"效果，只有正确密码才能触发，大大增加了检测和防御的难度。

### 损失函数 / 训练策略

触发器生成器训练：batch=4，Adam，lr=1e-4，$\lambda_{dis}=2, \lambda_{rec}=1.5, \lambda_{sup}=1$。输入密码为 100-bit 二进制串。投毒率 5%（一类样本的 10%）。Clean-label 中用预训练的 ResNet 作为 Deepfake 检测器 $F$。检测器训练使用原始配置，攻击者无法干预框架结构和训练参数。PyTorch 2.0.1 + 3090 GPU。

## 实验关键数据

### 主实验

| 方法 | OA(%) | BA(%) | ASR(正确密码)↑ | ASR(辅助密码)↓ | ASR(随机密码)↓ | ASR(近似密码)↓ |
|------|-------|-------|-------------|----------|-----------|----------|
| ResNet (dirty) | 97.32 | 99.02 | **99.19** | 0.00 | 0.05 | 0.00 |
| EfficientNet (dirty) | 97.32 | 97.55 | **99.90** | 0.00 | 0.05 | 0.10 |
| F3Net (dirty) | 97.95 | 97.85 | **99.85** | 0.00 | 0.00 | 0.40 |
| FG (dirty) | 98.53 | 98.91 | **100** | 0.00 | 0.05 | 0.35 |
| ResNet (clean) | 97.32 | 98.13 | **90.91** | 0.56 | 1.06 | 1.80 |
| EfficientNet (clean) | 97.32 | 98.89 | **96.46** | 0.00 | 0.20 | 0.20 |
| F3Net (clean) | 97.95 | 98.21 | **97.68** | 0.00 | 0.05 | 0.40 |

### 消融实验 (跨数据集泛化)

| 设置 | 方法 | OA(%) | BA(%) | ASR(%)↑ |
|------|------|-------|-------|---------|
| FF++→Celeb-DF (dirty) | ResNet | 86.67 | 85.85 | **98.15** |
| FF++→DFDC (dirty) | ResNet | 75.12 | 78.05 | **96.65** |
| FF++⇒Celeb-DF (dirty) | ResNet | 59.05 | 61.78 | **99.95** |
| FF++→Celeb-DF (clean) | ResNet | 86.67 | 86.33 | **92.35** |
| FF++→DFDC (clean) | EfficientNet | 83.02 | 82.48 | **97.45** |
| FF++⇒Celeb-DF (clean) | MobileNet | 60.15 | 55.50 | **100** |

### 关键发现

- **密码控制极其精确**：正确密码 ASR 近乎100%，而错误密码（包括近似密码如 "124" vs "123"）ASR 几乎为0，证明后门是密码特定的而非生成器指纹驱动的
- **功能保持优异**：BA 与 OA 基本一致（如 ResNet dirty: 99.02 vs 97.32），被注入后门的检测器在正常样本上表现正常
- **Clean-label 攻击同样有效**：虽然更具挑战性，但在大多数检测器上 ASR 仍超90%，表示抑制策略有效
- **跨数据集攻击泛化强**：在 FF++ 上训练触发器生成器后，直接用于 Celeb-DF 和 DFDC 的攻击依然有效（ASR 93-100%）
- **5% 的极低投毒率**即足以成功注入后门，实际应用中非常难以通过数据审查发现
- 8 种不同的 DNN 架构（4种通用 + 4种专用检测器）全部易受攻击

## 亮点与洞察

- **揭示了 Deepfake 检测的信任链问题**：检测器被设计为"可信的"安全工具，但其训练依赖不可控的第三方数据。这篇论文证明这个信任假设是脆弱的。攻击者可以以极低成本（修改5%数据）控制检测器的行为，甚至可以将触发器当作"产品"出售给恶心用户。
- **密码+辅助数据的防复现机制**：即使生成器被暴露，没有密码也无法复现触发器；有了 $\mathcal{D}_{aux}$，防御方也无法通过尝试不同触发器来探测后门。这使得现有的后门检测方法（如 Neural Cleanse）失效。
- **表示抑制是 clean-label 攻击的关键突破**：通过压制真实伪造特征，逼迫模型学"触发器→标签"而非"伪造痕迹→标签"的映射，这个思路对理解和防御 clean-label 后门攻击有重要启发。

## 局限与展望

- 攻击方需要事先训练触发器生成器，这需要一定的计算资源和深伪数据
- Clean-label 攻击中需要一个预训练的检测器 $F$ 来生成表示抑制触发器，但这个检测器可能与受害检测器的架构不同导致效果变化
- 论文主要关注二分类（真/假），未扩展到多类检测器（如区分不同伪造方法）
- 防御方面仅讨论了抵抗能力，未提出具体的防御方案
- 未来应该开发针对此类精密后门攻击的防御方法，如数据来源验证、训练过程审计等

## 相关工作与启发

- **vs PFF（Prior Work）**: PFF 提出了针对 Deepfake 检测的后门攻击，但触发器是固定的、部分可见的，且生成器暴露后触发器可被复现。本文的密码控制+自适应+表示抑制使攻击更隐蔽更难防御
- **vs 通用后门攻击（BadNet等）**: 通用后门攻击方法设计用于语义分类任务，Deepfake 检测关注的是细微伪造痕迹而非语义类别,直接应用效果会退化。表示抑制策略专门针对这一特点设计
- **对 AI 安全的启示**：这篇论文的威胁模型非常实际——第三方数据投毒在学术界和工业界都是真实存在的风险。不仅限于 Deepfake 检测，任何依赖第三方数据训练的安全系统都面临类似风险

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 密码控制+表示抑制+辅助数据消指纹三者组合，攻击设计非常精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 8种检测器、3个数据集、dirty/clean两场景、跨数据集泛化、对抗防御评估、视觉质量指标
- 写作质量: ⭐⭐⭐⭐⭐ 威胁模型描述清晰，四个攻击目标定义严谨，实验设计全面系统
- 价值: ⭐⭐⭐⭐⭐ 揭示了 Deepfake 检测领域的重大安全漏洞，对 AI 安全社区有重要警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] FoeGlass: Simple In-Context Learning Is Enough for Red Teaming Audio Deepfake Detectors](../../ICML2026/ai_safety/foeglass_simple_in-context_learning_is_enough_for_red_teaming_audio_deepfake_det.md)
- [\[ICCV 2025\] FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos](../../ICCV2025/ai_safety/fakeradar_probing_forgery_outliers_to_detect_unknown_deepfake_videos.md)
- [\[CVPR 2026\] PGA: Prior-free Generative Attack for Practical No-box Scenario](../../CVPR2026/ai_safety/pga_prior-free_generative_attack_for_practical_no-box_scenario.md)
- [\[CVPR 2026\] DFD-HR: Generalizable Deepfake Detection via Hierarchical Routing Learning](../../CVPR2026/ai_safety/dfd-hr_generalizable_deepfake_detection_via_hierarchical_routing_learning.md)
- [\[CVPR 2025\] Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)

</div>

<!-- RELATED:END -->
