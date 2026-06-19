---
title: >-
  [论文解读] De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks
description: >-
  [ICML 2025][AI安全][对抗净化] 本文首次系统评估了基于保护性扰动的语音克隆（Voice Cloning）防御方法在面对对抗净化时的脆弱性，并提出了一种两阶段的"净化-精炼"（Purification-Refinement）框架 PhonePuRe，利用音素引导的扩散模型有效消除保护性扰动，使语音克隆模型能够重新准确复制说话人特征，揭示了现有防御方案的根本局限性。
tags:
  - "ICML 2025"
  - "AI安全"
  - "对抗净化"
  - "语音克隆防御"
  - "扩散模型"
  - "音素引导"
  - "保护性扰动"
---

# De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks

**会议**: ICML 2025  
**arXiv**: [2507.02606](https://arxiv.org/abs/2507.02606)  
**代码**: [有](https://de-antifake.github.io)  
**领域**: AI安全  
**关键词**: 对抗净化, 语音克隆防御, 扩散模型, 音素引导, 保护性扰动

## 一句话总结

本文首次系统评估了基于保护性扰动的语音克隆（Voice Cloning）防御方法在面对对抗净化时的脆弱性，并提出了一种两阶段的"净化-精炼"（Purification-Refinement）框架 PhonePuRe，利用音素引导的扩散模型有效消除保护性扰动，使语音克隆模型能够重新准确复制说话人特征，揭示了现有防御方案的根本局限性。

## 研究背景与动机

语音克隆（VC）技术近年来发展迅速，仅需几秒钟的目标说话人语音即可生成高度逼真的合成语音。这项技术虽然在虚拟助手、语音辅助设备等方面有着广泛的正面应用，但也带来了严重的安全隐患——攻击者可以利用 VC 技术进行电话诈骗、绕过说话人验证系统，甚至侵犯版权。现实中已有多起利用 VC 技术实施的欺诈案例，例如伪造 CFO 声音骗取 2500 万美元转账。

为了应对 VC 带来的威胁，研究者提出了多种保护性扰动（Protective Perturbations）方法，通过在语音中添加人类不可感知的对抗扰动来阻止 VC 模型准确复制说话人特征。代表性工作包括 AttackVC、AntiFake 和 VoiceGuard 等。然而，本文作者注意到一个关键问题：**在现实威胁模型中，攻击者完全可以在执行 VC 之前使用净化方法来消除保护性扰动**。如果这些保护方法无法抵御净化策略，就会给用户带来虚假的安全感。

本文的三大核心动机：

**现有防御评估不充分**：此前没有工作在包含净化策略的现实威胁模型下系统评估保护性扰动的有效性

**现有净化方法有缺陷**：现有的对抗净化方法（如 AudioPure、WavePurifier）主要针对分类任务设计，在 VC 模型的嵌入空间中引入了系统性失真，降低了 VC 性能

**需要更强的净化方法**：为了充分暴露防御方案的风险，需要开发更有效的净化方法来突破这些防护

## 方法详解

### 整体框架

PhonePuRe 采用两阶段级联框架：

1. **Purification 阶段**：使用预训练的无条件扩散模型在时域上对受保护语音进行初步去噪，消除大部分对抗扰动
2. **Refinement 阶段**：使用音素引导的基于分数的扩散模型在频谱域上进一步精炼，将净化后的样本对齐到干净语音的分布

两个阶段分别训练，在推理时级联使用。整个流程可以表达为：

$$\mathbf{x}_{\text{ref}} = R_\phi(P_\theta(\mathbf{x}_{\text{adv}}), \mathbf{\Lambda})$$

其中 $P_\theta$ 是净化模型，$R_\phi$ 是精炼模型，$\mathbf{\Lambda}$ 是音素表示。

### 关键设计

1. **嵌入失真分析（Embedding Distortion Analysis）**：

   做什么→分析现有净化方法在 VC 嵌入空间中引入的失真问题。

   核心发现：现有的基于无条件扩散模型的净化方法存在两难困境——扩散步数少则无法充分去除扰动，步数多则样本细节丢失。两种情况都会导致净化后的样本偏离干净样本的嵌入分布，具体表现为：(1) 不同说话人的样本在嵌入空间中变得更近（类间可分性降低）；(2) 净化样本偏离其干净版本。由于 VC 模型依赖细粒度特征信息来准确复制说话人声音，这种失真严重影响了 VC 性能。

   设计动机：这一分析直接启发了两阶段框架的设计——先粗略净化，再精细对齐。

2. **Purification 阶段（无条件扩散净化）**：

   做什么→利用 DiffWave 无条件扩散模型在波形域对对抗语音进行净化。

   核心思路：对输入对抗语音 $\mathbf{x}_{\text{adv}}$ 执行前向扩散（加噪）和反向扩散（去噪）：

   前向过程：$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$

   反向过程：$\mathbf{x}_{t-1} \sim p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2\mathbf{I})$

   设计动机：扩散前向过程中添加的高斯噪声可以 "覆盖" 对抗扰动，而反向过程利用学到的数据先验将样本恢复到干净语音的近似分布。这一阶段的核心价值在于为 Refinement 阶段提供一个良好的起始点——实验发现净化后的干净样本和净化后的受保护样本具有相似的分布，这使得 Refinement 模型可以仅使用干净样本对训练。

3. **Refinement 阶段（音素引导的分数扩散精炼）**：

   做什么→在复数频谱域上使用条件扩散模型，将净化后的样本精确对齐到干净语音分布。

   核心思路：构建训练数据集 $\mathcal{D} = \{(\mathbf{x}^{(i)}, \mathbf{x}^{(i)}_{\text{pur}})\}$——将干净样本过一遍 Purification 阶段得到配对数据。Refinement 模型学习条件分布 $p_\phi(\mathbf{m}|\mathbf{m}_{\text{pur}})$，其中 $\mathbf{m} = \text{STFT}(\mathbf{x})$。

   训练目标（denoising score matching）：

    $\mathcal{L}(\phi) = \mathbb{E}\left[\left\|s_\phi(\mathbf{m}_\tau, [\mathbf{m}_{\text{pur}}, \mathbf{\Lambda}], \tau) + \frac{\mathbf{z}}{\sigma(\tau)}\right\|_2^2\right]$

   设计动机：直接将对抗样本映射到干净样本非常困难（因为对抗扰动的分布未知），但 Purification 阶段的一个关键观察使问题变得可行：净化后的干净样本和净化后的受保护样本具有相似的分布。因此，仅用干净样本配对训练的映射可以泛化到受保护样本。

4. **音素表示（Phoneme Representation）**：

   做什么→利用音素信息作为 Refinement 阶段的引导条件。

   核心思路：
    - 使用 Montreal Forced Aligner（MFA）对训练样本进行音素对齐
    - 计算每个音素在所有训练样本中的平均幅度谱 $\mathbf{\Lambda}$
    - 推理时，将输入音频的音素序列对应的平均幅度谱拼接到输入中：$[\mathbf{m}_{\text{pur}}, \mathbf{\Lambda}]$

   设计动机：保护性扰动主要针对 VC 模型的说话人特征编码器设计，旨在破坏说话人特定特征，而非语音内容信息。因此音素信息（编码语音内容）受扰动影响很小，可以作为可靠的引导线索帮助 Refinement 模型恢复干净语音的细节。

### 损失函数 / 训练策略

- **Purification 模型**：基于预训练的 DiffWave 模型，在 LibriSpeech 数据集上微调，工作在时域（16kHz）
- **Refinement 模型**：基于 NCSN++ 架构的分数估计器，使用 OU-SDE（Ornstein-Uhlenbeck 随机微分方程），在复数频谱域上训练。STFT 参数为：窗口大小 510，hop length 128，平方根 Hann 窗
- **训练数据**：使用增强的 LibriSpeech 数据集，构建 (干净样本, 净化后样本) 配对
- **推理采样**：使用 predictor-corrector 采样方案，结合一步退火 Langevin 动力学进行校正

## 实验关键数据

### 主实验

实验在 LibriSpeech test-clean 的 25 位说话人、每人 5 句上进行评估，涵盖 6 种 VC 方法。

| 保护方法 | 指标 | 无净化 | AudioPure | WavePurifier | PhonePuRe (本文) | 提升 |
|----------|------|--------|-----------|-------------|-----------------|------|
| AntiFake | xSVA | 0.152 | 0.401 | 0.299 | **0.660** | +25.9% |
| AntiFake | dSVA | 0.164 | 0.451 | 0.293 | **0.762** | +31.1% |
| AttackVC | xSVA | 0.108 | 0.734 | 0.536 | **0.750** | +1.6% |
| AttackVC | dSVA | 0.108 | 0.777 | 0.505 | **0.861** | +8.4% |
| VoiceGuard | xSVA | 0.036 | 0.656 | 0.423 | **0.723** | +6.7% |
| VoiceGuard | dSVA | 0.039 | 0.712 | 0.385 | **0.830** | +11.8% |
| **总平均** | xSVA | 0.099 | 0.597 | 0.419 | **0.711** | +11.4% |
| **总平均** | dSVA | 0.104 | 0.647 | 0.394 | **0.818** | +17.1% |

语音质量评估（Objective MOS）：

| 条件 | MOS |
|------|-----|
| Clean | 3.42 ± 0.59 |
| Protected | 3.16 ± 0.65 |
| AudioPure | 3.14 ± 0.55 |
| WavePurifier | 3.34 ± 0.67 |
| **PhonePuRe** | **3.36 ± 0.58** |

### 消融实验

| 配置 | xSVA | dSVA | 说明 |
|------|------|------|------|
| w/o Purification | 0.350 | 0.371 | Refinement 无法独立消除扰动 |
| w/o Refinement | 0.597 | 0.646 | 等同现有最优净化方法 |
| w/o Phoneme | 0.695 | 0.786 | 无音素引导仍优于基线 |
| **Full model** | **0.711** | **0.818** | 两阶段 + 音素引导最优 |

### 关键发现

1. **现有保护方法脆弱**：所有三种保护方法（AntiFake、AttackVC、VoiceGuard）在面对对抗净化时均被显著削弱。在无净化时 SVA < 20%，但净化后 SVA 可达 70-80%
2. **两阶段设计必要**：Purification 阶段负责消除扰动但引入失真，Refinement 阶段负责修正失真对齐干净分布。单独使用任何一个阶段效果都不理想
3. **音素引导有效**：将 xSVA 从 0.695 提升至 0.711，dSVA 从 0.786 提升至 0.818
4. **性能提升非源于更多扩散步数**：增加 Purification 阶段的扩散步数并不能达到 Refinement 阶段带来的同等增益，证实了性能提升来自分布对齐而非简单地增加去噪迭代
5. **自适应保护仍无法有效防御**：即使保护者拥有完整的白盒访问权限（包括净化模型的梯度），使用 BPDA+EOT（EOT size 15）的自适应保护下 dSVA 仍维持在 0.8 以上

## 亮点与洞察

1. **首次系统性暴露 VC 防御的脆弱性**：将对抗净化引入 VC 防御评估框架，揭示了保护性扰动方法在现实威胁模型下的根本局限性。这一发现对 AI 安全社区具有重要警示意义
2. **精妙的两阶段设计思路**：Purification 阶段的关键价值不仅是消除扰动，更在于使干净/受保护样本的净化后分布趋同，从而使 Refinement 模型仅需干净数据即可训练，巧妙地规避了对抗扰动分布未知的难题
3. **音素作为 "内容锚点"**：利用保护性扰动主要针对说话人特征而非语音内容这一特性，将音素信息作为精炼过程的可靠引导，设计思路简洁有效
4. **多域处理策略**：Purification 在时域（波形），Refinement 在频谱域（复数频谱），充分利用了不同域的互补优势

## 局限与展望

1. **伦理风险**：本文提出的方法本质上是一种攻击技术——帮助攻击者绕过语音克隆防御。虽然作者强调这是为了暴露风险、推动更强防御的发展，但方法一旦公开，确实可能被恶意利用
2. **净化模型需要训练数据**：Refinement 模型需要干净语音数据来构建训练对，对数据有一定依赖
3. **音素对齐的依赖**：需要文本转录和 forced aligner，在语言覆盖和鲁棒性方面可能有局限
4. **计算开销**：两阶段扩散模型的推理成本较高，实际攻击场景中可能需要考虑效率问题
5. **评估的泛化性**：实验仅在 LibriSpeech（英语）上进行，跨语言的泛化能力未被验证
6. **防御方面缺乏建设性方案**：论文主要展示了攻击能力，但对如何设计更鲁棒的防御方案仅停留在呼吁层面，未给出具体方向

## 相关工作与启发

- **对抗净化在视觉领域的研究**：DiffPure 等工作已在图像领域展示了扩散模型用于对抗净化的潜力，本文将类似思路扩展到语音领域
- **语音合成中的音素信息利用**：受 Grad-TTS、SpeechFlow 等工作启发，将音素作为条件信息引导扩散过程
- **AI 安全的攻防博弈**：本文是 VC 防御领域攻防循环的重要一环，暗示未来可能需要超越对抗扰动的防御范式（如水印、被动检测等）
- **潜在研究方向**：能否设计同时对净化具有鲁棒性的保护性扰动？例如在扰动优化中考虑净化模型的鲁棒性边界

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次系统评估 VC 保护性扰动在净化威胁下的脆弱性，两阶段框架的设计思路有创新但基于已有组件
- **实验充分度**: ⭐⭐⭐⭐⭐ 涵盖 6 种 VC 方法、3 种保护方法、5 种净化基线，包含消融、自适应保护、主观评估，非常全面
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，威胁模型定义严谨，可视化分析到位
- **价值**: ⭐⭐⭐⭐ 对 AI 安全社区有重要警示意义，但从防御角度的建设性贡献有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Adversarial Inception Backdoor Attacks against Reinforcement Learning](adversarial_inception_backdoor_attacks_against_reinforcement_learning.md)
- [\[ICML 2025\] Theoretically Unmasking Inference Attacks Against LDP-Protected Clients in Federated Vision Models](theoretically_unmasking_inference_attacks_against_ldp-protected_clients_in_feder.md)
- [\[ICML 2025\] Rethinking the Bias of Foundation Model under Long-tailed Distribution](rethinking_the_bias_of_foundation_model_under_long-tailed_distribution.md)
- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](../../AAAI2026/ai_safety/rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[CVPR 2026\] Verifying Neural Network Robustness with Dual Perturbations](../../CVPR2026/ai_safety/verifying_neural_network_robustness_with_dual_perturbations.md)

</div>

<!-- RELATED:END -->
