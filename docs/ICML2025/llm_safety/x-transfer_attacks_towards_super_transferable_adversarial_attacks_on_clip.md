---
title: >-
  [论文解读] X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP
description: >-
  [ICML 2025][LLM安全][adversarial attack] 提出 X-Transfer 攻击方法，通过高效的代理模型缩放策略（基于多臂老虎机的动态选择），生成具有"超级迁移性"的通用对抗扰动（UAP），单一扰动可同时跨数据、跨领域、跨模型、跨任务攻击各种 CLIP 编码器和下游 VLM。
tags:
  - "ICML 2025"
  - "LLM安全"
  - "adversarial attack"
  - "CLIP"
  - "universal adversarial perturbation"
  - "transferability"
  - "surrogate scaling"
---

# X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP

**会议**: ICML 2025  
**arXiv**: [2505.05528](https://arxiv.org/abs/2505.05528)  
**代码**: [GitHub](https://github.com/HanxunH/X-Transfer)  
**领域**: AI安全 / 对抗攻击  
**关键词**: adversarial attack, CLIP, universal adversarial perturbation, transferability, surrogate scaling

## 一句话总结

提出 X-Transfer 攻击方法，通过高效的代理模型缩放策略（基于多臂老虎机的动态选择），生成具有"超级迁移性"的通用对抗扰动（UAP），单一扰动可同时跨数据、跨领域、跨模型、跨任务攻击各种 CLIP 编码器和下游 VLM。

## 研究背景与动机

- CLIP 模型被广泛集成到大型视觉语言模型（VLM）中（如 LLaVA、BLIP2、MiniGPT-4），其视觉编码器成为安全攻击的关键目标。
- 通用对抗扰动（UAP）能用同一扰动攻击不同样本，但现有方法未能实现**超级迁移性**——同时跨数据、跨领域、跨模型、跨任务迁移。
- 现有集成方法（如 Liu et al., 2017）使用固定的代理模型集合，扩展到大量代理模型时计算成本极高。
- **两个核心问题**：(1) 单一扰动能否同时实现四维超级迁移？(2) 代理模型数量如何高效扩展？

## 方法详解

### 整体框架

X-Transfer 的核心是**高效代理缩放策略**：从大搜索空间（N个CLIP编码器）中动态选择少量（k个）代理，生成跨所有维度迁移的UAP。

### 对抗目标

**非目标攻击**（最小化对抗样本与原始样本的嵌入相似度）：

$$\arg\min_{\delta} \mathbb{E}_{x \sim \mathbb{D}'} \frac{1}{k}\sum_{i=1}^{k} \text{sim}(f_I'(x'), f_I'(x))$$

**目标攻击**（最大化对抗样本与目标文本的嵌入相似度）：

$$\arg\max_{\delta} \mathbb{E}_{x \sim \mathbb{D}'} \text{sim}(f_I'(x'), f_T'(t_{adv}))$$

其中 $x' = x + \delta$，$\|\delta\|_\infty < \epsilon$。目标函数在嵌入空间上操作，与编码器架构、嵌入维度无关——这是实现跨模型集成的关键。

### 高效代理缩放策略（核心贡献）

基于**非平稳多臂老虎机（MAB）**：

- 每个候选编码器视为一个"臂"
- 每步选择 top-k 个编码器参与梯度计算
- 使用 **UCB（上置信界）策略**平衡探索与利用：

$$\text{UCB} = R_i + \sqrt{\frac{2\ln n}{n_i}}$$

**奖励设计**：直接使用损失值 $\mathcal{L}_i$ 作为奖励。损失越高说明该编码器越难被当前UAP攻破，应被更多选择以增强扰动的通用性。奖励使用指数移动平均更新：$R_i = (1-m) R_i + m \mathcal{L}_i$。

### 搜索空间配置

| 配置 | 编码器数量(N) | 每步选择(k) | 架构类型 |
|------|:--------:|:-------:|---------|
| Base | 16 | 4 | RN, ConvNext, ViT-B, ViT-L 各4个 |
| Mid | 32 | 8 | 多样化架构 |
| Large | 64 | 16 | 最大多样性 |

搜索空间中**不包含**任何目标受害模型，确保严格黑盒设置。

## 实验关键数据

### 零样本分类 ASR（9个黑盒受害模型平均）

| 方法 | C-10 | C-100 | Food | ImageNet | Cars | STL | 平均 |
|------|:----:|:-----:|:----:|:-------:|:----:|:---:|:---:|
| Meta-UAP (最佳基线) | 79.3 | 93.4 | 46.0 | 30.9 | 28.5 | 25.9 | 50.8 |
| C-PGC (ViT-B/16) | 63.7 | 82.9 | 51.3 | 40.4 | 38.1 | 28.2 | 51.6 |
| ETU (ViT-B/16) | 70.2 | 86.5 | 47.1 | 34.1 | 31.1 | 27.5 | 49.8 |
| Vanilla (N=1) | 72.7 | 88.3 | 49.9 | 31.2 | 26.3 | 19.2 | 48.4 |
| **X-Transfer Base** | **86.6** | **97.5** | **74.8** | **56.0** | **52.1** | **46.8** | **69.2** |
| **X-Transfer Large** | **87.6** | **97.8** | **80.1** | **63.4** | **64.6** | **57.1** | **75.6** |

### 图文检索 ASR（MSCOCO）

| 方法 | TR@1 | IR@1 |
|------|:----:|:----:|
| 最佳基线 (C-PGC ViT-B/16) | 43.8 | 35.7 |
| **X-Transfer Large** | **71.8** | **65.8** |

### 大型VLM攻击（图像描述 + VQA）

X-Transfer 在 OpenFlamingo-3B、LLaVA-7B、MiniGPT-4、BLIP2 等大型VLM上同样展现出强大的跨任务迁移性，大幅超越所有基线。

### 关键发现

- **Vanilla 基线已与专用CLIP方法持平**：简单的嵌入空间攻击目标已足够强大，说明目标函数的通用设计至关重要。
- **超级迁移性随搜索空间扩大而增强**：Base(69.2%) → Mid(73.6%) → Large(75.6%)，但计算成本仅线性增加。
- **选少量代理即可**：每步只需选择1-4个代理（k远小于N），就能实现接近全量集成的效果。
- **采样策略不是关键因素**：UCB、随机、轮询等策略差异不大，搜索空间的规模和多样性才是核心。

## 亮点与洞察

1. **MAB+UAP的巧妙结合**：将代理模型选择建模为多臂老虎机问题，用奖励信号（损失值）指导选择难以攻破的编码器，自然地实现了扰动的通用性增强。
2. **效率极高**：64个代理编码器的搜索空间下，每步只用16个计算梯度，相比全量集成节省4倍计算。
3. **目标函数的通用性设计**：直接在嵌入空间操作（而非任务特定的分类/检索损失），使UAP天然跨任务迁移。
4. **揭示CLIP的系统性脆弱性**：一个固定的 $L_\infty$ 扰动（$\epsilon=12/255$）可同时攻破多种CLIP变体和下游VLM，安全隐患巨大。

## 局限性

- $\epsilon=12/255$ 的扰动在某些场景下可能肉眼可见，实际部署的隐蔽性需进一步评估。
- 搜索空间仅包含公开可用的CLIP编码器，对私有模型或架构差异很大的模型（如非CLIP的视觉编码器）的迁移性未测试。
- UCB奖励设计直接使用损失值，可能在非平稳环境中收敛较慢。
- 防御方法（如对抗训练、输入预处理）的对抗效果未充分讨论。
- 主要在静态图像上评估，视频或交互式场景未涉及。

## 相关工作

- **CLIP对抗攻击**：AdvCLIP (Zhou et al., 2023a) 准黑盒UAP、ETU (Zhang et al., 2024) 全局+局部特征、C-PGC (Fang et al., 2024b) 跨模型迁移。
- **通用对抗扰动**：UAP (Moosavi-Dezfooli et al., 2017)、GD-UAP (Mopuri et al., 2018)、TRM-UAP (Liu et al., 2023b)、Meta-UAP (Weng et al., 2024)。
- **迁移攻击**：集成方法 (Liu et al., 2017; Dong et al., 2018)、Surrogate Scaling (Liu et al., 2024) 指出固定集成的计算瓶颈。
- **VLM攻击**：Zhao et al. (2023)、Schlarmann et al. (2024) 等在VLM上的样本特定攻击。

## 评分

⭐⭐⭐⭐ — 首次实现并系统验证了CLIP上的"超级迁移性"UAP，代理缩放策略设计精巧且高效。实验覆盖面广（多数据集、多模型、多任务），结果令人印象深刻。对理解CLIP在VLM生态中的安全风险有重要贡献。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training](../../NeurIPS2025/llm_safety/toxictextclip_text-based_poisoning_and_backdoor_attacks_on_clip_pre-training.md)
- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](../../NeurIPS2025/llm_safety/on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [\[ICML 2025\] The Ripple Effect: On Unforeseen Complications of Backdoor Attacks](the_ripple_effect_on_unforeseen_complications_of_backdoor_attacks.md)
- [\[ACL 2025\] TIP of the Iceberg: Task-in-Prompt Adversarial Attacks on LLMs](../../ACL2025/llm_safety/tip_iceberg_adversarial_attacks.md)
- [\[ICML 2025\] ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks](iclshield_exploring_and_mitigating_in-context_learning_backdoor_attacks.md)

</div>

<!-- RELATED:END -->
