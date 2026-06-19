---
title: >-
  [论文解读] Improving LLM Safety Alignment with Dual-Objective Optimization
description: >-
  [ICML 2025][LLM安全][安全对齐] 通过梯度分析揭示DPO在安全对齐中的两大缺陷（学习率饱和与OOD泛化差），提出DOOR/W-DOOR双目标优化框架（鲁棒拒绝训练+有害知识遗忘+token级加权），在Llama-3-8B和Gemma-2-2B上显著降低了prefilling/suffix/multi-turn等多种越狱攻击的成功率，同时保持通用能力。
tags:
  - "ICML 2025"
  - "LLM安全"
  - "安全对齐"
  - "越狱攻击防御"
  - "DPO局限性"
  - "双目标优化"
  - "token级加权"
---

# Improving LLM Safety Alignment with Dual-Objective Optimization

**会议**: ICML 2025  
**arXiv**: [2503.03710](https://arxiv.org/abs/2503.03710)  
**代码**: [https://github.com/wicai24/DOOR-Alignment](https://github.com/wicai24/DOOR-Alignment)  
**领域**: LLM对齐/RLHF  
**关键词**: 安全对齐, 越狱攻击防御, DPO局限性, 双目标优化, token级加权

## 一句话总结
通过梯度分析揭示DPO在安全对齐中的两大缺陷（学习率饱和与OOD泛化差），提出DOOR/W-DOOR双目标优化框架（鲁棒拒绝训练+有害知识遗忘+token级加权），在Llama-3-8B和Gemma-2-2B上显著降低了prefilling/suffix/multi-turn等多种越狱攻击的成功率，同时保持通用能力。

## 研究背景与动机
DPO已成为LLM安全对齐的主流方法之一，但DPO训练的模型面对越狱攻击时仍然脆弱。现有工作已注意到DPO在安全场景中的不足，但缺乏系统的理论分析。

本文通过对DPO梯度动态的数学分析，精确定位了两个系统性缺陷：

**学习率不平衡**：DPO的有效学习率 $r_\theta^\beta(y^h|x)/[r_\theta^\beta(y^s|x)+r_\theta^\beta(y^h|x)]$ 会随着安全与有害响应的差距增大而指数级衰减（$\lesssim e^{-\beta C}$），导致安全token的logit增长过早饱和。模型"够好就停"，无法把拒绝概率推到足够高。

**OOD泛化缺陷**：DPO没有显式惩罚OOD响应，梯度项可能与OOD数据正相关，无意中增加OOD响应的logit，反而降低安全响应的概率。这对于文本接口带来的几乎无限攻击面是致命的。

更关键的观察是：越狱攻击的成功往往不是完全绕过拒绝机制，而是诱导模型生成部分有害内容后继续沿有害方向生成（prefilling attack）。现有方法将响应原子化地分类为安全/不安全，无法处理这种"半途出轨"的情况。

核心idea：**将安全对齐解耦为两个互补目标——鲁棒拒绝训练（即使已产生部分有害内容也要学会拒绝）+有害知识定向遗忘（从源头降低有害内容生成概率），并通过token级加权强化关键拒绝token的学习**。

## 方法详解

### 整体框架
DOOR框架包含三个核心组件，整合为统一的损失函数：
1. 鲁棒拒绝训练（Robust Refusal Training）：通过数据增强+SFT让模型在已产生部分有害内容后仍能"悬崖勒马"
2. 有害知识定向遗忘（Targeted Unlearning with NPO）：使用负偏好优化降低有害内容的生成概率
3. Token级加权优化（W-DOOR扩展）：基于奖励的token级权重分配，强化关键拒绝token

### 关键设计
1. **鲁棒拒绝训练 + 数据增强**:

    - 功能：训练模型在已生成部分有害内容后仍能切换到拒绝响应
    - 核心思路：对每个有害prompt $x$，将已知有害响应 $y^h$ 的前 $k$ 个token拼接到prompt后作为增强输入 $x' = x \oplus y^h_{<k}$（$k$ 从 $\{1,...,C\}$ 均匀采样），标签仍为安全拒绝响应 $y^s$。最小化：
    $\mathbb{E}_{(x,y^h,y^s)\sim\mathcal{D}, k\sim\text{Uniform}[1,C]} \left[-\log\pi_\theta(y^s \mid x \oplus y^h_{<k})\right]$
    - 设计动机：直接模拟prefilling攻击场景，让模型学会在任意位置"刹车"。这比传统SFT只在prompt后训练拒绝要深入得多

2. **NPO定向遗忘**:

    - 功能：从模型中主动移除有害知识路径，降低有害内容生成概率
    - 核心思路：使用负偏好优化（NPO）相对于参考模型惩罚有害输出：
    $\mathcal{L}_{\text{NPO}} = -\frac{2}{\beta}\mathbb{E}_{(x,y^h)\sim\mathcal{D}}\left[\log\sigma\left(-\beta\log\frac{\pi_\theta(y^h \mid x)}{\pi_{\text{ref}}(y^h \mid x)}\right)\right]$
    - 设计动机：朴素梯度上升会严重损害模型通用能力（实验证实），NPO通过参考模型约束避免了训练不稳定
    - 有害响应数据通过先微调一个"越狱模型"来可扩展地生成

3. **Token级加权（W-DOOR）**:

    - 功能：在SFT拒绝训练中对不同token赋予不同权重，优先强化关键拒绝token
    - 核心思路：基于KL正则化优化原理，定义token级奖励 $r(s_t, a_t) = \log\frac{\pi^*(y_t|x,y_{<t})}{\pi_{\text{ref}}(y_t|x,y_{<t})}$，权重为：
    $\beta_t = \exp\left(\frac{1}{\tau}r(s_t, a_t)\right) = \left(\frac{\pi^*(y_t|x,y_{<t})}{\pi_{\text{ref}}(y_t|x,y_{<t})}\right)^{1/\tau}$
      其中 $\pi^*$ 用DPO对齐模型近似，$\tau$ 为温度参数
    - 设计动机：关键拒绝token（如"Sorry"、"cannot"等）在 $\pi^*$ 和 $\pi_{\text{ref}}$ 间概率比最大，应获得最强梯度更新。非重要token（可能与有害token重叠）不应被过度强化，避免干扰遗忘过程

### 损失函数 / 训练策略
**DOOR总损失**：
$$\mathcal{L}_{\text{DOOR}} = \mathbb{E}_{(x,y^s,y^h),k}\left[-\log\pi_\theta(y^s \mid x\oplus y^h_{<k}) - \frac{2}{\beta}\log\sigma\left(-\beta\cdot\log\frac{\pi_\theta(y^h|x)}{\pi_{\text{ref}}(y^h|x)}\right)\right]$$

**W-DOOR总损失**（加入token级加权）：
$$\mathcal{L}_{\text{W-DOOR}} = \mathbb{E}\left[\sum_{t=1}^T\left(-\beta_t\log\pi_\theta(y_t^s|x,y_{<t}^s) - \frac{2}{\beta}\log\sigma\left(-\beta\log\frac{\pi_\theta(y_t^h|x,y_{<t}^h)}{\pi_{\text{ref}}(y_t^h|x,y_{<t}^h)}\right)\right)\right]$$

**能力保持损失**：
$$\mathcal{L}_{\text{Total}} = \alpha\mathcal{L}_{\text{Align}} + (1-\alpha)\mathcal{L}_{\text{Retain}}$$

训练设置：10 epochs, H100 GPU, batch size 2, lr $1\times10^{-5}$, $\beta=0.5$, $\alpha=0.2$, W-DOOR中$\tau=5$。

**DOOR梯度优势分析**：
- 安全token的有效学习率恒为1（DPO中会指数衰减），保证持续强化拒绝行为
- 梯度中包含 $\mathbb{E}_{y\sim\pi_\theta}[\nabla s_{\theta,y}(x)]$ 项，相当于对所有响应的logit施加正则化，改善OOD泛化

## 实验关键数据

### 主实验

| 方法 | Multi-turn ASR↓ | Prefilling ASR↓ | GCG ASR↓ | AutoDAN ASR↓ | HellaSwag↑ | XSTest Refusal↓ |
|------|----------------|----------------|----------|-------------|-----------|----------------|
| **Llama-3-8B** | | | | | | |
| Original | 0.521 | 0.547 | 0.307 | 0.198 | 0.577 | 0.409 |
| DPO | 0.521 | 0.210 | 0.133 | 0.138 | 0.564 | 0.456 |
| RR | 0.213 | 0.338 | 0.045 | 0.000 | 0.574 | 0.404 |
| SFT (Qi et al.) | 0.511 | 0.071 | 0.143 | 0.136 | 0.564 | 0.396 |
| DOOR | 0.489 | 0.055 | 0.093 | 0.095 | 0.565 | 0.407 |
| W-DOOR | 0.447 | 0.034 | 0.093 | 0.088 | 0.573 | 0.440 |
| **Gemma-2-2B** | | | | | | |
| Original | 0.554 | 0.346 | 0.190 | 0.098 | 0.536 | 0.422 |
| DPO | 0.446 | 0.060 | 0.148 | 0.048 | 0.478 | 0.438 |
| SFT (Qi et al.) | 0.505 | 0.010 | 0.156 | 0.020 | 0.513 | 0.400 |
| DOOR | 0.525 | 0.009 | 0.106 | 0.015 | 0.504 | 0.407 |
| W-DOOR | 0.347 | 0.005 | 0.103 | 0.020 | 0.507 | 0.440 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 数据增强 vs 无增强 | ASR大幅降低 | 增强对SFT系方法效果显著，DPO获益较小 |
| 梯度上升遗忘 | HellaSwag/MMLU严重下降 | 朴素GA会导致模型退化，NPO更稳定 |
| W-DOOR τ=1/3/5/10 | 鲁棒性对τ不敏感 | 指数/sigmoid变体效果相近 |
| 越狱模型替代参考模型 | 轻微提升 | 提供更强对比信号但需更多考量 |

### 关键发现
- **数据增强是关键**：暴露模型于有害前缀后仍训练拒绝，是提升prefilling攻击鲁棒性的核心因素
- **DPO确实不如SFT系方法学拒绝**：DPO的ASR模式更接近NPO而非SFT，验证了理论分析
- **W-DOOR在multi-turn攻击中独特优势**：其他方法ASR随轮数增加而上升，W-DOOR反而保持稳定甚至下降
- **W-DOOR最好地保持通用能力**：HellaSwag准确率几乎不降，DPO能力损失最大
- **KL散度与鲁棒性强相关**：W-DOOR相对基座模型的KL散度最大，且对深层token位置的影响更均匀
- **Pareto分析**：大部分鲁棒性增益在第1个epoch获得，后续训练主要恢复通用能力
- **过度拒绝不随训练增加**：延长训练反而同时降低ASR和过度拒绝率

## 亮点与洞察
- **梯度分解分析**非常精彩：从数学上清晰说明了DPO在安全场景的两个根本缺陷，不是经验性观察而是有理论支撑
- **DOOR的梯度有效学习率恒为1**：这是相比DPO最关键的优势，确保安全响应概率能持续增长
- **token级加权的思路**有普适价值：通过 $\pi^*/\pi_{\text{ref}}$ 比值自动识别关键拒绝token，无需人工定义
- W-DOOR通过降低不重要token权重，**避免了拒绝训练和遗忘训练之间的梯度冲突**（不重要安全token可能与有害token重叠）
- **t-SNE可视化**直观展示了W-DOOR实现了"更深层"的对齐——安全/有害表征分离更明显
- Prefilling ASR可作为整体鲁棒性的可靠代理指标，简化评估

## 局限与展望
- 多轮攻击防御效果仍有限（所有方法ASR都偏高），可能需要多轮/长上下文训练数据
- 数据增强中均匀采样前缀长度 $k$ 可能导致过度拒绝（部分良性前缀被错误学习为有害信号）
- token级权重参数 $\tau$ 虽不敏感但缺乏原则性选择方法
- 仅在Gemma-2-2B和Llama-3-8B上验证，未测试更大规模模型
- 训练数据规模小（~400安全+400有害+400通用），大规模数据下表现有待验证
- W-DOOR的过度拒绝率略高于DOOR，加权设计引入了安全性-可用性的微妙权衡

## 相关工作与启发
- **vs DPO**: DOOR从理论上修正了DPO在安全场景的两个根本缺陷，实验全面验证
- **vs Decoupled Refusal Training (Yuan et al.)**: 类似的数据增强思路，但DOOR额外加入NPO遗忘和token级加权
- **vs Safe Unlearning (Zhang et al.)**: 也结合遗忘和SFT，但未用数据增强
- **vs Circuit Breakers (Zou et al.)**: 表征工程方法，与训练级方法正交互补
- **vs TAR (Tamirisa et al.)**: 防篡改方法，但实验显示其对suffix攻击反而更脆弱

## 评分
- 新颖性: ⭐⭐⭐⭐ 梯度分析提供了理论洞察，DOOR框架设计合理，token级加权有创新
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖4种攻击类型、2个模型、多个baseline，含Pareto分析和表征可视化
- 写作质量: ⭐⭐⭐⭐⭐ 理论分析→方法设计→实验验证逻辑链清晰，图表丰富且信息量大
- 价值: ⭐⭐⭐⭐ 对LLM安全对齐领域有实际指导意义，DOOR/W-DOOR可直接用于改进现有安全训练pipeline

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Safety Alignment Can Be Not Superficial With Explicit Safety Signals](safety_alignment_can_be_not_superficial_with_explicit_safety_signals.md)
- [\[ICML 2025\] Align-then-Unlearn: Embedding Alignment for LLM Unlearning](align-then-unlearn_embedding_alignment_for_llm_unlearning.md)
- [\[NeurIPS 2025\] Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](../../NeurIPS2025/llm_safety/learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)
- [\[CVPR 2026\] Machine Unlearning via Adaptive Gradient Reweighting and Multi-stage Objective Optimization](../../CVPR2026/llm_safety/machine_unlearning_via_adaptive_gradient_reweighting_and_multi-stage_objective_o.md)
- [\[AAAI 2026\] SafeNlidb: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces](../../AAAI2026/llm_safety/safenlidb_a_privacy-preserving_safety_alignment_framework_for_llm-based_natural_.md)

</div>

<!-- RELATED:END -->
