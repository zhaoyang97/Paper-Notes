---
title: >-
  [论文解读] Long-Context Generalization with Sparse Attention
description: >-
  [ICLR 2026][目标检测][稀疏注意力] 提出 ASEntmax（Adaptive-Scalable Entmax），用可学习温度的 α-entmax 替代 softmax 注意力，从理论和实验两方面证明稀疏注意力能实现 1000× 长度外推，解决 softmax 在长上下文下的注意力弥散（dispersion）问题。
tags:
  - "ICLR 2026"
  - "目标检测"
  - "稀疏注意力"
  - "长上下文泛化"
  - "α-entmax"
  - "长度外推"
  - "Transformer"
---

# Long-Context Generalization with Sparse Attention

**会议**: ICLR 2026  
**arXiv**: [2506.16640](https://arxiv.org/abs/2506.16640)  
**代码**: [deep-spin/asentmax](https://github.com/deep-spin/asentmax)  
**领域**: 目标检测  
**关键词**: 稀疏注意力, 长上下文泛化, α-entmax, 长度外推, Transformer

## 一句话总结

提出 ASEntmax（Adaptive-Scalable Entmax），用可学习温度的 α-entmax 替代 softmax 注意力，从理论和实验两方面证明稀疏注意力能实现 1000× 长度外推，解决 softmax 在长上下文下的注意力弥散（dispersion）问题。

## 研究背景与动机

**Softmax 的注意力弥散问题**：随着上下文长度 $n$ 增大，softmax 将概率质量分散到所有 token 上，导致相关 token 的注意力权重趋近于零。理论上，当 $n \to \infty$ 时，softmax 的归一化熵趋近 1（完全均匀分布），即 **complete dispersion**。

**长度外推失败的根本原因**：模型在短序列上训练时学到的注意力模式无法迁移到长序列——softmax 在长序列中的权重分布与短序列截然不同，导致检索和推理能力崩溃。

**已有长上下文方案的局限**：RoPE 外推、ALiBi 等位置编码方法只处理位置信息，不解决注意力分布本身的弥散问题；Scalable Softmax (SSMax) 通过缩放因子缓解但缺乏理论保证。

**稀疏注意力的理论优势**：α-entmax 等稀疏变换能将不相关 token 的注意力精确置零，天然避免弥散。但此前缺乏严格的理论分析来解释为何稀疏注意力有助于长度外推。

**三大理论性质的缺失**：需要形式化证明稀疏注意力在以下方面优于 softmax：(1) 非消失注意力；(2) 集中度韧性（concentration resilience）；(3) 表征保持（representational preservation）。

**自适应稀疏度的需求**：不同注意力头在不同层可能需要不同程度的稀疏性，固定 α 过于僵硬，需要可学习的自适应机制。

## 方法详解

### 整体框架

ASEntmax 不改动 Transformer 的整体结构，只把每个注意力头里的 softmax 换成稀疏的 α-entmax，再给它配一套随上下文长度自适应缩放的温度。论文分两步推进：先从理论上说清"为什么稀疏注意力天生抗长度外推失败"——α-entmax 能把无关 token 的权重精确置零，由此推出三条形式化性质（非消失、非弥散、表征保持）；再针对"固定稀疏度在超长序列上会过度尖锐"这一副作用，提出 Adaptive-Scalable Entmax（ASEntmax），让每个头的反温度（inverse temperature）随序列长度 $n$ 以 $\delta + \beta(\log n)^\gamma$ 的形式可学习地变化，在稀疏（聚焦固定模式）与稠密（接近 softmax）之间平滑插值。最终在短序列上训练、远超训练长度的序列上测试时，注意力分布的形态不随长度发生质变，从而实现最高 1000× 的长度外推。

### 关键设计

**1. α-entmax 稀疏变换：让不相关 token 的权重精确归零**

softmax 的根本问题在于它的输出永远是全正的稠密分布，每个 token 都分到一点概率，长序列里相关 token 的权重因此被无关 token 稀释殆尽。α-entmax 是 softmax 的连续推广，$\alpha = 1$ 时退化为 softmax、$\alpha = 2$ 时即 sparsemax，而当 $\alpha > 1$ 它的输出会包含精确的零值，自动把得分低于阈值的无关 token 排除出注意力支撑集（support）。这个变换仍然可微，能端到端训练，因此只是把 softmax 这一个算子换掉，就从源头堵住了概率质量向无关 token 流失的通道。

**2. 三条理论性质：从数学上证明稀疏注意力为何能外推**

本文的核心理论贡献是为"稀疏注意力有利于长度外推"给出形式化证明，归结为三条性质。其一是**非消失注意力（Non-vanishing Attention）**：当 $\alpha > 1$ 时，往序列里添加得分低于阈值的无关 token，相关 token 的权重完全不变；而 softmax 无论新 token 是否相关，都会按比例削减所有已有权重。其二是**集中度韧性 / 非弥散（Concentration Resilience）**：用归一化熵 $H(z)/\log n$ 度量注意力的弥散程度——softmax 在 $n \to \infty$ 时该值趋近 1（完全弥散为均匀分布），而 α-entmax 的熵上界是 $O(\log s)$（$s$ 为支撑集大小、$s \ll n$），与序列长度 $n$ 无关，归一化熵被压在 1 以下、不随长度抬升。这意味着序列即便放大 1000×，只要真正相关的 token 数 $s$ 不变，注意力的集中程度就保持不变——这个"不弥散"正是短序列注意力模式能迁移到长序列的根本原因。其三是**表征保持（Representational Preservation）**：在 $L$ 层网络中，softmax 的梯度路径数为 $O(n^L)$，深层会因路径组合爆炸而表征坍缩、加剧过度挤压（over-squashing），α-entmax 把它压到 $O(s^L)$，从而强化了长程依赖的梯度流、在长序列下仍能区分不同输入。这三条共同解释了为什么换一个算子就能换来量级的外推能力。

**3. ASEntmax 自适应可伸缩温度：随长度调节稀疏度，避免长序列过度尖锐**

固定的 $\alpha$ 与固定温度有个反向的隐患：当真正相关的 token 很多时，固定稀疏度会"忽略得太狠"，在长序列上让注意力过度尖锐（overly peaky）。ASEntmax 的做法是把缩放因子做成序列长度 $n$ 的函数，对每个头独立可学习：

$$\text{ASEntmax}(z) = \alpha\text{-entmax}\big((\delta + \beta(\log n)^\gamma)\,z\big)$$

其中 $\delta, \beta, \gamma$ 是每个头各自学习的标量（即反温度系数）。$\gamma > 0$ 让温度随长度缓慢上升、$\gamma < 0$ 则随长度衰减，模型据此学出"稀疏度该如何随长度演化"的调度；$\beta = 0$ 时退化为标准 α-entmax，因此缩放与不缩放之间能平滑切换。把缩放写成 $\log n$ 的形式而非直接乘 $n$，是为了不干扰位置编码。实验里不同头学到了不同的调度系数，正说明这种按头、按长度的自适应是有必要的——它在 α-entmax 天然的集中性之上，额外给了"稀疏模式如何随长度变化"的精确控制。

### 损失函数 / 训练策略

训练沿用标准语言模型目标，即 next-token 预测的交叉熵损失，每个头的反温度系数 $\delta, \beta, \gamma$ 通过反向传播与模型参数联合优化。$\alpha$ 一般固定为实验验证的较优值 1.5（也可设为可学习，但会引入不稳定，见消融）。整个评测的关键设定是在短序列（如长度 64）上训练、直接在远超训练长度的序列（如 65K）上测试，以此考查纯粹的外推能力。

## 实验关键数据

### 主实验

Associative Recall 任务（训练长度 64）的长度外推准确率：

| 方法 | 64 | 256 | 1K | 4K | 16K | 65K |
|------|-----|------|------|------|------|------|
| Softmax | 99.8% | 52.1% | 12.3% | 3.1% | 0.8% | 0.2% |
| SSMax | 99.7% | 89.4% | 71.2% | 45.6% | 28.3% | 15.1% |
| Adaptive Temp | 99.6% | 91.2% | 78.5% | 52.3% | 34.7% | 21.4% |
| **ASEntmax** | **99.9%** | **99.5%** | **99.1%** | **98.2%** | **96.8%** | **95.3%** |

### 消融实验

α 和温度可学习性的影响（Associative Recall, 测试长度 16K）：

| 配置 | 准确率 | 说明 |
|------|--------|------|
| ASEntmax (α=1.5, θ 可学习) | **96.8%** | 最优配置 |
| α-entmax (α=1.5, 固定温度) | 88.4% | 缺乏自适应能力 |
| α-entmax (α=2.0, 固定温度) | 82.1% | 过度稀疏导致信息损失 |
| ASEntmax (α 可学习, θ 可学习) | 95.2% | α 学习不稳定，略有下降 |
| Softmax + Adaptive Temp | 34.7% | 温度无法解决 softmax 的根本弥散问题 |

### 关键发现

1. **1000× 外推**：训练长度 64 → 测试长度 65K，ASEntmax 保持 95.3% 准确率，softmax 降至 0.2%
2. **语言建模优势**：在长上下文 LM 评估中，ASEntmax 在 8× 训练长度时的困惑度趋势显著优于 softmax 和 SSMax
3. **检索能力保持**：在远超训练长度的 needle-in-a-haystack 测试中，ASEntmax 保持高检索成功率
4. **稀疏度自适应**：不同层和头学到了不同的温度调度系数，验证了按头、按长度自适应机制的必要性

## 亮点与洞察

- **理论深度扎实**：三大性质（non-vanishing, concentration resilience, representational preservation）的形式化证明是论文最大贡献，为稀疏注意力的长度外推优势提供了严格的数学基础
- **Dispersion 概念的提出**：将 softmax 的长上下文失败统一归因为"弥散"，并用归一化熵定量刻画，概念清晰且有说服力
- **$O(s^L)$ vs $O(n^L)$ 的洞察**：揭示了稀疏注意力在深层网络中的本质优势——梯度路径的组合爆炸被稀疏性有效抑制
- **简洁的实现**：仅替换 softmax 为 α-entmax + 可学习温度，无需额外架构修改，工程实现友好

## 局限与展望

1. **计算效率**：α-entmax 的前向/反向传播涉及排序操作，复杂度为 $O(n \log n)$，比 softmax 的 $O(n)$ 更高；尽管稀疏输出可加速后续计算，但注意力计算本身更慢
2. **预训练成本**：需要从头预训练或全量微调，不能简单作为 drop-in replacement 应用于已有预训练模型
3. **大规模验证不足**：实验主要在中等规模模型上进行，尚未在 7B+ 参数的大模型上验证
4. **与 FlashAttention 的兼容性**：稀疏注意力的不规则访存模式可能与 FlashAttention 等硬件优化方法冲突
5. **α 值的选择**：虽然实验表明 1.5 较优，但缺乏理论指导来确定最优 α

## 相关工作与启发

- **Scalable Softmax (SSMax)**：通过 $\log n$ 偏置项缩放 softmax logits，缓解弥散但不根治——本文的理论分析解释了为何 SSMax 效果有限
- **RoPE / ALiBi / YaRN**：位置编码层面的长度外推方法，与 ASEntmax 是正交的改进方向，可组合使用
- **Entmax (Peters et al., 2019)**：α-entmax 的原始工作，主要用于 NLP 分类和翻译任务，本文首次将其与长上下文外推联系起来
- **Sparse Transformer (Child et al., 2019)**：结构化稀疏注意力，与 α-entmax 的数据驱动稀疏不同
- **Gated Attention / Linear Attention**：替代 softmax 的其他方案，但缺乏 α-entmax 的理论保证

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 三大理论性质的形式化证明具有开创性，将稀疏注意力与长度外推建立了严格的数学联系
- **实验充分度**: ⭐⭐⭐⭐ — 合成任务和语言建模均有覆盖，1000× 外推结果令人印象深刻，但缺乏大规模模型验证
- **写作质量**: ⭐⭐⭐⭐⭐ — 理论推导清晰，概念层次分明，dispersion 的定义和可视化非常直观
- **价值**: ⭐⭐⭐⭐ — 为长上下文 LLM 提供了一个理论上有保证的新方向，但工程落地仍需解决效率和兼容性问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Enhancing Vision Transformers for Object Detection via Context-Aware Token Selection and Packing](enhancing_vision_transformers_for_object_detection_via_context-aware_token_selec.md)
- [\[ICLR 2026\] SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection](spwood_sparse_partial_weakly-supervised_oriented_object_detection.md)
- [\[CVPR 2026\] Bridge: Basis-Driven Causal Inference Marries VFMs for Domain Generalization](../../CVPR2026/object_detection/bridge_basis-driven_causal_inference_marries_vfms_for_domain_generalization.md)
- [\[ICCV 2025\] Adversarial Attention Perturbations for Large Object Detection Transformers](../../ICCV2025/object_detection/adversarial_attention_perturbations_for_large_object_detection_transformers.md)
- [\[AAAI 2026\] CountSteer: Steering Attention for Object Counting in Diffusion Models](../../AAAI2026/object_detection/countsteer_steering_attention_for_object_counting_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
