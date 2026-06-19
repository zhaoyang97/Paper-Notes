---
title: >-
  [论文解读] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference
description: >-
  [NeurIPS 2025][图像恢复][扩散模型] 提出 E2D2，一种面向离散扩散语言模型的编码器-解码器架构，通过轻量解码器迭代去噪、大型编码器定期更新表征，同时实现更快推理（~3× vs MDLM）和更高效的 block diffusion 训练（FLOPs 减半）。 离散扩散模型在语言建模中能实现并行 token…
tags:
  - "NeurIPS 2025"
  - "图像恢复"
  - "扩散模型"
  - "encoder-decoder"
  - "language model"
  - "KV caching"
---

# Encoder-Decoder Diffusion Language Models for Efficient Training and Inference

**会议**: NeurIPS 2025  
**arXiv**: [2510.22852](https://arxiv.org/abs/2510.22852)  
**代码**: [kuleshov-group/e2d2](https://github.com/kuleshov-group/e2d2)  
**领域**: 图像生成  
**关键词**: discrete diffusion, encoder-decoder, block diffusion, language model, KV caching

## 一句话总结

提出 E2D2，一种面向离散扩散语言模型的编码器-解码器架构，通过轻量解码器迭代去噪、大型编码器定期更新表征，同时实现更快推理（~3× vs MDLM）和更高效的 block diffusion 训练（FLOPs 减半）。

## 背景与动机

离散扩散模型在语言建模中能实现并行 token 采样，推理速度优于自回归方法。然而现有方法存在关键效率瓶颈：

1. **全网络调用**：decoder-only 架构在每步去噪时都需要完整前向传播，计算开销大
2. **Block diffusion 训练代价**：BD3LM 需同时处理完整 clean 序列和 noised 序列（2L tokens），训练前向 FLOPs 是标准扩散模型的 2×
3. **缺乏 KV 缓存支持**：全序列扩散模型（如 MDLM、LLaDA）使用双向注意力，限制了 KV caching 的使用

作者的核心洞察：离散扩散去噪过程本质上执行两类计算——(1) 表征 clean tokens 和 (2) 去噪 masked tokens，这两类任务可以用不同的模块分别处理。

## 核心问题

如何利用编码器-解码器分离设计，在不牺牲生成质量的前提下，同时加速离散扩散模型的训练和推理？

## 方法详解

### 1. 架构设计

**编码器**：$N_{\text{Enc}}$ 层 Transformer，接收 clean tokens（prompt + 已解码 tokens），输出特征表示 $\mathbf{h}_t = \text{Encoder}(\mathbf{x}_{t,\text{Enc}})$。

**解码器**：$N_{\text{Dec}}$ 层轻量 Transformer（$N_{\text{Dec}} \ll N_{\text{Enc}}$），接收当前被去噪的 noised 序列，通过交叉注意力与编码器输出交互：

$$\mathbf{x}_{\text{logit}} = \text{Decoder}(\mathbf{z}_{t,\text{Dec}}, \mathbf{h}_t)$$

**关键**：解码器可**多次调用**进行迭代去噪，而编码器仅在有新 tokens 生成后**定期调用**一次更新表征。

### 2. 两种连接方式

- **Last Hidden State**：将编码器最后一层输出拼接到解码器每一层的输入前，类似 T5
- **Shared KV Cache**：解码器第 $i$ 层复用编码器第 $j$ 层的 KV cache——适合从预训练 decoder-only 模型微调

两者都使用**融合注意力核**，将 self-attention 和 cross-attention 合并为单次调用，减少内核启动开销。

### 3. 采样算法

```
1. 编码 prompt → 获得编码器特征 h 和 KV cache
2. for 每个 block b:
     初始化 block 为全 [MASK]
     for 每步去噪:
       z_t ← Sample(Decoder(z_t, h))  // 仅调用轻量解码器
     将生成的 block 送入编码器更新 KV cache
```

由于每个 block 的 $T$ 步去噪仅需调用轻量解码器，主要计算瓶颈从 $BT \cdot O(\theta_{\text{full}})$ 降为 $B \cdot O(\phi) + BT \cdot O(\theta_{\text{small}})$，其中 $O(\theta_{\text{small}}) \ll O(\phi)$。

### 4. 训练算法

通过定制的注意力掩码实现**单次前向传播**训练所有 block：
- 编码器使用 block-causal mask，clean tokens 只关注当前及之前 block
- 解码器使用跨越 $2L$ 个 KV 的掩码，noised tokens 关注自身 block + 之前 clean blocks 的编码器表征

**FLOPs 对比**（$N$ 总层数，$L$ 序列长度，$S$ block 大小）：

| 模型 | Attention FLOPs | MLP FLOPs |
|------|----------------|-----------|
| MDLM | $4NDL^2$ | $24NLD^2$ |
| BD3LM | $4ND(L^2 + LS)$ | $48NLD^2$ |
| **E2D2** | $4(N_E+N_D)D\frac{L^2+LS}{2}$ | $24(N_E+N_D)LD^2$ |

相同总层数下，E2D2 训练 FLOPs 比 BD3LM **减少 2×**。

## 实验关键数据

### 文本摘要（CNN/DailyMail）

| 模型 | 层数 | 吞吐量 (tok/s) | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------|------|--------------|---------|---------|---------|
| AR | 28 | 89.1 | 31.7 | 11.7 | 22.1 |
| MDLM | 28 | 49.3 | 30.6 | 12.5 | 22.7 |
| BD3LM | 12 | 135.1 | 35.8 | 13.7 | 23.7 |
| **E2D2** | 20/8 | **155.8** | **36.0** | **14.1** | **23.9** |

E2D2 比 AR 吞吐量高 ~75%，ROUGE-L 更高；比 MDLM 快 ~3×。

### 机器翻译（WMT14 de-en）

| 模型 | 层数 | 吞吐量 (tok/s) | BLEU |
|------|------|--------------|------|
| AR | 32 | 77.6 | 25.2 |
| MDLM | 32 | 60.4 | 18.4 |
| BD3LM | 16 | 102.4 | 24.7 |
| **E2D2** | 24/8 | **124.3** | **25.1** |

### 数学推理（GSM8K）

通过从预训练 Qwen3 1.7B 微调，E2D2 在 pass@1 准确率上达到竞争力水平，证明了框架在推理任务上的有效性。

## 亮点

1. **解耦 clean 表征与去噪计算**的洞察简洁且有效，是对离散扩散建模的重要架构创新
2. Block diffusion 训练 **FLOPs 减半**，这对扩展到更大模型/更长序列有直接价值
3. 支持 **KV caching**，解决了扩散语言模型推理效率的核心瓶颈
4. 通过调节编码器/解码器的层数比例，可以灵活**映射质量-吞吐量的 Pareto 前沿**
5. **Shared KV Cache 变体**可从预训练 AR 模型直接微调，降低了实践门槛

## 局限与展望

1. 实验规模相对较小（<2B 参数），未在 7-8B 级模型上验证 E2D2 的扩展性
2. 编码器需定期重新编码已生成 tokens，对非常长的序列可能成为瓶颈
3. 当前仅在任务特定模型上评估，通用语言建模的效果有待进一步验证
4. 融合注意力核强制解码器在编码器表征和自身隐状态之间分配注意力，可能限制某些场景下的表达能力

## 与相关工作的对比

| 维度 | MDLM | BD3LM | LLaDA | E2D2 |
|------|------|-------|-------|------|
| 架构 | decoder-only | decoder-only | decoder-only | **enc-dec** |
| KV Cache | ✗ | ✓ | 近似 | **✓** |
| 训练 FLOPs（vs BD3LM） | 1× | 2× | 1× | **1×** |
| 推理吞吐（相对） | 低 | 中 | 低 | **高** |
| Block decoding | 可选 | 原生 | 推理时 | **原生** |

## 启发与关联

1. **"表征"与"生成"的分离**是一个通用原则：在 VLM、语音等多步迭代生成场景中，类似的编码器-解码器分离都可能带来效率提升
2. **Block diffusion 作为 AR 与扩散的桥梁**：E2D2 进一步强化了这一范式，表明分块自回归 + 块内扩散可能是扩散语言模型的最优实践
3. 从预训练 AR 模型初始化扩散训练的策略在多个工作中被验证有效，暗示 AR 预训练可能为扩散模型提供了良好的权重初始化

## 评分

- 新颖性: ⭐⭐⭐⭐ — 编码器-解码器用于离散扩散的想法自然但此前未被探索
- 实验充分度: ⭐⭐⭐⭐ — 多任务覆盖、Pareto 前沿分析、FLOPs 理论分析
- 写作质量: ⭐⭐⭐⭐⭐ — 论文结构优秀，算法伪代码清晰，FLOPs 推导严谨
- 价值: ⭐⭐⭐⭐ — 为扩散语言模型的实用化提供了有效的架构方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Consistent Diffusion Language Models](../../ICML2026/image_restoration/consistent_diffusion_language_models.md)
- [\[NeurIPS 2025\] Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates](rethinking_circuit_completeness_in_language_models_and_or_and_adder_gates.md)
- [\[NeurIPS 2025\] Elucidated Rolling Diffusion Models for Probabilistic Forecasting of Complex Dynamics](elucidated_rolling_diffusion_models_for_probabilistic_forecasting_of_complex_dyn.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](../../ICLR2026/image_restoration/activation_steering_for_masked_diffusion_language_models.md)
- [\[ICML 2026\] Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models](../../ICML2026/image_restoration/plan_for_speed_dilated_scheduling_for_masked_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
