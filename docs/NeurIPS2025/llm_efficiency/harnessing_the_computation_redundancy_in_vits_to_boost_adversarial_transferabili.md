---
title: >-
  [论文解读] Harnessing the Computation Redundancy in ViTs to Boost Adversarial Transferability
description: >-
  [NeurIPS 2025][LLM效率][对抗迁移性] 深入挖掘 ViT 中数据级和模型级的计算冗余，提出注意力稀疏化、注意力头置换、干净 token 正则化、Ghost MoE 多样化和鲁棒化 token 五种技术，结合在线学习策略动态选择操作，在 ImageNet-1K 上以 86.9% 平均 fooling rate 大幅超越所有基线。
tags:
  - "NeurIPS 2025"
  - "LLM效率"
  - "对抗迁移性"
  - "ViT 计算冗余"
  - "注意力稀疏化"
  - "Ghost MoE"
  - "鲁棒化 token"
---

# Harnessing the Computation Redundancy in ViTs to Boost Adversarial Transferability

**会议**: NeurIPS 2025  
**arXiv**: [2504.10804](https://arxiv.org/abs/2504.10804)  
**代码**: 无  
**领域**: 机器人  
**关键词**: 对抗迁移性, ViT 计算冗余, 注意力稀疏化, Ghost MoE, 鲁棒化 token

## 一句话总结
深入挖掘 ViT 中数据级和模型级的计算冗余，提出注意力稀疏化、注意力头置换、干净 token 正则化、Ghost MoE 多样化和鲁棒化 token 五种技术，结合在线学习策略动态选择操作，在 ImageNet-1K 上以 86.9% 平均 fooling rate 大幅超越所有基线。

## 背景与动机

1. **ViT 的对抗迁移优势**：已有研究表明基于 ViT 生成的对抗样本比 CNN 有更强的迁移性，但对其根本原因缺乏深入理解。
2. **计算冗余的洞察**：ViT 中存在大量冗余——数据级（token 可以被大量剪枝而不降低性能）和模型级（注意力头冗余、FFN 中 dropout 训练带来的神经元冗余）。
3. **核心假设**：这些冗余不是浪费，而是可以被重新利用来增强对抗迁移性——通过多样化模型的中间表示来减少对代理模型的过拟合。

## 核心问题
如何系统地利用 ViT 固有的计算冗余来提升对抗样本从代理模型到黑盒受害模型的迁移性？

## 方法详解

### 技术 1：注意力稀疏化
对注意力 logits 施加随机二值掩码，丢弃比例为 $r$：

$$\text{MHA}(\mathbf{z}) = \text{softmax}\left(\left(\frac{\mathbf{z}\mathbf{W}_Q(\mathbf{z}\mathbf{W}_K)^\top}{\sqrt{d_k}}\right) \odot \mathbf{M}\right)\mathbf{z}\mathbf{W}_V$$

其中 $\mathbf{M}\in\{0,1\}^{N\times N}$ 是丢弃比 $r$ 的随机掩码。结果：$r\leq 0.4$ 时白盒成功率几乎不变，黑盒迁移性显著提升。

### 技术 2：注意力头置换
随机打乱不同注意力头的 QK 权重，保持 V 不变：

$$\text{MHA}(\mathbf{z}) = \text{Concat}\left(\text{softmax}\left(\pi\left(\frac{\mathbf{Q}_1\mathbf{K}_1^\top}{\sqrt{d_k}},...,\frac{\mathbf{Q}_H\mathbf{K}_H^\top}{\sqrt{d_k}}\right)\right)[\mathbf{V}_1,...,\mathbf{V}_H]^T\right)$$

每层以概率 $p$ 被选中，被选层中 $r$ 比例的头被随机置换。验证了冗余头学习到相似的注意力模式。

### 技术 3：干净 token 正则化
在每个 Transformer block 中插入少量来自干净样本的 token，作为稳定锚点正则化对抗表示。采样比 $r=0.3\sim0.5$ 效果最佳。

### 技术 4：Ghost MoE 多样化
用不同 dropout 掩码实例化同一 FFN 的多个"ghost 专家"：

$$\text{MoE}(\mathbf{z}) = \frac{1}{q}\sum_{e=1}^q \text{FFN}_{\theta_e}(\mathbf{z}), \quad q\sim\mathcal{U}(1,E)$$

最佳配置：drop rate 0.3，增加专家数持续提升性能。

### 技术 5：鲁棒化 token
在 patch embedding 后附加 $N_r$ 个可学习 token，通过测试时对抗训练优化：

$$\min_{\mathbf{z}_r}\max_\delta \mathcal{L}(f(x+\delta;\mathbf{z}_r), y)$$

动态版本（逐样本优化）和全局版本（在校准集上离线预训练）均有效，最高提升 14%+。

### 在线学习策略
初始化采样矩阵 $\mathbf{M}\in\mathbb{R}^{L\times O}$，每层 $l$ 以概率选择操作 $\phi_o$，用 REINFORCE 优化操作分布：

$$\max_\mathbf{M}\;\mathbb{E}_{\phi\sim\mathbf{M}}[\mathcal{L}(f(x+\delta(\phi)),y)]$$

自适应地为每个 Transformer block 选择最优冗余利用策略。

## 实验关键数据

### ViT-B/16 作为代理模型

| 方法 | RN-50 | VGG-16 | MN-V2 | Inc-v3 | ViT | PiT | Vis-S | Swin | 平均 |
|------|-------|--------|-------|--------|-----|-----|-------|------|------|
| MI-FGSM | 39.4 | 58.4 | 57.9 | 42.2 | 97.4 | 40.4 | 42.0 | 55.0 | 54.1 |
| PGN | 68.9 | 75.7 | 76.3 | 72.4 | 97.6 | 75.6 | 75.5 | 80.0 | 77.8 |
| TGR | 53.4 | 72.5 | 72.4 | 55.5 | 97.7 | 59.2 | 61.8 | 74.5 | 68.4 |
| **Ours** | **77.7** | **90.6** | **91.1** | **79.9** | **99.7** | **78.9** | **83.5** | **93.5** | **86.9** |

### PiT-B / Swin-T 作为代理

| 代理 | 方法 | 平均 fooling rate |
|------|------|-------------------|
| PiT-B | PGN | 75.1% |
| PiT-B | **Ours** | **87.4%** |
| Swin-T | PGN | 85.3% |
| Swin-T | **Ours** | **88.9%** |

### 攻击 VLLMs（ViT 代理 → LLaVA/Qwen/InternVL/DeepSeek）
平均提升 2.2%，在 Qwen 上超越 runner-up 5.5%。

## 亮点
- **系统性分析**：首次从"计算冗余"视角统一理解 ViT 对抗迁移性
- **五种互补技术**：从数据级（token）到模型级（FFN/MHA）全面覆盖
- **在线学习策略**：自适应组合操作，避免人工调参
- **跨架构有效**：从 CNN 到 ViT 到 VLLM 全面验证
- 比最强基线 PGN 平均提升 9+ 个百分点

## 局限性
- 五种技术各有超参数（丢弃比、专家数、token 数等），联合调参空间大
- 鲁棒化 token 的逐样本优化版本计算开销较高
- 在线学习策略引入 REINFORCE 估计器，方差可能较大
- 防御方法（如对抗训练后的模型）的鲁棒性未充分验证

## 评分
- 新颖性: ⭐⭐⭐⭐ — "冗余即资源"视角新颖，技术组合全面
- 实验充分度: ⭐⭐⭐⭐⭐ — 8 种模型、3 种代理、VLLM、详尽消融
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，由分析到设计逻辑通顺
- 综合价值: ⭐⭐⭐⭐ — 对抗迁移攻击领域的重要进展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Scaling and Transferability of Annealing Strategies in Large Language Model Training](../../AAAI2026/llm_efficiency/scaling_and_transferability_of_annealing_strategies_in_large_language_model_trai.md)
- [\[AAAI 2026\] Harnessing the Unseen: The Hidden Influence of Intrinsic Knowledge in Long-Context Language Models](../../AAAI2026/llm_efficiency/harnessing_the_unseen_the_hidden_influence_of_intrinsic_knowledge_in_long-contex.md)
- [\[ICLR 2026\] SwingArena: Adversarial Programming Arena for Long-context GitHub Issue Solving](../../ICLR2026/llm_efficiency/swingarena_competitive_programming_arena_for_long-context_github_issue_solving.md)
- [\[ACL 2026\] Small Data, Big Noise: Adversarial Training for Robust Parameter-Efficient Fine-Tuning](../../ACL2026/llm_efficiency/small_data_big_noise_adversarial_training_for_robust_parameter-efficient_fine-tu.md)
- [\[NeurIPS 2025\] 3-Model Speculative Decoding (PyramidSD)](3model_speculative_decoding.md)

</div>

<!-- RELATED:END -->
