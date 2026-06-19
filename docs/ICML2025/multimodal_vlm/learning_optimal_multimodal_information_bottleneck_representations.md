---
title: >-
  [论文解读] Learning Optimal Multimodal Information Bottleneck Representations
description: >-
  [ICML2025][多模态VLM][多模态学习] 提出 OMIB 框架，通过理论推导正则化参数 β 的上界并动态调整各模态权重 r，保证多模态信息瓶颈表示的最优性（包含全部任务相关信息、排除冗余信息）。 领域现状 领域现状：多模态信息瓶颈（MIB）方法通过最大化表示与标签的互信息、最小化与输入的互信息来平衡充分性与简洁性…
tags:
  - "ICML2025"
  - "多模态VLM"
  - "多模态学习"
  - "信息瓶颈"
  - "变分推断"
  - "regularization bound"
  - "注意力机制"
---

# Learning Optimal Multimodal Information Bottleneck Representations

**会议**: ICML2025  
**arXiv**: [2505.19996](https://arxiv.org/abs/2505.19996)  
**代码**: 未公开  
**领域**: 多模态VLM  
**关键词**: 多模态学习, 信息瓶颈, 变分推断, regularization bound, cross-attention fusion

## 一句话总结
提出 OMIB 框架，通过理论推导正则化参数 β 的上界并动态调整各模态权重 r，保证多模态信息瓶颈表示的最优性（包含全部任务相关信息、排除冗余信息）。

## 研究背景与动机

### 领域现状

**领域现状**：多模态信息瓶颈（MIB）方法通过最大化表示与标签的互信息、最小化与输入的互信息来平衡充分性与简洁性

### 现有痛点

**现有痛点**：现有问题一**：正则化参数 β 通常靠手工调参，取值不当会导致要么保留冗余信息、要么丢失任务相关信息，无法保证最优 MIB

### 核心矛盾

**核心矛盾**：现有问题二**：各模态的正则化权重固定，忽略了模态间任务相关信息不平衡问题（弱模态可能携带少量但关键信息）

### 解决思路

**解决思路**：现有问题三**：已有理论未同时覆盖一致性、互补性、特异性、充分性、简洁性五个信息论因素

### 补充说明

**补充说明**：作者首次给出了 β 的理论上界 $M_u$，在此范围内优化可保证达到最优 MIB

## 方法详解

### 整体框架（OMIB）
OMIB 包含两阶段训练：
1. **Warm-up 阶段**：Task Relevance Branch（TRB）为每个模态 $v_i$ 训练编码器 $Enc_i$ 提取充分表示 $z_i$，拼接高斯噪声后经预测头 $Dec_i$ 预测标签
2. **Main Training 阶段**：增加 Optimal Multimodal Fusion（OMF）模块，将 $z_i$ 通过 VAE 生成 $\zeta_i$，再用 Cross-Attention Network（CAN）融合为 MIB 表示 $\xi$

### 关键设计
- **VAE + 重参数化**：$\mu_i, \Sigma_i = VAE_i(z_i)$，$\zeta_i = \mu_i + \Sigma_i \times \epsilon_i$
- **Cross-Attention 融合**：$\xi = CAN(\zeta_1, \zeta_2)$

### 损失函数
OMF 的损失：

$$L_{OMF} = \frac{1}{N}\sum_{n=1}^{N} \mathbb{E}_{\epsilon_1}\mathbb{E}_{\epsilon_2}[-\log q(y^n|\xi^n)] + \beta(KL[p(\zeta_1^n|z_1^n)||\mathcal{N}(0,I)] + r \cdot KL[p(\zeta_2^n|z_2^n)||\mathcal{N}(0,I)])$$

其中 β 控制冗余约束强度，r 动态平衡两模态正则化。

### 动态权重 r 的计算
$$r = 1 - \tanh\left(\ln \frac{1}{N}\sum_{n}\frac{KL(p(\hat{y}_2^n|\xi^n,z_2^n)||p(\hat{y}^n|\xi^n))}{KL(p(\hat{y}_1^n|\xi^n,z_1^n)||p(\hat{y}^n|\xi^n))}\right)$$

当 $v_2$ 的任务相关信息仍大量未被编码时 r 偏小，鼓励从 $v_2$ 纳入更多信息。

### β 的理论上界
$$M_u = \frac{1}{(1+r)(H(v_1)+H(v_2)-I(v_1;v_2))}$$

设 $\beta \in (0, M_u]$ 即可保证 $F(\xi) = \{a_0,a_1,a_2\}$（最优 MIB），通过 MINE 从训练数据先验计算。

## 实验关键数据


### 主实验

| 任务 | 数据集 | OMIB | 最佳基线 | 提升 |
|------|--------|------|----------|------|
| 情感识别 | CREMA-D | **最优** | 次优基线 | 显著 |
| 情感分析 | CMU-MOSI | **最优** | 次优基线 | 回归+分类均领先 |
| 异常组织检测 | 10x-hBC | **最优** | DMIB | AUC 提升 |
| 合成数据 | SIM-I/III | 0.892/0.890 | 真实最优MIB 0.909/0.908 | 接近上界 |

- 合成数据验证：OMIB 生成的 MIB 准确率（0.892）接近理论最优 MIB（0.909）
- β 超出理论上界时性能显著下降，验证了 Proposition 5.7

## 亮点与洞察
- **理论创新突出**：首次给出最优 MIB 的可达性证明（Proposition 5.7），为 β 设定提供数学保障
- **动态权重 r**：优雅地解决了模态间信息不平衡问题，无需手工调整
- **五因素统一**：理论分析完整覆盖一致性、互补性、特异性、充分性、简洁性
- **合成+真实验证**：合成数据可精确验证理论性质

## 局限与展望
- β 上界的计算依赖 MINE 估计互信息，对高维复杂数据可能不精确
- 主要验证了两模态场景，多模态（≥3）的扩展虽在附录讨论但缺乏实验
- 合成数据场景相对简单（高斯分布），实际数据的信息结构远更复杂
- CAN 融合结构较固定，未探索其他融合策略的影响
- Warm-up 训练阶段使用随机噪声替代 MIB，过渡到 Main Training 的稳定性未充分讨论
- 对于大规模视觉-语言多模态任务（如 VQA、图文检索），scalability 是开放问题
- 动态权重 r 的 tanh 映射引入了人为的上界 (0,2)，可能限制极端不平衡场景
- 理论分析假设模态间信息结构符合 Venn 图式分解，实际模态关系可能更复杂
- 未与近年基于 Transformer 的端到端多模态融合方法（如 Perceiver、CoCa）做对比
- 训练包含 TRB + OMF 两阶段，增加了训练复杂度和超参数调优难度

## 相关工作与启发
- **L-MIB/E-MIB/C-MIB**（Mai et al., 2023）：探索不同融合阶段的 MIB，但 β 靠经验
- **DMIB**（Fang et al., 2024）：过滤冗余但缺少最优性保证
- **VIB**（Alemi et al., 2017）：变分信息瓶颈框架，OMIB 的变分近似基础
- 启发：理论约束+动态调整的结合是未来多模态学习的重要方向

## 评分
- 新颖性: ⭐⭐⭐⭐ (理论贡献扎实，首次证明最优 MIB 可达性)
- 实验充分度: ⭐⭐⭐⭐ (合成+多个真实任务，消融完整)
- 写作质量: ⭐⭐⭐⭐ (理论推导清晰，图示直观)
- 价值: ⭐⭐⭐⭐ (为 MIB 方法提供了坚实理论基础)

### 补充实验细节
- 合成数据上，当 β 设为理论上界 $M_u$ 时准确率最高，超过后急剧下降
- CREMA-D 情感识别：音视频双模态，6类情感分类
- CMU-MOSI 情感分析：视觉+音频+文本三模态，回归(-3到3)和分类任务
- 10x-hBC 异常组织检测：基因表达+组织学图像，SVDD异常检测
- 与 L-MIB、E-MIB、C-MIB、MMIB-Zhang、DMIB 等基线全面对比
- 非 MIB 基线包括 Concat、BiGated、MISA 等融合方法
- 变分近似使用 KL 散度，重参数化技巧保证可微性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Graph4MM: Weaving Multimodal Learning with Structural Information](graph4mm_weaving_multimodal_learning_with_structural_information.md)
- [\[AAAI 2026\] Conditional Information Bottleneck for Multimodal Fusion: Overcoming Shortcut Learning in Sarcasm Detection](../../AAAI2026/multimodal_vlm/conditional_information_bottleneck_for_multimodal_fusion_overcoming_shortcut_lea.md)
- [\[ACL 2026\] From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck](../../ACL2026/multimodal_vlm/from_verbatim_to_gist_distilling_pyramidal_multimodal_memory_via_semantic_inform.md)
- [\[CVPR 2026\] SeD-UD: An Influence-Driven and Hierarchically-Decoupled Information Bottleneck for Multimodal Intent Recognition](../../CVPR2026/multimodal_vlm/sed-ud_an_influence-driven_and_hierarchically-decoupled_information_bottleneck_f.md)
- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](../../NeurIPS2025/multimodal_vlm/visual_instruction_bottleneck_tuning.md)

</div>

<!-- RELATED:END -->
