---
title: >-
  [论文解读] Enhancing Outcome Reward-Based RL Training of MLLMs with Self-Consistency Sampling
description: >-
  [NeurIPS 2025][多模态VLM][强化学习] 针对多模态多选题中"结果奖励 RL 训练导致不忠实推理轨迹"的问题，提出 Self-Consistency Sampling (SCS)，通过截断-重采样和视觉扰动获得一致性奖励来惩罚虚假推理，搭载 RLOO 后在六个基准上平均提升 7.7 个百分点。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "强化学习"
  - "自一致性采样"
  - "推理忠实性"
  - "多模态推理"
  - "结果奖励"
---

# Enhancing Outcome Reward-Based RL Training of MLLMs with Self-Consistency Sampling

**会议**: NeurIPS 2025  
**arXiv**: [2511.10648](https://arxiv.org/abs/2511.10648)  
**代码**: [GitHub](https://github.com/GenuineWWD/SCS)  
**领域**: 多模态VLM  
**关键词**: 强化学习, 自一致性采样, 推理忠实性, 多模态推理, 结果奖励

## 一句话总结

针对多模态多选题中"结果奖励 RL 训练导致不忠实推理轨迹"的问题，提出 Self-Consistency Sampling (SCS)，通过截断-重采样和视觉扰动获得一致性奖励来惩罚虚假推理，搭载 RLOO 后在六个基准上平均提升 7.7 个百分点。

## 研究背景与动机

结果奖励 RL（如 GRPO、RLOO、REINFORCE++）是提升 MLLM 推理能力的主流方法。然而在多选题（multimodal reasoning benchmarks 的主要格式）中存在一个被忽视的关键问题：**不忠实轨迹蒙混奖励** — 模型在错误的推理链之后猜中正确选项，却获得与真正推理相同的满分奖励。

论文通过探索性实验揭示问题的严重性：

**多选格式增益不足**：在 Geometry3K 上，多选训练仅提升 5.6%，比开放式问答的 12.0% 低 6.4 个百分点

**截断-续写发散**：对已生成的推理轨迹在不同位置截断后续写，同一前缀频繁产生不同的最终选项，说明不忠实推理轨迹普遍存在

**定性分析**：模型经常生成错误推理但碰巧选对答案

过程奖励模型（PRM）虽然可以缓解此问题，但计算代价高昂。SCS 的目标是在不引入额外奖励模型的前提下，通过自一致性检查来识别和降权不可靠的推理轨迹。

## 方法详解

### 整体框架

SCS 基于三个关键假设将推理过程建模为树结构：

1. **正确轨迹唯一性**：推理树中恰好存在一条正确推理轨迹
2. **叶节点/选项对齐**：每条轨迹最终都指向一个答案选项
3. **正确/错误轨迹与选项的关系**：正确轨迹必然导向正确选项；错误轨迹可能猜中正确选项、也可能选错

基于这些假设，当只用准确率奖励 $r = r_{\text{acc}}$ 时，不忠实推理仍有非零概率获得满分：

$$P(\tau^- \mid y^+) = \frac{P(y^+, \tau^- \mid x)}{P(y^+, \tau^- \mid x) + P(y^+, \tau^+ \mid x)}$$

### 关键设计

#### 一致性奖励

SCS 引入一致性奖励 $r_{\text{con}}$ 惩罚不一致的推理模式。给定推理轨迹 $\tau$，模型重复采样 $N$ 次得到答案集合 $\mathcal{A}$，一致性奖励定义为：

$$r_{\text{con}} = \frac{1}{N}(N - |\mathcal{A}|) \cdot c$$

其中 $c$ 是缩放系数。直觉：如果推理正确，重复采样应始终指向同一答案（$|\mathcal{A}|=1$，奖励最大）；如果推理错误，答案会发散（$|\mathcal{A}|$ 增大，奖励降低）。

正确轨迹的一致性奖励：$r_{\text{con}}^+ = \frac{1}{N}(N-1) \cdot c$（最大值）。
错误轨迹的一致性奖励：$r_{\text{con}}^- = \frac{1}{N}(N-|\mathcal{A}^-|) \cdot c$（因为 $\mathbb{E}(|\mathcal{A}^-|) > 1$，所以 $r_{\text{con}}^+ > r_{\text{con}}^-$）。

总奖励：$r = r_{\text{for}} + r_{\text{acc}} + r_{\text{con}}$（格式奖励 + 准确率奖励 + 一致性奖励）。

#### 截断-重采样（Truncation-Resampling, TR）

对初始推理轨迹 $\tau$ 按截断比例 $k$ 截断到不完整轨迹 $\tau^<$，以此为前缀进行 $m$ 次重采样，每次续写生成一个新答案 $a_t$。收集所有答案组成 $\mathcal{A}$。

核心思想：如果推理过程是忠实的，从同一前缀续写应该得到一致的答案；如果推理过程是虚假的，续写结果会发散。

#### 视觉扰动（Visual Perturbation, VP）

在每次重采样时对输入图像添加轻微高斯噪声：

$$\tilde{\mathbf{x}}_i = \mathbf{x} + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_i^2), \quad \sigma_i \sim \mathcal{U}(\sigma_{\min}, \sigma_{\max})$$

每次采样的扰动强度 $\sigma_i$ 从均匀分布中独立采样，让模型面对多样化的视觉变体。这迫使策略基于扰动的视觉证据进行推理，正确推理对小扰动应保持鲁棒。

### 训练策略

SCS 与多种 RL 算法兼容：RLOO、GRPO、REINFORCE++、REINFORCE++-baseline。训练配置：

- 基础模型：Qwen2.5-VL-7B-Instruct
- 训练数据：M³CoT (7.8k) + Geometry3K (2.1k) + ScienceQA (6.2k, 过滤后) — 仅保留包含图像的多选题
- 批大小：128，每个 prompt 采样 16 条轨迹
- 学习率：1e-6，温度：1.0
- 截断比例 $k$=0.8（RLOO/REINFORCE++）或 0.4（GRPO），重采样数 $m$=4（RLOO）或 8（GRPO）
- 8 张 A800 GPU，约 24 小时训练

## 实验关键数据

### 主实验

**表1：SCS 搭配不同 RL 算法的效果（Qwen2.5-VL-7B-Instruct）**

| 方法 | SCS | Overall | M3CoT | MMMU | SciQA | WeMath | MathVerse | MathVision |
|------|-----|---------|-------|------|-------|--------|-----------|------------|
| Baseline (pretrained) | - | 54.9 | 65.5 | 45.7 | 73.7 | 62.5 | 57.7 | 24.1 |
| SFT | - | 58.6 | 78.7 | 52.6 | 51.0 | 90.7 | 49.4 | 29.3 |
| RLOO | ✕ | 57.8 | 67.6 | 51.5 | 53.9 | 86.4 | 56.8 | 30.4 |
| RLOO | ✓ | **65.5 (+7.7)** | 75.7 | 59.1 | 68.8 | 88.1 | 67.1 | 34.0 |
| GRPO | ✕ | 63.6 | 72.6 | 57.2 | 66.6 | 88.3 | 64.2 | 32.8 |
| GRPO | ✓ | **64.5 (+0.9)** | 73.9 | 58.0 | 66.4 | 88.7 | 67.0 | 33.1 |
| REINFORCE++ | ✕ | 60.9 | 66.8 | 54.9 | 64.8 | 84.3 | 60.9 | 33.4 |
| REINFORCE++ | ✓ | **62.9 (+2.0)** | 65.7 | 54.6 | 76.1 | 85.4 | 61.6 | 34.0 |

**表2：跨模型泛化（RLOO + SCS）**

| 模型 | 无 SCS | 有 SCS | 提升 |
|------|--------|--------|------|
| Qwen2.5-VL-7B-Instruct | 57.8 | 65.5 | +7.7 |
| Qwen2.5-VL-3B-Instruct | 54.7 | 57.9 | +3.2 |
| InternVL3-8B | 61.7 | 63.3 | +1.6 |

### 消融实验

**组件有效性（基于 RLOO）**：

| TR | VP | Overall | 提升 |
|----|----|---------|----|
| ✕ | ✕ | 57.8 | - |
| ✓ | ✕ | 63.0 | +5.2 |
| ✕ | ✓ | 62.8 | +5.0 |
| ✓ | ✓ | 65.5 | +7.7 |

两个组件各自贡献约 5 个点，组合后恢复全部 7.7 点增益，说明两者互补。

**超参数敏感性**：

- 截断比例 $k$：从 0.1 到 0.8 上升，0.8 峰值后下降。过小比例一致性信号不够，过大比例探索空间不足
- 重采样数 $m$：从 2 到 4 上升，$m=4$ 峰值后下降。过多采样引入随机性且增加计算开销
- 两个超参数变化范围内性能波动在 4 个点以内，方法比较鲁棒

**推理可靠性量化分析**：
每个基准随机抽 100 个正确回答的样本，人工检查推理忠实性。SCS 训练后不忠实推理比例降低约 15%（人工评判：25.0 → 21.2；o3-mini 评判：22.0 → 19.0）。

### 关键发现

1. RLOO 在 vanilla RL 设置下表现甚至不如 SFT（57.8 vs 58.6），但加入 SCS 后跳升至 65.5 — SCS 对 RLOO 的增益最大（+7.7）
2. Vanilla RL 相比 SFT 的优势微弱（约 2-3 个点），说明标准结果奖励在多选题上效率低下
3. SCS 的一致性奖励实质上起到了"穷人版过程奖励"的作用，无需额外奖励模型
4. 两个组件贡献近乎对称（TR +5.2 vs VP +5.0），但组合后有协同效应

## 亮点与洞察

- 精准定位了一个被广泛忽视的问题：多选题格式下的不忠实推理轨迹导致结果奖励 RL 退化
- SCS 设计优雅且轻量：不需要额外奖励模型、不修改 RL 算法结构，只增加一个一致性奖励项
- 理论推导清晰：从树结构建模到一致性奖励的数学推导逻辑严谨
- 截断-重采样的思路妙在利用了推理轨迹的"因果连贯性" — 正确推理的前缀应能一致地导向正确结论

## 局限与展望

- 仅验证了多选题格式，开放式问答场景的效果未知
- 训练数据规模较小（约 16k 样本），更大规模数据下的表现待验证
- 一致性奖励的数学假设（正确轨迹唯一性）在开放式推理中可能不成立
- 视觉扰动使用简单的高斯噪声，更复杂的数据增强（几何变换、遮挡）可能效果更好
- SCS 对 GRPO 的增益较小（+0.9），可能因为 GRPO 的组相对优势估计已部分抑制了不忠实轨迹

## 相关工作与启发

- Self-Consistency（Wang et al., 2022）通过多次采样取众数来提升推理一致性，SCS 将一致性思想融入 RL 训练奖励
- 过程奖励模型（PRM）提供逐步反馈但计算代价高，SCS 是低成本替代方案
- RLOO 获益最大可能是因为其 leave-one-out 基线方差本来就低，SCS 的一致性信号进一步稳定了训练
- 截断-重采样机制可推广到其他需要验证推理忠实性的场景

## 评分

- 新颖性：⭐⭐⭐⭐ — 问题定位精准，一致性奖励设计原创性强
- 技术深度：⭐⭐⭐⭐ — 理论推导完整，两个组件设计合理
- 实验充分度：⭐⭐⭐⭐ — 4 个 RL 算法、3 个模型、6 个基准、多组消融
- 实用价值：⭐⭐⭐⭐ — 即插即用，计算开销可忽略，适合多选题推理训练

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism](../../ACL2025/multimodal_vlm/vrest_tree_search_vlm_reasoning.md)
- [\[NeurIPS 2025\] What Can RL Bring to VLA Generalization? An Empirical Study](what_can_rl_bring_to_vla_generalization_an_empirical_study.md)
- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)
- [\[AAAI 2026\] Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View](../../AAAI2026/multimodal_vlm/revisiting_the_data_sampling_in_multimodal_post-training_from_a_difficulty-disti.md)
- [\[ICML 2025\] ERL-VLM: Enhancing Rating-Based RL to Leverage Feedback from Large VLMs](../../ICML2025/multimodal_vlm/enhancing_rating-based_reinforcement_learning_to_effectively_leverage_feedback_f.md)

</div>

<!-- RELATED:END -->
