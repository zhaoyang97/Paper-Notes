---
title: >-
  [论文解读] CoreMatching: A Co-adaptive Sparse Inference Framework with Token and Neuron Pruning for Comprehensive Acceleration of Vision-Language Models
description: >-
  [ICML 2025][多模态VLM][推理加速] 首次揭示 VLM 中 token 稀疏与神经元稀疏之间的内在关联——核心神经元与核心 token 相互决定、相互强化，并据此提出 CoreMatching 协同稀疏推理框架，在 pre-filling 和 decoding 两阶段同时实现加速，达到 5× FLOPs 降低和 10× 整体加速。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "推理加速"
  - "Token剪枝"
  - "神经元稀疏"
  - "视觉语言模型"
  - "协同稀疏"
---

# CoreMatching: A Co-adaptive Sparse Inference Framework with Token and Neuron Pruning for Comprehensive Acceleration of Vision-Language Models

**会议**: ICML 2025  
**arXiv**: [2505.19235](https://arxiv.org/abs/2505.19235)  
**代码**: [GitHub](https://github.com/wangqinsi1/2025-ICML-CoreMatching/tree/main)  
**领域**: 多模态VLM  
**关键词**: 推理加速, Token剪枝, 神经元稀疏, 视觉语言模型, 协同稀疏

## 一句话总结

首次揭示 VLM 中 token 稀疏与神经元稀疏之间的内在关联——核心神经元与核心 token 相互决定、相互强化，并据此提出 CoreMatching 协同稀疏推理框架，在 pre-filling 和 decoding 两阶段同时实现加速，达到 5× FLOPs 降低和 10× 整体加速。

## 研究背景与动机

VLM（如 LLaVA、BLIP、LLaMA）在图像问答等任务上表现出色，但因需要处理大量 image token，推理的时间和内存开销远超纯文本 LLM，限制了实际部署。

现有两条加速路线各有局限：

**Token 稀疏**：利用 image token 的高冗余度，只保留少量关键 token。代表方法如 PruMerge（保留 20% token）、FastV（从第 2 层开始丢弃 >50% token）。但主要加速 **pre-filling 阶段**，decoding 阶段仅通过缩减 KV cache 获得有限收益。

**神经元稀疏**：利用 FFN 层中大量神经元处于未激活状态，跳过不活跃神经元的计算。代表方法如 DejaVu、PowerInfer、CoreInfer。但主要加速 **decoding 阶段**，在 pre-filling 阶段因 token 数量大、稀疏度低而增速有限。

**核心问题**：这两种稀疏空间之间是否存在深层关联？能否将它们统一起来实现全面加速？

本文是第一个系统研究该问题的工作。作者发现：被核心神经元激活模式最匹配的 token 恰恰是对输出最重要的 token，这一 "匹配机制" 将两种稀疏范式有机联系在一起。

## 方法详解

### 整体框架

CoreMatching 在一次 pre-filling 前向传播中同时计算 **Core Neurons** 和 **Core Tokens**，随后在 pre-filling 阶段只处理 Core Tokens（token 维度稀疏），在 decoding 阶段只使用 Core Neurons 进行 FFN 计算且 KV cache 中仅保留 Core Tokens（神经元维度稀疏）。整个流程分三步：

1. **计算 Core Neurons**：在 FFN 块中统计所有 token 的激活分布，选出最频繁激活的一组神经元。
2. **匹配 Core Tokens**：逐 token 计算其激活神经元与 Core Neurons 的交集大小，选出交集最大的 token 子集作为 Core Tokens。
3. **稀疏推理**：后续层仅传递 Core Tokens；decoding 阶段仅使用 Core Neurons 做 FFN 运算。

### 关键设计

#### 1. Core Neurons 定义与验证

对单个 token $x$，其 token-wise core neurons 定义为激活值最高的 top-$\rho$ 神经元：

$$\mathcal{C}_\rho(x) = \{n \mid a_n \geq \text{Percentile}(A^+, \rho)\}$$

其中 $A^+ = \{a_n \mid a_n > 0\}$ 为正激活集合。

对整个句子 $\mathbf{s} = [x_1, \ldots, x_M]$，sentence-wise core neurons 为所有 token 的 core neurons 中出现频率最高的 top-$\beta$ 神经元：

$$\mathcal{C}_\rho^\beta(\mathbf{s}) = \{n \mid f_\rho(n; \mathbf{s}) \geq \text{Percentile}(f_\rho(\mathbf{s}), \beta)\}$$

实验验证（LLaVA-1.5-7B, TextVQA）：仅保留 60% core neurons 即可达到 55.8% 准确率（完整模型 57.8%），说明极少量核心神经元即可维持性能。

此外可视化证明 core neurons 具有**可预测性**：当输入语义充分时，core neurons 几乎不变。

#### 2. Core Tokens 定义——从神经元到 token 的匹配

核心洞察：token $x$ 的激活神经元集合 $\Gamma(x)$ 与 sentence-wise core neurons $\mathcal{C}_\rho^\beta(\mathbf{s})$ 的交集大小反映了该 token 的重要性。交集越大的 token，对模型输出贡献越大。

Core Tokens 定义为交集大小超过阈值 $\tau$ 的 token 集合：

$$\mathcal{T}_{\text{core}} = \{x \mid |\Gamma(x) \cap \mathcal{C}_\rho^\beta(\mathbf{s})| \geq \tau\}$$

阈值 $\tau$ 通过**最大几何距离法**（Maximum Geometric Distance）自适应确定：对所有 token 的交集大小排序后，找到排序曲线上离对角线最远的点作为分界。这避免了需要手动设定保留比例，使得不同样本可以保留不同数量的 token。

#### 3. Projection-guided Criterion——理论解释

为什么 Core Tokens 比传统基于 attention score 的方法更优？作者提出 **Projection-guided Criterion** 进行理论分析。

传统方法仅用 attention score（即 softmax 后的注意力权重）衡量 token 重要性。而作者指出，token 对输出的真正贡献不仅取决于 attention 权重大小，还取决于 value 向量的**角度信息**（方向对齐度）。

具体来说，token $x_i$ 对最终输出隐状态 $h$ 的贡献可分解为：

$$\text{Contribution}(x_i) \propto \alpha_i \cdot \|v_i\| \cdot \cos\theta_i$$

其中 $\alpha_i$ 是 attention 权重，$v_i$ 是 value 向量，$\theta_i$ 是 $v_i$ 与输出方向的夹角。

Core Tokens 天然倾向于选出那些 $\cos\theta_i$ 大（方向一致）且激活模式与核心神经元匹配度高的 token，因此在理论上优于仅依赖 $\alpha_i$ 的方法。

#### 4. 两阶段加速机制

| 阶段 | 稀疏维度 | 加速原理 |
|------|----------|----------|
| Pre-filling | Token 稀疏 | 只处理 Core Tokens，减少 attention 和 FFN 计算量 |
| Decoding | 神经元稀疏 + KV cache 稀疏 | FFN 仅用 Core Neurons 计算；KV cache 仅存 Core Tokens |

两种稀疏**协同增强**：Core Neurons 指导 Core Tokens 的选取，Core Tokens 又决定了后续 Core Neurons 的计算范围，形成正反馈闭环。

### 损失函数 / 训练策略

CoreMatching 是 **training-free** 的推理加速框架，无需任何额外训练或微调。核心神经元和核心 token 均通过前向传播中的统计量在线计算。这是其一大优势——可以直接应用于已有的 VLM 模型。

## 实验关键数据

### 主实验

在 10 个图像理解 benchmark 上评估，基础模型为 LLaVA-1.5-7B：

| 数据集 | 指标 | CoreMatching | FastV | PruMerge | 完整模型 |
|--------|------|-------------|-------|----------|----------|
| TextVQA | Acc | 55.8% | 54.2% | 53.5% | 57.8% |
| VQAv2 | Acc | 保持竞争力 | 下降明显 | 中等下降 | 基准 |
| GQA | Acc | 优于基线 | 基线 | 基线 | 基准 |
| ScienceQA | Acc | 优于基线 | 基线 | 基线 | 基准 |

硬件加速效果（NVIDIA Titan Xp）：

| 指标 | CoreMatching | 说明 |
|------|-------------|------|
| FLOPs 降低 | 5× | 综合 token + neuron 稀疏 |
| 整体加速 | 10× | 端到端推理时间 |
| Pre-filling 加速 | 2.1× | Token 稀疏主导 |
| Decoding 加速 | 9.2× | Neuron 稀疏主导 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| 仅 Token 稀疏 | pre-filling 加速，decoding 有限 | 传统方法的上界 |
| 仅 Neuron 稀疏 | decoding 加速，pre-filling 有限 | CoreInfer 路线 |
| CoreMatching（协同） | 两阶段同时加速 | 1+1 > 2 的效果 |
| Attention-based Token 选择 | 低于 CoreMatching | 缺少角度信息 |
| Core Neuron Matching 选择 | 最优 | 激活模式对齐 |
| 固定阈值 vs 自适应阈值 | 自适应更稳定 | 最大几何距离法泛化好 |

Core Neurons 保留比例验证（TextVQA / LLaVA-1.5-7B）：

| 保留比例 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|----------|-----|-----|-----|-----|-----|
| Accuracy | 45.1% | 53.2% | 55.8% | 56.3% | 57.8% |

### 关键发现

1. **5× FLOPs 降低 + 10× 整体加速**（NVIDIA Titan Xp 上）
2. **Pre-filling 2.1× 加速 + Decoding 9.2× 加速**，首次实现两阶段均获显著加速
3. 在 3 种硬件设备上均超越 SOTA 基线
4. Core neurons 在输入语义充分后几乎保持不变（可预测性），保留 40% 即可达到 53.2% 准确率
5. Core tokens 可视化精确覆盖了图像中与文本问题最相关的语义区域

## 亮点与洞察

- **填补了重要空白**：首次系统研究 token sparsity 和 neuron sparsity 的关联，发现二者不是独立的，而是可以相互指导
- **优雅的统一框架**：仅在 pre-filling 阶段增加一步匹配计算，就同时获得两个维度的稀疏，设计简洁
- **Training-free**：无需训练预测器或额外微调，直接利用前向传播统计量，部署门槛极低
- **理论贡献**：Projection-guided Criterion 从理论上解释了为何纯 attention score 不够好——需同时考虑 value 方向信息
- **自适应阈值**：最大几何距离法避免手动调比例，不同样本自动确定保留 token 数量

## 局限与展望

1. **模型规模**：主要在 LLaVA-1.5-7B 上验证，对更大模型（13B/70B）和更新架构的泛化性有待验证
2. **视频/多图场景**：当前仅针对单图理解任务，视频 VLM 场景是否适用需要探索
3. **与量化等压缩方法结合**：CoreMatching + 量化的联合方案可能进一步降低部署成本
4. **分层自适应策略**：不同层的稀疏特性可能不同，分层策略值得研究
5. **长文本多轮对话**：text token 很长时 core neurons 的稳定性需要更多验证

## 相关工作与启发

- **CoreInfer** (Wang et al., 2024)：核心 neuron 概念的来源，仅关注 LLM 神经元稀疏。本文将其扩展到 VLM 并与 token 稀疏联合
- **FastV** (Chen et al., 2025)：基于 attention score 的 token 剪枝 baseline，本文证明了 attention score 不够准确
- **PruMerge** (Shang et al., 2024)：用 image-text attention 平均分做 token 选择和合并
- **DejaVu / PowerInfer**：MLP 预测器驱动的神经元稀疏方法
- **启发**：两种看似独立的效率优化范式存在深层联系，可推广到 attention head 稀疏 + token 稀疏、MoE 专家选择 + token 路由等

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|-------------|------|
| 创新性 | 8 | 首次揭示两种稀疏范式的内在关联，视角新颖 |
| 理论深度 | 7 | Projection-guided Criterion 提供了有意义的理论解释 |
| 实验充分性 | 7 | 10 个 benchmark + 3 种硬件，但模型规模较单一 |
| 实用价值 | 8 | Training-free，10× 加速，部署门槛低 |
| 写作质量 | 7 | 结构清晰，图示直观 |
| **综合** | **7.5** | 推理加速领域的扎实工作，核心洞察有启发性 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](../../ICCV2025/multimodal_vlm/feather_the_throttle_revisiting_visual_token_pruning_for_vision-language_model_a.md)
- [\[ICML 2025\] SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference](sparsevlm_visual_token_sparsification_for_efficient_vision-language_model_infere.md)
- [\[ACL 2025\] Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?](../../ACL2025/multimodal_vlm/token_pruning_in_multimodal_large_language_models_are_we_solving_the_right_probl.md)
- [\[ACL 2025\] EffiVLM-Bench: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models](../../ACL2025/multimodal_vlm/effivlm_bench_vlm_acceleration.md)
- [\[ACL 2025\] FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation](../../ACL2025/multimodal_vlm/flagevalmm_a_flexible_framework_for_comprehensive_multimodal_model_evaluation.md)

</div>

<!-- RELATED:END -->
