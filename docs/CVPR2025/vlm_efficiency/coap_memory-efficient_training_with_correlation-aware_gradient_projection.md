---
title: "COAP: Memory-Efficient Training with Correlation-Aware Gradient Projection"
authors: "Yijiang Liu, Huixue Jia, Lunhao Duan, Jiwen Lu"
affiliations: "ByteDance, Rutgers University"
venue: "CVPR 2025"
date: 2024-11-30
tags: ["memory-efficient training", "gradient projection", "LLM", "optimizer", "low-rank"]
arxiv: "2412.00071"
code: "https://byteaigc.github.io/coap/"
---

# COAP: Memory-Efficient Training with Correlation-Aware Gradient Projection

## 研究背景与动机

大型语言模型（LLM）的训练面临严峻的内存瓶颈。以 LLaMA-7B 为例，模型参数本身占用约14GB（FP16），但 Adam 优化器需要维护一阶矩（momentum）和二阶矩（variance）两组与参数同等大小的状态变量，导致优化器状态额外占用约28GB显存。这使得在单张消费级GPU（如RTX 4090, 24GB）上训练中大模型几乎不可能。

现有的内存高效训练方法主要有两类：

**低秩优化器**（如 GaLore、Flora）：将梯度投影到低秩子空间，减少优化器状态的维度。但投影矩阵的更新依赖 SVD 分解，计算开销极大

**量化优化器**（如 Q-Adam）：对优化器状态进行量化压缩。但量化误差会累积，影响训练质量

低秩投影方法的核心问题在于**投影矩阵 $P_t$ 的更新策略**。GaLore 每隔 $T$ 步对当前梯度做完整 SVD 来更新 $P_t$，这一操作对于大矩阵极其耗时（7B模型约540秒一次完整SVD），严重拖慢训练速度。

本文的核心洞察：**相邻更新周期的投影矩阵之间存在高度相关性**。利用这一性质，可以用极低代价的增量更新替代昂贵的全量SVD。

## 方法详解

### 问题形式化

标准低秩梯度投影将 $m 	imes n$ 的梯度矩阵 $G_t$ 投影到秩-$r$ 子空间：

$$	ilde{G}_t = P_t P_t^T G_t$$

其中 $P_t \in \mathbb{R}^{m 	imes r}$ 是投影矩阵。优化器状态（momentum, variance）维护在低秩空间 $\mathbb{R}^{r 	imes n}$ 中，内存从 $O(mn)$ 降至 $O(rn + mr)$。

### 相关性感知投影更新

COAP 的核心创新是将投影矩阵的更新分为两个阶段：

#### 阶段1：SGD 增量更新（每步执行）

利用投影间的相关性，通过简单的 SGD 步骤增量更新 $P_t$：

$$P_{t+1} = P_t - \eta_P 
abla_{P} \mathcal{L}$$

这一更新的计算复杂度仅为 $O(mr)$，远小于 SVD 的 $O(m^2n)$。

#### 阶段2：偶发性低成本 SVD（每 $T$ 步执行）

每隔 $T$ 步，执行一次**热启动SVD**：以当前 $P_t$ 为初始化，对梯度做部分SVD分解。由于初始化已经接近最优解，收敛只需极少迭代。

| 操作 | GaLore SVD | COAP 热启动 SVD | 加速比 |
|------|-----------|----------------|-------|
| 单次耗时 (LLaMA-7B) | ~540s | ~23s | **~20×** |
| 更新频率 | 每200步 | 每200步 | - |
| 均摊每步开销 | 2.7s | 0.12s | **~23×** |

### Inter-Projection 相关性分析

本文通过实验验证了相邻投影矩阵间的高相关性：

$$	ext{sim}(P_t, P_{t+T}) = rac{\|P_t^T P_{t+T}\|_F}{\|P_t\|_F \|P_{t+T}\|_F} > 0.95$$

这一观察为SGD增量更新提供了理论基础：投影空间变化缓慢，小步增量更新即可跟踪最优子空间。

### 内存分析

| 方法 | 优化器内存 (LLaMA-1B) | 相对标准Adam |
|------|---------------------|-------------|
| Adam (FP16) | 4.0 GB | 100% |
| Adam (BF16) | 4.0 GB | 100% |
| GaLore (r=256) | 1.8 GB | 45% |
| Flora (r=256) | 1.6 GB | 40% |
| **COAP (r=256)** | **1.56 GB** | **39%** |

COAP 实现了 **-61%** 的优化器内存节省。

## 实验结果

### LLaMA 预训练

| 方法 | LLaMA-1B PPL↓ | LLaMA-7B PPL↓ | 训练速度 (tokens/s) |
|------|--------------|--------------|-------------------|
| Adam | 14.89 | 12.31 | 1× |
| GaLore | 16.12 | 13.05 | 0.72× |
| Flora | 15.98 | 12.87 | 0.81× |
| **COAP** | **15.56** | **12.58** | **0.93×** |

### LLaVA-7B 微调

| 方法 | 训练时间 | 准确率 | GPU内存 |
|------|---------|--------|---------|
| LoRA | 12.3h | 88.1% | 18GB |
| Full fine-tuning (Adam) | 47.1h | 82.4% | 62GB |
| GaLore | 15.2h | 87.3% | 24GB |
| **COAP** | **7.6h** | **92.3%** | **22GB** |

COAP 在 LLaVA-7B 微调上实现了**6.2×加速**（7.6h vs 47.1h），同时准确率从82.4%提升至92.3%。

### 下游任务评估

| 任务 | Adam | GaLore | COAP |
|------|------|--------|------|
| MMLU (5-shot) | 46.2 | 43.8 | 45.7 |
| HellaSwag | 72.1 | 69.4 | 71.5 |
| ARC-Challenge | 41.3 | 38.9 | 40.8 |
| WinoGrande | 67.4 | 65.1 | 66.9 |

## 总结与展望

COAP 通过观察投影矩阵间的高相关性，设计了一种高效的两阶段投影更新策略：SGD增量更新 + 偶发性热启动SVD。这一设计将SVD的计算开销降低约20倍，同时保持了与全量SVD相当的投影质量。在LLaMA-1B预训练中实现PPL 15.56、节省61%优化器内存，在LLaVA-7B微调中实现6.2×加速和9.9%的准确率提升。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Efficient Motion-Aware Video MLLM](efficient_motion-aware_video_mllm.md)
- [\[ICLR 2026\] Multimodal Classification via Total Correlation Maximization](../../ICLR2026/multimodal_vlm/multimodal_classification_via_total_correlation_maximization.md)
- [\[CVPR 2026\] ZOO-Prune: Training-Free Token Pruning via Zeroth-Order Gradient Estimation in Vision-Language Models](../../CVPR2026/multimodal_vlm/zoo-prune_training-free_token_pruning_via_zeroth-order_gradient_estimation_in_vi.md)
- [\[ICCV 2025\] G2D: Boosting Multimodal Learning with Gradient-Guided Distillation](../../ICCV2025/multimodal_vlm/g2d_boosting_multimodal_learning_with_gradient-guided_distillation.md)
- [\[ACL 2025\] Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference](../../ACL2025/multimodal_vlm/activating_distributed_visual_region_within_llms_for_efficient_and_effective_vis.md)

</div>

<!-- RELATED:END -->
