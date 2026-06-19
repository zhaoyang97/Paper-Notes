---
title: >-
  [论文解读] DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration
description: >-
  [NeurIPS 2025][模型压缩][剪枝] 提出 DenoiseRotator，在剪枝前通过可学习正交变换最小化参数重要性分数的信息熵，将重要性集中到少数参数上，使 LLaMA3-70B 在 2:4 半结构化稀疏下困惑度退化缩小 58%（8.1→3.4），可即插即用组合 Magnitude/Wanda/SparseGPT。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "剪枝"
  - "orthogonal transformation"
  - "entropy minimization"
  - "importance concentration"
  - "semi-structured sparsity"
---

# DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration

**会议**: NeurIPS 2025  
**arXiv**: [2505.23049](https://arxiv.org/abs/2505.23049)  
**代码**: [Axel-gu/DenoiseRotator](https://github.com/Axel-gu/DenoiseRotator)  
**领域**: 图像复原  
**关键词**: LLM pruning, orthogonal transformation, entropy minimization, importance concentration, semi-structured sparsity

## 一句话总结

提出 DenoiseRotator，在剪枝前通过可学习正交变换最小化参数重要性分数的信息熵，将重要性集中到少数参数上，使 LLaMA3-70B 在 2:4 半结构化稀疏下困惑度退化缩小 58%（8.1→3.4），可即插即用组合 Magnitude/Wanda/SparseGPT。

## 研究背景与动机

**领域现状**：剪枝是 LLM 压缩的主流手段，SparseGPT 和 Wanda 等方法利用 Taylor 展开近似估计每个参数的重要性，移除最低分的权重。但在半结构化稀疏（2:4 模式）约束下，性能退化仍然严重。

**现有痛点**：已有方法只关注"选哪些权重剪掉"，在预训练模型的固定参数空间内操作，无法改变参数重要性的底层分布。当重要性分散在整个权重矩阵中时，无论怎么选择剪枝集合，总剪掉的重要性都不可避免地较大。

**核心矛盾**：如何在不改变模型输出的前提下重新分配参数重要性？若能将重要性集中到少数参数上，剪掉剩余大量低重要性参数的代价就会很小。

**切入视角**：Transformer 的计算不变性（computational invariance）允许在权重矩阵上施加正交变换而不改变模型输出。正交变换保范数——层的总重要性不变，但重要性在参数间的分布可以被重新塑造。

**核心 idea**：在剪枝前学习正交矩阵 $R$，最小化归一化重要性分数的信息熵 $\mathcal{H}(P) = -\sum p_{ij} \log p_{ij}$，使重要性分布从均匀变为尖峰；训练完毕后将 $R$ 合并回权重矩阵，再执行标准剪枝流程。

## 方法详解

### 整体流程

三步走：(1) 将 RMSNorm 权重融合到相邻线性层，在每个 Transformer 层插入层级正交矩阵 $R_1$ 和注意力级正交矩阵 $R_2$；(2) 冻结原始权重，仅通过熵最小化损失训练 $R_1, R_2$（2000 步，lr=0.01）；(3) 将 $R$ 合并回权重矩阵后执行任意剪枝方法（Magnitude/Wanda/SparseGPT）。

### 关键设计

1. **熵引导的重要性集中**：

    - 将每个线性层的重要性分数 $S_{ij}$ 归一化为离散概率分布 $p_{ij} = S_{ij} / \sum S_{ij}$
    - 最小化该分布的信息熵作为代理目标：熵是置换不变的（与参数排列顺序无关），且作为概率单纯形上的凹函数，最小化时自然促进分布向少数元素集中
    - 归一化组的划分依据正交矩阵的位置：右乘则按行归一化，左乘则按列归一化，两侧都乘则行列熵求和

2. **正交矩阵的插入位置**：

    - **层级旋转 $R_1$**：一对 $(d_{hidden}, d_{hidden})$ 的正交矩阵分别插在 Transformer 层的输入端和输出端（残差连接和 RMSNorm 之间），影响 Q/K/Up/Gate/Down 投影
    - **注意力级旋转 $R_2$**：在自注意力内部，对 Value 和 Output 投影施加额外旋转，进一步集中注意力内部的重要性
    - 以 Output 投影为例：变换后权重 $W' = R_1^\top W R_2$，输入 $X' = R_2^\top X$，乘积 $W'X' = R_1^\top WX$ 保持不变

3. **总重要性不变性**：

    - 正交变换 $\|Rx\| = \|x\|$ 保证 $\sum \mathcal{T}_S(S) = \sum S$，即层的总重要性在变换前后严格不变
    - 重要性只被重新分配而非被创造或消灭，这保证了算法的稳定性

4. **QR 分解参数化**：

    - 直接用梯度下降优化正交矩阵会破坏正交性约束
    - 引入无约束矩阵 $A$，前向传播时做 QR 分解 $A = QR$，只用正交分量 $Q$；反向传播对 $A$ 求梯度
    - PyTorch 的 `torch.qr` 原生支持自动求导，无需 Stiefel 流形优化器

5. **即插即用兼容性**：

    - DenoiseRotator 与剪枝步骤完全解耦，作为剪枝前预处理
    - 训练好的 $R$ 合并进权重后，下游可接任何剪枝方法
    - Hessian 矩阵 $H = XX^\top$ 在训练 $R$ 和后续剪枝中复用，不需额外校准

### 训练细节

- 优化器：Adam，学习率 0.01，训练 2000 步
- 精度：bfloat16（QR 分解除外）
- 初始化：$R$ 初始化为单位矩阵，确保训练起点不改变模型行为
- 损失函数：$\text{Loss}(R_{1,i}, R_{2,i}) = \sum_{\ell \in \mathcal{L}_i} \sum_{\mathcal{G} \in \ell} \mathcal{H}(P_\mathcal{G})$，即层内所有线性层的所有归一化组的熵之和
- LLaMA3-70B + SparseGPT 训练约 28 小时，单卡 A100，约 30 GB 显存

## 实验关键数据

### 困惑度结果（WikiText-2）

| 模型 | 稀疏模式 | 剪枝方法 | 基线 PPL | +DenoiseRotator PPL | Dense PPL |
|------|---------|---------|---------|-------------------|----------|
| LLaMA3-70B | 2:4 | SparseGPT | 10.97 | **6.25** | 2.86 |
| LLaMA3-70B | 50% | SparseGPT | 5.99 | **4.61** | 2.86 |
| LLaMA3-8B | 2:4 | SparseGPT | 17.67 | **10.01** | 6.14 |
| LLaMA3-8B | 50% | SparseGPT | 9.57 | **7.60** | 6.14 |
| Qwen2.5-72B | 2:4 | SparseGPT | 7.19 | **5.85** | 3.88 |
| Qwen2.5-72B | 50% | SparseGPT | 4.94 | **4.78** | 3.88 |
| Mistral-7B | 2:4 | Wanda | 10.18 | **7.80** | 5.95 |
| Mistral-7B | 50% | Wanda | 6.92 | **6.52** | 5.95 |

### Zero-shot 准确率（5 任务平均）

| 模型 | 稀疏模式 | 剪枝方法 | 基线 Acc | +DenoiseRotator Acc | Dense Acc |
|------|---------|---------|---------|-------------------|----------|
| LLaMA3-70B | 2:4 | SparseGPT | 69.16% | **76.37%** | 80.05% |
| LLaMA3-70B | 50% | SparseGPT | 76.66% | **78.54%** | 80.05% |
| Qwen2.5-72B | 2:4 | SparseGPT | 75.43% | **77.16%** | 78.69% |
| Qwen2.5-72B | 50% | SparseGPT | 78.25% | **78.42%** | 78.69% |

### 熵降低与性能的关系（LLaMA3-8B, SparseGPT 50%）

| 训练步数 | 平均熵 | 困惑度 | Zero-shot Acc |
|---------|-------|--------|-------------|
| 0（基线） | 457280 | 9.567 | 66.88% |
| 100 | 396992 (-13%) | 7.701 | 70.54% |
| 400 | 387904 | 7.619 | 70.12% |
| 2000 | 384128 | **7.597** | 69.58% |

### 推理开销（LLaMA3-8B, A100）

| 配置 | 单层时延 | 加速比 |
|------|---------|--------|
| Dense | 5.80 ms | 1.00× |
| 2:4 Sparse | 4.37 ms | 1.33× |
| 2:4 Sparse + 正交矩阵 | 4.69 ms | 1.24× |

正交矩阵仅增加 0.32 ms/层开销，参数量增加约 6.7%（LLaMA3-8B 约增加 0.5B 参数）。

### 关键发现

- **LLaMA3-70B 2:4 场景最亮眼**：困惑度退化从 8.11 降到 3.39（缩小 58%），zero-shot 准确率从 69.16% 提升至 76.37%
- **对 Magnitude 剪枝的提升最大**：如 Qwen2.5-72B 的 Magnitude 2:4 基线困惑度 287.70，加 DenoiseRotator 后降至 8.81，从不可用变为可用
- **跨模型一致有效**：LLaMA3、Qwen2.5、Mistral 三个模型族，7B 到 72B 规模均显著改善
- **熵降低 13% 即可带来显著收益**：100 步训练即可将困惑度从 9.57 降到 7.70
- **超参数不敏感**：学习率在 0.01-0.1 范围内、步数 400-4000 均能接近最优结果

## 亮点与洞察

- **信息论视角的新颖性**：将剪枝鲁棒性问题转化为重要性分布的熵最小化问题，提供了理论清晰的优化目标，摆脱了直接优化 NP-hard 组合问题的困境
- **正交变换的巧妙利用**：Transformer 的计算不变性保证变换不改变模型输出，正交性保证总重要性不变——两个性质的结合使得"免费"重分配重要性成为可能
- **即插即用的工程价值**：作为纯预处理步骤不修改任何后续剪枝流程，兼容 Magnitude/Wanda/SparseGPT，降低了采用门槛
- **让"不可用"变"可用"**：对 Magnitude 这种简单方法的提升幅度最大，如 Qwen2.5-72B 2:4 稀疏从 287.70 PPL 降到 8.81

## 局限与展望

- **推理开销非零**：层间正交矩阵无法完全合并，单层增加约 0.32 ms，参数量增加约 6.7%；论文探索了块对角正交矩阵来降低开销，但这是精度-效率的权衡
- **未显式优化半结构化约束**：2:4 稀疏的收益来源于正交变换的"随机置换效应"而非针对性优化，未来可将结构约束纳入训练目标
- **训练计算成本较高**：LLaMA3-70B 需约 28 小时 A100 单卡训练，对于快速迭代部署场景可能偏慢
- **仅验证 post-training 场景**：论文提到将重要性集中整合到预训练或持续训练中可能更有效，但未实验验证

## 评分

- 新颖性: ⭐⭐⭐⭐ 熵最小化 + 正交变换的组合视角在剪枝领域是新的，从"重分配重要性"而非"选择哪些权重"切入
- 实验充分度: ⭐⭐⭐⭐⭐ 7 个模型（7B-72B），3 种剪枝方法，2 种稀疏模式，消融实验和超参数分析完整
- 写作质量: ⭐⭐⭐⭐ 动机清晰、理论推导完整、图示直观
- 价值: ⭐⭐⭐⭐ 对 LLM 部署有直接实用价值，尤其是 2:4 硬件加速场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](duogpt_training-free_dual_sparsity_through_activation-aware_pruning_in_llms.md)
- [\[NeurIPS 2025\] QuadEnhancer: Leveraging Quadratic Transformations to Enhance Deep Neural Networks](quadenhancer_leveraging_quadratic_transformations_to_enhance_deep_neural_network.md)
- [\[ICML 2025\] VocabTrim: Vocabulary Pruning for Efficient Speculative Decoding in LLMs](../../ICML2025/model_compression/vocabtrim_vocabulary_pruning_for_efficient_speculative_decoding_in_llms.md)
- [\[ACL 2025\] Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching](../../ACL2025/model_compression/flipping_kd_small_to_large.md)
- [\[ICCV 2025\] CIARD: Cyclic Iterative Adversarial Robustness Distillation](../../ICCV2025/model_compression/ciard_cyclic_iterative_adversarial_robustness_distillation.md)

</div>

<!-- RELATED:END -->
