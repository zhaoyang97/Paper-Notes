---
title: >-
  [论文解读] DictPFL: Efficient and Private Federated Learning on Encrypted Gradients
description: >-
  [NeurIPS 2025][AI安全][联邦学习] 提出 DictPFL 框架，通过将模型权重分解为静态字典+可训练查找表，并结合加密感知剪枝，在联邦学习中实现全梯度同态加密保护的同时，将通信开销降低 402–748 倍、训练速度提升 28–65 倍，运行时间仅为明文 FL 的 2 倍以内。 联邦学习（FL）允许多个机构协…
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "联邦学习"
  - "Homomorphic Encryption"
  - "Privacy-Preserving"
  - "剪枝"
  - "Dictionary Decomposition"
---

# DictPFL: Efficient and Private Federated Learning on Encrypted Gradients

**会议**: NeurIPS 2025  
**arXiv**: [2510.21086](https://arxiv.org/abs/2510.21086)  
**代码**: [UCF-ML-Research/DictPFL](https://github.com/UCF-ML-Research/DictPFL)  
**领域**: AI安全  
**关键词**: federated learning, Homomorphic Encryption, Privacy-Preserving, Gradient Pruning, Dictionary Decomposition

## 一句话总结

提出 DictPFL 框架，通过将模型权重分解为静态字典+可训练查找表，并结合加密感知剪枝，在联邦学习中实现全梯度同态加密保护的同时，将通信开销降低 402–748 倍、训练速度提升 28–65 倍，运行时间仅为明文 FL 的 2 倍以内。

## 背景与动机

联邦学习（FL）允许多个机构协作训练模型而无需共享原始数据，但共享梯度仍面临隐私泄露风险——梯度反演攻击（Gradient Inversion Attack）可从共享梯度中重建客户端的原始训练数据。

同态加密（HE）是保护梯度隐私的理想方案：客户端加密梯度后上传，服务器直接在密文上进行聚合，无需解密。然而 HE 带来的开销极其巨大：

- **密文膨胀**：通信开销增加 1–3 个数量级
- **计算开销**：加解密和同态聚合消耗大量时间
- 在 ViT 训练中，HE 相关操作（加密、解密、聚合、通信）占据了绝大部分训练时间

现有方法 FedML-HE 采用"选择性加密"策略：仅加密最敏感的 10% 梯度，其余明文传输。这虽降低了开销，但未加密的梯度仍然暴露隐私信息——实验表明，当 30% 梯度未加密时，攻击者可恢复出与原图相似度高达 23% 的图像。

## 核心问题

如何在 HE-based FL 中**同时实现**：

1. **完全隐私保护**：所有传输的梯度必须全部加密，不留任何明文梯度
2. **高效率**：将 HE 带来的通信和计算开销降低到接近明文 FL 的水平

这两个目标在此前被认为是矛盾的——全加密意味着高开销，低开销则需要牺牲部分隐私。

## 方法详解

DictPFL 包含两个核心模块：

### 1. Decompose-for-Partial-Encrypt (DePE) — 字典分解

**核心思想**：将权重矩阵 $W \in \mathbb{R}^{n \times m}$ 分解为静态字典 $D \in \mathbb{R}^{n \times r}$ 和可训练查找表 $T \in \mathbb{R}^{r \times m}$，其中 $r \ll \min(n, m)$。

**具体步骤**：

- 对初始权重 $W_0$ 进行截断 SVD 分解：$W_0 \approx U_r \Sigma_r V_r^\top$
- 字典 $D = U_r \Sigma_r$（冻结，各客户端一致，永不传输）
- 查找表初始化为零矩阵，实际权重构造为 $W = W_0 + D \cdot T$
- 仅对 $T$ 的梯度加密传输进行聚合

**关键设计**：保留原始 $W_0$ 并将 $T$ 初始化为零（而非直接用 $V_r^\top$），避免 SVD 截断带来的信息损失。当 $r=4$ 时，可训练参数量大幅减少，直接减少需要加密的密文数量。

### 2. Prune-for-Minimum-Encrypt (PrME) — 加密感知剪枝

在 DePE 基础上进一步减少需要加密传输的参数。

**HE 环境下剪枝的独特挑战**：

- 各客户端独立剪枝会导致位置不一致，而 HE 的 SIMD 批处理机制要求各客户端密文槽对齐
- 加密后的索引无法在服务器端进行非线性比较操作

**Temporal Inactivity Pruning (TIP)**：

- 使用前 $\tau$ 轮的全局梯度历史（各客户端一致）作为共享剪枝指标
- 仅当参数在连续 $\tau$ 轮中梯度幅度都位于最小 $s\%$ 时才剪枝
- 剪枝掩码公式：当 $\sum_{k=1}^{\tau} \mathbf{1}(|\delta w_{i,t-k}| < \theta_{s,t-k}) = \tau$ 时 $M_{i,t}=0$（剪枝）

**Holistic Reactivation Correction (HRC)**：

- 解决 TIP 中被剪枝参数永久失活的问题
- 为每个被剪枝参数分配动态重激活概率 $p_i$
- 重激活后，若累积全局梯度仍小则降低 $p_i$（乘以衰减因子 $\beta$），否则增大 $p_i$
- 通过共享随机种子确保各客户端重激活一致性
- 剪枝掩码无需发送到服务器，避免通过明文掩码推断隐私

### 默认超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| $r$ | 4 | 字典大小 |
| $s\%$ | 70% | 剪枝比例 |
| $\tau$ | 3 | 剪枝耐心窗口 |
| $\beta$ | 0.2 | 重激活概率衰减因子 |

## 实验关键数据

### 效率对比（vs 全加密 FedHE-Full）

- 通信开销降低 **402–748×**
- 训练速度提升 **28–65×**
- 运行时间仅为明文 FL 的 **< 2×**

### 效率对比（vs 选择性加密 FedML-HE，加密 10%）

- 通信开销降低 **51–155×**
- 训练速度提升 **4–19×**
- 同时 DictPFL 提供**完全**隐私保护（FedML-HE 有隐私泄露风险）

### 准确率（ViT，3 客户端同质场景）

| 方法 | CIFAR-10 | GTSRB | Diabetic Retinopathy |
|------|----------|-------|---------------------|
| FedHE-Full | 基准 | 基准 | 82.74% |
| FedHE-Top2 | — | 58.9% | — |
| DictPFL ($r$=4) | 同等水平 | 95.27% | 81.99% |

### 隐私保护

- FedML-HE（30% 未加密）：攻击者恢复图像相似度达 23%
- DictPFL：所有梯度加密，可抵御任何梯度反演攻击

### 消融实验

- **字典大小 $r$**：$r=4$ 即可达到接近全模型的准确率，$r=2$ 性能显著下降
- **剪枝比例**：70% 剪枝 + HRC 重激活可达到 20% 剪枝的准确率，同时获得 70% 剪枝的通信效率
- **剪枝耐心 $\tau$**：$\tau=3$ 已足够平衡准确率和通信效率

## 亮点

1. **首次证明 HE-based FL 可实际部署**：运行时间仅为明文 FL 的 2 倍以内，此前被认为不可能
2. **零隐私泄露设计**：所有传输梯度全部加密，未传输参数（字典）留在本地，剪枝掩码也不发送到服务器
3. **DePE 的巧妙设计**：保留 $W_0$ + 零初始化 $T$ 避免 SVD 截断信息损失，字典 $D$ 在各客户端自然一致无需通信
4. **HRC 重激活机制**：优雅解决了 HE 环境下剪枝不可逆的问题，通过动态概率在效率和收敛性之间取得平衡
5. **跨任务通用性**：在图像分类、文本分类、文本生成任务上均有效，覆盖 ViT、BERT、TinyLlama 等模型

## 局限与展望

1. **固定字典**：字典在训练前一次性构建并冻结，无法适应高度异质性的客户端数据分布，未来可探索动态字典
2. **场景限制**：仅评估了 cross-silo 场景（少量客户端），尚未验证 cross-device 场景（大量资源受限设备）
3. **模型家族**：实验集中在 Transformer 架构，CNN 等其他架构未覆盖
4. **SVD 分解开销**：虽然是一次性操作，但对超大模型的 SVD 分解本身可能带来显著计算成本
5. **$r$ 的选择**：字典大小 $r$ 对准确率影响较大（$r=2$ 到 $r=4$ 准确率跳跃明显），如何自动选择最优 $r$ 未探讨

## 与相关工作的对比

| 方法 | 隐私级别 | 通信开销 | 训练速度 | 准确率 |
|------|---------|---------|---------|--------|
| FedHE-Full | 完全加密 | 极高（基准） | 极慢 | 最高 |
| FedHE-Top2 | 完全加密（仅末层） | 中等 | 一般 | 较低 |
| FedML-HE (10%) | 部分加密，有泄露 | 高 | 慢 | 高 |
| DP-based FL | 噪声保护 | 低 | 快 | 有损 |
| MPC-based FL | 聚合保护 | 依赖 | 依赖 | 无损 |
| **DictPFL** | **完全加密** | **极低** | **接近明文** | **高** |

与 FedML-HE 的核心区别：FedML-HE 是"减少加密量"（选择性加密），DictPFL 是"减少传输量"（全部加密但传输量极小）。思路完全不同，DictPFL 从根源上避免了隐私泄露。

## 启发与关联

- **字典分解思路**的一般性：将大参数空间分解为共享静态部分+个性化可训练部分，类似于 LoRA 的低秩分解思想，但冻结的是 SVD 的左侧而非右侧
- **HE 友好的模型设计**：启发未来在设计 FL 算法时考虑加密方案的约束（如 SIMD 槽对齐），而非事后适配
- **剪枝一致性问题**可推广到其他需要分布式一致性的场景，如分布式训练中的稀疏通信
- 与安全聚合（Secure Aggregation）可互补：DictPFL 保护的是客户端到服务器的传输，SA 保护的是聚合过程中的个体贡献

## 评分

- 新颖性: ⭐⭐⭐⭐ — DePE+PrME 的组合设计新颖，首次实现 HE-FL 的实用化
- 实验充分度: ⭐⭐⭐⭐ — 多数据集、多模型、多场景消融，隐私攻击验证完整
- 写作质量: ⭐⭐⭐⭐ — 图示清晰，动机和方法逻辑连贯
- 价值: ⭐⭐⭐⭐⭐ — 解决了 HE-FL 的核心实用性瓶颈，对实际部署有重大意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[NeurIPS 2025\] MARS: A Malignity-Aware Backdoor Defense in Federated Learning](mars_a_malignity-aware_backdoor_defense_in_federated_learning.md)
- [\[NeurIPS 2025\] DESIGN: Encrypted GNN Inference via Server-Side Input Graph Pruning](design_encrypted_gnn_inference_via_server-side_input_graph_pruning.md)
- [\[ICML 2025\] An Efficient Private GPT Never Autoregressively Decodes](../../ICML2025/ai_safety/an_efficient_private_gpt_never_autoregressively_decodes.md)
- [\[ICML 2025\] Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](../../ICML2025/ai_safety/clients_collaborate_flexible_differentially_private_federated_learning_with_guar.md)

</div>

<!-- RELATED:END -->
