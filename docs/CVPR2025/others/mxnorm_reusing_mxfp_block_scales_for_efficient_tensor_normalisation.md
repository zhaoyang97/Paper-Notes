# MXNorm: Reusing MXFP block scales for efficient tensor normalisation

**会议**: CVPR2025  
**arXiv**: [2603.13180](https://arxiv.org/abs/2603.13180)  
**代码**: 无公开代码（Graphcore 内部实现，附录含 PyTorch 伪代码）  
**领域**: 高效训练 / 量化  
**关键词**: RMSNorm, MXFP8, 低精度训练, 量化感知训练, LLM预训练, 归一化加速

## 一句话总结
MXNorm 提出复用 MXFP 量化过程中已计算的 block absmax 来近似 RMS，将归一化与 MX 量化融合为单次统计收集操作，实现 RMSNorm 的 drop-in 替换，在 Llama 3 8B 预训练中保持训练精度的同时获得最高 2.4× 的 kernel 加速。

## 研究背景与动机
1. **领域现状**：过去 8 年 GPU 矩阵乘法加速提升了 80×（V100→GB200），但内存带宽仅提升 8.9×，CUDA core 仅提升 5.1×。随着 MXFP8/FP4 等低精度矩阵乘法的普及，非矩阵乘法操作（归一化、逐元素计算、规约）正成为新的吞吐瓶颈。
2. **核心矛盾**：Pre-Norm Transformer（Llama 系列）在每个 QKV 投影和 FFN 投影前都执行 RMSNorm，而 RMSNorm 需要对整个隐层维度做规约（求平方均值），紧接着又要对同一张量做 MX 量化（求 block absmax）——这是两次独立的统计收集操作。
3. **关键观察**：RMSNorm 和 MXCast 都在隐层维度上收集统计量来缩放元素。当概率分布被线性缩放时，其期望 absmax 也等比缩放——因此 block absmax 的广义幂均值可以用来估计 RMS。
4. **本文目标**：融合归一化和量化为单次操作，将规约大小降低 32×（从 D 维降到 K=D/B 个 block absmax）。

## 方法详解

### 核心思想
对于输入张量 $X \in \mathbb{R}^{T \times K \times B}$（T 个 token，D=KB 维被分为 K 个大小 B 的块）：
- **RMSNorm** 需要对所有 KB 个元素求平方均值：$\rho_t = (\frac{1}{KB}\sum_{kb} X_{tkb}^2)^{-1/2}$
- **MXCast** 需要对每个 block 求 absmax：$m_{tk} = \max_b |X_{tkb}|$
- **MXNorm** 复用 block absmax，用其广义 p-mean 估计 RMS：$\tilde{\rho}_t = c^{(p,B)} \cdot (\frac{1}{K}\sum_k m_{tk}^p)^{-1/p}$

其中 $c^{(p,B)}$ 是依赖于 p、B 和标准化分布的修正常数，通过蒙特卡洛采样从高斯分布估计。

### 理论保证（Theorem 1）
当输入来自尺度族分布（如高斯分布）时，block absmax 的广义 p-mean 与 RMS 之比几乎必然收敛到常数 $c^{(p,B)}$。这意味着只需一个乘法修正因子即可将 block absmax 统计量转换为 RMS 的近似。

### MXNormLinear
- 归一化的 gain 参数 $\gamma$ 无法直接应用于 MX 格式的张量
- 利用线性运算的结合律，将 $\gamma$ 融入后续线性层的权重：$H = \text{MXNorm}(X) \cdot \text{MXCast}(W\gamma)^\top$
- 反向传播使用 RMSNorm 的梯度计算作为 straight-through estimator

### p 值选择
- **p=1**（算术均值）：规约更简单，但输出上界为 $O(K)$，对异常值不够鲁棒
- **p=2**（RMS 均值）：输出上界为 $O(\sqrt{K})$，与 RMSNorm 的 $O(\sqrt{D})$ 同阶，提供更紧的值域约束

### 输出上界分析
- RMSNorm 输出上界：$\|x\|_\infty \leq \sqrt{D}$
- MXNorm(p=2) 输出上界：$\|x\|_\infty \leq \sqrt{K}/c$
- MXNorm(p=1) 输出上界：$\|x\|_\infty \leq K/c$
- 更紧的上界约束对训练稳定性至关重要

## 实验关键数据

### Llama 3 预训练对比（SlimPajama 数据集）

| 模型规模 | RMSNorm Loss | MXNorm(p=1) Loss | MXNorm(p=2) Loss |
|---------|-------------|-----------------|-----------------|
| 125M | 3.090±0.004 | 3.113±0.012 | 3.116±0.010 |
| 1B | 2.692±0.011 | 2.684±0.009 | 2.691±0.007 |
| 8B (300B tokens) | 2.132 | 2.175 (❌ 显著更差) | **2.126** (✅ 匹配) |

### 零样本下游任务（OLMES，8B 模型）
- MXNorm(p=2) 在 5/10 任务上领先 RMSNorm，5/10 任务上 RMSNorm 领先
- MXNorm(p=1) 在多数任务上显著落后（ARC-C: 39.1 vs 45.3, BoolQ: 56.6 vs 73.2）

### Kernel 加速（GB200, torch.compile）
- 孤立 kernel 最大加速：**2.4×**
- MXFP8 (B=32) 平均加速：41.7%
- NVFP4 (B=16) 平均加速：31.2%
- Llama 3 8B Transformer 层端到端加速：MXFP8 **1.3%**, NVFP4 **2.6%**

### 训练稳定性发现
- MXNorm(p=1) 在 8B 规模训练时出现 loss spike，由异常特征值（outlier features）触发
- 原因不是 RMS 估计不准（spike 前估计很准确），而是输出值域约束过松（$O(K)$ vs $O(\sqrt{K})$），导致异常特征值引发的权重更新过大并复利累积

### 近似质量验证
- 对高斯分布采样生成的输入，MXNorm 产生的 MX scale 和 value 分布与 RMSNorm+MXCast 几乎完全一致
- 反量化后的 $r^2$ 拟合优度随 block 数增加渐近趋向 1.0
- 仅 1024 个元素（32 个 block × 32）即可达到优秀的近似质量

### 与已有加速归一化方法的对比
- **nGPT**：通过约束权重在超球面上消除归一化，但在优化器步骤中引入额外开销
- **FlashNorm**：异步计算 RMS 并用原始输入乘权重，但有累加器溢出风险
- **部分元素 RMS**：仅用前 k 个元素估计 RMS，容易遗漏 outlier 值
- **MXNorm 的优势**：不改变训练配方，无溢出风险，利用已有计算而非引入近似采样

## 亮点
- **理论-工程结合的典范**：从数学定理（Theorem 1 证明 absmax 幂均值可估计 RMS）到实用 kernel 实现，逻辑链完整
- **问题定义精准**：识别到归一化和量化的统计收集冗余，提出融合方案降低 32× 规约大小
- **p=1 vs p=2 的深度分析**：不仅展示了 p=2 更好，还通过输出上界理论和 loss spike 分析解释了"为什么" p=1 会失败——这对理解归一化层的稳定性机制有普适价值
- **无额外超参数**：MXNorm 是 RMSNorm 的 drop-in 替换，修正常数 c 自动从分布估计
- **前瞻性**：明确指出随着 NVFP4/INT2 等更低精度格式的普及，非 matmul 瓶颈将更严重，MXNorm 的价值会放大

## 局限性
- 端到端加速有限：Llama 3 8B 层级仅 1.3%~2.6%，因为归一化本身在总计算中占比不大
- 仅验证了 Llama 3 架构和 SlimPajama 数据集，未测试其他架构（如 MoE、Vision Transformer）
- 理论依赖高斯分布假设，对于训练后期激活分布偏离高斯的情况未充分讨论
- 反向传播使用 RMSNorm 梯度的 straight-through estimator，引入了近似误差
- 目前仅支持 Pre-Norm 架构中 Norm→Linear 的模式
- 计算资源需求大：8B 实验使用 64 块 H100 训练 8 天
- 未讨论 MXNorm 对推理阶段（非训练）的影响和适用性

## 评分
- 新颖性: ⭐⭐⭐⭐ 观察到归一化和量化的统计收集可复用，视角独到
- 实验充分度: ⭐⭐⭐⭐⭐ 从 125M 到 8B 全面验证，学习率扫描、稳定性分析、kernel benchmark 一应俱全
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨清晰，Algorithm 1/2 的 diff 对比设计精妙
- 价值: ⭐⭐⭐⭐ 随着低精度格式的普及接口价值会持续增长，但当前端到端收益有限
