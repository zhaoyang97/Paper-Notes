# MXNorm: Reusing MXFP block scales for efficient tensor normalisation

**会议**: CVPR 2026  
**arXiv**: [2603.13180](https://arxiv.org/abs/2603.13180)  
**代码**: 无（Graphcore 内部实现，基于 TorchTitan + TorchAO）  
**领域**: 模型压缩 / 高效训练  
**关键词**: RMSNorm, MXFP量化, 低精度训练, 归一化融合, LLM预训练  

## 一句话总结
MXNorm 提出将 RMSNorm 与 MXFP 量化融合：利用 MXFP 量化过程中已经计算好的 block absmax 来近似 RMS 值，从而省掉单独的归一化 reduction 操作，在 Llama 3 最高 8B 参数的预训练中保持训练精度，同时在 GB200 上实现最高 2.4 倍的 kernel 加速。

## 背景与动机
近年来 AI 加速器在低精度矩阵乘法上取得了巨大进步（V100 到 GB200 提升了 80 倍），但逐元素运算和 reduction 的带宽/吞吐量只提高了 5-9 倍。这种差距导致以前不是瓶颈的操作（如归一化层）现在成为新的性能瓶颈。RMSNorm 是 Llama 系列等主流 LLM 的标配归一化层，它需要对整个隐藏维度做 reduction 计算 RMS 值。在 Pre-Norm transformer 中，RMSNorm 紧跟着就是 MXFP 量化（MXCast），而 MXCast 本身也要计算 block absmax——两者都在做类似的统计量聚合。这种冗余计算是优化的切入点。

## 核心问题
矩阵乘法加速远超其他运算的提升速度，RMSNorm 中的 reduction 操作正成为低精度 transformer 训练的新瓶颈。能否利用 MXFP 量化中已有的 block 级统计信息来替代 RMSNorm 的全维度 reduction，从而融合归一化和量化？

## 方法详解

### 整体框架
在 Pre-Norm transformer 的典型 "RMSNorm → Linear" 模式中，MXNorm 将 RMSNorm 和 MXCast 融合为一个操作。输入是 BF16 的激活张量 $X \in \mathbb{R}^{T \times D}$，输出是 MXFP8 格式的 $(S, V)$ 元组（block scales + quantized values）。关键是用 block absmax 的广义 p-均值来估计 RMS，省掉了对整个 $D$ 维做 reduction 的步骤，只需对 $K = D/B$ 个 block 的 absmax 做 reduction（$B=32$ 时减少 32 倍）。

### 关键设计
1. **Block Absmax 近似 RMS（定理1）**: 论文严格证明：对于满足尺度族分布的 i.i.d. 元素，block absmax 的广义 p-均值与 RMS 的比值在 $K \to \infty$ 时以概率 1 收敛到一个只依赖于 $p$、$B$ 和分布形状的常数 $c(p,B)$。因此只需预计算一个校正系数 $\tilde{c}(p,B)$（用高斯蒙特卡洛估计），就能将 block absmax 的 p-均值线性缩放为 RMS 的近似。

2. **MXNorm 前向计算**: 对输入张量分块后计算 block absmax $\tilde{m}_{tk}$，然后用 p-均值估计逆 RMS $\tilde{\rho}_t = \tilde{c}(p,B) \cdot (\frac{1}{K}\sum_k \tilde{m}_{tk}^p)^{-1/p}$。直接用 $\tilde{\rho}_t$ 调整 block scales 和 values，一步完成归一化+量化。这样 reduction 的规模从 $D$（全隐藏维度）降低到 $K = D/B$（block 数目），缩小了 $B$ 倍（典型 $B=32$）。

3. **p=2 vs p=1 的稳定性差异**: p=1（算术平均）和 p=2（RMS 均值）在小模型上表现相当，但在 8B 模型上 p=1 出现 loss spike 导致最终性能退化。作者分析原因是 MXNorm(p=1) 的输出上界为 $O(K)$，而 p=2 为 $O(\sqrt{K})$，更松的上界导致异常特征（outlier feature）引发训练不稳定。这个发现也揭示了归一化层的"截断"效应对训练稳定性的关键作用。

4. **MXNormLinear（Gain 参数处理）**: RMSNorm 有可学习 gain $\gamma$，但在 MX 量化后的张量上做逐元素广播不方便。作者利用线性操作的结合律，将 $\gamma$ 吸收到后续 Linear 层的权重 $W$ 中：$H = \text{MXNorm}(X) \cdot \text{MXCast}(W \gamma)^\top$。推理时可以预先融合，训练时在梯度计算中单独处理。

### 损失函数 / 训练策略
- 反向传播使用 RMSNorm 的梯度作为直通估计器（STE），保证梯度的平滑性
- 前向 MXNorm，反向 RMSNorm gradient，不引入额外超参数
- 反向需要缓存输入 $X$ 和逆 RMS 估计 $\tilde{\rho}$，但这与标准 RMSNorm+MXCast 的缓存需求相同

## 实验关键数据
| 模型规模 | 指标 | RMSNorm | MXNorm(p=1) | MXNorm(p=2) |
|--------|------|---------|-------------|-------------|
| 125M | Training Loss | 3.090±0.004 | 3.113±0.012 | 3.116±0.010 |
| 1B | Training Loss | 2.692±0.011 | 2.684±0.009 | 2.691±0.007 |
| 8B (300B tokens) | Training Loss | 2.132 | 2.175 | 2.126 |

| 设置 | 指标 | MXNorm vs RMSNorm 加速 |
|------|------|----------------------|
| Kernel (MXFP8, B=32) | 几何平均加速 | 41.7% |
| Kernel (NVFP4, B=16) | 几何平均加速 | 31.2% |
| Kernel (最大) | 单次加速 | 2.4× |
| Llama 3 8B 层 (MXFP8) | 端到端层加速 | 1.3% |
| Llama 3 8B 层 (NVFP4) | 端到端层加速 | 2.6% |

8B 模型 OLMES 零样本评估：MXNorm(p=2) 在 10 个任务中赢 5 个，RMSNorm 赢 5 个，整体持平。

### 消融实验要点
- p=1 在 8B 规模出现 loss spike，最终 loss 明显劣于 p=2 和 RMSNorm
- p=2 在所有模型规模上都能匹配 RMSNorm 基线
- Post-Round MXNorm（用 E8M0 取整后的 scales 估计 RMS）在 8B 规模严重不稳定，说明取整带来的量化噪声破坏了近似质量
- 近似质量随 block 数量增加而渐近改善，1024 个元素（32 blocks）就已有很好的 $r^2$
- 输出上界是训练稳定性的关键指标：$O(\sqrt{K})$ 的 p=2 明显优于 $O(K)$ 的 p=1

## 亮点
- **极简设计**：零额外超参数的 drop-in 替换，只需改归一化层实现，不改架构
- **理论+实践闭环**：从严格的数学证明（定理1 + 上界分析）到实际 kernel 加速，每一步都有理论支撑
- **对训练稳定性的深入分析**：通过 p=1 vs p=2 的对比，揭示了归一化层的"截断效应"对抑制 outlier feature 的重要性，这个 insight 超越了 MXNorm 本身
- **适用于更低精度格式**：MXNorm 的优势随精度降低（FP4、INT2、ternary）会更加显著，方向非常 forward-looking

## 局限性 / 可改进方向
- 端到端加速相对有限（MXFP8 下 Llama 3 8B 层仅 1.3%），因为在 matmul 主导的架构中归一化占比本来就小
- 目前只验证了 Pre-Norm transformer 的 "Norm → Linear" 模式，Post-Norm 或其他位置的归一化未探索
- 仅在语言模型上验证，视觉 transformer 和 VLM 场景未涉及
- 未探索与其他非 matmul 操作（RoPE、门控线性单元等）的联合融合优化
- 需要 MX 格式硬件支持（目前限于 Blackwell 及以后的 GPU）

## 与相关工作的对比
- **FlashNorm**（异步计算 RMS + 用未归一化输入做矩阵乘法）：有精度风险，可能冲爆累加器；MXNorm 更安全
- **Partial RMS**（只用前 k 个元素估计 RMS）：容易漏掉 outlier；MXNorm 用所有 block 的 absmax，覆盖全部元素
- **无归一化训练**（如 nGPT 将权重约束到超球面、tanh 替代归一化）：引入额外开销（优化器步骤更慢、逐元素特殊函数）；MXNorm 不改变优化器，开销更低
- MXNorm 的独特优势是"搭车"已有的量化计算，理论上零额外开销

## 启发与关联
- 这个"复用已有计算的中间结果"的思路可以推广：比如 attention 中的 softmax 也需要 max/sum reduction，能否与量化融合？
- 归一化输出上界与训练稳定性的关系值得进一步研究，可能启发新的自适应归一化设计
- 方向上与 `ideas/model_compression/` 中的量化和高效训练想法相关

## 评分
- 新颖性: ⭐⭐⭐⭐ 核心 idea 简洁优雅（复用 block scales），但属于工程导向的增量创新
- 实验充分度: ⭐⭐⭐⭐⭐ 125M/1B/8B 三个规模 + LR sweep + 稳定性分析 + kernel benchmark + OLMES 评估，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 数学严谨，叙事清晰，图表精美，从理论到实验层层递进
- 价值: ⭐⭐⭐⭐ 对低精度训练生态有实际贡献，但受限于 MX 格式硬件普及度和端到端加速幅度
