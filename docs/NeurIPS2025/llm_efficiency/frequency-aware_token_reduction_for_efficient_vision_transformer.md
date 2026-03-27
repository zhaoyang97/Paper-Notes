# Frequency-Aware Token Reduction for Efficient Vision Transformer

**会议**: NeurIPS 2025
**arXiv**: [2511.21477](https://arxiv.org/abs/2511.21477)
**代码**: [GitHub](https://github.com/jhtwosun/frequency-aware-token-pruning)
**领域**: 模型压缩 / Vision Transformer
**关键词**: token reduction, rank collapse, over-smoothing, frequency analysis, vision transformer

## 一句话总结
从频域视角提出 frequency-aware token reduction，将 token 分为高频（HF）和低频（LF）两组，选择性保留 HF token 并将 LF token 聚合为 DC token，在缓解 rank collapse 的同时减少 ViT 的计算量，在 30% token 减少率下多个模型上超越现有 SOTA。

## 研究背景与动机
1. **领域现状**：Vision Transformer 的二次复杂度推动了 token reduction 研究——主要分为 merging（融合相似 token）和 pruning（丢弃不重要 token）两大类，已有 ToME、EViT、DynamicViT 等方法。
2. **现有痛点**：现有方法忽略了 self-attention 的频域特性——SA 本质上是低通滤波器，堆叠 SA 层会导致 rank collapse（所有 token 表示趋同）。Token reduction 会加剧这一问题：merging 直接平均掉高频信号，pruning 如果移除含高频信息的 token 也加速 collapse。
3. **核心矛盾**：减少 token 数量以提高效率 vs 保留高频信息以维持 ViT 表达能力，二者看似矛盾。
4. **本文要解决什么？** 设计一种在 token reduction 中显式保护高频信息的方法，在提高效率的同时缓解 rank collapse。
5. **切入角度**：将 attention 矩阵分解为低频分量 $A^{LP} = \frac{1}{n}\mathbf{11}^T$ 和高频分量 $A^{HP} = A - A^{LP}$，根据 $A^{HP}$ 中每个 token 对高频贡献的大小选择保留/聚合。
6. **核心idea一句话**：保留对输出高频分量贡献最大的 token，将低频 token 聚合为 DC token 保留零频信息。

## 方法详解

### 整体框架
在每个 reduction 层：(1) 从 attention 矩阵中分解出高频分量 $A^{HP}$；(2) 按列求和识别 HF token 和 LF token；(3) 保留 top-r 个 HF token，将 LF token 按空间局部组聚合为 DC token；(4) 用可学习参数 $\omega_1, \omega_2$ 调整后续 attention 权重以缓解 collapse。

### 关键设计

1. **频域 Token 分选**:
   - 做什么：将 token 分为高频（HF）和低频（LF）两组
   - 核心思路：对多头注意力矩阵求高频分量 $A^{HP} = A - \frac{1}{n}\mathbf{11}^T$，按列求和得到每个 token 的高频贡献分数 $\tilde{A}_k$。分数最高的 r 个为 HF token，最低的 r 个为 LF token
   - 设计动机：只需简单的列平均运算（比 FFT 或余弦相似度计算高效得多），直接利用已有的 attention 矩阵，零额外计算开销

2. **Local DC Token 聚合**:
   - 做什么：将 LF token 按 $w^2$ 个空间局部组聚合为 DC token，保留零频信息
   - 核心思路：$x_{DC}^j = \frac{1}{|N_{LF}^j|} \sum_{i \in N_{LF}^j} x_i$，多层 reduction 时递归更新 DC token
   - 设计动机：直接丢弃 LF token 会损失 DC 信号（图 2b 验证 LF token 确实主导 DC 分量）；局部 DC token（$w>1$）在早期层保留 LF token 中残余的高频空间局部信息

3. **注意力权重调整**:
   - 做什么：修改 attention 矩阵以强调 HF token 并补偿 DC token 的低 attention 权重
   - 核心思路：$\hat{A} = A^{LP} + (\omega_1+1)A^{HP} + (\omega_2+1)A^{N_{DC}}$，$\omega_1$ 增强高频信号，$\omega_2$ 补偿 DC token 因 Jensen 不等式导致的偏低 attention score
   - 设计动机：仅减少 token 不够，还需主动抑制后续层对剩余 token 的 rank collapse 趋势

### 理论支撑
Proposition 3.1 证明：无论 pruning 还是 merging 都使 $\|H_f[SA(MX)]\|_F \leq \|H_f[SA(X)]\|_F$，即 token reduction 加速 rank collapse。本方法通过选择性保留 HF token 减缓这一趋势。

## 实验关键数据

### DeiT 系列主实验（ImageNet-1K, 30% token reduction per layer）
| 模型 | 方法 | MACs | Accuracy |
|------|------|------|----------|
| DeiT-S | Baseline | 4.6G | 79.8% |
| DeiT-S | ToME | 2.9G | 79.5% |
| DeiT-S | EViT | 3.0G | 79.5% |
| DeiT-S | DiffRate | 2.9G | 79.6% |
| DeiT-S | **本文** | **2.9G** | **80.0%** |
| DeiT-B | Baseline | 17.6G | 81.8% |
| DeiT-B | ToME | 11.2G | 81.7% |
| DeiT-B | **本文** | **11.5G** | **82.1%** |

### 自监督模型
| 模型 | 基线 Acc | 本文 Acc | MACs 减少 |
|------|---------|---------|---------|
| MAE ViT-B | 83.6% | **83.5%** | ~35% |
| DINO ViT-S | 81.5% | **81.5%** | ~35% |

### 消融实验
| 配置 | Accuracy |
|------|----------|
| 仅 HF token 保留 | 79.6% |
| + DC token | 79.8% |
| + Local DC | 79.9% |
| + Attention 调整（$\omega_1, \omega_2$）| **80.0%** |

### 关键发现
- HF token 确实包含更多高频信号（图 2a 频率分析验证），LF token 主导 DC 分量（图 2b 相似度验证）
- 对 HF token 加噪声对精度影响远大于对 LF token 加噪声（图 2c），证实 HF token 对模型更关键
- 现有 pruning 方法（如 EViT）仅在最后几层才倾向保留 HF token，在中间层行为不一致；本方法在所有层显式保留 HF token
- 方法在减少 30%+ token 的情况下，多数模型精度反而提升（超过未减少 token 的基线），说明缓解 rank collapse 的正向效应大于信息损失

## 亮点与洞察
- **频域理解 token reduction**：首次将 token reduction 与 ViT 的 rank collapse/over-smoothing 理论联系起来，提供了新的设计视角
- **简洁高效的 HF/LF 分选**：仅用 attention 矩阵的列平均，零额外计算，就能有效区分高/低频 token
- **精度提升的反直觉结果**：减少 token 反而提升精度，说明 rank collapse 是 ViT 的真实瓶颈，而非参数或容量不足

## 局限性 / 可改进方向
- **仅验证分类任务**：检测、分割等密集预测任务可能对低频信息更敏感
- **reduction 层位置固定**（第4/7/10层）：目前硬编码，可能不是所有模型的最优选择
- **$\omega_1, \omega_2$ 需要微调**：需要 30 epoch 微调，非完全 training-free

## 相关工作与启发
- **vs ToME（merging）**: ToME 融合相似 token 本质上是低通滤波，加速 rank collapse；本文保留 HF token 抵消 collapse
- **vs EViT（CLS-based pruning）**: EViT 按 CLS attention 选择 token，但不显式考虑频率特性
- **vs DiffRate**: DiffRate 学习自适应 reduction 率，但同样忽略频域特性

## 评分
- 新颖性: ⭐⭐⭐⭐ 频域视角的 token reduction 是新贡献，与 rank collapse 理论的结合优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 多种模型（DeiT/ViT/MAE/DINO）+ 多种训练策略 + 详细消融 + 频域分析可视化
- 写作质量: ⭐⭐⭐⭐ 理论动机清晰，实验与理论对应紧密
- 价值: ⭐⭐⭐⭐ 实用的 ViT 加速方法，rank collapse 视角对 ViT 效率研究有启发
