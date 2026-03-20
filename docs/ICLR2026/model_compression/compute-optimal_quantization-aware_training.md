# Compute-Optimal Quantization-Aware Training

**会议**: ICLR 2026  
**arXiv**: [2509.22935](https://arxiv.org/abs/2509.22935)  
**代码**: 无  
**领域**: 模型压缩 / LLM效率  
**关键词**: 量化感知训练, Scaling Law, 计算最优分配, tokens-per-parameter-byte, 低比特量化

## 一句话总结
本文通过 757 组 QAT 实验（86M-2.2B 参数，1-6 bit）发现：QAT 的最优训练比例随总计算量增长而增大（与先前认为固定 10% 的结论相反），并提出 tokens-per-parameter-byte 统计量和新的 loss scaling law 来精确预测最优 QAT 分配策略和最终损失。

## 研究背景与动机

1. **领域现状**：QAT 是训练高质量量化模型的主流方法，通常采用"先全精度（FP）训练→再 QAT 微调"的两阶段流程。Liu et al. (2025) 建议 QAT 阶段占总训练步数的 10%。
2. **现有痛点**：
   - 先前关于"10%最优"的结论在有限计算预算下得出，未验证是否在更大规模下成立
   - PTQ 引入的量化误差随预训练数据量增长而增大（Kumar et al.），暗示 FP-QAT 分配应与规模相关
   - 现有 QAT scaling law（Chen et al.）将 Dfp=0（从头开始 QAT），不处理"FP→QAT"的两阶段场景
   - 没有跨 bit-width 的统一 scaling law
3. **核心矛盾**：QAT 步数太少→模型无法适应低精度；QAT 步数太多→压缩 FP 阶段→用噪声梯度训练太久。随着总计算量增长，这个平衡点如何变化？
4. **本文要解决什么？**
   - 最优 QAT fraction 如何随模型大小、总 token 数、bit-width 变化？
   - 能否用一个统一的 scaling law 预测所有配置下的最终损失？
   - 能否进一步优化训练流程（如合并 cooldown 和 QAT）？
5. **切入角度**：引入 tokens-per-parameter-byte $S = D/(N \cdot B/8)$ 作为统一的缩放变量——它同时编码了模型大小、数据量和量化精度的信息。
6. **核心idea一句话**：QAT 的最优时间分配不是固定的 10%，而是随 tokens-per-parameter-byte 增长的函数，可用一个统一 scaling law 精确建模。

## 方法详解

### 整体框架
两阶段训练：FP 阶段（$D_{fp}$ tokens）→ QAT 阶段（$D_{qat}$ tokens），总计 $D_{total} = D_{fp} + D_{qat}$。核心问题：给定 $N, D_{total}, B$，最优 $f^* = D_{qat}^*/D_{total}$ 是多少？

### 关键设计

1. **Tokens-per-Parameter-Byte 统计量**:
   - 做什么：统一不同模型大小和 bit-width 的 QAT 最优分配预测
   - 核心思路：$S_{total} = D_{total}/(N \cdot B/8)$，将模型参数量按量化后的字节数归一化。大模型更容易量化（$N$ 大→$S$ 小），低 bit 更难量化（$B$ 小→$S$ 大），训练更长更难量化（$D$ 大→$S$ 大）。
   - 设计动机：图 2 对比显示，在 token 坐标下不同 bit-width 的最优点分散，而在 tokens-per-parameter-byte 坐标下不同 bit-width 的最优点几乎落在同一条线上。

2. **最优 QAT Fraction 预测**:
   - 做什么：直接拟合最优 QAT fraction 与 $S_{total}$ 的关系
   - 核心思路：$\hat{f}(D_{total}, N, B) = \frac{\exp(\log S_{total} - a/\log S_{total})}{S_{total}}$，其中 $a=6.7297$ 是唯一需要拟合的参数。
   - 设计动机：观察到 $S_{total}$ 与最优 $S_{qat}$ 在对数坐标下近似线性，加约束 $D_{qat} \leq D_{total}$。MAE 仅 0.091。

3. **统一 Loss Scaling Law**:
   - 做什么：跨模型大小、token 数、bit-width 预测最终损失
   - 核心思路：$L = \text{Chinchilla-like} + \delta(N, D_{qat}, D_{fp}, B)$，其中 QAT 惩罚项 $\delta$ 分解为三部分：
     - **不可约 QAT 误差** $\theta \cdot 2^{-\kappa B}$：bit-width 决定的精度下限
     - **纯 QAT 惩罚** $\frac{\phi \cdot 2^{-\chi B}}{N^\psi \cdot S_{qat}^\omega}$：QAT 步数不够时的误差
     - **FP/QAT 交互项** $\frac{\lambda \cdot 2^{-\mu B}}{N^\nu \cdot S_{fp}^\xi \cdot S_{qat}^\rho}$：FP 阶段过长导致量化更困难
   - 设计动机：之前的 scaling law 在 $D \to \infty$ 时 loss 趋向无穷（不合理）；新公式的所有惩罚项随 $S$ 增大而减小，保证 loss 最终收敛。757 个实验点拟合后准确预测最优 fraction 和 loss。

4. **Cooldown + QAT 融合**:
   - 做什么：将 learning rate 衰减阶段与 QAT 合并，消除冗余的 FP 更新
   - 核心思路：在 FP 阶段的最高学习率处直接开始 QAT，同时执行学习率衰减，而不是先完成 FP cooldown 再开始 QAT。
   - 设计动机：标准 FP+cooldown+QAT 中，cooldown 阶段的 FP 更新对最终量化模型价值不大，可以省掉。

### 损失函数 / 训练策略
- QAT 使用 straight-through estimator 处理量化操作的不可导性
- 757 组实验覆盖 86M-2.2B 模型、1/2/4/6-bit、2.3B-1.4T tokens
- Huber loss + 梯度下降拟合 scaling law 参数

## 实验关键数据

### 主发现：最优 QAT Fraction 随规模增长

| 模型大小 | 总 tokens | 2-bit 最优 f* | 4-bit 最优 f* | 6-bit 最优 f* |
|---------|----------|-------------|-------------|-------------|
| 86M | 短 | ~10% | ~8% | ~5% |
| 86M | 长 | ~40% | ~25% | ~15% |
| 396M | 中 | ~25% | ~15% | ~10% |
| 759M | 长 | ~30%+ | ~20% | ~12% |

### Scaling Law 预测精度

| 预测对象 | 误差 |
|---------|------|
| 最优 QAT fraction (直接拟合) | MAE = 0.091 |
| Loss scaling law (757 实验) | 准确预测所有配置 |
| 跨 bit-width 最优选择 | 正确预测 |
| Cooldown+QAT 融合 vs 标准 | 达到或超越标准方案 |

### 关键发现
- **推翻"10%通适"结论**：在大计算量下，最优 QAT fraction 可达 30-40%（低 bit）
- **低 bit 需要更多 QAT**：2-bit 比 6-bit 在相同规模下需要更多 QAT 步数
- **大模型更容易量化**：相同 $D_{total}$ 下，大模型需要更少的 QAT fraction
- **Cooldown 融合有效**：在不增加总 token 数的前提下，融合方案达到或超越标准 FP+QAT
- **4-bit QAT 最具性价比**：在大多数内存约束下，4-bit 在 loss 和内存之间最优

## 亮点与洞察
- **tokens-per-parameter-byte 是极巧妙的统一变量**：一个量就同时编码了模型大小、数据量和精度——不同配置在此坐标下呈现统一规律。这种"找对的变量"的方法论值得借鉴。
- **推翻先前结论的方法论价值**：通过更大规模的系统性实验证明先前的结论是局部的——这在 scaling law 研究中是常见且重要的工作。
- **Scaling law 的工程实用性**：拟合 757 个实验后，可以直接回答"给定计算预算和内存约束，应该用多少 bit 的量化、花多少比例做 QAT"——对大规模训练计划有直接指导价值。

## 局限性 / 可改进方向
- 仅测试到 2.2B 参数，未在 7B+ 大模型上验证
- 仅考虑 weight quantization，未涉及 activation quantization
- Scaling law 有 15+ 个可拟合参数，可能存在过拟合风险
- cooldown+QAT 融合的学习率调度策略较简单，可能有更优的调度方案
- 未考虑 MoE 架构、不同数据质量等变量

## 相关工作与启发
- **vs Chen et al. (2025b)**: 他们的 QAT scaling law 仅管从头训练（$D_{fp}=0$）且每个 bit-width 单独拟合；本文统一处理 FP→QAT 且跨 bit-width
- **vs Kumar et al. (2025)**: 他们发现 PTQ 误差随数据量增长——本文在 QAT 场景下验证了类似趋势，并给出了如何应对的方案
- **vs Chinchilla**: 本文的 scaling law 在 Chinchilla 框架上加入 QAT 感知项，是 Chinchilla 在量化训练场景的自然推广

## 评分
- 新颖性: ⭐⭐⭐⭐ 推翻已有结论+提出统一scaling law，但方法论并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 757组实验，多模型大小/bit-width/token数，工程量巨大
- 写作质量: ⭐⭐⭐⭐⭐ 图表设计精美，逻辑推导清晰，变量选择有充分动机说明
- 价值: ⭐⭐⭐⭐⭐ 对LLM量化训练的实际计划有直接指导价值，Apple出品质量保证
