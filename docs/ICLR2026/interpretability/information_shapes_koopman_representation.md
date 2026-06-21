# Information Shapes Koopman Representation

**会议**: ICLR 2026  
**arXiv**: [2510.13025](https://arxiv.org/abs/2510.13025)  
**代码**: [https://github.com/Wenxuan52/InformationKoopman](https://github.com/Wenxuan52/InformationKoopman)  
**领域**: 可解释性  
**关键词**: koopman operator, information bottleneck, dynamical systems, representation learning, von neumann entropy

## 总结

本文从信息瓶颈（Information Bottleneck, IB）的视角重新审视 Koopman 算子的有限维表示学习问题。Koopman 算子将非线性动力系统提升为无穷维线性演化，但实际应用需要在有限维子空间中近似，导致表示学习面临"简洁性 vs 表达力"的根本矛盾。作者证明：(1) 潜在互信息控制预测误差的上界，但过度最大化会导致模态坍塌（mode collapse）；(2) von Neumann 熵可防止坍塌并保持有效维度。基于此，提出了一个信息论 Lagrangian 公式，统一平衡时间一致性（temporal coherence）、预测充分性（predictive sufficiency）和结构一致性（structural consistency）三大目标，并推导出可计算的损失函数。在物理仿真、视觉控制和图结构动力学三类任务上均优于现有 Koopman 方法。

## 动机

1. **Koopman 算子的无穷维困境**：Koopman 算子理论上可将非线性动态线性化，但其无穷维特性使得在深度网络中寻找合适的有限维子空间极为困难，现有方法常出现不稳定或模态坍塌
2. **缺乏通用的表示学习原则**：已有工作依赖领域先验（对称性、守恒律等）来约束 Koopman 表示，但缺少一般性的指导原则来平衡简洁性与表达力
3. **信息瓶颈视角的自然适配**：IB 框架天然适合描述"压缩输入同时保留预测信息"的权衡，但标准 IB 未考虑动态系统的线性演化约束
4. **潜在空间的线性约束更严格**：与 VAE 不同，Koopman 学习要求潜在空间不仅编码当前状态，还需支持线性前向传播，对表示施加了更强的结构约束
5. **简单增加维度无法解决问题**：先前研究表明，盲目增大潜在空间维度并不能提升性能，反而可能破坏时间一致性
6. **误差在自回归预测中累积放大**：Koopman 表示中的微小偏差会随时间步传播和放大，需要理论工具来量化和控制这种累积误差

## 方法

### 整体框架

作者把 Koopman 表示学习放进信息瓶颈的天平上：编码器把状态压进有限维潜空间、线性 Koopman 算子在潜空间里向前演化、解码器再重建回原状态，整条链路的好坏取决于"潜在动态保留了多少原始动态的信息"。围绕这个核心量，论文走的是一条层层递进的推导链——先用概率轨迹分布刻画 Koopman 表示并导出预测误差的信息上界，再把潜在互信息按 Koopman 谱结构拆成"该留 / 该压"的成分，最后凝练成一个同时平衡时间一致性、预测充分性与结构表达力的信息论 Lagrangian，并落地为架构无关、可训练的损失。

### 关键设计

**1. 概率轨迹分布与自回归误差界：把"预测准不准"翻译成"潜在互信息够不够"**

要谈"丢了多少信息"，先得有一个概率对象。作者将 Koopman 表示诱导的轨迹分布写为 $p^{KR}(x_{1:t}|x_0) = \int p(z_0|x_0) \prod_{n=1}^{t} p(z_n|z_{n-1}) p(x_n|z_n) dz_{0:t}$，其中编码器 $p(z_0|x_0)$ 把状态映射到潜空间，线性高斯转移 $p(z_n|z_{n-1}) = \mathcal{N}(z_n|\mathcal{K}z_{n-1}, \Sigma)$ 完成 Koopman 演化，解码器 $p(x_n|z_n)$ 重建状态。这一形式让"编码—线性演化—解码"三段都成为可写互信息、可比较的随机变量链。

在此基础上，作者用全变差距离把 Koopman 表示沿时间步累积放大的预测漂移钉死在一个上界里：

$$\|p(x_{1:t}|x_0) - q^{KR}(x_{1:t}|x_0)\|_{TV} \leq \sqrt{\frac{1}{2}\sum_{n=1}^{t}\big(I(x_{n-1};x_n) - I(z_{n-1};z_n)\big) + \mathcal{E}}$$

式中逐步信息间隙 $I(x_{n-1};x_n) - I(z_{n-1};z_n)$ 度量了潜在线性转移相比真实状态转移丢失的动态耦合信息。这一步把抽象的"预测准不准"翻译成"潜在互信息够不够大"，也解释了为何最大化互信息能直接收紧预测误差——它是后面整个目标函数的理论支点。

**2. 信息分解与谱对应：分清哪些信息该留、哪些该压**

一味抬高互信息会逼着表示坍缩到少数模态，因此作者把潜在互信息 $I(z_t; x_t)$ 按 Koopman 特征值 $\lambda$ 的结构拆成三块：时间一致信息 $I(z_{t-n}; z_t)$ 对应 $|\lambda|\approx 1$ 的模态，是能长期保持、值得保留的成分；快速耗散信息 $I(z_t; x_{t-1}|z_{t-n})$ 对应 $|\lambda|<1$ 的模态，随时间指数衰减；残差信息 $I(z_t; x_t|x_{t-1})$ 无谱对应、属于噪声等不可预测成分，可放心压缩。这套分解把笼统的"压缩 vs 表达"细化为对不同谱成分的差异化处理，为目标函数指明了该奖励谁、惩罚谁。

**3. 信息论 Lagrangian 与可计算损失：把三类信息揉成一个可训练目标**

基于上述分解，论文提出统一优化目标

$$\max_z\ \alpha \log I(z_{t-n};z_t) - \beta\, I(z_t;x_t|z_{t-n}) + \gamma\, S\!\left(\frac{\mathcal{C}}{\text{tr}(\mathcal{C})}\right) + \log p(x_t|z_t)$$

其中 $\alpha$ 项奖励时间一致信息以保住长期可预测的模态，$\beta$ 项压缩耗散与残差成分以保持简洁，$\gamma$ 项用归一化协方差矩阵 $\mathcal{C}/\text{tr}(\mathcal{C})$ 上的 von Neumann 熵 $S(\cdot)$ 顶住模态坍塌、维持有效维度，末项是重建损失。von Neumann 熵的引入是点睛之笔：当表示坍缩到少数方向时熵急剧下降，该项会把梯度推回"维度铺得更开"的方向，从而和最大化互信息形成对偶制衡。

落地时，三项都被写成不依赖特定网络结构的可计算形式：时间一致信息用闭式互信息或 InfoNCE 估计；结构一致性写成 Koopman 线性转移的似然 $\mathbb{E}_{p_\theta(z_n|x_n)}[\log q_\psi(z_n|z_{n-1})]$；von Neumann 熵则从小批量的归一化协方差矩阵直接算出。正因为目标与架构无关，整套框架既能套在 VAE 上、也能套在确定性 AE 上，保证了方法的通用性。

## 实验

### 表1：物理仿真任务性能对比（NRMSE ↓ / SSIM ↑ / SDE ↓）

| 任务 | 指标 | VAE | KAE | KKR | PFNN | **Ours** |
|------|------|-----|-----|-----|------|----------|
| Lorenz 63 | 5-NRMSE | 0.005 | 0.006 | 0.004 | 0.005 | **0.003** |
| Lorenz 63 | 50-NRMSE | 0.019 | 0.023 | 0.017 | 0.017 | **0.013** |
| Lorenz 63 | KLD | 1.047 | 0.464 | 0.342 | 0.293 | **0.285** |
| Kármán Vortex | 5-NRMSE | 0.127 | 0.149 | 0.114 | 0.075 | **0.068** |
| Kármán Vortex | 5-SSIM | 0.743 | 0.719 | 0.868 | 0.920 | **0.936** |
| Kármán Vortex | SDE | 0.538 | 0.620 | 0.799 | 0.278 | **0.256** |
| Dam Flow | 50-NRMSE | 0.034 | 0.046 | 0.031 | – | **0.026** |
| Dam Flow | SDE | 0.563 | 0.488 | 0.373 | – | **0.244** |
| ERA5 Weather | 5-NRMSE | – | 0.055 | 0.058 | 0.049 | **0.028** |
| ERA5 Weather | 5-SSIM | – | 0.666 | 0.664 | 0.697 | **0.867** |

### 表2：消融实验——各正则项对 Pendulum 流形学习的影响

| 配置 | 时间一致性 (α) | 结构一致性 (β) | von Neumann 熵 (γ) | 流形质量 |
|------|:-:|:-:|:-:|------|
| 完整模型 | ✓ | ✓ | ✓ | 最接近真实 $\mathcal{S}^1 \times \mathbb{R}$ |
| α=0 | ✗ | ✓ | ✓ | 退化为散点，无几何结构 |
| β=0 | ✓ | ✗ | ✓ | 流形坍塌，丧失动力学结构 |
| γ=0 | ✓ | ✓ | ✗ | 保留 $\mathcal{S}^1$ 但丢失 $\mathbb{R}$ 维度 |
| 仅增大 α | ↑↑ | ✓ | ✗ | 表示集中于 $\mathcal{S}^1$ 分量 |
| α + γ | ✓ | ✓ | ✓ | 恢复完整 $\mathcal{S}^1 \times \mathbb{R}$ |

## 亮点

- **理论深度突出**：首次建立了 Koopman 表示的信息论框架，将互信息与自回归误差界、谱性质严格关联，揭示了 MI 促进简洁性但可能导致模态坍塌、von Neumann 熵维持表达力的对偶关系
- **信息分解具有洞察力**：将潜在信息分解为时间一致/快速耗散/残差三个成分并与 Koopman 特征值对应，为理解动力系统表示提供了新的分析工具
- **架构无关的通用框架**：所提 Lagrangian 可适配 VAE/AE 结构，且在物理仿真、视觉控制、图结构动力学三类不同任务上均有效
- **消融实验直观有力**：通过 Pendulum 流形可视化清晰展示了每个正则项的作用，理论预测与实验观察完美吻合

## 局限

- **计算开销未充分讨论**：von Neumann 熵需要计算协方差矩阵的特征分解，在高维潜在空间中可能成为瓶颈
- **超参数调节依赖经验**：Lagrangian 乘子 $\alpha, \beta, \gamma$ 的选择对性能有显著影响，但论文未提供系统的选择指南
- **实验规模相对有限**：物理仿真任务维度适中（最大 64×64×2），未在更大规模或更复杂的实际系统上验证
- **线性 Koopman 假设的局限**：框架本质假设潜在演化为线性，对强非线性或混沌系统（如湍流）的适用边界未深入分析
- **缺乏与现代基础模型的对比**：未与基于 Transformer 的时序预测方法（如 FourCastNet）比较

## 相关工作

- **Koopman 算子学习**：KAE (Pan et al., 2023) 是经典的 Koopman 自编码器；KKR (Bevanda et al., 2023) 基于核方法；PFNN (Cheng et al., 2025) 针对混沌系统设计了 Poincaré 流结构——本文从信息论角度统一和超越了这些方法
- **信息瓶颈方法**：Tishby et al. (2000) 提出经典 IB 框架；β-VAE (Burgess et al., 2018) 将 IB 引入变分自编码器——本文将 IB 扩展到动力系统的时序 Koopman 表示
- **动力系统表示学习**：E2C (Banijamali et al., 2019) 和 PCC (Levine et al., 2020) 从 VAE 出发学习可控表示——本文通过信息论约束获得更好的流形结构
- **谱分析与有效维度**：von Neumann 熵在量子信息中用于度量纠缠——本文创新性地将其引入 Koopman 表示防止模态坍塌

## 评分

| 维度 | 分数 (1-10) |
|------|:-----------:|
| 新颖性 | 8 |
| 理论深度 | 9 |
| 实验充分性 | 7 |
| 写作质量 | 8 |
| 实用价值 | 7 |
| **总分** | **7.8** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] The Deleuzian Representation Hypothesis](the_deleuzian_representation_hypothesis.md)
- [\[ICLR 2026\] Gauge-invariant Representation Holonomy](gauge-invariant_representation_holonomy.md)
- [\[ICLR 2026\] Concepts' Information Bottleneck Models](concepts_information_bottleneck_models.md)
- [\[ICLR 2026\] MICLIP: Learning to Interpret Representation in Vision Models](miclip_learning_to_interpret_representation_in_vision_models.md)
- [\[ICLR 2026\] The Geometry of Reasoning: Flowing Logics in Representation Space](the_geometry_of_reasoning_flowing_logics_in_representation_space.md)

</div>

<!-- RELATED:END -->
