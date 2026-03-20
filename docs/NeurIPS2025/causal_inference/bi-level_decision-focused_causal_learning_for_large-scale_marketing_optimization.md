# Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2510.19517](https://arxiv.org/abs/2510.19517)  
**代码**: 无（已在美团部署）  
**领域**: 因果推理/运筹优化  
**关键词**: decision-focused learning, causal inference, bi-level optimization, marketing, uplift modeling

## 一句话总结
提出 Bi-DFCL，通过双层优化框架联合利用观测数据和 RCT 实验数据来训练营销资源分配模型：上层用 RCT 数据的无偏决策损失端到端训练 Bridge Network 来动态纠正下层在观测数据上的偏差，同时设计了基于原始问题的可微代理决策损失（PPL/PIFD）和隐式微分算法，解决了传统两阶段方法的预测-决策不一致和偏差-方差困境。已在美团大规模在线部署。

## 研究背景与动机

1. **领域现状**：在线平台的营销优化（如优惠券分配）是一个经典的资源分配问题。主流方案是两阶段方法（TSM）：第一阶段用 ML 预测个体处理效果，第二阶段用运筹（OR）优化做分配决策。
2. **现有痛点**：(1) **预测-决策不一致**：ML 优化预测精度，OR 优化决策质量，但更好的预测不一定产生更好的决策，尤其在非凸 NP-hard 资源分配问题中预测误差会被放大；(2) **偏差-方差困境**：观测数据（OBS）丰富但有偏（选择偏差、位置偏差），RCT 数据无偏但稀缺高方差。
3. **核心矛盾**：Decision-Focused Learning (DFL) 可以缩小预测-决策差距，但因反事实问题只能在稀缺的 RCT 数据上计算决策损失，反而加剧了偏差-方差困境。
4. **本文要解决什么？** 同时解决 TSM 和现有 DFL 的两个挑战——预测-决策对齐 + 偏差-方差平衡。
5. **切入角度**：利用 RCT 数据的无偏性来指导 OBS 数据上的学习方向——将决策损失和预测损失分别分配到双层优化的上层和下层。
6. **核心 idea 一句话**：双层优化中，上层用 RCT 数据上的无偏决策损失训练 Bridge Network，下层用 OBS 数据上的（被 Bridge 纠正的）预测损失训练 Target Network，实现数据互补和目标对齐。

## 方法详解

### 整体框架
Target Network $\mathcal{F}_\theta$ 在大量 OBS 数据上训练预测损失（下层），Bridge Network $\mathcal{G}_\phi$ 在 RCT 数据上训练决策损失（上层）。Bridge 输出门控系数 $w$，混合 Teacher Network 和 Target Network 的预测来生成反事实伪标签，动态纠正 OBS 数据的偏差。

### 关键设计

1. **无偏决策损失估计**:
   - 做什么：利用 RCT 数据的随机化性质推导决策质量的无偏估计量。
   - 核心思路：通过强可忽略性假设，将决策损失重写为 $\mathcal{L}_\text{DL} = -\mathbb{E}[\frac{N}{N_{t_i}} \cdot z^*_{it_i} \cdot r_{it_i}]$，其中权重 $N/N_{t_i}$ 修正了 RCT 中不同处理组的不平衡。
   - 设计动机：真实的决策损失因反事实不可观测无法直接计算，无偏估计量让我们可以在 RCT 数据上可靠评估决策质量。

2. **Primal Policy Learning (PPL) 代理损失**:
   - 做什么：将离散的 argmax 决策松弛为连续可微的损失。
   - 核心思路：用 Softmax 松弛替代 indicator 函数：$z'_{it_i} = \frac{\exp[\hat{r}_{it_i} - \lambda^* \hat{c}_{it_i}]}{\sum_j \exp[\hat{r}_{ij} - \lambda^* \hat{c}_{ij}]}$。这直接在原始问题的约束（特定预算 $B$）下操作，比 DFCL 的对偶决策损失（考虑所有预算）更对齐实际场景。
   - 设计动机：离散优化的 indicator 函数不可微，需要可微代理来让梯度流过决策层。

3. **双层优化 + 隐式微分**:
   - 做什么：解决上层梯度计算中的 Jacobian $\partial\theta^*/\partial\phi$ 不可解析的问题。
   - 核心思路：利用隐函数定理，从下层最优性条件 $\partial\mathcal{L}_\text{PL}/\partial\theta|_{\theta=\theta^*} = 0$ 出发，求解 Hessian 方程得到 Jacobian。用共轭梯度算法避免显式计算 Hessian 逆，仅需 Hessian-vector product。
   - 设计动机：显式微分（unrolling gradient steps）依赖优化路径且梯度容易消失；隐式微分与路径无关，更稳定。

4. **Bridge Network 门控机制**:
   - 做什么：自适应混合 Teacher 和 Target 的预测来生成 OBS 数据的反事实伪标签。
   - 核心思路：门控系数 $w = \text{sigmoid}(\mathcal{G}_\phi(i,j))$ 控制对 Teacher（无偏但高方差）和 Target（有偏但低方差）的依赖程度。通过上层决策损失的梯度信号动态调整。
   - 设计动机：直接用 Teacher 预测做伪标签高方差（RCT 少），直接用 Target 自身预测没有修正作用。Bridge 在两者之间自适应平衡。

### 损失函数 / 训练策略
下层：MSE 预测损失在 OBS 数据上。上层：PPL 或 PIFD 决策代理损失在 RCT 数据上。每 k=5 个 batch 更新一次 Bridge Network。

## 实验关键数据

### 主实验

| 方法 | CRITEO-UPLIFT | 美团营销I | 美团营销II |
|------|-------------|----------|----------|
| TSM (最优 uplift) | baseline | baseline | baseline |
| DHCL | 改善但过拟合 | 改善 | 改善 |
| DFCL | 改善 | 改善 | 改善 |
| **Bi-DFCL** | **显著优于所有** | **显著优于所有** | **显著优于所有** |

### 在线 A/B 测试
- 在美团大规模线上 A/B 测试中，Bi-DFCL 显示统计显著的收入提升
- 已在多个真实营销场景中部署

### 消融实验

| 配置 | 说明 |
|------|------|
| w/o 双层优化（只在 RCT 上训练） | 高方差，过拟合 |
| w/o Bridge Network | 无法纠正 OBS 偏差 |
| PPL vs PIFD | PIFD 保留原始优化景观，有时更优 |
| 显式微分 vs 隐式微分 | 隐式微分更稳定 |

### 关键发现
- **双层优化框架有效解决偏差-方差困境**：OBS 提供低方差泛化信号，RCT 提供无偏决策指导。
- **原始问题的决策损失比对偶更有效**：PPL 直接在特定预算约束下操作，比 DFCL 的对偶损失更实际。
- **隐式微分优于显式微分**：更稳定，不受优化路径影响。
- **自适应平衡替代手动调参**：Bridge 自动学习 OBS 和 RCT 的混合比例，消除了 DFCL 中 $\alpha$ 的手动搜索。

## 亮点与洞察
- **双层优化将数据类型和损失类型精确对齐**：决策损失-RCT（保证无偏），预测损失-OBS（保证泛化）——这种设计比简单混合两个损失在同一数据上要深刻得多。
- **Bridge Network 的门控机制**有效实现了"用 RCT 的无偏信号修正 OBS 学习方向"，是一个可迁移到其他因果推断场景的设计模式。
- **大规模工业部署验证**：在美团这样的大规模平台上的在线 A/B 测试提供了强有力的实际验证。

## 局限性 / 可改进方向
- **假设 RCT 数据可获得**：很多场景中 RCT 数据获取成本极高或不可行。
- **MCKP 的 NP-hard 性质**：虽然 Lagrangian 松弛提供了近似解，但在极端场景下近似质量可能下降。
- **处理维度有限**：当前只考虑了有限的离散处理选项，未扩展到连续处理空间。

## 相关工作与启发
- **vs DFCL (Zhou et al.)**: DFCL 只用 RCT 数据且需手动调 $\alpha$，Bi-DFCL 联合利用 OBS+RCT 且自适应平衡。
- **vs 标准 DFL**: 通用 DFL 方法不处理反事实和营销特定约束，Bi-DFCL 针对性设计了无偏估计量和代理损失。
- **vs Meta-learning**: 双层优化在形式上类似 MAML，但目标不同。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 双层优化 + Bridge Network 的设计新颖深刻
- 实验充分度: ⭐⭐⭐⭐⭐ 公开数据集 + 工业数据集 + 在线 A/B 测试
- 写作质量: ⭐⭐⭐⭐ 技术细节完整但公式密集
- 价值: ⭐⭐⭐⭐⭐ 已在大规模平台部署，实际价值高
