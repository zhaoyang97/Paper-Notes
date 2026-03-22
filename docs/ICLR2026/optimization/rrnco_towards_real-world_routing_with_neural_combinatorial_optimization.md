# RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization

**会议**: ICLR 2026  
**arXiv**: [2503.16159](https://arxiv.org/abs/2503.16159)  
**代码**: [https://github.com/ai4co/real-routing-nco](https://github.com/ai4co/real-routing-nco)  
**领域**: 组合优化 / 神经路由规划  
**关键词**: Neural Combinatorial Optimization, Vehicle Routing Problem, Asymmetric Routing, Sim-to-Real Gap, Attention-Free Module

## 一句话总结

提出 RRNCO 架构，通过自适应节点嵌入（ANE）和神经自适应偏置（NAB）两大创新，首次在深度路由框架中联合建模非对称距离、时长和方向角，并构建了基于 100 个真实城市的 VRP 基准数据集，显著缩小了 NCO 方法从仿真到真实世界部署的差距。

## 研究背景与动机

1. **车辆路由问题（VRP）是物流优化的核心**：VRP 是一类 NP-hard 组合优化问题，广泛应用于末端配送、灾害救援、城市出行等场景。2025 年全球物流市场规模超过 10 万亿美元，路由效率的提升具有巨大的成本节约和环保价值。

2. **NCO 方法在合成数据上表现优秀但脱离实际**：神经组合优化通过强化学习自动学习启发式策略，在合成 VRP 实例上取得了令人印象深刻的结果，但主要依赖简化的对称欧式距离数据，无法反映真实道路网络的非对称性（单行道、交通模式、转弯限制等）。

3. **Sim-to-Real Gap 的两个根源**：
   - **数据层面**：训练和测试使用的合成数据集（如 TSPLIB、CVRPLIB）假设对称距离 $d_{ij}=d_{ji}$，与现实不符
   - **架构层面**：现有 NCO 架构基于节点级 Transformer，本质上无法高效处理边特征（非对称距离/时长矩阵）

4. **已有真实数据集的局限**：少数已有工作依赖商业 API、静态不可在线生成、速度慢且常不公开，同时缺少行驶时长信息。

5. **现有边特征编码方法不足**：MatNet 的行/列嵌入、GOAL 的交叉注意力虽然引入了部分边信息，但通常只处理单一代价矩阵，无法融合距离、时长、方向角等多模态非对称特征。

## 方法详解

### 整体框架

RRNCO 采用编码器-解码器架构。编码器构建综合节点表示，解码器自回归地生成路由解。核心创新集中在编码器端，包含两大模块：

- **Adaptive Node Embedding (ANE)**：融合坐标与距离矩阵信息
- **Neural Adaptive Bias (NAB)**：为 Adaptation Attention-Free Module (AAFM) 提供学习得到的非对称偏置

### 关键设计

**1. 自适应节点嵌入（ANE）**

- **做什么**：将距离矩阵中的边特征与节点坐标特征融合为综合的节点表示
- **核心思路**：对距离矩阵做概率加权采样 + Contextual Gating 融合
- **设计动机**：直接处理完整 $N \times N$ 距离矩阵计算量过大，而仅用坐标会丢失真实距离信息

具体步骤：
- 对距离矩阵 $\mathbf{D} \in \mathbb{R}^{N \times N}$，按距离倒数采样 $k$ 个邻居节点：$p_{ij} = \frac{1/d_{ij}}{\sum_{j} 1/d_{ij}}$
- 采样距离通过线性投影得到 $\mathbf{f}_{\text{dist}}$，坐标通过另一线性投影得到 $\mathbf{f}_{\text{coord}}$
- 通过 MLP 学习门控权重 $\mathbf{g} = \sigma(\text{MLP}([\mathbf{f}_{\text{coord}}; \mathbf{f}_{\text{dist}}]))$
- 融合：$\mathbf{h} = \mathbf{g} \odot \mathbf{f}_{\text{coord}} + (1 - \mathbf{g}) \odot \mathbf{f}_{\text{dist}}$
- 生成行嵌入 $\mathbf{h}^r$ 和列嵌入 $\mathbf{h}^c$（借鉴 MatNet），分别编码非对称关系的两个方向

**2. 神经自适应偏置（NAB）**

- **做什么**：替代 AAFM 中手工设计的偏置 $A = -\alpha \cdot \log(N) \cdot d_{ij}$，学习融合距离、方向角和时长的非对称偏置矩阵
- **核心思路**：分别嵌入三种边特征，通过 softmax 门控融合，输出标量偏置矩阵
- **设计动机**：手工偏置只能编码距离，无法建模方向（单行道效应）和时长与距离的非线性关系

$$\mathbf{D}_{emb} = \text{ReLU}(\mathbf{D}\mathbf{W}_D)\mathbf{W}'_D$$
$$\mathbf{\Phi}_{emb} = \text{ReLU}(\mathbf{\Phi}\mathbf{W}_\Phi)\mathbf{W}'_\Phi$$
$$\mathbf{T}_{emb} = \text{ReLU}(\mathbf{T}\mathbf{W}_T)\mathbf{W}'_T$$

其中 $\phi_{ij} = \text{arctan2}(y_j - y_i, x_j - x_i)$ 编码方向角。然后通过带可学习温度 $\tau$ 的 softmax 门控融合，最终投影为标量偏置：$\mathbf{A} = \mathbf{H}\mathbf{w}_{out} \in \mathbb{R}^{B \times N \times N}$

**3. AAFM（Adaptation Attention-Free Module）**

- 基于 Zhou et al. (2024a) 的无注意力模块
- 操作定义：$\text{AAFM}(Q,K,V,A) = \sigma(Q) \odot \frac{\exp(A) \cdot (\exp(K) \odot V)}{\exp(A) \cdot \exp(K)}$
- 将 NAB 产生的 $\mathbf{A}$ 作为偏置注入，使模块能感知非对称路由约束

### 损失函数 / 训练策略

- 使用 REINFORCE + POMO 基线的策略梯度方法
- 训练目标：$\max_\theta J(\theta) = \mathbb{E}_{\mathbf{x} \sim \mathcal{D}} \mathbb{E}_{\mathbf{a} \sim \pi_\theta(\cdot|\mathbf{x})} [R(\mathbf{a}, \mathbf{x})]$
- 奖励 $R$ 定义为路径代价的负值
- 利用 OSRM 引擎从 100 个城市在线采样训练实例，支持高效的在线数据生成

## 实验关键数据

### 主实验

在真实世界路由基准上的表现（ATSP 任务，50 节点）：

| 方法 | In-Dist Cost | Gap(%) | OOD-City Cost | Gap(%) | 时间 |
|------|-------------|--------|---------------|--------|------|
| LKH3 | 38.387 | *(best) | 38.903 | *(best) | 1.6h |
| POMO | 51.512 | 34.19 | 50.594 | 30.05 | 10s |
| MatNet | 39.915 | 3.98 | 40.548 | 4.23 | 27s |
| GOAL | 41.976 | 9.35 | 42.590 | 9.48 | 91s |
| **RRNCO** | **39.078** | **1.80** | **39.785** | **2.27** | **23s** |

ACVRP 任务（有容量约束）：

| 方法 | In-Dist Cost | Gap(%) | OOD-City Cost | Gap(%) | 时间 |
|------|-------------|--------|---------------|--------|------|
| PyVRP | 69.739 | *(best) | 70.488 | *(best) | 7h |
| MatNet | 74.801 | 7.26 | 75.722 | 7.43 | 30s |
| AAFM | 76.663 | 9.93 | 77.811 | 10.39 | 11s |
| **RRNCO** | **72.145** | **3.45** | **73.010** | **3.58** | **26s** |

### 消融实验

| 配置 | ATSP Gap(%) | ACVRP Gap(%) |
|------|------------|-------------|
| Full RRNCO | 1.80 | 3.45 |
| 去掉 NAB（用手工偏置） | ~9.35 | ~9.93 |
| 去掉 ANE（仅坐标） | ~34.19 | ~23.16 |
| 去掉方向角 $\Phi$ | 性能下降显著 | - |
| 去掉时长矩阵 $T$ | 性能下降显著 | - |

### 关键发现

1. **RRNCO 在所有真实世界任务上均为 NCO 方法 SOTA**：在 ATSP/ACVRP/ACVRPTW 三种任务、In-Distribution/OOD-City/OOD-Cluster 三种分布下全面领先
2. **与传统求解器差距极小**：ATSP 仅 1.8% 差距于 LKH3，但速度快约 250 倍（23s vs 1.6h）
3. **NAB 是关键创新**：GOAL 和 AAFM 使用手工偏置的差距分别为 9.35% 和 19.81%，而 RRNCO 仅 1.80%
4. **联合建模距离+时长+方向角的收益巨大**：去掉任一模态性能均显著下降
5. **泛化能力强**：OOD-City 和 OOD-Cluster 场景仅有微小性能损失

## 亮点与洞察

1. **首次将距离、时长、方向角三模态联合建模引入 NCO**：NAB 机制不仅技术新颖，更揭示了真实路由中这三者的耦合关系对求解质量的关键影响
2. **概率加权采样是高效处理距离矩阵的优雅方案**：避免了 $O(N^2)$ 的完整矩阵处理，同时保留了关键的非对称邻域信息
3. **开源数据集的价值**：100 个城市的真实数据集 + 在线采样框架，为后续 NCO 研究提供了标准化的真实世界基准
4. **Contextual Gating 的通用性**：ANE 和 NAB 中的门控融合机制可推广到其他需要融合异构特征的场景

## 局限性 / 可改进方向

1. **目前仅考虑静态路由**：未涉及动态交通流、实时路况变化等更复杂的真实场景
2. **节点规模有限**：实验主要在 50-100 节点范围，大规模（1000+）真实场景的可扩展性未验证
3. **数据集覆盖面**：虽然 100 个城市已是显著进步，但不同区域（农村、高速公路）的道路类型覆盖仍可扩展
4. **与传统求解器的差距**：在大规模实例上与 LKH3/PyVRP 的差距可能放大
5. **多目标优化**：实际物流中需要同时优化距离、时间、油耗等多目标，目前只优化单一代价函数

## 相关工作与启发

- **MatNet (Kwon et al., 2021)**：首次引入行/列嵌入处理非对称性，RRNCO 的 ANE 在此基础上增加了距离采样和门控融合
- **AAFM (Zhou et al., 2024a)**：提供了无注意力的高效框架，但使用手工偏置；RRNCO 的 NAB 将其升级为数据驱动的学习偏置
- **GOAL (Drakulic et al., 2024)**：使用交叉注意力编码边信息，但只处理单一代价矩阵
- 启发：Sim-to-Real Gap 问题不仅存在于路由领域，在机器人控制、自动驾驶等领域也普遍存在；RRNCO 的数据生成+架构创新的双管齐下思路具有普适价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ — NAB 机制首次联合建模三种非对称特征，概念清晰且实现优雅；ANE 的概率采样也有一定新意
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三种任务 × 三种分布 × 多种基线，消融实验完整，真实城市数据集令人信服
- **写作质量**: ⭐⭐⭐⭐ — 问题动机清晰，框架图直观，实验表格丰富；数学公式密集但组织有序
- **价值**: ⭐⭐⭐⭐⭐ — 开源数据集和代码为 NCO 社区提供了首个真实世界标准基准，推动该领域从玩具问题走向实际应用
