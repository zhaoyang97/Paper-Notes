# Inverse Optimization Latent Variable Models for Learning Costs Applied to Route Problems

**会议**: NeurIPS 2025  
**arXiv**: [2509.15999](https://arxiv.org/abs/2509.15999)  
**代码**: 无  
**领域**: 组合优化 / 逆优化 / 生成模型  
**关键词**: inverse optimization, latent variable model, Fenchel-Young loss, route planning, TSP, VAE

## 一句话总结
提出 IO-LVM（Inverse Optimization Latent Variable Model），用 VAE 式编码器映射观测的 COP 解到潜在成本空间，通过 Fenchel-Young 损失和黑盒求解器（Dijkstra/TSP solver）在解码端保证可行性，无需 agent 标签即可从路径数据中学到成本函数的分布，成功不可监督地分离不同 agent 的导航偏好。

## 研究背景与动机
1. **领域现状**：逆优化（Inverse Optimization）从观测的最优解反推目标/成本函数，在路径规划、IRL 等领域广泛应用。标准方法假设单一成本函数或依赖上下文条件。
2. **现有痛点**：
   - **单成本假设**：现实中不同 agent（如不同出租车司机、不同船舶）有不同的隐含成本偏好，标准 IO/IRL 方法需要 agent 标签来区分
   - **VAE 无法保证可行性**：对结构化输出（如图上的路径），VAE 解码出的样本难以满足约束（如连通性、到达指定终点）
   - **离散解的梯度问题**：COP 的离散解对参数的梯度几乎处处为零，阻碍端到端学习
3. **核心矛盾**：需要一个生成模型既能学习成本函数的分布（不同 agent），又能保证生成的解满足 COP 约束，且能通过非可微求解器反传梯度
4. **核心 idea**：用 VAE 编码器映射到低维潜在**成本空间**（而非解空间），解码器 = 参数化的成本映射 + 黑盒 COP 求解器。通过 Fenchel-Young 损失的扰动梯度估计绕过求解器的不可微问题

## 方法详解

### 整体框架
观测路径 $\mathbf{x}_i$（+ 问题需求 $\mathbf{p}_i$，如起终点）→ 编码器 $q_\phi(\mathbf{z}|\mathbf{x})$ → 潜在空间 $\mathcal{Z} \subseteq \mathbb{R}^k$ → 解码器 $g_\theta: \mathcal{Z} \to \mathcal{Y}$（映射到成本空间）→ 黑盒求解器 $\omega(\mathbf{y}^\theta + \epsilon, \mathbf{p})$（Dijkstra/TSP solver）→ 重建路径 $\hat{\mathbf{x}}$

### 关键设计

1. **成本空间作为潜在表示**
   - 做什么：VAE 的潜在空间编码的不是路径的几何形状，而是生成该路径的成本函数
   - 核心思路：相同 agent 的不同路径（不同起终点）由相似的成本函数生成 → 在潜在空间中聚类在一起。而普通 VAE 按路径几何聚类，无法分离成本因素
   - 设计动机：成本函数是更底层、更紧凑的表示——几何空间高维且依赖起终点，但成本空间可以很低维（如 2D 就够分离 3 类 agent）

2. **Fenchel-Young 损失 + 扰动梯度估计**
   - 做什么：绕过非可微求解器，获得关于成本参数 $\theta$ 的梯度
   - 核心思路：$l_{FY}^\epsilon(\mathbf{y}, \mathbf{x}) = \langle \mathbf{y}, \mathbf{x} \rangle - \mathbb{E}_\epsilon[\min_{\mathbf{x} \in \mathcal{X}} \langle \mathbf{y} + \epsilon, \mathbf{x} \rangle]$。梯度为 $\nabla_\mathbf{y} l_{FY} = \mathbf{x} - \hat{\mathbf{x}}_\epsilon$——观测路径与扰动成本下求解器输出的差。通过链式法则传到 $\theta$
   - SGD 技巧：利用期望线性性，每个样本只需运行求解器一次（而非蒙特卡洛多次），通过双重期望（$\mathbb{E}_{q_\phi} \mathbb{E}_\epsilon$）得到无偏梯度估计
   - 设计动机：计算效率是关键——求解器是训练瓶颈，每样本单次调用使方法可扩展

3. **约束保证的生成**
   - 做什么：通过在解码端放置真实 COP 求解器，生成的路径一定满足约束（如连通性、经过所有节点）
   - 对比 VAE：标准 VAE 的解码器是神经网络，无法保证输出是合法路径，尤其在 TSP 等复杂约束下
   - 设计动机：对路径规划等安全关键应用，可行性保证是必须的

### 损失函数 / 训练策略
- 总损失：$l(\theta, \phi) = \mathbb{E}_{q_\phi}[l_{FY}(\mathbf{y}^\theta, \mathbf{x})] + \beta D_{KL}(q_\phi \| P(\mathbf{z}))$
- $\beta$-VAE 风格的 KL 权重：同时控制后验坍缩和去噪能力
- 重参数化技巧 + SGD
- 求解器：Dijkstra（最短路径）/ TSP solver（哈密顿回路）

## 实验关键数据

### 数据集
- **合成 Waxman 图**：700 节点，3 种 agent（南部高成本/北部高成本/无偏），6000 条路径
- **船舶 AIS 数据**：丹麦海域，2513 节点，2500 条船舶轨迹
- **出租车轨迹**：旧金山 Cabspotting，1125 节点，101344 条轨迹
- **TSPLIB**：burma14 和 bayg29 图的哈密顿回路

### 主要results

| 任务 | IO-LVM 特点 | VAE 对比 |
|------|-----------|---------|
| 潜在空间聚类 | 按成本/agent 聚类（无监督分离 3 类 agent） | 按路径几何聚类（无法分离 agent） |
| 路径重建（TSP） | 可行解，高匹配率 | 常产生不可行解 |
| 路径分布预测 | 通过 KDE 采样生成合理路径分布 | — |
| 未见起终点泛化 | 从潜在空间采样 + 求解器生成新路径 | 无约束保证 |

### 关键发现
- 2D 潜在空间就能无监督分离 3 种不同成本偏好的 agent（合成数据）
- 船舶数据中发现：潜在空间的某些区域对应更宽的船舶——宽船避开厄勒海峡走大贝尔特海峡（虽然更远但更安全），这是 IO-LVM 自动发现的未标注物理因素
- IO-LVM 可用于异常检测（测试路径的潜在编码偏离训练分布）和去噪（通过 $\beta$ 控制正则化强度）

## 亮点与洞察
- **"解码器中放求解器"的设计范式**：这是一个优雅的思路——让生成模型的约束保证完全委托给经典求解器，神经网络只负责学习成本空间的分布。可推广到任何有高效求解器的 COP
- **成本空间 vs 解空间的洞察**：传统 VAE 学解空间的分布，IO-LVM 学成本空间的分布——后者更紧凑且物理意义更明确。一个成本函数可以生成无穷条路径（不同起终点），但反过来不行
- **无监督 agent 分离**：不需要知道"谁走的这条路"，IO-LVM 自动从路径中反推成本偏好并聚类

## 局限性 / 可改进方向
- 依赖高效求解器——TSP 等 NP-hard 问题的大规模实例可能成为瓶颈
- 假设观测路径是成本下的最优解（最短路径），实际中人类决策可能次优
- Fenchel-Young 梯度估计的 Proposition 1 需要所有可行路径等长，这只在 TSP（哈密顿回路）中严格成立，在最短路径中是近似
- 缺乏与其他逆优化/IRL 方法的定量对比（如 MaxEntIRL、可微 Floyd-Warshall 等）

## 相关工作与启发
- **vs 可微 Floyd-Warshall (Vlastelica 2020)**：直接使图算法可微，但随图规模增大可扩展性差；IO-LVM 用黑盒求解器 + Fenchel-Young 梯度，更灵活
- **vs 标准逆优化**：假设单一成本函数，需要 agent 标签；IO-LVM 自动学习成本分布
- **vs VAE for COPs**：标准 VAE 无约束保证且学几何分布；IO-LVM 通过求解器保证可行性且学成本分布

## 评分
- 新颖性: ⭐⭐⭐⭐ VAE + COP 求解器的架构设计新颖，成本空间作为潜在表示的洞察有价值
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据集覆盖两种 COP，定性+定量分析
- 写作质量: ⭐⭐⭐⭐ 框架清晰，ship 例子的物理发现很有说服力
- 价值: ⭐⭐⭐⭐ 对路径分析、行为建模和异常检测有实际应用价值
