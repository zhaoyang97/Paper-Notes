---
title: "AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization"
authors: "Saeed Hedayatian, Stefanos Nikolaidis"
affiliations: "University of Southern California, Archimedes AI"
venue: "ICLR 2026"
arxiv: "2506.05634"
code: "https://github.com/conflictednerd/autoqd-code"
tags: ["quality-diversity", "occupancy measure", "random Fourier features", "MMD", "CMA-MAE", "behavior descriptor", "policy diversity"]
rating:
  novelty: 4
  experiments: 4
  writing: 4
  value: 4
---

# AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization

## 一句话总结

提出 AutoQD，利用占用度量 (occupancy measure) 的随机 Fourier 特征嵌入自动生成行为描述子 (behavioral descriptor)，替代传统 QD 优化中的手工设计描述子，在 6 个连续控制任务上展现了强大的多样化策略发现能力。

## 研究动机

Quality-Diversity (QD) 优化旨在发现一组既高性能又行为多样的解（如机器人控制器中的行走、跳跃、爬行等不同行为）。QD 算法的核心依赖是**行为描述子 (BD)**——将策略映射到低维行为空间的函数。然而现有方法存在关键瓶颈：

1. **手工设计成本高**：传统 QD 方法（如 MAP-Elites）需要领域专家预定义 BD（如双足机器人的脚部接触模式），随着任务复杂度增加，设计 BD 越来越困难
2. **探索被预设维度约束**：手工 BD 限定了多样性的定义，可能错过领域专家未能预见的有趣行为变化
3. **现有自动化方法理论基础薄弱**：Aurora 使用自编码器学习状态表示作为 BD，但缺乏行为差异性的理论保证；DIAYN 等无监督 RL 方法需要预设技能数量、且不利用任务奖励

核心问题：**如何在无需领域知识的前提下，自动生成理论有保证的行为描述子？**

## 方法详解

### 核心思想：占用度量 ↔ 策略唯一对应

AutoQD 的关键洞察来自 RL 理论：在标准假设下，**策略与其占用度量 (occupancy measure) 一一对应**。占用度量 $\rho^\pi(\mathbf{s}, \mathbf{a})$ 刻画了策略 $\pi$ 下状态-动作对的折扣访问频率分布，是策略行为的完整刻画。因此，度量两个策略行为差异可以转化为度量它们占用度量之间的距离。

### 第一步：随机 Fourier 特征嵌入

选用 **Maximum Mean Discrepancy (MMD)** 配合高斯核来衡量占用度量之间的距离。然而高斯核对应无穷维特征映射，无法直接计算。AutoQD 使用 **Random Fourier Features (RFF)** 近似：

- 采样随机频率 $\mathbf{w}_i \sim \mathcal{N}(0, \sigma^{-2}I)$ 和偏移 $\mathbf{b}_i \sim \mathcal{U}(0, 2\pi)$
- 构造 $D$ 维特征映射 $\phi(\mathbf{s}, \mathbf{a}) = \sqrt{2/D}[\cos(\mathbf{w}_1^T[\mathbf{s};\mathbf{a}] + \mathbf{b}_1), \ldots, \cos(\mathbf{w}_D^T[\mathbf{s};\mathbf{a}] + \mathbf{b}_D)]$
- 策略嵌入 $\psi^\pi$ 为所有采样轨迹的折扣加权特征均值

**理论保证 (Theorem 1)**：嵌入间的 $\ell_2$ 距离以高概率逼近真实 MMD 距离，近似误差随样本数 $n$ 和嵌入维度 $D$ 指数衰减，且状态-动作维度 $d$ 仅出现在指数项分母中，说明扩展到高维问题只需 $D$ 线性增长。

### 第二步：cwPCA 降维

$D$ 维嵌入对 QD 存档来说维度过高（存档格子数随维度指数增长）。AutoQD 使用**校准加权 PCA (cwPCA)** 投影到 $k$ 维行为描述子空间：

- **加权**：根据策略适应度 (fitness) 对嵌入加权，使高质量策略对主成分方向影响更大，促进在高性能行为附近的探索
- **校准**：缩放各轴使投影值落在 $[-1, 1]$，保证存档边界稳定

最终行为描述子为仿射变换 $\mathrm{desc}(\pi) = \mathbf{A}\psi^\pi + \mathbf{b}$。

### 第三步：迭代优化

AutoQD 与 CMA-MAE（一种 state-of-the-art 黑盒 QD 算法）结合，交替执行：

1. **QD 优化**：使用当前 BD 在存档中发现多样策略（CMA-ES 采样策略参数 → 环境评估 → 根据存档改进排名更新分布）
2. **BD 更新**：定期基于存档中策略的嵌入重新计算 cwPCA 投影，更新行为描述子定义

这种迭代机制使 BD 随着存档扩展而自适应调整，捕捉逐渐丰富的行为变化。

## 实验设计

### 环境与基线

在 6 个连续控制任务上评估：**Ant, HalfCheetah, Hopper, Swimmer, Walker2d, BipedalWalker**（前 5 个为 MuJoCo 基准）。对比 5 个基线：

| 方法 | 策略 |
|------|------|
| RegularQD | 使用手工设计的 BD + CMA-MAE |
| Aurora | 自编码器学习状态表示作为 BD |
| LSTM-Aurora | LSTM 编码完整轨迹作为 BD |
| DvD-ES | 进化策略联合优化性能与多样性 |
| SMERL | 基于 SAC + 判别器鼓励技能多样性 |

### 评估指标

- **GT QD Score**：使用手工 BD 定义的存档计算 QD 分数，评估在专家定义行为空间中的覆盖质量
- **Vendi Score (VS)**：基于占用度量嵌入的成对核矩阵计算有效种群大小，度量种群多样性
- **Quality-Weighted Vendi Score (qVS)**：VS 乘以平均适应度，综合衡量质量与多样性

## 主要结果

### 策略发现

AutoQD 在 6 个环境的 18 个指标中取得 **12 个最优**：

- **Ant**：GT QD Score 361.43（RegularQD 仅 182.58），VS 和 qVS 也大幅领先，说明自动 BD 发现了比手工 BD 更丰富的行为多样性
- **Swimmer**：GT QD Score 21.31 vs RegularQD 11.09，VS 达 16.92（RegularQD 4.67），展现压倒性优势
- **HalfCheetah**：GT QD Score 最高但 qVS 较低——发现了大量通过微妙关节运动"滑行"的新颖策略，虽多样但速度较慢导致奖励低
- **Walker2d**：排名第二，过度关注脚部关节导致在手工 BD 定义的多样性维度上稍逊

### 动态适应实验

在 BipedalWalker 上测试摩擦系数和质量变化下的适应能力：

- AutoQD 种群在两种变化下均取得**最高 AUC**（摩擦 1429.66 vs 第二名 Aurora 1309.41；质量 295.65 vs LSTM-Aurora 271.83）
- 在严格阈值 $p=0.9$ 下，AutoQD 始终拥有最多成功适应的策略
- 验证了行为多样性带来的鲁棒适应优势

## 局限性

1. **高随机性环境**中准确估计策略嵌入需要大量轨迹，降低采样效率
2. **低维 BD** 可能使探索集中于简单稳定行为，阻碍复杂行为发现
3. **核带宽固定**，动态调整可能进一步提升嵌入质量
4. 继承 CMA-MAE 的可扩展性限制，对大型策略网络和高维行为空间挑战较大

## 个人评价

### 优点
- **理论优雅**：基于占用度量-策略一一对应的经典 RL 理论，RFF 近似 MMD 有严谨的收敛保证（Theorem 1），是无监督 QD 领域少见的理论驱动工作
- **方法简洁有效**：RFF 嵌入 + cwPCA + CMA-MAE 的流水线每步都有明确动机，无需训练额外神经网络
- **实验全面**：6 个环境、5 个基线、3 个指标、适应性实验，评估维度多角度覆盖
- **在无领域知识的前提下超越使用手工 BD 的 RegularQD**，充分证明了自动 BD 生成的价值

### 不足
- 核带宽 $\sigma$ 的选择缺乏自适应机制，且不同环境可能需要不同的超参
- 未与梯度基 QD 方法（PGA-ME, PPGA）结合实验，留待未来工作
- HalfCheetah 和 Walker2d 上 qVS 不占优，说明质量-多样性权衡仍有提升空间
