# DiffusionNFT: Online Diffusion Reinforcement with Forward Process

**会议**: ICLR 2026  
**arXiv**: [2509.16117](https://arxiv.org/abs/2509.16117)  
**代码**: [https://research.nvidia.com/labs/dir/DiffusionNFT](https://research.nvidia.com/labs/dir/DiffusionNFT)  
**领域**: 扩散模型 / 强化学习对齐  
**关键词**: online RL, forward process, negative-aware finetuning, flow matching, CFG-free  

## 一句话总结
提出 DiffusionNFT，一种全新的扩散模型在线 RL 范式：不在反向采样过程上做策略优化（如 GRPO），而是在前向过程上通过 flow matching 目标对正样本和负样本做对比式训练，定义隐式的策略改进方向，比 FlowGRPO 快 3-25×，且无需 CFG。

## 研究背景与动机

1. **领域现状**：FlowGRPO/DanceGRPO 将反向采样离散化为 MDP，使用 SDE 采样器 + GRPO 实现扩散模型的在线 RL 对齐，取得了显著效果。

2. **现有痛点**：GRPO 式方法有三大根本性限制：(a) **前向不一致**——只优化反向过程，模型可能退化为级联高斯；(b) **求解器限制**——只能用一阶 SDE 采样器，无法使用更高效的 ODE/高阶求解器；(c) **CFG 复杂性**——需要同时优化有条件和无条件模型，效率低且工程复杂。

3. **核心矛盾**：反向过程 RL 需要似然估计，但扩散模型的似然不可精确计算。离散化近似引入系统性偏差。

4. **本文要解决什么？** 能否在**前向过程**（flow matching 目标）上做 RL，完全避开似然估计、求解器限制和 CFG 依赖？

5. **切入角度**：一个扩散策略有唯一的前向过程但多个反向过程（不同求解器）。在前向过程上优化更本质——直接用正/负样本对比定义策略改进方向，嵌入 flow matching 的监督学习框架中。

6. **核心 idea 一句话**：把 RL 信号转化为前向过程中正负样本的对比式 flow matching 目标，用隐式参数化将 reinforcement guidance 直接整合进单一策略模型。

## 方法详解

### 整体框架

每轮迭代：
1. **数据收集**：用任意求解器从当前模型采样 $K$ 张图像，对每张用奖励函数评分
2. **正负划分**：每张图像以概率 $r$ 归入正集 $\mathcal{D}^+$，以概率 $1-r$ 归入负集 $\mathcal{D}^-$
3. **策略优化**：在前向过程上同时训练正分支（flow matching on $\mathcal{D}^+$）和负分支（flow matching on $\mathcal{D}^-$），通过隐式参数化提取改进方向
4. **只需保存干净图像**——无需存储整个采样轨迹

### 关键设计

1. **改进方向定理 (Theorem 3.1)**:
   - 做什么：证明正/负/旧策略三个速度场之间的差异方向成比例
   - 核心思路：$\Delta := \alpha(\mathbf{x}_t)[\mathbf{v}^+(\mathbf{x}_t) - \mathbf{v}^{\text{old}}(\mathbf{x}_t)] = [1-\alpha(\mathbf{x}_t)][\mathbf{v}^{\text{old}}(\mathbf{x}_t) - \mathbf{v}^-(\mathbf{x}_t)]$，其中 $\alpha$ 是与正策略密度比相关的标量
   - 设计动机：建立了"远离负样本 = 接近正样本"的等价关系，形式类似 CFG，但来自 RL 原理

2. **策略优化目标 (Theorem 3.2)**:
   - 做什么：设计一个同时利用正负数据的 flow matching 损失
   - 核心思路：$\mathcal{L}(\theta) = \mathbb{E}[r \|\mathbf{v}_\theta^+ - \mathbf{v}\|^2 + (1-r)\|\mathbf{v}_\theta^- - \mathbf{v}\|^2]$，其中 $\mathbf{v}_\theta^+ = (1-\beta)\mathbf{v}^{\text{old}} + \beta \mathbf{v}_\theta$（隐式正策略），$\mathbf{v}_\theta^- = (1+\beta)\mathbf{v}^{\text{old}} - \beta \mathbf{v}_\theta$（隐式负策略）
   - 设计动机：通过隐式参数化，只训练一个模型 $\mathbf{v}_\theta$，但等价于同时让它接近正策略、远离负策略。最优解 $\mathbf{v}_{\theta^*} = \mathbf{v}^{\text{old}} + \frac{2}{\beta}\Delta$——reinforcement guidance 自动整合入策略中

3. **前向一致性**:
   - 做什么：保证训练后的模型仍对应有效的前向过程
   - 核心思路：DiffusionNFT 用标准 flow matching 损失（前向过程），而非反向 SDE 的策略梯度
   - 设计动机：FlowGRPO 只优化反向过程可能破坏前向-反向一致性

4. **CFG-free 训练**:
   - 做什么：不使用 CFG，reinforcement guidance 替代了 CFG 的功能
   - 核心思路：Theorem 3.1 中的 $\Delta$ 形式上等价于 guidance——相当于 RL 自动学到了"引导方向"
   - 设计动机：避免 GRPO 中需要同时训练有/无条件模型的复杂性

### 损失函数 / 训练策略

- 基于 SD3.5-Medium，rectified flow 参数化
- 每轮采样 $K$ 图像，按奖励分正负
- $\beta$ 控制 guidance 强度（类似 CFG 强度）
- 支持多奖励模型联合训练

## 实验关键数据

### 主实验（SD3.5-Medium, 单奖励 Head-to-Head vs FlowGRPO）

| 任务 | DiffusionNFT | FlowGRPO | 效率提升 |
|------|-------------|----------|---------|
| GenEval | 0.98 (1k steps) | 0.95 (5k steps) | **25×** |
| PickScore | 更高 | — | 3-5× |
| Aesthetic | 更高 | — | 3× |
| OCR | 更高 | — | 5× |

**多奖励联合训练（SD3.5-Medium → SD3.5-Medium-NFT）**:
- GenEval: 0.63 → **0.98** (w/o CFG)
- DPG-Bench: 81.65 → **92.82**
- T2I-CompBench: 0.54 → **0.75**
- HPS v2.1: 29.95 → **32.52**

### 消融实验

| 配置 | 效果 |
|------|-----|
| 只训练正样本（RFT） | 快速坍缩 |
| DiffusionNFT（完整） | 稳定提升 |
| 增大 $\beta$ | 更激进但可能过拟合 |
| 不同求解器（ODE/高阶） | 都兼容，性能无降 |

### 关键发现
- **负样本至关重要**：只在正样本上训练（RFT）会导致模式坍缩，加入负样本后稳定
- DiffusionNFT 完全无 CFG 但从极低起点（GenEval 0.24 w/o CFG）快速提升到 0.98，超过 FlowGRPO + CFG 的 0.95
- 可以用任意求解器（ODE/高阶），且不需要存储采样轨迹，训练效率显著更高
- 对域外奖励也有泛化提升

## 亮点与洞察
- **前向 vs 反向 RL** 的视角转换是核心贡献。扩散模型的前向过程唯一确定而反向过程依赖求解器选择，在前向过程上做 RL 更本质且避开了似然估计的困难。
- **隐式参数化** 技巧极为巧妙——通过 $\mathbf{v}_\theta^+ = (1-\beta)\mathbf{v}^{\text{old}} + \beta \mathbf{v}_\theta$，只需训练一个模型就等价于同时做"向好靠拢+离坏远走"。这比显式训练 guidance 模型高效得多。
- **NFT vs GRPO** 的类比类似 LLM 中的 DPO vs PPO——将 RL 转化为监督学习框架，工程实现更简单。

## 局限性 / 可改进方向
- $\beta$ 的选择需要调优，过大导致过拟合奖励
- 正负样本划分基于采样概率而非硬阈值，可能引入噪声
- 仅在 SD3.5-Medium 上验证，未在其他架构（SDXL/Flux/DiT）上测试
- 理论分析假设无限数据和模型容量，实际中近似误差未量化
- 多奖励联合训练的奖励权重设置未被系统研究

## 相关工作与启发
- **vs FlowGRPO**: 根本性不同——前向 RL vs 反向 RL。DiffusionNFT 快 3-25×，无需 SDE 采样器和 CFG。
- **vs DPO/DRaFT**: DiffusionNFT 是在线的（on-policy 采样），避免了离线方法的分布偏移问题。
- **vs LLM NFT (Chen et al., 2025c)**: 将 NFT 范式从语言模型引入扩散模型，利用 flow matching 的特性进行适配。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 前向过程 RL 是全新范式，隐式参数化优雅地统一了正负数据训练
- 实验充分度: ⭐⭐⭐⭐⭐ 与 FlowGRPO 的 head-to-head 对比清晰，多奖励联合训练全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨清晰，与 CFG 的类比直觉化，图表设计优秀
- 价值: ⭐⭐⭐⭐⭐ 解决了扩散 RL 的多个根本性问题（求解器限制、CFG 依赖、效率），有望成为新标准
