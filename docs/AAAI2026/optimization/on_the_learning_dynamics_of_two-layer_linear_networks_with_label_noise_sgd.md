# On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD

**会议**: AAAI 2026  
**arXiv**: [2603.10397](https://arxiv.org/abs/2603.10397)  
**代码**: [https://github.com/a-usually/Label-Noise-SGD](https://github.com/a-usually/Label-Noise-SGD)  
**领域**: 对齐RLHF / 优化理论  
**关键词**: Label Noise SGD, Learning Dynamics, Lazy-to-Rich Transition, Implicit Bias, Sharpness-Aware Minimization  

## 一句话总结

在二层过参数化线性网络上理论分析 Label Noise SGD 的学习动力学，揭示了两阶段行为——Phase I 中权重范数逐渐缩小使模型从 lazy regime 逃逸到 rich regime，Phase II 中权重与真实插值器对齐并收敛——并将该理论扩展到 SAM 优化器。

## 研究背景与动机

1. **领域现状**：深度学习的成功部分源于梯度训练算法中噪声引发的隐式偏置。近来有实证发现，训练时注入标签噪声反而能提升泛化性能（如 CIFAR-10 + ResNet-18 上约 1.5% 测试准确率提升），且训练出的模型更稀疏。

2. **现有痛点**：现有理论工作（Blanc et al. 2020; Damian et al. 2021; HaoChen et al. 2021）大多关注 label noise SGD 在全局极小值附近的局部隐式正则化效应（如正则化 sharpness），或在对角线性网络这种极简模型上分析。对于有两个可训层、存在层间耦合效应的更现实网络，缺乏对完整学习动力学（从初始化到收敛）的理论分析。

3. **核心矛盾**：过参数化网络在 NTK 初始化下倾向于停留在 lazy regime（参数几乎不动、等价于线性核方法），无法解释深度学习的泛化优势。然而实践中模型确实学到了有意义的特征（rich regime）。标签噪声如何驱动这一关键转变？
4. **本文要解决什么？** 严格刻画二层线性网络在 label noise SGD 下的完整学习轨迹，解释标签噪声如何以及为什么能驱动 lazy-to-rich 过渡。
5. **切入角度**：从第二层参数的振荡效应出发——标签噪声加速了第二层权重 $\mathbf{a}$ 的振荡，这些振荡通过层间耦合导致第一层权重 $\mathbf{W}$ 范数逐渐缩小，从而实现从 NTK 初始化的 lazy regime 到小初始化行为的 rich regime 的过渡。
6. **核心idea一句话**：标签噪声通过放大第二层参数的振荡，间接驱动第一层权重范数持续衰减，使网络从 lazy regime 自然过渡到 rich regime，最终收敛到稀疏的 ground-truth 插值器。

## 方法详解
### 整体框架
考虑二层线性网络 $\hat{y}_i = \mathbf{a}^\top \mathbf{W} \mathbf{x}_i$，其中 $\mathbf{W} \in \mathbb{R}^{m \times d}$，$\mathbf{a} \in \mathbb{R}^m$。使用 NTK 初始化（$w_{i,j}(0) \sim \frac{1}{\sqrt{d}} \mathcal{N}(0, I)$，$a_i(0) \sim \frac{1}{\sqrt{m}} \mathcal{N}(0, I)$）。训练采用 Label Noise SGD：每步随机翻转标签 $\tilde{y}_i = y_i + \epsilon$（$\epsilon \sim \{-\sigma, +\sigma\}$），在带噪标签上计算梯度更新。

### 关键设计
1. **Phase I：逐渐缩小与 Lazy-to-Rich 过渡（Theorem 4.2, Lemma 4.3）**
   - 做什么：证明在 Phase I 期间，所有神经元的第一层权重范数 $\|\mathbf{w}_i(t)\|$ 单调递减，最终模型逃逸 lazy regime。
   - 核心思路：权重范数的变化量 $\Delta W_i(j) = -\nabla\hat{\ell}^2 \cdot ((\mathbf{x}^\top \mathbf{w}_i)^2 - a_i^2 \|\mathbf{x}\|^2)$。由于 $\mathbf{a}(0)$ 初始化很小，$(\mathbf{x}^\top \mathbf{w}_i)^2$ 项占主导，使得 $\Delta W_i(j)$ 以高概率为负。关键等式 $\nabla\hat{\ell}^2 \cdot (\mathbf{x}^\top \mathbf{w}_i)^2 = (a_i(j+1) - a_i(j))^2$ 表明第一层范数的衰减由第二层的振荡幅度直接控制，而标签噪声正是振荡的来源。
   - 设计动机：这是论文的核心发现——建立了"标签噪声 → 第二层振荡 → 第一层范数衰减 → lazy-to-rich 过渡"的因果链。Theorem 4.2 给出逃逸时间 $T_1 = O(\frac{\sqrt{\log m}}{\sigma^2 \eta^2 \sqrt{m}})$，明确依赖噪声强度 $\sigma$。

2. **Phase II：对齐与收敛（Lemma 4.5, 4.6）**
   - 做什么：证明当权重范数足够小后（$\|\mathbf{w}_i\|, |a_i| \leq \sqrt{\eta}$），神经元快速与 ground-truth 插值器 $\theta^*$ 对齐并收敛。
   - 核心思路：Lemma 4.5 证明经过 $T_2 = \frac{1}{\|\theta^*\|} \ln(1/\eta)$ 步后，对齐度 $\frac{|\langle \theta^*, \mathbf{w}_i \rangle|}{\|\theta^*\| \cdot \|\mathbf{w}_i\|} \geq 1 - O(\ln(1/\eta) \cdot \sqrt{\eta})$。Lemma 4.6 证明完美对齐后再经 $T_3 = O(\frac{-\ln\eta}{\eta})$ 步收敛到 $\|\theta(t_3) - \theta^*\| \leq O(\eta \ln(1/\eta))$。
   - 设计动机：Phase II 类似于小初始化下的学习行为——这正是 Phase I 的效果，label noise SGD 将大初始化"变成了"等效的小初始化。

3. **扩展到 SAM**
   - 做什么：验证 SAM（Sharpness-Aware Minimization）也展现相同的两阶段动力学。
   - 核心思路：SAM 的内层对抗扰动放大了梯度噪声，类似于标签噪声的效应。在合成和 CIFAR-10 实验中，SAM 训练的 WideResNet 损失曲线与线性化模型显著偏离（rich regime 特征），同时第一层权重范数明显衰减。
   - 设计动机：将发现的机制从一种特定噪声源推广到更一般的"噪声放大"优化策略。

### 损失函数 / 训练策略
使用均方损失 $\hat{\ell}_i(\theta(t)) = \frac{1}{2} |f(\theta(t); \mathbf{x}_i) - y_i - \epsilon|^2$，其中 $\epsilon \sim \{-\sigma, +\sigma\}$。关键条件包括：过参数化 $m = \Omega(1/\sqrt{\eta})$，小学习率 $\eta \leq 1/C^{96}$，充足数据 $n \geq 1/\eta^2$，稀疏 ground-truth $\|\theta^*\| \leq m^{-1/4}$。这些条件确保 Phase I 足够长以实现充分的范数衰减，并且网络宽度足够大以保证高概率结果。训练采用在线 SGD（批大小 1）加标签噪声，每步随机采样一个训练样本并随机翻转标签。

## 实验关键数据

### 主实验（CIFAR-10 + ResNet-18）

| 配置 | 测试准确率 | 测试损失 | 说明 |
|------|----------|---------|------|
| Vanilla SGD | ~93.5% | 较高 | 基准 |
| Label Noise SGD ($\tau=0.05$) | ~94.5% | 较低 | +1.0% |
| Label Noise SGD ($\tau=0.1$) | ~95.0% | 最低 | +1.5% |
| Label Noise SGD ($\tau=0.2$) | ~94.8% | 低 | +1.3% |

标签噪声概率 $\tau \in \{0.05, 0.1, 0.2\}$ 均优于 vanilla SGD。

### 消融实验（合成二层线性网络，两阶段验证）

| 阶段 | 关键指标 | 观测 |
|------|---------|------|
| Phase I | 平均神经元范数 $\text{Avg}(\|\mathbf{w}_i\|)$ | 初始下降，验证 progressive diminishing |
| Phase I | 训练/测试损失 | 训练损失波动但测试损失稳定下降 |
| Phase II | 平均对齐度 $\text{Avg}(\langle \mathbf{w}_i, \theta^* \rangle)$ | 快速上升至接近 1 |
| Phase II | 参数距离 $\|\theta(t) - \theta^*\|$ | 收敛至 $O(\eta \ln(1/\eta))$ |
| Lazy regime（无噪声 GD） | 损失曲线 | 与线性化模型几乎重合，确认 lazy |
| Rich regime（有噪声） | 损失曲线 | 偏离线性化模型，确认 rich |
| 模拟振荡（Markov Chain） | 权重范数 | 证实振荡 → 范数衰减 |

### 关键发现
- **标签噪声驱动 lazy→rich 过渡**：无噪声 GD 训练的 WideResNet 损失曲线与其线性化模型几乎完全重合（lazy regime），而加入标签噪声后显著偏离（rich regime），且第一层权重范数明显衰减。
- **稀疏性优势**：在相同剪枝比例（$\alpha\%$ 参数保留）下，label noise SGD 训练的模型准确率始终高于 vanilla SGD，说明学到了更稀疏的表示。
- **SAM 展现相同行为**：在合成和 CIFAR-10 上，SAM 的学习动力学定性上与 label noise SGD 一致（权重范数先降后对齐），支持理论的可推广性。
- **噪声-第二层振荡-第一层衰减的因果链**：三状态 Markov 过程模拟验证了即使去除 SGD 采样噪声，只要第二层有振荡就能驱动第一层范数衰减（Lemma 4.4）。

## 亮点与洞察
- **隐式偏置的新机制**：不同于之前关注"label noise 正则化 sharpness"的视角，本文揭示了一个更基本的效应——噪声驱动 regime 过渡。这个视角更具解释力，因为它解释了为什么 noise 不仅降低 sharpness 还能促进 feature learning。
- **层间耦合的精妙分析**：第二层振荡驱动第一层衰减的分析，抓住了深度网络中层间依赖的本质。这种分析范式可以推广到更深网络。
- **从 SGD to SAM 的统一视角**：将 label noise SGD 和 SAM 统一在"噪声放大促进 rich regime"的框架下，为理解不同优化器的隐式偏置提供了统一理论基础。- **实验可视化出色**：Figure 2 中每个神经元在范数-对齐度空间的轨迹图直观展现了两阶段动力学，是理论与实验结合的典范。
## 局限性 / 可改进方向
- 理论限于**二层线性网络**——没有非线性激活函数，离实际网络架构有距离。非线性如何影响两阶段动力学是重要开放问题。
- 理论分析在 Phase II 切换到了 GD（无噪声），简化了分析但牺牲了对完整 label noise SGD 收敛阶段的刻画。
- 实验规模偏小：CIFAR-10 + ResNet-18/WideResNet 的 64 图像子集，未在大规模任务上验证。
- 理论条件较强：$\eta \leq 1/C^{96}$ 和 $\|\theta^*\| \leq m^{-1/4}$ 在实践中难以精确判断是否满足。
- 仅分析**回归任务**，如何扩展到分类任务的 cross-entropy 损失仍是开放挑战。

## 相关工作与启发
- **vs HaoChen et al. (2021) / Vivien et al. (2023)**：他们在对角线性网络（单层有效参数）上分析 label noise SGD，证明它会恢复稀疏 ground-truth。本文在二层网络上分析，处理了更复杂的层间耦合。
- **vs Blanc et al. (2020) / Damian et al. (2021)**：他们关注 label noise 在全局极小值附近的隐式正则化（trace of Hessian），是 Phase II 附近的局部分析。本文给出从初始化到收敛的全局动力学。
- **vs Geiger et al. (2020)**：他们表明初始化尺度决定 lazy vs rich regime。本文补充：即使大初始化（NTK），label noise 也能将系统驱动到 rich regime。
- **vs Weight Decay / Large LR 方法**：Li et al. 和 Lewkowycz et al. 证明 weight decay 和大学习率也能诱导 rich regime。本文的标签噪声机制不同——通过第二层振荡间接作用，提供了理解 regime 转换的新视角。
- **vs Varre et al. (2023)**：他们证明 label noise SGD 倾向降低参数矩阵的秩，与本文 Phase I 的权重范数衰减一致，但他们未分析完整的两阶段动力学。
- 启发：振荡-衰减机制可能适用于分析 Dropout、Mixup 等其他正则化技术的隐式偏置。

## 评分

- 新颖性: ⭐⭐⭐⭐ lazy→rich 过渡的理论分析是新贡献，层间耦合的振荡机制有洞察力
- 实验充分度: ⭐⭐⭐ 合成实验充分验证理论，但真实实验规模受限于 NTK 计算复杂度
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，两阶段图景清晰，但符号较重
- 价值: ⭐⭐⭐⭐ 对理解 SGD 隐式偏置和 feature learning 有理论贡献，实践影响需进一步验证
