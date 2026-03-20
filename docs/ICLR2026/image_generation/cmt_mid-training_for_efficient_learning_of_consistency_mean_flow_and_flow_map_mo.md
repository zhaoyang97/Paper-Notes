# CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models

**会议**: ICLR 2026  
**arXiv**: [2509.24526](https://arxiv.org/abs/2509.24526)  
**代码**: [https://github.com/sony/cmt](https://github.com/sony/cmt)  
**领域**: 扩散模型 / 少步生成  
**关键词**: flow map, consistency model, mid-training, few-step generation, diffusion distillation  

## 一句话总结
提出 Consistency Mid-Training (CMT)，在预训练扩散模型和 flow map 后训练之间插入一个轻量级中间训练阶段，通过让模型学习将 ODE 轨迹上的任意点映射回干净样本来获得轨迹对齐的初始化，从而大幅降低训练成本（最多 98%）并达到 SOTA 两步生成质量。

## 研究背景与动机

1. **领域现状**：扩散模型生成质量高但推理慢（需要多步 ODE 求解）。Flow map 模型（如 Consistency Models、Mean Flow）通过学习 PF-ODE 的解映射来实现少步（1-2 步）生成，是当前加速扩散模型的主流方向。

2. **现有痛点**：Flow map 模型训练不稳定、对超参数敏感、计算成本高昂。核心原因是缺乏真实的回归目标——当前方法依赖 stop-gradient 的伪目标，这些目标随训练动态漂移，导致优化信号有偏且不稳定。

3. **核心矛盾**：从预训练扩散模型初始化虽然有帮助，但扩散模型学的是无穷小步长的去噪，而 flow map 需要学习大跨度的轨迹跳跃。这种"微分 vs 积分"的 mismatch 使得扩散初始化脆弱，仍然需要大量启发式技巧（时间采样、损失权重调度等），训练仍然缓慢且不稳定。

4. **本文要解决什么？** (a) 如何为 flow map 模型提供一个轨迹对齐的高质量初始化？(b) 如何避免 stop-gradient 带来的伪目标偏差？(c) 如何大幅降低 flow map 训练成本？

5. **切入角度**：受 LLM 领域 mid-training 概念启发，在预训练和后训练之间插入一个中间阶段。利用预训练模型的 ODE solver 生成参考轨迹，这些轨迹提供了确定性的、无 stop-gradient 的回归目标。

6. **核心 idea 一句话**：用预训练模型的 ODE 轨迹作为固定监督信号，通过简单回归让模型学会"沿着轨迹跳到终点"，从而为 flow map 后训练提供轨迹感知的初始化。

## 方法详解

### 整体框架

CMT 提出三阶段 pipeline：**预训练** → **中间训练 (CMT)** → **后训练 (Flow Map)**。

- **输入**：预训练好的扩散模型 $\mathbf{D}_\phi$（或 flow matching 模型）
- **中间训练**：从先验分布 $p_{\text{prior}}$ 采样 $\mathbf{x}_T$，用预训练模型的 ODE solver（如 DPM-Solver++ 16 步）生成离散轨迹 $\{\hat{\mathbf{x}}_{t_i}\}_{i=0}^M$，训练模型将轨迹上任意点映射回干净终点
- **后训练**：用 CMT 的权重初始化 flow map 模型（ECT/ECD/MF），正常训练
- **输出**：1-2 步即可生成高质量图像的 flow map 模型

### 关键设计

1. **CMT-CM 损失（针对 Consistency Model）**:
   - 做什么：学习将轨迹上任意中间点 $\hat{\mathbf{x}}_{t_i}$ 直接映射到干净样本 $\hat{\mathbf{x}}_{t_0}$
   - 核心思路：$\mathcal{L}_{\text{CMT-CM}}(\theta) = \mathbb{E}_i \mathbb{E}_{\mathbf{x}_T \sim p_{\text{prior}}} [d(\mathbf{f}_\theta(\hat{\mathbf{x}}_{t_i}, t_i), \hat{\mathbf{x}}_{t_0})]$，其中 $\hat{\mathbf{x}}_{t_0}$ 是 ODE solver 生成的确定性"干净"样本，$d$ 可以是 LPIPS 或 $\ell_2$ 距离
   - 设计动机：这是 oracle CM 损失的离散近似——由于 solver 生成点近似真实流映射 $\hat{\mathbf{x}}_{t_i} \approx \Psi_{T \to t_i}(\mathbf{x}_T)$，整个损失变成标准回归问题，**无需 stop-gradient、无需自定义时间采样、无需损失权重调度**。每个 $\mathbf{x}_T$ 确定唯一轨迹，但 $\mathbf{x}_T$ 可以任意多，避免了过拟合。

2. **CMT-MF 损失（针对 Mean Flow）**:
   - 做什么：学习轨迹点之间的平均漂移
   - 核心思路：$\mathcal{L}_{\text{CMT-MF}}(\theta) = \mathbb{E}_{i>j} \mathbb{E}_{\mathbf{x}_T} [\|\mathbf{h}_\theta(\hat{\mathbf{x}}_{t_i}, t_i, t_j) - \frac{\hat{\mathbf{x}}_{t_i} - \hat{\mathbf{x}}_{t_j}}{t_i - t_j}\|_2^2]$
   - 设计动机：将 MF 的复杂训练目标简化为轨迹点有限差分的回归。当 $t_j = 0$ 时退化为 CMT-CM，说明 CMT-MF 是更一般的形式。同样无需 stop-gradient 和 Jacobian-向量积（JVP），大幅降低计算成本。

3. **灵活的教师采样器**:
   - 做什么：CMT 的轨迹生成不限于扩散模型的 ODE solver
   - 核心思路：在 ImageNet 256 实验中，用一个小型 MF-B/4 模型（质量较差，8 步 FID=13.44）作为教师生成轨迹，训练大型 MF-XL/2
   - 设计动机：证明 CMT mid-training 是架构无关的——只要能生成 ODE 轨迹就行。这意味着可以先快速训练一个小模型，再用它加速大模型训练。

4. **轨迹复用机制**:
   - 做什么：DPM-Solver++ 等多步 solver 的中间状态可以复用
   - 核心思路：产生一条 $M$ 步轨迹后，每个中间点 $\hat{\mathbf{x}}_{t_i}$ 都可以和终点 $\hat{\mathbf{x}}_{t_0}$ 构成一个训练样本，一次 solver 调用产出 $M$ 个训练对
   - 设计动机：相比只用端点（Slow CMT），CMT 的数据效率高约 3 倍，GPU 时间开销更低

### 损失函数 / 训练策略

- CM 类实验用 LPIPS 感知损失（像素空间）或 ELatentLPIPS（潜空间）
- MF 类实验用 $\ell_2$ 损失
- ODE solver 统一用 DPM-Solver++ 16 步或 MF 教师 8 步
- 后训练可以去掉大量 ad-hoc 技巧（$\Delta t$ annealing、loss reweighting、自定义时间采样、EMA 变体、非线性学习率调度等）

## 实验关键数据

### 主实验

| 数据集 | 指标 | CMT (本文) | 之前 SOTA | 提升 |
|--------|------|-----------|-----------|------|
| CIFAR-10 32×32 | 2-step FID | **1.97** | 1.98 (IMM) | -0.01 |
| ImageNet 64×64 | 2-step FID | **1.32** (w/ ECD) | 1.25 (AYF) | +0.07 |
| ImageNet 64×64 | 2-step FID | **1.48** (w/ ECT) | 1.48 (sCT) | 持平但 98% 少训练 |
| ImageNet 512×512 | 2-step FID | **1.84** | 1.87 (AYF) | -0.03 |
| ImageNet 256×256 | 1-step FID | **3.34** | 3.43 (MF) | -0.09 |
| AFHQv2 64×64 | 2-step FID | **2.34** | 2.61 (ECT) | -0.27 |
| FFHQ 64×64 | 2-step FID | **2.75** | 4.02 (iCT) | -1.27 |

### 消融实验

| 配置 | 1-step FID | 2-step FID | 说明 |
|------|-----------|-----------|------|
| Full model (CMT) | **2.74** | **1.97** | 完整模型 |
| Vanilla ECT (51.2M) | 3.54 | 2.12 | 无 mid-training |
| CMT_short (1.28M mid + 49.92M post) | 3.42 | 2.11 | 短 mid-training |
| CMT_long (25.6M mid + 25.6M post) | 3.30 | 2.04 | 长 mid-training |
| KD 初始化 | 3.54 | 2.19 | 知识蒸馏初始化，不如 CMT |
| Slow CMT | 2.75 | 1.98 | 只用端点，质量相近但慢 3× |

### 关键发现
- CMT mid-training 越长效果越好，说明轨迹对齐的初始化至关重要
- 即使用质量较差的小模型做教师（MF-B/4, 8-step FID=13.44），CMT 仍然有效——将 MF-XL/2 训练时间减半，且 FID 更优
- 理论证明 CMT 初始化的梯度偏差为 $\mathcal{O}(\varepsilon + \Delta t^2)$，远小于扩散初始化和随机初始化
- 在 MS-COCO T2I 任务上也有效，减少 47% 训练时间

## 亮点与洞察
- **Mid-training 概念的跨领域迁移**：将 LLM 中的 mid-training 思想引入视觉生成领域，用极其简洁的方式解决了 flow map 训练不稳定的老问题。巧妙在于找到了一个天然存在且易于获取的固定回归目标——ODE 轨迹。
- **化繁为简的工程价值**：CMT 让后训练可以去掉几乎所有 ad-hoc 技巧（$\Delta t$ annealing、自定义时间采样、损失权重调度），大幅降低调参难度。这是工程上的重大简化。
- **弱教师也能用**：证明了 mid-training 的教师不需要很强，一个小模型就够用。这个发现可以迁移到其他蒸馏/初始化场景——先快速训练一个小模型提供粗略轨迹，再用它引导大模型。

## 局限性 / 可改进方向
- 仍然需要预训练扩散模型作为基础，无法完全从零开始
- 中间训练阶段的 ODE solver 步数（16 步）是固定的，未探索步数对最终质量的影响
- 在 T2I 任务上 1-step FID 仍然较大（15.12），可能受数据集限制
- 理论分析主要基于简化假设（均匀权重、$\ell_2$ 距离），实际使用感知损失时的理论保证未充分讨论
- 可以探索在 video generation 等更复杂的生成任务上是否同样有效

## 相关工作与启发
- **vs ECT/ECD**: CMT 以它们为后训练方法，通过增加 mid-training 阶段在相同或更低成本下显著提升性能。本质区别在于 CMT 提供了更好的初始化。
- **vs sCT/sCD**: 性能相当但 CMT 训练成本低 93-98%，因为不需要昂贵的 JVP 计算。
- **vs Knowledge Distillation**: KD 只学端点映射，而 CMT 利用中间轨迹信息，数据效率更高。
- **vs Mean Flow**: CMT 可以用 MF 小模型做教师，且初始化后 MF 训练加速 50%。

## 评分
- 新颖性: ⭐⭐⭐⭐ mid-training 概念在视觉生成中首次系统性提出，但核心技术（轨迹回归）相对简单
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖多种数据集、多种分辨率、像素和潜空间、CM 和 MF 两个框架、T2I 任务，消融全面
- 写作质量: ⭐⭐⭐⭐⭐ 统一了 CM/CTM/MF 的视角，理论分析清晰，实验组织有条理
- 价值: ⭐⭐⭐⭐⭐ 实际训练成本降低 90%+ 同时达到 SOTA，工程价值极高

