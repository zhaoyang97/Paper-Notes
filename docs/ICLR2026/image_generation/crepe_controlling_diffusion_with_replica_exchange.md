# CREPE: Controlling Diffusion with Replica Exchange

**会议**: ICLR 2026  
**arXiv**: [2509.23265](https://arxiv.org/abs/2509.23265)  
**代码**: 有（GitHub）  
**领域**: 扩散模型 / 推理时控制  
**关键词**: replica exchange, parallel tempering, inference-time control, SMC alternative, reward tilting, CFG debiasing  

## 一句话总结
提出 CREPE，一种基于 Replica Exchange（并行回火/Parallel Tempering）的扩散模型推理时控制方法，作为 SMC 的计算对偶——在去噪步维度上并行、在样本维度上串行生成，具有高样本多样性、可在线精炼、支持温度退火/奖励倾斜/模型组合/CFG 去偏等多种任务。

## 研究背景与动机

1. **领域现状**：推理时控制扩散模型（不重训练就满足新约束）是热门方向。目前主流方法是 SMC（序贯蒙特卡洛），通过在去噪轨迹上维护一批加权粒子来纠正启发式 guidance 的偏差。

2. **现有痛点**：SMC 有三大局限：(a) 需要在整个去噪轨迹中同时维护大量粒子，内存开销大；(b) 样本多样性差，尤其粒子数少时退化严重（重采样导致粒子坍缩）；(c) 采样完成后无法精炼——如果结果不满意或加入新约束，必须从头生成。

3. **核心矛盾**：SMC 的"并行粒子 + 串行时间步"的模式决定了它天然存在多样性和灵活性的瓶颈。需要一种计算上对偶的方案。

4. **本文要解决什么？** 提出 SMC 的替代方案，实现：(a) 粒子逐个生成而非批量 (b) burn-in 后保持高多样性 (c) 支持在线精炼和早停 (d) 覆盖 tempering、reward-tilting、model composition、CFG debiasing 等多种任务

5. **切入角度**：Replica Exchange / Parallel Tempering 恰好是 SMC 的计算对偶——它在不同去噪步上并行运行链，串行生成样本。将这个 MCMC 采样框架适配到扩散模型的设定中。

6. **核心 idea 一句话**：将 Parallel Tempering 的 swap move 适配到扩散模型路径空间上，利用 Radon-Nikodym Estimator 计算接受概率，实现无需显式目标密度的推理时控制。

## 方法详解

### 整体框架

CREPE 维护 $M+1$ 个粒子，每个粒子驻留在不同的扩散时间步 $t_0 < t_1 < ... < t_M$（从数据分布到噪声）。每轮迭代包含：
1. **Communication step**：相邻粒子通过 APT swap move 交换——生成前向和后向提议路径，计算接受概率后决定是否交换
2. **Local exploration step**：每个粒子在其时间步上做局部 MCMC 更新
3. 两步可并行化

### 关键设计

1. **Accelerated PT Swap Move 在扩散路径空间中的实现**:
   - 做什么：让驻留在时间步 $t$ 和 $t'$ 的粒子 $(x, x')$ 通过前向/后向扩散路径交换位置
   - 核心思路：从 $x$ 出发沿扩散前向过程走到 $t'$，从 $x'$ 出发沿反向过程走到 $t$，用 Metropolis-Hastings 接受概率 $\alpha_{t,t'}$ 决定是否交换。接受概率通过 Radon-Nikodym Estimator (RNE) 计算，RNE 利用预训练扩散模型的前向/后向转移概率之比
   - 设计动机：标准 PT 需要知道目标分布的未归一化密度，但推理时控制只有预训练模型。通过 RNE 关系 $p_{t'}(x_{t'})/p_t(x_t) = R_{t,t'}^{-1}$，可以避免直接评估密度

2. **退火路径设计**:
   - 做什么：为不同控制任务定义中间分布序列
   - 核心思路：
     - Tempering: $\pi_t(x) \propto p_t^j(x)^\beta$
     - Reward tilting: $\pi_t(x) \propto p_t^j(x) \exp(r_t(x))$
     - Model composition: $\pi_t(x) \propto \prod_j p_t^j(x)$
     - CFG debiasing: $\pi_t(x) \propto p_t(x)^{1-w} p_t(x|c)^w$
   - 设计动机：所有这些目标分布都可以用预训练模型密度比来表达，因此接受概率可通过 RNE 计算

3. **在线精炼能力**:
   - 做什么：在 MCMC 链运行过程中动态添加/修改约束
   - 核心思路：MCMC 链可以无限运行，任何时候加入新的奖励项只需修改退火路径，PT 自然会适应
   - 设计动机：SMC 是一次性的，结束后无法修改；CREPE 作为 MCMC 天然支持迭代精炼

4. **同时支持连续和离散扩散**:
   - 做什么：推导了高斯扩散（SDE）和离散掩码扩散（CTMC）两种情况下的 swap rate
   - 设计动机：覆盖图像生成（连续）和文本/离散数据（离散掩码扩散如 MDLM）

### 损失函数 / 训练策略

- 无需训练，完全在推理时运行
- 需要预训练扩散模型的前向和反向过程
- 计算开销与 SMC 可比但分布不同——PT 需 burn-in，但之后每个样本成本恒定

## 实验关键数据

### 主实验

**分子温度退火（Alanine Dipeptide/Tetrapeptide/Hexapeptide）**

| 方法 | Energy TVD ↓ | TICA MMD ↓ | 说明 |
|------|-------------|-----------|------|
| FKC (SMC) | 0.345 | 0.116 | SMC baseline |
| CREPE (Ours) | **0.224** | **0.096** | Dipeptide |
| CREPE | **0.122** | **0.035** | Tetrapeptide |

**CFG Debiasing（ImageNet-64）**

| 方法 | #Samples | IR ↑ | CLIP ↑ | FID ↓ |
|------|----------|------|--------|-------|
| FKC (SMC) | 8 | **-0.29** | **24.17** | **1.85** |
| CREPE | 8 | -0.30 | 24.10 | 1.92 |
| FKC | 512 | -0.08 | 24.31 | 1.96 |
| CREPE | 512 | **0.09** | 24.28 | **1.79** |

### 关键发现
- 少量样本时 SMC 更优（CREPE 需要 burn-in），但随样本数增加 CREPE 超越 SMC，尤其 FID 持续改善
- CREPE 的核心优势是**多样性**——SMC 的重采样导致粒子坍缩（同一 batch 内视觉相似），CREPE 的 MCMC 链天然探索更广
- 在线精炼实验中，添加新约束后 CREPE 仅需 1k 次迭代即可满足，展示了灵活性
- 在离散扩散（MNIST MDLM）上也有效，说明方法的通用性

## 亮点与洞察
- **SMC 的计算对偶视角**极为优雅——将"并行粒子×串行时间"翻转为"串行粒子×并行时间"，一句话就讲清了核心创新。这种对偶关系（Syed et al., 2024）来自采样理论的深层联系。
- **在线精炼**是 SMC 完全做不到的——对实际应用（交互式生成、迭代设计）非常有用。
- **统一框架**覆盖 tempering、reward-tilting、model composition、CFG debiasing 等多种任务，还可以自由组合。方法论上很通用。

## 局限性 / 可改进方向
- Burn-in 期间样本质量差，少量样本场景不如 SMC
- 每个 swap move 需要模拟前向+后向扩散路径，计算开销非平凡
- 高维图像（ImageNet-512）上主要展示 reward-tilting 的定性结果，缺少定量对比
- 接受率可能随维度增加而下降，需要更细的退火调度
- 未探索与 guidance 方法（如 DPS、FreeDoM）的组合

## 相关工作与启发
- **vs FKC (SMC)**: 计算对偶关系。少样本 SMC 优，多样本 CREPE 优。CREPE 多样性更好。
- **vs Twisted SMC/DDRM**: 都是推理时控制的纠偏方法，但 CREPE 基于 MCMC 而非重要性采样。
- **与 APT (Zhang et al., 2025)**: CREPE 将 APT 从已知未归一化密度的设定扩展到只有预训练扩散模型的设定。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将 Parallel Tempering 首次适配到扩散模型推理时控制，SMC 对偶视角非常优雅
- 实验充分度: ⭐⭐⭐⭐ 覆盖分子/图像/轨迹/离散数据多模态，但高分辨率图像定量实验较少
- 写作质量: ⭐⭐⭐⭐ 理论严谨但符号密度高，需要较强的随机过程背景
- 价值: ⭐⭐⭐⭐ 为扩散模型推理时控制提供了新的范式，尤其在多样性和在线精炼方面有独特优势
