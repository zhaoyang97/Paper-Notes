# Diffusion Alignment as Variational Expectation-Maximization

**会议**: ICLR 2026  
**arXiv**: [2510.00502](https://arxiv.org/abs/2510.00502)  
**代码**: [https://github.com/Jaewoopudding/dav](https://github.com/Jaewoopudding/dav)  
**领域**: 扩散模型 / 对齐  
**关键词**: diffusion alignment, expectation-maximization, test-time search, reward optimization, mode collapse prevention  

## 一句话总结
将扩散模型对齐形式化为变分 EM 算法：E-step 用 test-time search（soft Q 引导 + 重要性采样）探索高奖励多模态轨迹，M-step 通过 forward-KL 蒸馏将搜索结果写入模型参数，在图像生成和 DNA 序列设计上同时实现高奖励和高多样性。

## 研究背景与动机

1. **领域现状**：扩散模型对齐（使生成匹配外部奖励）主要有两条路线：RL（DDPO/DPOK）和直接反向传播（DRaFT/AlignProp）。

2. **现有痛点**：RL 方法用 reverse-KL 优化，导致 mode-seeking 行为→模式坍缩和多样性丧失；直接反向传播依赖奖励模型的梯度信号，容易 reward over-optimization。两类方法在训练后期都出现 reward 高但图像质量/多样性急剧下降的现象。

3. **核心矛盾**：reward 优化 vs 多样性保持的 trade-off。Reverse-KL 天然 mode-seeking，容易坍缩到单一模式。

4. **本文要解决什么？** 设计一个对齐框架，能有效优化奖励的同时保持样本多样性和自然性，且适用于连续（图像）和离散（DNA）扩散模型。

5. **切入角度**：将对齐问题形式化为变分 EM——引入最优性变量 $\mathcal{O}$ 和轨迹潜变量 $\tau$，E-step 找多模态后验，M-step 用 forward-KL（mode-covering）蒸馏。Forward-KL 天然鼓励覆盖所有模式而非聚焦单一模式。

6. **核心 idea 一句话**：E-step 用 test-time search 发现多模态高奖励样本，M-step 用 forward-KL 蒸馏保持多样性，循环迭代逐步改善。

## 方法详解

### 整体框架

DAV 在训练中交替执行：
1. **E-step（探索）**：从当前模型出发，用 test-time search（梯度引导采样 + 重要性采样）生成高奖励且多样的轨迹，近似变分后验 $\eta_k^*$
2. **M-step（蒸馏）**：用 E-step 发现的轨迹训练模型，最小化 forward-KL $D_{\text{KL}}(\eta_k^* || p_\theta)$，等价于最大化搜索轨迹的对数似然 $-\log p_\theta(\tau)$

### 关键设计

1. **变分 EM 形式化**:
   - 做什么：将 reward 优化转化为最优性变量的边际似然最大化
   - 核心思路：定义 $p(\mathcal{O}=1|\tau) \propto \exp(\sum r_t/\alpha)$，轨迹 $\tau$ 是潜变量。ELBO 为 $\mathcal{J}_{\alpha,\gamma}(\eta, p_\theta)$，引入折扣因子 $\gamma$ 衰减远离终端的时间步的信用
   - 设计动机：EM 框架天然将探索（E-step）和利用（M-step）解耦，且 M-step 的 forward-KL 是 mode-covering 的，防止坍缩

2. **E-step: Test-time search**:
   - 做什么：近似采样最优变分后验 $\eta_k^*(x_{t-1}|x_t) \propto p_{\theta_k}(x_{t-1}|x_t) \exp(Q_{\text{soft}}^*/\alpha)$
   - 核心思路：两阶段——先用梯度引导（$Q_{\text{soft}}$ 近似为 $\gamma^{t-1} r(\hat{x}_0(x_{t-1}))$，Tweedie's formula）采样 $M$ 个候选粒子，再用重要性采样精炼
   - 设计动机：单纯 on-policy 重加权（传统 EM-RL）在策略偏离后验时严重偏差。Test-time search 主动探索，发现策略分布外的高奖励区域

3. **M-step: Forward-KL 蒸馏**:
   - 做什么：将 E-step 的搜索轨迹蒸馏到模型参数
   - 核心思路：$\mathcal{L}_{\text{DAV}} = -\mathbb{E}_{\tau \sim \eta_k^*}[\log p_\theta(\tau)]$。可选加 KL 正则化 $\mathcal{L}_{\text{DAV-KL}} = \mathcal{L}_{\text{DAV}} + \lambda D_{\text{KL}}(p_\theta || p_{\theta^0})$ 约束对预训练模型的偏离
   - 设计动机：Forward-KL 最小化 = 最大化搜索样本的似然 = mode-covering。与 RL 的 reverse-KL (mode-seeking) 相反，自然保持多样性

4. **模块化设计**:
   - E-step 的搜索算法可替换为任何 test-time search 方法
   - 适用于连续和离散扩散模型

### 损失函数 / 训练策略

- 基于 SD v1.5，奖励为 LAION aesthetic score（可微）或 compressibility（不可微）
- EM 迭代 100 epochs
- E-step 每步采样 $M$ 个候选, 重要性采样选择
- 折扣因子 $\gamma$ 衰减早期时间步的信用分配

## 实验关键数据

### 主实验（Text-to-Image, SD v1.5, Aesthetic Reward）

| 方法 | Aesthetic ↑ | LPIPS-A ↑ | ImageReward ↑ | 类型 |
|------|-----------|---------|-------------|------|
| Pretrained | 5.40 | 0.65 | 0.90 | — |
| DDPO | 6.83 | 0.48 | 0.27 | RL |
| DRaFT | 7.22 | 0.46 | 0.19 | 反向传播 |
| **DAV** | **8.04** | 0.53 | **0.95** | EM |
| DAV-KL | 6.99 | **0.58** | 1.13 | EM+KL |
| DAS (search only) | 7.22 | 0.65 | 1.07 | 推理时 |
| DAV Posterior | **9.18** | 0.53 | 0.91 | EM+search |

### 消融实验

| 分析 | 关键发现 |
|------|---------|
| DAV ELBO 趋势 | ELBO 单调递增（近似），消融掉 E-step search 后 ELBO 下降 |
| DAV vs DAV-KL | KL 正则化牺牲奖励（8.04→6.99）换取多样性（0.53→0.58） |
| DDPO/DRaFT 100 epochs | 已严重坍缩，ImageReward 降至负值 |

### 关键发现
- DAV 的奖励（8.04）远超 DDPO（6.83）和 DRaFT（7.22），同时 ImageReward 保持 0.95（接近预训练的 0.90），说明没有 reward over-optimization
- DDPO 和 DRaFT 在训练后期 ImageReward 暴跌（0.27/0.19），说明严重过优化
- DAV Posterior（推理时加 search）达到 9.18 aesthetic，是所有方法最高
- 在 DNA 序列设计上也有效，在 reward-diversity-naturalness 三个维度上全面超越 baseline

## 亮点与洞察
- **Forward-KL vs Reverse-KL** 的选择是核心洞察。RL 方法用 reverse-KL（mode-seeking→坍缩），DAV 用 forward-KL（mode-covering→保持多样性）。这个选择的理论动机清晰且实验验证充分。
- **Test-time search amortization** 是一个通用思路——先搜索再蒸馏，将推理时计算转化为模型能力。可以迁移到任何需要在推理时做昂贵搜索的场景（如代码生成、分子设计等）。
- **跨模态适用**：同一框架同时处理连续（图像）和离散（DNA）扩散，说明方法论的通用性。

## 局限性 / 可改进方向
- E-step 的 test-time search 增加训练开销（每个 EM 迭代需要多次 ODE/forward pass）
- Tweedie's formula 估计 $Q^*_{\text{soft}}$ 只是近似，对高噪声时间步可能不准
- 只在 SD v1.5 上验证，缺少 SDXL/Flux 等更大模型的实验
- 折扣因子 $\gamma$ 的最优选择未被系统研究
- Forward-KL 蒸馏可能无法精确覆盖后验的所有模式（有限样本+有限训练步）

## 相关工作与启发
- **vs DDPO/DPOK**: 都是 RL 对齐，但 DAV 用 forward-KL 替代 reverse-KL，核心区别在于 mode-covering vs mode-seeking。DAV 奖励更高且不坍缩。
- **vs DRaFT/AlignProp**: 直接反向传播效率高但梯度信号脆弱。DAV 不要求可微奖励。
- **vs DAS (test-time search)**: DAS 只做推理时搜索不更新模型。DAV 将搜索结果蒸馏回模型，推理时无额外开销。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 变分 EM 视角统一了 RL 和 test-time search，forward-KL 蒸馏是关键创新
- 实验充分度: ⭐⭐⭐⭐ 图像+DNA 双域验证，训练动态分析充分，但仅 SD v1.5
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，动机-方法-实验逻辑严谨
- 价值: ⭐⭐⭐⭐⭐ 解决了扩散对齐的核心痛点（over-optimization + mode collapse），具有广泛适用性
