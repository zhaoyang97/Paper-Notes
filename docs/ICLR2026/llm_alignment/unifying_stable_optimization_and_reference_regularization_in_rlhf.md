# Unifying Stable Optimization and Reference Regularization in RLHF (DAR)

**会议**: ICLR 2026  
**arXiv**: [2602.11523](https://arxiv.org/abs/2602.11523)  
**代码**: https://github.com/tmllab/2026_ICLR_DAR  
**领域**: 对齐RLHF  
**关键词**: RLHF, 双KL正则化, 优势回归, 参考策略插值, 奖励黑客

## 一句话总结
提出DAR(Dual-regularized Advantage Regression)：发现标准RLHF中参考模型正则化(防reward hacking)和策略稳定约束(防崩溃)会逐步冲突导致优化空间过度受限，通过双KL目标在对数空间插值参考策略+回归变换消除策略比率不稳定性，在直接AI对齐和标准RLHF设置中达到92.42%平均胜率，超GRPO 7.27%。

## 研究背景与动机

1. **领域现状**：在线RLHF（PPO/RLOO/GRPO）通过RL优化LLM策略。两个核心难题：reward hacking(策略过度优化代理奖励)和训练不稳定(策略剧烈偏移导致崩溃)。
2. **现有痛点**：
   - 防reward hacking用KL(π_θ||π_0)约束到初始模型
   - 防训练不稳定用clip/KL(π_t||π_θ)约束到当前策略
   - **关键发现**：这两个约束逐步冲突——策略必须同时接近π_0和π_t，但随着训练推进π_t远离π_0，两者交集缩小，高奖励策略被排除在外
3. **核心矛盾**：稳定性约束和参考正则化的冲突导致优化空间过度受限
4. **核心idea一句话**：用对数空间插值的动态参考策略 $\pi_0^\alpha \cdot \pi_t^{1-\alpha}$ 统一两个约束 + 回归变换消除策略比率不稳定

## 方法详解

### 整体框架
双KL对齐目标：$\mathcal{J} = \max_{\pi_\theta} \mathbb{E}[A(x,y)] - \beta(\alpha \text{KL}[\pi_\theta||\pi_0] + (1-\alpha)\text{KL}[\pi_\theta||\pi_t])$，等价于对动态插值参考 $\pi_{\text{ref}} \propto \pi_0^\alpha \pi_t^{1-\alpha}$ 的单KL约束。然后转化为加权SFT(回归)损失消除RL的不稳定性。

### 关键设计

1. **双KL对齐目标**:
   - 做什么：统一防reward hacking和训练稳定性约束
   - 核心思路(Proposition 4.1)：$\alpha \text{KL}[\pi_\theta||\pi_0] + (1-\alpha)\text{KL}[\pi_\theta||\pi_t]$ 等价于 $\text{KL}[\pi_\theta || \frac{1}{C}\pi_0^\alpha \pi_t^{1-\alpha}]$
   - 效果：随着π_t演化，插值参考自动跟踪高奖励区域，提供更好的支持覆盖
   - α控制trade-off：α→1偏保守(接近初始模型)，α→0偏探索(接近当前策略)

2. **回归变换(Advantage Regression)**:
   - 做什么：将RL目标转化为加权SFT损失
   - 闭式最优策略(Theorem 4.2)：$\pi^* \propto \pi_0^\alpha \pi_t^{1-\alpha} \exp(\frac{1}{\beta}A)$
   - 实际损失：$\mathbb{E}[(w_{\text{reg}} \cdot w_{\text{adv}}) \cdot \log\pi_\theta(y|x)]$
     - $w_{\text{reg}} = (\pi_0/\pi_t)^\alpha$：正则化权重，惩罚偏离参考的回答
     - $w_{\text{adv}} = \exp(\frac{1}{\beta}A)$：优势权重，奖励好回答
   - 设计动机：避免PPO中策略比率的不稳定性，回归损失更平滑稳定
   - 权重裁剪：$\min(w_{\text{reg}} \cdot w_{\text{adv}}, w_{\text{clip}})$ 防止梯度爆炸

### 损失函数 / 训练策略
- Monte Carlo采样估计优势（避免单独的价值模型）
- 批次内优势归一化
- $w_{\text{clip}} = 20$, $\alpha = 0.1$, $\beta = 0.05$

## 实验关键数据

### 主实验：直接AI对齐（Qwen2-7B, GPT-4-Turbo评估）

| 方法 | TL;DR | Helpful | Harmless | 平均胜率 |
|------|-------|---------|----------|---------|
| DPO(offline) | 67.17% | 81.34% | 77.91% | 75.47 |
| Online DPO | 78.47% | 88.86% | 83.55% | 83.63 |
| GRPO | 83.03% | 86.93% | 85.50% | 85.15 |
| **DAR** | **98.27%** | **93.16%** | **85.84%** | **92.42** |

### 标准RLHF：Qwen2-7B-Instruct

| 方法 | MT-Bench(GPT-4) | LC% vs π₀ | 长度 |
|------|-----------------|-----------|------|
| GRPO | 8.425 | 50.50 | 1559 |
| RLOO | 8.409 | 52.25 | 1580 |
| **DAR** | **8.538** | **54.17** | **1358** |

### 消融：α的影响

| α | 效果 | 说明 |
|---|------|------|
| α=1.0 | 保守，低奖励 | 完全绑定初始模型 |
| α=0.1 | **最佳平衡** | 允许探索但有约束 |
| α=0.0 | 高奖励但reward hacking | 8% missing-EOS率 |

### 关键发现
- **DAR在TL;DR上达98.27%胜率**：几乎完美的偏好对齐
- **回归变换是关键**：直接RL的双KL(DAO)训练不稳定，双PPO高方差，只有DAR稳定优越
- **样本效率**：DAR用**一半的标注量**达到DAP方法同等效果
- **长度控制**：DAR生成长度(1358)接近原始模型(1340)，不会length hacking

## 亮点与洞察
- **两个约束冲突的深刻发现**：指出RLHF中两类正则化(防hacking vs 防崩溃)实际上在优化中逐步对抗。这个观察解释了为什么很多RLHF方法效果不如预期
- **对数空间插值的优雅解法**：将两个KL项统一为对插值参考的单KL，理论上等价且实践上释放了优化空间
- **回归变换消除RL不稳定性**：将RL问题转化为加权SFT，避免了策略比率估计的方差问题；权重裁剪提供了进一步的稳定性

## 局限性 / 可改进方向
- **需要在线采样**：每步需要从当前策略采样计算优势，开销比offline DPO大
- **α和β需要联合调优**：Pareto前沿依赖(α,β)的选择
- **改进思路**：可结合NSPO的零空间投影——在DAR的加权SFT中确保安全梯度不损害通用能力

## 相关工作与启发
- **vs PPO**：PPO独立处理两类约束→冲突；DAR统一→Pareto前沿提升
- **vs GRPO**：GRPO无价值模型+组相对优势，但仍用策略比率做RL；DAR用回归变换消除比率不稳定
- **vs DPO(offline)**：DPO在固定偏好数据上训练，DAR在线采样+动态参考→更强泛化

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 两类约束冲突的发现+对数插值的解法都很深刻
- 实验充分度: ⭐⭐⭐⭐⭐ 多设置(直接对齐+标准RLHF)×多模型×多评估器，消融详尽
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，问题动机清晰
- 价值: ⭐⭐⭐⭐⭐ 为RLHF训练稳定性提供了新理论视角和实用解决方案
