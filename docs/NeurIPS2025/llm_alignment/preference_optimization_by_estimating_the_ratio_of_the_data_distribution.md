# Preference Optimization by Estimating the Ratio of the Data Distribution

**会议**: NeurIPS 2025  
**arXiv**: [2505.19601](https://arxiv.org/abs/2505.19601)  
**代码**: [GitHub](https://github.com/aailab-kaist/BPO)  
**领域**: 对齐RLHF  
**关键词**: DPO, Bregman divergence, likelihood ratio estimation, preference optimization, alignment

## 一句话总结
将 DPO 重新解释为似然比估计（ratio matching）问题，基于 Bregman 散度框架提出 BPO（Bregman Preference Optimization），包含 DPO 为特例的广义损失函数族，并设计了 SBA（Scaled Basu's Power Divergence）实例，在 Llama-3-8B 上实现 55.9% AlpacaEval2 length-controlled win rate 的 SOTA。

## 研究背景与动机
1. **领域现状**：DPO 是最流行的直接偏好优化方法，将 RLHF 简化为偏好数据上的逻辑回归。后续工作（f-DPO、f-PO）扩展了 DPO 的损失函数，但各有不足。
2. **现有痛点**：
   - **f-DPO**：扩展了损失函数形式但**丧失了目标最优性保证**——最小化 f-DPO 不一定收敛到 DPO 定义的最优策略
   - **f-PO**：保留最优性但**需要额外训练奖励模型+Monte Carlo 估计配分函数**，增加了大量计算开销
   - 没有方法能同时满足：(O) 最优性保证、(S) 简洁性（无额外训练开销）、(G) 通用性（多种目标函数）
3. **核心矛盾**：扩展 DPO 损失时，最优性和简洁性似乎不可兼得——f-PO 保优但不简洁，f-DPO 简洁但不保优。
4. **本文要解决什么？** 找到一种既保持目标最优性、又不需要额外计算开销、同时支持多种损失函数实例的通用偏好优化框架。
5. **切入角度**：从似然比估计的视角重新理解 DPO——最优策略可以通过其**似然比**来唯一确定（无需奖励模型或配分函数），因此问题转化为用 Bregman 散度做比率匹配。
6. **核心idea一句话**：DPO 本质上在匹配模型比率 $R_\theta$ 到数据比率 $R_{\text{data}}$，选择不同的 Bregman 散度 $h$ 就得到不同的损失函数，所有实例都保最优性且无额外开销。

## 方法详解

### 整体框架
将偏好优化重新建模为两个比率之间的匹配问题。$R_{\text{data}} = \frac{p_{\text{data}}(\mathbf{y}_w \prec \mathbf{y}_l | \mathbf{x})}{p_{\text{data}}(\mathbf{y}_w \succ \mathbf{y}_l | \mathbf{x})}$ 是数据偏好比率，$R_\theta = \left[\frac{\pi_\theta(\mathbf{y}_l|\mathbf{x})\pi_{\text{ref}}(\mathbf{y}_w|\mathbf{x})}{\pi_\theta(\mathbf{y}_w|\mathbf{x})\pi_{\text{ref}}(\mathbf{y}_l|\mathbf{x})}\right]^\beta$ 是模型比率。最小化 $D_h(R_{\text{data}} || R_\theta)$ 即可让 $\pi_\theta$ 收敛到最优策略。关键技巧是通过类似 implicit score matching 的方法推导出不含 $R_{\text{data}}$（不可直接计算）的等价目标。

### 关键设计

1. **Proposition 1：最优策略的似然比表示**：
   - 做什么：证明最优策略可以仅通过参考模型和偏好数据分布来刻画（不需要奖励模型和配分函数）
   - 核心公式：$\frac{\pi_{\theta^*}(\mathbf{y}_w|\mathbf{x})}{\pi_{\theta^*}(\mathbf{y}_l|\mathbf{x})} = \frac{\pi_{\text{ref}}(\mathbf{y}_w|\mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}_l|\mathbf{x})} \times \left(\frac{p_{\text{data}}(\mathbf{y}_w \succ \mathbf{y}_l|\mathbf{x})}{p_{\text{data}}(\mathbf{y}_w \prec \mathbf{y}_l|\mathbf{x})}\right)^{1/\beta}$
   - 设计动机：似然比（concrete score）具有完备性——能唯一确定分布，因此匹配似然比就足以恢复目标策略

2. **BPO 目标函数（Theorem 2 & 3）**：
   - 做什么：构造可计算的广义损失函数
   - 核心公式：$\mathcal{L}^h_{\text{BPO}}(R_\theta; p_{\text{data}}) = \mathbb{E}_{p_{\text{data}}}[h'(R_\theta)R_\theta - h(R_\theta) - h'(R_\theta^{-1})]$
   - Theorem 2 证明任意严格凸 $h$ 下最优解都是 $\pi_{\theta^*}$（保最优性）；Theorem 3 证明 $\mathcal{L}^h_{\text{BPO}}$ 与不可计算的 $D_h(R_{\text{data}} || R_\theta)$ 仅差常数（保可计算性）
   - 当 $h(R) = \frac{R\log R - (1+R)\log(1+R)}{2}$ 时恢复标准 DPO

3. **梯度分析（Proposition 4）**：
   - 做什么：分析不同 $h$ 的学习动态差异
   - 核心发现：$\nabla_\theta \mathcal{L} = \mathbb{E}[G_h(R_\theta) \nabla_\theta R_\theta]$——所有 BPO 实例的梯度方向相同（由 $\nabla_\theta R_\theta$ 决定），只有梯度大小 $G_h(R_\theta)$ 不同。$h$ 控制的是不同置信度样本的权重分配
   - 设计动机：解释了为什么不同 $h$ 都能收敛到最优解但实际训练表现不同——关键在于样本加权

4. **SBA（Scaled Basu's Power Divergence）**：
   - 做什么：提出一种新的 BPO 实例，解决 BA 散度的梯度尺度问题
   - 核心思路：$G_{\text{SBA}_\lambda}(R_\theta) = (R_\theta^\lambda + R_\theta^{-\lambda-1})/s$，设 $s=4$ 使初始化时梯度尺度与 DPO 一致。超参 $\lambda$ 控制对高/低置信度样本的敏感度
   - 设计动机：BA 散度的梯度大小随 $\lambda$ 线性放大（$(\lambda+1)$ 倍），导致需要重新调整超参。SBA 消除了这个问题

### 损失函数 / 训练策略
- BPO 是 DPO 的 drop-in replacement，只需修改几行求损失的代码
- 可以与其他 DPO 变体正交组合：将 f-DPO 的模型比率 $R_\theta^{f\text{-DPO}}$ 代入 BPO 框架即可

## 实验关键数据

### 主实验
Dialogue generation（Pythia-2.8B, Anthropic-HH）：

| 方法 | Win Rate vs Preferred ↑ | Win Rate vs SFT ↑ | Entropy ↑ |
|------|----------------------|-------------------|-----------|
| DPO | 48.5% | 71.5% | 2.801 |
| f-DPO (χ²) | 53.5% | 72.0% | 2.369 ↓ |
| f-PO (JS) | 54.5% | 76.0% | 2.531 ↓ |
| **BPO-SBA** | **57.0%** | **77.0%** | **3.010** ↑ |

Llama-3-8B-Instruct on AlpacaEval2：

| 方法 | LC Win Rate |
|------|------------|
| DPO | 51.3% |
| SimPO | 53.7% |
| **BPO-SBA** | **55.9%** |

### 消融实验
| BPO 实例 | Win Rate vs Pref | Entropy | 说明 |
|---------|-----------------|---------|------|
| LR (= DPO) | 48.5% | 2.801 | baseline |
| KLIEP | 48.5% | 2.901 | 多样性提升但生成质量持平 |
| LSIF | 50.5% | 2.908 | 均有改善 |
| BA | 51.0% | 2.803 | 需要调 lr |
| **SBA** | **57.0%** | **3.010** | 兼顾质量和多样性 |

### 关键发现
- **BPO 的核心优势**：其他扩展（f-DPO、f-PO）在 win rate 和 diversity 之间存在 trade-off，而 BPO-SBA 同时提升了两者
- **梯度尺度很关键**：BA 散度理论上和 SBA 等价，但实际中因梯度尺度问题表现差很多，说明偏好优化对超参极其敏感
- **$\lambda$ 的效果**：较大的 $\lambda$ 增强对高置信度样本（$R_\theta$ 远离 1）的关注，适合数据质量较好的场景

## 亮点与洞察
- **似然比视角重新理解 DPO** 非常优雅：DPO 不是在"学习奖励"也不是在"分布匹配"，而是在"匹配偏好比率"。这个视角直接消除了对奖励模型和配分函数的依赖，使得扩展变得自然。
- **Bregman 散度框架统一了 DPO 的所有扩展**：DPO = logistic regression，KLIEP/LSIF/BA 都是不同的 $h$ 选择。这给从业者提供了一个清晰的"菜单"来选择损失函数。
- **SBA 的梯度尺度归一化** 是一个实用的工程贡献：通过简单的缩放使不同 $\lambda$ 可以用相同的超参训练。

## 局限性 / 可改进方向
- **最优 $\lambda$ 需要调参**：虽然框架统一，但选择最好的 $h$（或 $\lambda$）仍然依赖实验，缺乏自动选择机制
- **理论分析均在无限容量假设下**：实际模型的有限容量如何影响不同 $h$ 的选择未被分析
- **实验规模有限**：主要在 Pythia-2.8B 和 Llama-3-8B 上实验，更大模型上的效果未知
- **改进方向**：(1) 自适应的 $h$ 选择策略；(2) 有限容量下不同 BPO 实例的理论比较；(3) 与 online DPO/RLHF 结合

## 相关工作与启发
- **vs DPO**: BPO 包含 DPO 作为特例（$h$ = logistic regression），同时提供了更多损失函数选择
- **vs f-DPO**: f-DPO 扩展了损失函数但丧失最优性，BPO 保持最优性
- **vs f-PO**: f-PO 保最优性但需要额外奖励模型+配分函数估计，BPO 无需任何额外开销
- **vs SimPO/ORPO 等工程变体**: 互补关系——BPO 可以与这些方法的模型比率定义组合使用

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 似然比估计视角非常新颖，Bregman 框架统一且自然
- 实验充分度: ⭐⭐⭐⭐ 对话+摘要+AlpacaEval2，与多个基线对比，消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，Table 1 总结精炼，代码改动量小
- 价值: ⭐⭐⭐⭐⭐ 为偏好优化提供了统一理论框架和实用 SOTA 方法
