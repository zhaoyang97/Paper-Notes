# SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety

**会议**: ICLR 2026 Oral  
**arXiv**: [2505.20065](https://arxiv.org/abs/2505.20065)  
**代码**: 无  
**领域**: AI Safety / LLM对齐  
**关键词**: safety alignment, DPO, constrained optimization, safety margin, PKU-SafeRLHF  

## 一句话总结
重新审视安全约束 RLHF 目标并证明其存在闭式最优策略，据此推导出等价的可处理目标 SafeDPO，仅需在标准 DPO 上加入安全感知数据变换和安全 margin 项（1 个额外超参数），无需奖励/代价模型，在 PKU-SafeRLHF-30K 上实现 96.87% 无害率且保持竞争力的有用性，训练速度比 SafeRLHF 快 25×。

## 研究背景与动机

1. **领域现状**：LLM 部署需要同时保证有用性（helpfulness）和安全性（safety），主流做法是将安全约束引入 RLHF，如 SafeRLHF 使用拉格朗日方法约束代价函数。

2. **现有痛点**：(a) SafeRLHF 需要训练奖励模型+代价模型+两个 value 网络+在线采样，共 6 个网络，复杂度极高（训练 35,200 秒 vs DPO 的 1,388 秒）；(b) SACPO 等方法依赖近似放松，无法保证收敛到原始安全约束问题的解；(c) 直接在 DPO 上分别训练有用性/无害性数据效果不佳——DPO-HELPFUL 有用但不安全，DPO-HARMLESS 安全但不有用。

3. **核心矛盾**：有用性和安全性之间存在天然张力——模型越"配合"用户请求越有用，但也越容易生成有害内容。如何在单阶段训练中同时兼顾？

4. **本文要解决什么？** 能否找到安全约束优化问题的闭式解，从而像 DPO 一样用监督学习直接训练，避免复杂的多阶段管线？

5. **切入角度**：将安全约束转化为代价增强奖励 $r_c(x,y) = r(x,y)$ if safe, $-\infty$ if unsafe，这使得不安全响应在最优策略中概率为零，且该问题有闭式最优策略。

6. **核心idea一句话**：通过安全感知的数据变换（交换不安全 winner）+ 安全 margin 项，将安全约束精确编码进 DPO 损失中，无需额外模型。

## 方法详解

### 整体框架
SafeDPO 在标准 DPO pipeline 上做两个修改：(1) 安全感知数据变换 $T$：当 winner 不安全而 loser 安全时交换它们，当两者都不安全时丢弃该对；(2) 在 DPO loss 中加入安全 margin $\Delta$，增大安全/不安全响应对之间的 margin。

### 关键设计

1. **代价增强奖励与闭式最优策略**
   - 做什么：将硬安全约束转化为显式的奖励修改
   - 核心思路：$r_c(x,y) = r(x,y)$ if $c(x,y) \leq 0$（安全），$-\infty$ otherwise（不安全）。由此闭式最优策略为 $\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp(\frac{1}{\beta} r_c(x,y))$，不安全响应自动获得零概率
   - 设计动机：将看似不可处理的硬约束问题转化为标准的 KL 正则化框架，允许 DPO 式重参数化

2. **安全感知数据变换 $T$**
   - 做什么：根据安全标签重新整理偏好对
   - 核心思路：给定 $(x, y_w, y_l, h_w, h_l)$，$h=1$ 表示不安全：(a) $h_w=0$: 保持不变；(b) $h_w=1, h_l=0$: **交换** winner 和 loser；(c) $h_w=1, h_l=1$: **丢弃**
   - 设计动机：消融实验证明这一步是最关键的——仅给其他 DPO 变体加 margin 效果有限，但安全感知变换可显著提升安全性

3. **SafeDPO 损失与安全 Margin**
   - 做什么：在标准 DPO loss 中增加安全 margin 项
   - 核心思路：$\mathcal{L}(\theta; \Delta) = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(\tilde{y}_w|x)}{\pi_{\text{ref}}(\tilde{y}_w|x)} - \beta \log \frac{\pi_\theta(\tilde{y}_l|x)}{\pi_{\text{ref}}(\tilde{y}_l|x)} - (\tilde{h}_l - \tilde{h}_w)\Delta)]$。Margin $\Delta \geq 0$ 仅在安全-不安全对上激活（$\tilde{h}_l - \tilde{h}_w = 1$），增大安全响应的优势
   - 设计动机：**Proposition 4.4** 证明 $\Delta$ 不改变最优解集（optimality invariance），但改善优化动态——加速远离不安全区域

### 损失函数 / 训练策略
基于标准 DPO 训练框架，$\beta=0.1$, $\Delta=10$（默认），3 epochs, lr=1e-6, cosine schedule。仅需偏好数据 + 二值安全标签（$h \in \{0,1\}$），不需要有害性偏好标签。

## 实验关键数据

### 主实验

| 方法 | 有用性 | 无害率 (%) | 无害性 | 训练时间 |
|------|--------|-----------|-------|---------|
| SFT | 0.00 | 45.49 | -0.77 | — |
| DPO-HELPFUL | 10.00 | 37.59 | -2.23 | — |
| DPO-HARMLESS | 0.52 | 75.69 | 3.14 | — |
| SafeRLHF | 4.23 | 88.97 | 3.63 | 35,200s |
| SACPO | 2.80 | 89.60 | 4.34 | — |
| **SafeDPO** | **4.61** | **96.87** | **5.97** | **1,388s** |

### 消融实验

| 配置 | 关键发现 |
|------|---------|
| $\Delta=0$ (无 margin) | 仍达到高无害率，证明数据变换 $T$ 是核心 |
| $\Delta=10$ (默认) | 最佳权衡 |
| $\Delta=50$ (过大) | 有用性退化（梯度饱和） |
| DPO+只加 margin | 安全性提升有限，远不及 SafeDPO |
| 1.5B → 13B | 安全性和有用性均随模型规模提升 |

### 关键发现
- **数据变换是核心**：消融证明安全感知变换 $T$ 贡献了大部分安全性提升，$\Delta$ 主要改善优化动态
- **25× 训练加速**：SafeDPO 1,388s vs SafeRLHF 35,200s，且只需 2 个网络（policy + reference）vs 6 个
- **GPT-4 评估达到 100% 无害率**：在 GPT-4 评判下 SafeDPO 的无害率达到 100%
- **XSTest over-refusal**：SafeDPO 的 over-refusal 率为 12.4%（SafeRLHF 3.2%），说明安全性提升伴随一定的过度拒绝

## 亮点与洞察
- **理论驱动的简洁方法**：从安全约束问题的闭式解自然推导出方法，不是 ad-hoc 设计。Proposition 4.4 证明 margin 不改变最优解是优雅的理论保证
- **数据变换思路可广泛复用**：安全感知的 winner/loser 交换策略可以应用到任何偏好学习方法中（IPO、KTO 等），不限于 DPO
- **仅需二值安全标签**：不需要训练代价模型或细粒度安全评分，一个简单的 $h \in \{0,1\}$ 标签就够了，大大降低数据标注成本

## 局限性 / 可改进方向
- **Over-refusal 问题**：12.4% 的 over-refusal 率高于 SafeRLHF（3.2%），对某些应用场景可能不可接受
- **实验局限于 PKU-SafeRLHF 数据集**：未在更多安全基准（如 Anthropic HH、BeaverTails）上验证
- **二值安全标签的局限**：现实中安全性是连续谱，$h \in \{0,1\}$ 的粗粒度可能丢失信息
- **与 AuxDPO 的互补**：SafeDPO 解决安全性，AuxDPO 解决误指定问题，两者能否结合？

## 相关工作与启发
- **vs SafeRLHF**：SafeRLHF 使用完整的约束 RL 管线（奖励模型+代价模型+PPO），SafeDPO 证明闭式解可以完全避免这些复杂组件
- **vs Why DPO is Misspecified**：SafeDPO 在 DPO 框架内解决安全性，但未解决 AuxDPO 指出的参数化策略误指定问题。两者正交互补
- **vs SACPO**：SACPO 使用代理目标做松弛，SafeDPO 证明可以直接解原始问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 闭式解推导新颖，但数据变换思路相对直觉化
- 实验充分度: ⭐⭐⭐⭐ 多尺度验证+消融充分，但数据集单一
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，proposition 链条完整
- 价值: ⭐⭐⭐⭐⭐ 极简可用，大幅降低安全对齐门槛，适合工业部署
