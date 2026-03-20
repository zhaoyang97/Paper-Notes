# Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences

**会议**: ICLR 2026  
**arXiv**: [2602.06788](https://arxiv.org/abs/2602.06788)  
**代码**: 无  
**领域**: LLM对齐 / 偏好优化  
**关键词**: DPO, f-divergence, likelihood displacement, preference optimization, SquaredPO  

## 一句话总结
发现 f-DPO 的可解性不需要 f 凸（仅需 $\lim_{t\to 0^+} f'(t) = -\infty$），进一步证明 $\arg\min f(t) \geq 1$ 是抵抗概率位移的必要条件，由此提出 SquaredPO（$f(t) = \frac{1}{2}(\log t)^2$，非凸），在保持性能的同时显著缓解 winner 概率下降问题。

## 研究背景与动机
1. **领域现状**：DPO 及其变体是 LLM 对齐的主流方法，本质上是在 RLHF 目标中用 KL 散度约束策略偏离参考模型。Wang et al. (2024) 将 KL 推广为 f-divergence，但仅限于凸 f。
2. **现有痛点**：DPO 存在"概率位移"（probability displacement）现象——训练过程中 winner 和 loser 的概率都趋近零。这导致过训练时性能急剧下降，是 DPO 最广为诟病的实际问题。
3. **核心矛盾**：KL 散度对应的 $f_{KL}(t) = t\log t$，其 $\arg\min = e^{-1} < 1$，这在理论上决定了 DPO 必然导致 winner 概率下降至少 $e^{-1}$ 倍。凸 f-divergence 类中很难找到同时满足可解性和抗位移的 f。
4. **本文要解决什么？**（1）f-DPO 的可解性条件到底是什么？（2）哪些 f 能从理论上防止概率位移？（3）能否设计一个同时可解且抗位移的损失？
5. **切入角度**：放弃凸性要求，在更广的函数类中寻找满足两个条件的 f。
6. **核心idea一句话**：用 $f(t) = \frac{1}{2}(\log t)^2$（非凸、抗位移）替换 $f(t) = t\log t$（凸、会位移），得到理论更优的 SquaredPO 损失。

## 方法详解

### 整体框架
从一般化的 RLHF 目标出发：$\max_{\pi_\theta} \mathbb{E}[r(x,y)] - \beta D_f[\pi_\theta \| \pi_{ref}]$。f-DPO 损失为 $-\log\sigma(\beta f'(\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}) - \beta f'(\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}))$。论文分两步：(1) 确定 f 的可解性条件 (2) 确定 f 的抗位移条件。

### 关键设计

1. **DPO-Inducing 条件（可解性）**：
   - 做什么：精确刻画哪些 f 能让 RLHF 问题保持可解。
   - 核心结果 (Corollary 1)：f 是 DPO-inducing 等价于 $\lim_{t\to 0^+} f'(t) = -\infty$。
   - 意义：凸性不是必要条件！只要 f 在 0 附近导数趋负无穷（保证最优策略对所有 response 赋正概率），就可以用。这大幅扩展了可用的 f 类。

2. **Displacement-Resistant 条件（抗位移）**：
   - 做什么：刻画哪些 f 能防止 winner 概率下降。
   - 核心结果 (Lemma 2)：若 $\arg\min_{t \geq 0} f(t) < 1$，则最优策略对 in-sample response 的概率必然低于 $c \cdot \pi_{ref}$。因此抗位移的必要条件是 $\arg\min f(t) \geq 1$。
   - DPO 的问题：$f_{KL}(t) = t\log t$ 的最小值在 $t = e^{-1} < 1$，所以 DPO 理论上必然位移。
   - 关键洞察 (Lemma 1)：f-DPO 不仅解完整的 RLHF 问题 (5)，也同时解一个退化问题 (7)——其正则化仅覆盖 in-sample responses。这意味着 f-DPO 对 out-of-sample 行为没有约束，是位移的根本原因。

3. **SquaredPO 损失**：
   - 做什么：一个满足两个条件的具体损失。
   - $f(t) = \frac{1}{2}(\log t)^2$，是非凸函数，$\lim_{t\to 0^+} f'(t) = -\infty$（DPO-inducing），$\arg\min f(t) = 1$（displacement-resistant）。
   - 损失形式：相当于 "DPO with adaptive $\beta$"，$\beta_\theta(y,x) = \beta / \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$。当 winner 概率下降时，其 $\beta$ 自动增大，强化正则化，抑制进一步下降。
   - 与 SimPO/$\beta$-DPO 的区别：SimPO 的自适应 $\beta$ 仅依赖长度且训练中固定；$\beta$-DPO 引入额外超参数。SquaredPO 的自适应 $\beta$ 从理论自然推导，无额外超参数。

## 实验关键数据

### 概率位移缓解
| 指标 | SquaredPO | DPO |
|------|-----------|-----|
| Epoch 1 chosen log-ratio 中位数 | 更高（位移更小） | 更低（位移更严重） |
| 单调下降的 winner 占比（4 epoch） | **4.21%** | **99.63%** |

关键发现：DPO 中 99.63% 的 winner 概率一旦在第 1 个 epoch 下降，后续每个 epoch 都继续下降（单调下降）。SquaredPO 将这一比例降至 4.21%。

### 过训练鲁棒性（TL;DR Win Rate vs Base Model）
| Epochs | SquaredPO | χPO | DPO |
|--------|-----------|-----|-----|
| 1 | 50.8% | 51.2% | 51.8% |
| 2 | 50.6% | 48.9% | 45.0% |
| 4 | **51.0%** | 48.3% | **34.7%** |

DPO 在 4 epoch 后 win rate 降至 34.7%（严重过训练），SquaredPO 保持 51.0%。

### 标准基准（1 epoch）
| 方法 | AlpacaEval LC↑ | AlpacaEval WR↑ | MT-Bench↑ |
|------|---------------|----------------|-----------|
| SquaredPO | 29.2 | 24.5 | 7.924 |
| DPO | 29.6 | 24.8 | 7.925 |

性能基本持平，但 SquaredPO 未调超参（使用 DPO 默认值）。

## 亮点与洞察
- **从理论推导出的"自适应 $\beta$"**：SquaredPO 的核心直觉极为简单——当 winner 概率下降时自动加大正则化。但这不是启发式设计，而是从 f-divergence 理论中自然推导出来的。
- **Lemma 1 的深刻揭示**：f-DPO 同时解决完整问题和退化问题，意味着所有 f-DPO 变体都具有对 out-of-sample 行为缺乏约束的结构性缺陷。位移不是 bug，而是数学上的必然。
- **99.63% 单调下降**：首次报告 DPO 中 winner 概率单调下降的现象，这比之前"平均概率下降"的报告更精确和令人震惊。

## 局限性 / 可改进方向
- 仅在单一数据集（TL;DR）和单一模型（Llama-3-8B）上验证并使用 LoRA。
- Displacement-resistant 条件被证明是必要条件，但不是充分条件——满足条件并不保证完全消除位移。
- SquaredPO 在第 1 个 epoch 略逊于 DPO，超参数（$\beta$）未针对 SquaredPO 调优。
- 仅探索了一个具体的 f（$(\log t)^2/2$），还有许多满足两个条件的 f 值得探索。

## 相关工作与启发
- **vs DPO (Rafailov et al., 2023)**：DPO 是 $f_{KL}(t) = t\log t$ 的特例，$\arg\min = e^{-1}$，理论上必然位移。SquaredPO 用 $f(t) = \frac{1}{2}(\log t)^2$ 保证 $\arg\min = 1$，从根上解决。
- **vs χPO (Huang et al., 2025)**：χPO 也用 f-DPO 框架（$\chi^2$ 散度），有过训练鲁棒性但不如 SquaredPO。本文理论更一般（覆盖所有 f），χPO 只分析一个特例。
- **vs SimPO/β-DPO**：这些方法用启发式自适应 $\beta$，SquaredPO 的自适应 $\beta$ 从理论推导，无额外超参。
- **vs RCPO (Beyond Pairwise)**：RCPO 关注偏好数据格式（pairwise → ranked），SquaredPO 关注正则化的数学性质。两者正交，可以组合。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 完全刻画 DPO-inducing 条件 + 首次提出 displacement-resistant 条件，理论贡献深刻
- 实验充分度: ⭐⭐⭐ 仅一个数据集/模型，但位移分析很详尽
- 写作质量: ⭐⭐⭐⭐⭐ 理论结构清晰，定义→引理→定理的逻辑链完美，Venn 图直观
- 价值: ⭐⭐⭐⭐ 为 DPO 类方法提供了设计原则（两个条件），对未来偏好优化研究有指导意义
