# A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation

**会议**: NeurIPS 2025  
**arXiv**: [2404.11577](https://arxiv.org/abs/2404.11577)  
**代码**: 无  
**领域**: AI安全 / 机器遗忘评估 / 隐私保护  
**关键词**: machine unlearning, evaluation metric, cryptographic game, membership inference attack, SWAP test  

## 一句话总结
将机器遗忘的评估问题建模为密码学博弈（unlearning sample inference game），通过定义adversary的"advantage"来衡量遗忘质量，克服了传统MIA准确率作为评估指标的多种缺陷（不以retrain为零基准、对数据划分敏感、对MIA选择敏感），并提出SWAP test作为高效的实用近似方案。

## 背景与动机
机器遗忘（Machine Unlearning）由GDPR等数据保护法规中的"被遗忘权"驱动——用户有权要求组织从模型中删除其个人数据。朴素的解决方案是重新训练，但代价太高。因此出现了大量近似遗忘算法，但**如何可靠评估这些算法的数据移除有效性**仍是一个关键的开放问题（NeurIPS 2023的机器遗忘竞赛也强调了这点）。

现有评估方式主要有三类：
- **重训练对比**（比参数差异/后验差异）：受训练随机性影响大，不可靠
- **理论保证**（certified removal）：依赖凸性等强假设，实践中难用
- **攻击基评估**（MIA）：最常见但问题最多——MIA准确率本身不校准，retrain不能保证获得最低MIA准确率，且结果依赖MIA算法选择和数据划分方式

## 核心问题
如何设计一个理论可靠、实践可用的机器遗忘评估指标？具体来说，好的评估指标应满足：(1) retrain作为理想遗忘方法应获得完美分数；(2) 与certified removal等理论保证一致；(3) 不受特定MIA算法选择的影响。现有的MIA准确率/AUC都无法同时满足这三个性质。

## 方法详解

### 整体框架
核心思想来自密码学中的安全博弈：定义一个"遗忘样本推断博弈"（Unlearning Sample Inference Game），包含挑战者（遗忘算法）和对手（MIA攻击者），通过对手的"advantage"来量化遗忘质量。这个advantage经过精心设计，具备理论上可证的优良性质。

博弈 $\mathcal{G} = (\text{Ul}, \mathcal{A}, \mathcal{D}, \mathbb{P}_\mathcal{D}, \alpha)$ 包含三个阶段：

### 关键设计

1. **初始化阶段**：将数据集 $\mathcal{D}$ 随机划分为 retain set $\mathcal{R}$、forget set $\mathcal{F}$、test set $\mathcal{T}$，满足遗忘比例 $\alpha = |\mathcal{F}|/(|\mathcal{R} \cup \mathcal{F}|)$ 且 $|\mathcal{F}| = |\mathcal{T}|$。构建随机预言机 $\mathcal{O}_s(b)$，当 $b=0$ 时从 $\mathcal{F}$ 采样，$b=1$ 时从 $\mathcal{T}$ 采样。引入敏感性分布 $\mathbb{P}_\mathcal{D}$ 使得更敏感的数据被采样概率更高。

2. **挑战者阶段**：遗忘算法 Ul 在 $\mathcal{R} \cup \mathcal{F}$ 上训练的模型上执行遗忘 $\mathcal{F}$，输出遗忘后模型 $m = \text{Ul}(\text{Lr}(\mathcal{R} \cup \mathcal{F}), \mathcal{F})$。

3. **对手阶段**：对手 $\mathcal{A}$ 只能访问遗忘模型 $m$ 和预言机 $\mathcal{O}$，需猜测 $b \in \{0, 1\}$，即判断预言机给出的数据到底来自 $\mathcal{F}$ 还是 $\mathcal{T}$。

4. **Advantage 定义**：对所有可能的数据划分取平均并取绝对值：
$$\text{Adv}(\mathcal{A}, \text{Ul}) = \frac{1}{|\mathcal{S}_\alpha|} \left| \sum_{s \in \mathcal{S}_\alpha} \Pr(\mathcal{A}^{\mathcal{O}_s(0)}(m)=1) - \sum_{s \in \mathcal{S}_\alpha} \Pr(\mathcal{A}^{\mathcal{O}_s(1)}(m)=1) \right|$$
**Unlearning Quality** 定义为 $\mathcal{Q}(\text{Ul}) = 1 - \sup_\mathcal{A} \text{Adv}(\mathcal{A}, \text{Ul})$。

5. **SWAP Test（核心实用工具）**：直接枚举所有划分不可行。SWAP test只需两个对称划分 $s = (\mathcal{R}, \mathcal{F}, \mathcal{T})$ 和 $s' = (\mathcal{R}, \mathcal{T}, \mathcal{F})$（交换forget和test集），用这对swap pair的平均advantage近似完整指标。关键洞察：swap pair保留了原始定义的对称性，因此保留了零基准等性质；而随机选两个划分如果有重叠，会导致简单的查表攻击就能获得advantage=1，完全失效。

6. **弱对手假设**：由于SOTA的MIA都是对每个数据点独立决策，实际计算时限制对手只与预言机交互一次（weak adversary），通过多个SOTA MIA中选最强的来近似supremum。

### 核心定理

- **零基准定理 (Theorem 3.3)**：对任意对手 $\mathcal{A}$，$\text{Adv}(\mathcal{A}, \text{Retrain}) = 0$，即 $\mathcal{Q}(\text{Retrain}) = 1$。证明依赖对称性——每个数据点在所有划分中出现在 $\mathcal{F}$ 和 $\mathcal{T}$ 的次数完全对称，MIA的偏差会被抵消。
- **Certified Removal保证 (Theorem 3.5)**：若 Ul 是 $(\epsilon, \delta)$-certified removal，则 $\text{Adv}(\mathcal{A}, \text{Ul}) \leq 2(1 - \frac{2 - 2\delta}{e^\epsilon + 1})$。这建立了提出的指标与理论保证之间的联系。

## 实验关键数据

在CIFAR10上使用ResNet-20，比较了Retrain/Fisher/Ftfinal/Retrfinal/NegGrad/SalUn/SSD等7种遗忘方法。

| 设置 | 关键发现 |
|------|---------|
| 数据集大小消融 | 不同大小下遗忘方法排序一致，小规模实验可可靠预测大规模表现 |
| α消融 | 不同遗忘比例下排序一致 |
| DP预算验证 | Retrain的 $\mathcal{Q} \approx 1$（验证零基准）；$\mathcal{Q}$ 与 $\epsilon$ 负相关（验证Theorem 3.5）|
| 与IC Score对比 | IC Score在DP预算变化时无法产生负相关结果，且排序不一致 |
| 与MIA AUC对比 | MIA AUC的区分度差，大部分方法得分挤在一起（0.4-0.5），难以区分 |
| 更多数据集 | CIFAR100/MNIST/SST5上排序一致，且反映数据集难度（MNIST上Q更高，CIFAR100上更低）|
| 模型架构 | ResNet-20/56/110上排序一致 |

### 消融实验要点
- SSD在所有设置下显著优于其他近似遗忘方法，与其SOTA地位一致
- SWAP test相比随机划分方法标准差显著更低，更稳定
- 弱对手限制不影响评估实用性，因为当前SOTA MIA都是弱对手

## 亮点
- **密码学视角的巧妙迁移**：把安全博弈的advantage概念引入遗忘评估，通过对称性天然地解决了retrain零基准问题，而不是人为强制offset
- **SWAP test的简洁设计**：只需两次训练+遗忘就能近似完整指标，且保留核心理论性质，实用性极强
- **统一框架**：同时兼容黑盒/白盒对手、弱/强对手，理论推广性好
- **与certified removal的桥接**：Theorem 3.5建立了实证指标与理论保证之间的量化联系，是少见的理论-实践桥接工作

## 局限性 / 可改进方向
- **弱对手的能力瓶颈**：即使None（不做任何遗忘）的 $\mathcal{Q}$ 仍高达0.587，说明当前SOTA MIA还不够强。如果未来出现更强的MIA，评估指标的区分度会更好
- **仅限i.i.d.遗忘场景**：均匀划分限制了对非i.i.d.遗忘（如整类遗忘）的适用性，作者也承认推广到非均匀划分会破坏现有理论
- **实验计算成本高**：虽然SWAP test已简化，但训练多个模型（含DP模型）的总实验成本仍不低（需约6天GPU时间来做标准差消融）
- **数据集/任务范围有限**：主要在图像分类上验证，缺少生成模型、大语言模型等场景的验证

## 与相关工作的对比
- **vs Triantafillou & Kairouz (2023)**：使用假设检验+MIA估计隐私预算，理论更严格但计算成本极高（需对每个遗忘样本训练MIA），本文方法更实用
- **vs Goel et al. (2023) IC Test**：IC Test通过混淆类标签来间接评估遗忘，但对DP预算不敏感且排序不一致；本文直接衡量隐私泄露
- **vs MIA AUC (Golatkar et al. 2021)**：MIA AUC不满足零基准，对数据划分敏感，区分度差；本文方法克服所有这些问题
- **vs Brimhall et al. (2025) Computational Unlearning**：类似的密码学启发，但在不可区分性博弈中比较retrained和unlearned模型，技术假设不同

## 启发与关联
- 与 `ideas/ai_safety/20260316_structure_faithful_unlearning.md` 直接相关——该idea中"遗忘验证协议"可以考虑采用本文的SWAP test框架作为评估标准
- SWAP test的对称性设计思想可以迁移到其他对比评估场景（如数据增强的有效性评估）
- 如果要做面向大语言模型的遗忘评估，需要解决非i.i.d.遗忘的理论推广问题，这可能是一个重要的研究方向

## 评分
- 新颖性: ⭐⭐⭐⭐ 密码学博弈视角评估遗忘是新的，但advantage的概念在密码学中已很成熟
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多架构、多消融、与多种baseline对比，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 定理清晰、证明完整、discussion透彻，写作水平很高
- 价值: ⭐⭐⭐⭐ 为遗忘评估提供了更可靠的工具，但受限于SOTA MIA的能力和i.i.d.假设
