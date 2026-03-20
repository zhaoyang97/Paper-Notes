# Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation

**会议**: ACL 2025  
**arXiv**: [2502.10762](https://arxiv.org/abs/2502.10762)  
**代码**: [https://github.com/andyclsr/BoneSoups](https://github.com/andyclsr/BoneSoups)  
**领域**: 文本生成  
**关键词**: 模型合并, 多目标生成, 可控生成, Pareto最优, 强化学习  

## 一句话总结

提出 Bone Soup 模型合并方法，通过先构造"骨架奖励"（多目标奖励的组合）训练骨架模型、再用对称循环矩阵映射确定合并系数，解决了 Rewarded Soup 中单目标模型合并的次优性问题，在三个多目标生成任务上实现更好的 Pareto 前沿和可控性。

## 研究背景与动机

1. **领域现状**：可控多目标生成（CMOG）要求 LLM 在推理时根据用户偏好动态调整生成策略（如在事实性 vs 完整性之间权衡）。模型合并（Model Merging/Soup）是主流方案——分别训练各目标专用模型，推理时按用户偏好权重线性插值合并参数。
2. **现有痛点**：Rewarded Soup 等方法为每个目标单独训练一个专用模型，合并时直接用偏好权重作为合并系数。但这忽略了竞争目标之间的相互影响——当不同目标的奖励函数曲率不同时，简单加权合并的结果会显著偏离最优解。
3. **核心矛盾**：用单目标奖励训练的专用模型对各自目标过度特化，模型参数空间中的最优点分布不均匀。线性插值只能到达两个特化点之间的直线上的点，而真正的 Pareto 最优解可能在曲线上。
4. **本文要解决什么？** 如何选择更好的"基底模型"（骨架模型）使得它们的线性组合能更好地逼近 Pareto 前沿。
5. **切入角度**：不直接优化单个奖励，而是先将多个奖励组合为"骨架奖励"，让骨架模型同时考虑多个目标。用数学定理证明了这种方法在大部分偏好区间上优于 Rewarded Soup。
6. **核心idea一句话**：先"选好料"（构造骨架奖励训练骨架模型），再"煲汤"（按用户偏好合并骨架模型）。

## 方法详解

### 整体框架
给定 $n$ 个目标及其奖励函数 $\{r_i\}_{i=1}^n$，和用户偏好权重 $\bm{\mu}$：(1) 通过基向量构造方法生成骨架奖励 $h_j = \sum_i B_{ji} r_i$（$B$ 为组合权重矩阵）；(2) 用骨架奖励通过 MORL 训练骨架模型 $\{\bm{\theta}_j^{\text{bone}}\}$；(3) 推理时用对称循环矩阵映射从用户偏好 $\bm{\mu}$ 计算合并系数 $\bm{\lambda}$，合并骨架模型。

### 关键设计

1. **骨架奖励构造（Backbone Reward Construction）**:
   - 做什么：将原始单目标奖励组合为考虑多目标交互的骨架奖励
   - 核心思路：构造组合矩阵 $B$，使 $[h_1,...,h_n]^T = B[r_1,...,r_n]^T$。$B$ 采用对称循环矩阵形式，参数化为 $\beta \in (1/2, 1)$。例如双目标时 $h_1 = \beta r_1 + (1-\beta)r_2$，$h_2 = (1-\beta)r_1 + \beta r_2$
   - 设计动机：用数学证明（Theorem 1）表明，对于有区分度的二次奖励函数，Bone Soup 在偏好区间 $\mu \in (\frac{1-\sqrt{2\beta^2-2\beta+1}}{2}, \frac{1+\sqrt{2\beta^2-2\beta+1}}{2})$ 上严格优于 Rewarded Soup，区间长度至少 $\frac{\sqrt{2}}{2} \approx 0.71$

2. **骨架模型训练（Backbone Model Training）**:
   - 做什么：用骨架奖励通过多目标强化学习训练骨架模型
   - 核心思路：将骨架奖励 $h_j$ 作为 PPO 训练的奖励信号，得到骨架模型 $\bm{\theta}_j^{\text{bone}}$
   - 设计动机：骨架模型已经在训练中考虑了多目标的权衡，位于参数空间中更靠近真正 Pareto 前沿的区域

3. **合并系数映射（Merging Coefficient Mapping）**:
   - 做什么：根据用户偏好权重自动计算合并系数
   - 核心思路：利用对称循环矩阵 $B$ 的逆映射，从偏好权重 $\bm{\mu}$ 计算合并系数 $\bm{\lambda} = B^{-1}\bm{\mu}$。最终合并模型 $\bar{\bm{\theta}} = \sum_j \lambda_j \bm{\theta}_j^{\text{bone}}$
   - 设计动机：确保合并系数与骨架奖励的构造方式一致，当用户偏好恰好等于某个骨架奖励的组合权重时，直接输出对应骨架模型即为最优解

### 损失函数 / 训练策略
- 骨架模型通过 PPO + 骨架奖励训练
- $\beta$ 从 $\{0.6, 0.7, 0.8\}$ 中选取，仅需训练 20% 步数即可用 hypervolume 评估选择
- 可选外推步骤：$\hat{\bm{\theta}}^b = (1+\alpha)\hat{\bm{\theta}} - \alpha\bm{\theta}_{\text{sft}}$ 进一步提升

## 实验关键数据

### 主实验

| 任务 | 方法 | Pareto 前沿 | 可控性 |
|------|------|-----------|--------|
| Long Form QA (事实性vs完整性) | Rewarded Soup | 次优前沿 | 10-11点 |
| Long Form QA (事实性vs完整性) | **Bone Soup** | **接近甚至超越 Oracle MORLHF** | 10-11点 |
| Helpful Assistant (helpful vs harmless) | Rewarded Soup | 标准前沿 | - |
| Helpful Assistant (helpful vs harmless) | **Bone Soup** | **显著优于 RS 和 MOD** | - |
| Reddit Summary (faithful vs preference) | Rewarded Soup | 标准前沿 | - |
| Reddit Summary (faithful vs preference) | **Bone Soup** | **显著优于 RS** | - |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| ABA（仅骨架训练，直接用偏好合并） | 不稳定 | 部分 trade-off 劣于 RS，验证了两阶段的必要性 |
| β=0.6 | 偏弱 | 骨架奖励过于接近原始奖励 |
| β=0.7 | 最佳 | 适度混合 |  
| β=0.8 | 较好 | 混合更强 |
| 外推（α=0.1-0.5） | 进一步提升 | 减少 SFT 初始化模型的影响 |

### 关键发现
- Bone Soup 在三目标设定下同样有效，Pareto 前沿甚至主导 Oracle MORLHF 的前沿
- GPT-4 评估与奖励模型评估结论一致，证实了改进的鲁棒性
- 直接合并骨架模型（ABA，不用映射系数）不稳定——说明合并系数的正确映射是关键
- 训练开销与 Rewarded Soup 相当（同样训练 $n$ 个模型），推理开销完全相同

## 亮点与洞察

- **数学证明模型合并的次优性**并给出修复方案——Theorem 1 严格证明了 Bone Soup 在至少 71% 的偏好区间上优于 Rewarded Soup，这不是经验观察而是理论保证。
- **"煲汤先选料"的隐喻**准确传达了核心思想——不是随便拿几个模型合并，而是要精心选择基底模型。
- **对称循环矩阵**的设计简洁优雅——只引入一个超参 $\beta$，既保证了理论性质又易于实现。
- 该方法可直接迁移到视觉、多模态等其他领域的模型合并场景。

## 局限性 / 可改进方向

- 仅用自动评估器（奖励模型+GPT-4），无人工评估
- 仅实验了 2-3 个目标，更多目标的扩展性未验证
- $\beta$ 仍需搜索，缺乏自适应选择机制
- 理论分析限于二次奖励函数，实际奖励函数更复杂

## 相关工作与启发

- **vs Rewarded Soup**: RS 用单目标模型直接合并，忽略目标间交互；Bone Soup 通过骨架奖励在训练时考虑目标交互，合并结果更优
- **vs MORLHF**: MORLHF 对每个偏好点单独训练一个模型（枚举），成本极高且推理时不可控；Bone Soup 只需训练 $n$ 个骨架模型
- **vs MOD（解码时方法）**: MOD 在 logit 层面合并多模型输出，灵活但引入推理开销；Bone Soup 在参数层面合并，推理零开销

## 评分

- 新颖性: ⭐⭐⭐⭐ 理论驱动的方法设计——Theorem 1 严格证明 Bone Soup 在至少 71% 偏好区间上优于 Rewarded Soup，对称循环矩阵参数化简洁优雅
- 实验充分度: ⭐⭐⭐⭐ 三个任务（Long Form QA/Helpful Assistant/Reddit Summary）八个奖励模型两个基础模型，消融充分，但缺少人工评估
- 写作质量: ⭐⭐⭐⭐ 动机用 Example 1 具体说明 Rewarded Soup 的次优性很有说服力，Figure 3 的oracle对比直观
- 价值: ⭐⭐⭐⭐ 对模型合并领域有实质性贡献——"煲汤先选料"的思想通用且可迁移到视觉、多模态等其他领域的模型合并
