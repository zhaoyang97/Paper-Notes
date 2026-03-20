# A Unifying View of Linear Function Approximation in Off-Policy RL Through Matrix Splitting and Preconditioning

**会议**: NeurIPS 2025 (Spotlight, top 3%)  
**arXiv**: [2501.01774](https://arxiv.org/abs/2501.01774)  
**代码**: 无  
**领域**: 强化学习 / 策略评估 / 理论分析  
**关键词**: temporal difference learning, fitted Q-iteration, matrix splitting, preconditioning, convergence analysis  

## 一句话总结
将线性函数逼近下的TD、FQI和PFQI统一为求解同一线性系统的迭代方法（仅预条件子不同），首次引入矩阵分裂理论来分析它们的收敛性，给出了各算法收敛的充要条件，并揭示了TD收敛不一定意味着FQI收敛（反之亦然）。

## 背景与动机
在Off-Policy策略评估（OPE）中，TD学习可能发散，FQI通常被认为更稳定。传统观点认为TD、FQI、PFQI的区别仅在于对目标值函数的更新次数（TD=1次，FQI=∞次，PFQI=有限次）。但这一直觉性理解无法正确解释为什么TD收敛时FQI可能发散，也无法建立三种算法之间的严格收敛关系。此外，现有理论分析通常依赖于"特征线性独立"等强假设，限制了理论的适用范围。

## 核心问题
1. TD、FQI和PFQI在数学上有什么本质联系？
2. 每种算法收敛的充要条件到底是什么（不依赖特征线性独立假设）？
3. 三种算法的收敛性之间有怎样的蕴含关系？Target network技术的理论本质是什么？

## 方法详解

### 整体框架
核心洞察：TD、FQI、PFQI都是求解同一个目标线性系统 $(Σ_{cov} - γΣ_{cr})θ = θ_{ϕ,r}$ 的迭代方法，区别**仅在于预条件子M**：
- TD: $M_{TD} = αI$（常数预条件子）
- FQI: $M_{FQI} = Σ_{cov}^{-1}$（数据-特征自适应预条件子）  
- PFQI: $M_{PFQI} = α\sum_{i=0}^{t-1}(I - αΣ_{cov})^i$（从TD到FQI的过渡）

迭代格式统一为：$θ_{k+1} = (I - MA)θ_k + Mb$

### 关键设计
1. **Rank Invariance条件（秩不变性）**：提出新条件 $Rank(Φ) = Rank(Φ^⊤D(I - γP_π)Φ)$，证明它是目标线性系统对任意奖励函数都有解的充要条件。该条件等价于 $γΣ_{cov}^†Σ_{cr}$ 没有等于1的特征值，在实践中几乎总是满足的。
2. **预条件子连续变换**：随着PFQI中更新次数t的增加，$M_{PFQI}$ 从 $αI$（t=1时=TD）连续过渡到 $Σ_{cov}^{-1}$（t→∞时=FQI）。这揭示了target network技术的本质：从常数预条件子过渡到数据自适应预条件子。
3. **Proper Splitting**：当rank invariance成立时，$Σ_{cov}$ 和 $Σ_{cr}$ 构成 $(Σ_{cov} - γΣ_{cr})$ 的proper splitting，使FQI的收敛条件放松为 $ρ(γΣ_{cov}^†Σ_{cr}) < 1$，并保证不动点唯一。这从理论上解释了FQI比TD更稳定的实验观察。

### 核心理论结果
- **FQI收敛充要条件**（Theorem 5.1）：线性系统一致 + $H_{FQI}$ 半收敛
- **TD收敛充要条件**（Theorem 6.1）：线性系统一致 + $H_{TD}$ 半收敛
- **TD稳定性**（Corollary 6.2）：存在使TD收敛的学习率 ⟺ 一致性 + 正半稳定性 + $A_{LSTD}$ 的index ≤ 1
- **学习率形成区间**（Corollary 6.3）：首次证明当大学习率不行时，小学习率可能有效——可行学习率形成区间(0,ε)
- **On-policy TD无需线性独立特征**（Theorem 6.4）：经典结论要求特征线性独立，本文证明可以去掉这一假设
- **PFQI增加更新次数可能发散**（Section 7）：当特征不线性独立时，增加t（target network更新频率）可能导致发散

## 实验关键数据
本文是纯理论工作，无实验数据。通过反例构造证明：
- TD收敛但FQI发散的例子存在
- FQI收敛但TD发散的例子存在

### 消融实验要点
- Rank invariance单独即可保证FQI线性系统非奇异，而目标线性系统需要rank invariance + 特征线性独立
- 线性独立特征假设对FQI收敛不是关键的，但决定了FQI求解的是哪个线性系统
- 在on-policy设定下，rank invariance自动成立

## 亮点
- **统一框架的优雅性**：用一个简单的预条件子差异就统一了三种算法，非常简洁
- **首次引入矩阵分裂理论**：将数值线性代数中的经典工具（matrix splitting, proper splitting, semiconvergent matrices）引入RL收敛分析
- **充要条件而非充分条件**：比之前的工作更锋利，且修正了文献中的多处错误
- **实用洞察**：(1) 学习率形成区间→调参有理论依据；(2) target network是预条件子变换→为DQN中的target network提供理论解释；(3) rank invariance→新的温和假设取代线性独立
- **Encoder-decoder视角**：提供了理解TD收敛的新视角

## 局限性 / 可改进方向
- **仅限线性函数逼近**：核心理论依赖线性结构，无法直接推广到神经网络（虽然最后一层通常是线性的）
- **仅限策略评估**：未涉及控制（policy improvement）场景
- **无实证验证**：纯理论工作，缺少对实际问题规模的经验验证
- **Expected TD为主**：虽然声称结果可推广到stochastic TD和batch TD，但主体分析在expected TD（确定性版本）上

## 与相关工作的对比
- **vs Tsitsiklis & Van Roy (1996)**：经典结论需要on-policy + 特征线性独立，本文证明可去掉线性独立假设
- **vs Fellows et al. (2023)**：仅给了PFQI的充分条件，本文给出充要条件
- **vs Asadi et al. (2024) / Xiao (2021)**：声称给出了FQI收敛的充要条件，但本文指出那些实际上只是充分条件
- **vs Ghosh et al. (2020)**：声称线性独立特征足以保证off-policy TD不动点唯一，本文指出还需要rank invariance

## 启发与关联
- 矩阵分裂和预条件子的视角可能启发新的RL算法设计：选择更好的预条件子可以加速收敛
- Rank invariance条件有可能成为RL理论分析中新的标准假设
- 对DQN中target network的理论理解（预条件子变换）可能启发更好的target network更新策略

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 矩阵分裂视角完全是新的，统一三种算法的优雅性很高
- 实验充分度: ⭐⭐⭐ 纯理论工作，反例构造有效但无实证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但数学密度极高，附录50+页
- 价值: ⭐⭐⭐⭐⭐ Spotlight论文，在RL理论领域有重要影响，修正了多处文献错误
