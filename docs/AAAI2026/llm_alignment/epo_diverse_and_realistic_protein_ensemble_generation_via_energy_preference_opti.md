# EPO: Diverse and Realistic Protein Ensemble Generation via Energy Preference Optimization

**会议**: AAAI 2026  
**arXiv**: [2511.10165](https://arxiv.org/abs/2511.10165)  
**代码**: 无  
**领域**: 对齐RLHF / 蛋白质生成  
**关键词**: 蛋白质构象集合, 能量偏好优化, SDE采样, Boltzmann分布, 无MD模拟

## 一句话总结
提出EPO（Energy Preference Optimization），将反向SDE采样与listwise能量排序偏好优化结合，用能量信号对齐预训练蛋白质生成器与目标Boltzmann分布，在Tetrapeptides/ATLAS/Fast-Folding三个基准9个指标上达到SOTA，完全消除了昂贵的分子动力学（MD）模拟需求。

## 研究背景与动机
1. **领域现状**：蛋白质功能依赖于构象集合体（而非单一结构），理解这些集合对药物设计至关重要。传统方法依赖昂贵的MD模拟生成构象。
2. **现有痛点**：(a) MD模拟计算成本极高（一个蛋白质需要数天到数周）；(b) 预训练生成模型能生成构象但不遵循Boltzmann分布；(c) 基于DPO的pairwise偏好优化会忽视高能量但重要的亚稳态。
3. **核心矛盾**：想要多样且物理真实的构象集合，但(a)MD太慢、(b)生成模型不遵循热力学、(c)pairwise优化损害多样性。
4. **本文要解决什么？** 仅用能量信号（无需MD轨迹）对齐生成器产生物理真实且多样的蛋白质构象集合。
5. **切入角度**：(1) ODE→SDE转换增加随机性逃离局部最小值；(2) listwise替代pairwise保留ensemble多样性；(3) Jensen不等式推导可计算的上界。
6. **核心idea一句话**：listwise能量排序偏好优化+SDE随机性=多样且物理真实的构象集合，无需MD模拟。

## 方法详解

### 整体框架
输入：预训练蛋白质生成器（ODE-based）+ 能量函数。输出：Boltzmann分布对齐的构象集合生成器。

### 关键设计

1. **ODE→SDE转换**：
   - 做什么：将确定性ODE采样转为随机SDE采样
   - 核心思路：$dx_t = v(x_t,t)dt + \frac{1}{2}w_t s(x_t,t)dt + \sqrt{w_t}d\bar{W}_t$，其中$w_t$控制随机性水平
   - 设计动机：ODE陷在单一构象的局部最小值，SDE随机性帮助逃离能量屏障探索多个亚稳态

2. **Listwise能量排序偏好优化**：
   - 做什么：用listwise替代pairwise，更公平地对待不同亚稳态
   - 核心思路：$\mathcal{L}_{LiPO} = -\mathbb{E}\sum_{k=1}^K \log\frac{\exp[s_\theta(\tau^{(k)})]}{\sum_{j=k}^K \exp[s_\theta(\tau^{(j)})]}$，基于Plackett-Luce选择概率
   - 设计动机：pairwise DPO在多构象场景会渐进忽略高能量但功能重要的亚稳态——listwise给所有排序位置相等的梯度权重

3. **可计算轨迹概率上界**：
   - 做什么：推导连续时间模型中不可解轨迹概率的实用上界
   - 核心思路：用Jensen不等式近似，$s_\theta(\tau^{(i)}) = \beta(\text{MSE}_t(y_0^{(i)}, y_1^{(i)}; \theta_{ref}) - \text{MSE}_t(y_0^{(i)}, y_1^{(i)}; \theta_{opt}))$
   - 设计动机：精确轨迹概率在连续生成模型中不可计算

4. **蛋白质结构表示**：
   - SE(3)×R³旋转-平移 + 7个扭转角(ψ,φ,ω,χ₁,...,χ₄)
   - 每残基token $\xi_t^j \in \mathbb{R}^{7+14}$

### 损失函数 / 训练策略
Listwise LiPO损失+LoRA在线迭代更新。能量通过力场计算（无需MD）。

## 实验关键数据

### 主实验

| 基准 | 指标 | EPO | MDGen(基线) | 说明 |
|------|------|-----|----------|------|
| Tetrapeptides | Pairwise RMSD相关 | **SOTA** | 0.48 | 改善明显 |
| ATLAS | Global RMSF | **SOTA** | 0.50 | 全局柔性 |
| Fast-Folding | Per-target RMSF | **SOTA** | 0.71 | 局部柔性 |
| 综合 | 9个指标 | **全部SOTA** | - | 全面领先 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| EPO-List (完整) | **SOTA** | Listwise偏好 |
| EPO-Pair (pairwise) | 较差 | 忽视高能亚稳态 |
| ODE采样 | 单一构象 | 困在局部最小值 |
| SDE采样 | **多构象** | 逃离能量屏障 |

### 关键发现
- **能量信号单独即可替代MD轨迹**——大幅降低了构象集合生成的计算成本
- **Listwise > Pairwise**：pairwise DPO渐进忽略高能量亚稳态，listwise保持多样性
- **SDE随机性是关键**：确定性ODE困在单一构象，SDE有效探索能量景观
- 成功捕捉了预训练模型遗漏的关键亚稳态（图3可视化）

## 亮点与洞察
- **RLHF→蛋白质的跨域迁移**——将LLM对齐中的偏好优化思路应用到分子生成，展示了对齐范式的通用性。
- **Listwise的多样性保持**在所有需要ensemble多样性的场景都适用——不仅是蛋白质，也适用于分子生成、材料设计等。
- **ODE→SDE的简单转换**带来了质的飞跃——在能量景观中增加随机性就能探索到稀有但重要的构象。

## 局限性 / 可改进方向
- 仅在小蛋白质上验证（Tetrapeptides/Fast-Folding），大蛋白质的适用性未知
- 能量函数的准确性是上限——如果力场不准则对齐目标也不准
- LoRA在线更新的稳定性在长时间训练中可能退化
- 可以探索将EPO与AlphaFold3等结构预测工具结合

## 相关工作与启发
- **vs RLHF/DPO**：EPO将偏好优化从文本对齐迁移到分子对齐，用能量替代人类偏好。核心洞察：偏好排序的数学框架与物理系统的能量排序天然对应
- **vs DistributionalGraphormer**：DG预测平衡态分布但需要MD轨迹作为训练数据，EPO仅需能量函数即可直接对齐到Boltzmann分布
- **vs 传统MD**：EPO完全跳过MD步骤——传统MD需要数天到数周计算一个蛋白质的构象集合，EPO在分钟级生成
- **vs Boltzmann Generator**：BG直接学习平衡分布但训练困难，EPO通过偏好优化间接对齐更稳定
- **启发**：对齐范式的通用性——任何有"评分函数"的生成任务都可以用偏好优化来改善生成质量。这从蛋白质可推广到材料设计、药物分子、甚至建筑结构优化
- **方法论意义**：listwise偏好优化保持多样性的特性在所有需要ensemble/diversity的场景都适用——不仅限于蛋白质
- **SDE随机性的工程启示**：在确定性模型中引入少量受控随机性可以显著扩大探索空间——这是一个跨领域的通用trick
- **计算成本对比**：传统MD需要数天到数周生成一个蛋白质的构象集合，EPO在分钟级完成，效率提升3-4个数量级

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 偏好优化在蛋白质生成的首次listwise应用，理论贡献（上界推导）
- 实验充分度: ⭐⭐⭐⭐ 3基准9指标+消融（pairwise vs listwise + ODE vs SDE）
- 写作质量: ⭐⭐⭐⭐ 跨域动机清晰
- 价值: ⭐⭐⭐⭐⭐ 对药物设计/结构生物学有直接影响
