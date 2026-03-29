# Attribution-Guided Model Rectification of Unreliable Neural Network Behaviors

**会议**: CVPR 2026  
**arXiv**: [2603.15656](https://arxiv.org/abs/2603.15656)  
**代码**: 无  
**领域**: LLM效率  
**关键词**: model editing, attribution, backdoor defense, spurious correlation, rank-one update

## 一句话总结
提出基于归因引导的动态模型纠正框架，利用Integrated Gradients量化各层"可编辑性"来自动定位导致不可靠行为的嫌疑层，结合rank-one编辑在仅需1个清洁样本的情况下修复后门攻击、虚假相关和特征泄漏等模型不可靠行为。

## 研究背景与动机

1. **领域现状**：神经网络在遭遇分布不一致（后门触发器、虚假特征、特征泄漏）时表现出不可靠行为。现有修复策略主要依赖数据清洗+模型重训，计算和人工成本巨大。
2. **现有痛点**：Rank-one model editing（如ROME）已在生成模型和判别模型中展现知识编辑能力，但直接应用于不可靠行为修复存在两个结构性问题：(1) out-of-span残差——新key可能不在训练key的span内，(2) 样本复杂度——分布偏移下需要大量样本估计key。
3. **核心矛盾**：现有编辑方法固定在最后一层操作，但不同层的可编辑性（editability）差异显著——没有哪一层是普遍最优的。
4. **本文要解决什么**：(1) 在data-efficient条件下（甚至1个清洁样本）修复模型不可靠行为；(2) 自动找到导致不可靠行为的最关键层进行编辑。
5. **切入角度**：将rank-one editing从"领域适配"重新定位为"行为纠正"，利用clean-corrupted pair的结构性质（rectifiability + span-aligned control）绕开标准编辑的局限；再用归因方法量化层级可编辑性。
6. **核心idea一句话**：通过计算corrupted→clean路径上各层的归因得分并投影到rank-one编辑方向，自动定位嫌疑层并进行动态迭代修复。

## 方法详解

### 整体框架
给定一个有不可靠行为的模型 $f$，一对corrupted样本 $\tilde{x}$ 和clean样本 $x$，框架执行三步循环：(1) 用Integrated Gradients计算各层归因 $M^l(x, \tilde{x})$；(2) 将归因投影到rank-one编辑方向后比较标量得分 $\hat{G}_l$，选出最佳层 $l^*$；(3) 对该层执行rank-one编辑建立新的key-value关联。循环直到预测差距 $\delta^*$ 足够小或性能退化超出预算。

### 关键设计

1. **纠正设定的理论优势（Rectifiability & Span-Aligned Control）**
   - 做什么：证明行为纠正设定相比领域适配设定的结构性优势
   - 核心思路：因为模型训练时已经见过clean和corrupted样本，corrupted key $k^*$ 已在训练key的span内（Proposition 4.1），所以out-of-span残差消失；同时因为paired supervision，不需要大量新域样本即可精确估计key（Proposition 4.2）
   - 设计动机：这两个性质从理论上解释了为什么本方法只需1个清洁样本就能有效——不是启发式的luck，而是有结构保证

2. **归因引导的层定位（Attribution-Guided Layer Localization）**
   - 做什么：自动找到对不可靠行为贡献最大的层
   - 核心思路：用Integrated Gradients计算从 $\tilde{x}$ 到 $x$ 路径上各层归因 $M^l(x, \tilde{x})$，满足Completeness公理（所有层的归因和等于输出变化）。然后将归因投影到rank-one编辑方向：$M^{*,l} = M^l \cdot (C^{-1}k^*)^T$，取Frobenius范数 $\hat{G}_l = \|M^{*,l}\|_F$ 作为可编辑性得分，选 $l^* = \arg\max_l \hat{G}_l$
   - 设计动机：不同层对不同类型的不可靠行为敏感度不同（Fig. 2明确展示了这一点），而归因投影到编辑方向后才能衡量"这层有多容易被修好"，而不仅是"这层有多重要"

3. **动态模型纠正框架（Dynamic Model Rectification）**
   - 做什么：迭代地定位和修复多个层
   - 核心思路：在while循环中，每轮重新定位嫌疑层（因为前一轮的编辑可能改变最优层），执行rank-one编辑，直到 $\delta^* \leq \delta$（预测差距足够小）或 $\epsilon^* > \epsilon$（性能退化超出预算）
   - 设计动机：静态编辑（只改一层）可能不够充分，动态框架允许模型在修复过程中自适应地探索多层修复路径

### 损失函数 / 训练策略
Rank-one编辑的优化目标：$\min_\Lambda \|v^* - f_l(k^*; W')\|$，约束 $W' = W + \Lambda(C^{-1}k^*)^T$，其中 $C = KK^T$ 是key的二阶矩统计量。这是一个rank-one约束最小二乘问题，有闭式解。

## 实验关键数据

### 主实验

**后门攻击修复（CIFAR-10 & ImageNet）**

| 方法 | #样本 | CIFAR-10 OA↑ | CIFAR-10 ASR↓ | ImageNet OA↑ | ImageNet ASR↓ |
|------|------|-------------|--------------|-------------|--------------|
| Trojaned模型 | - | 93.67 | 99.94 | 69.05 | 87.24 |
| Fine-tune | 1 | 90.83 | 73.07 | 65.95 | 79.91 |
| Fine-tune | 20 | 91.58 | 13.22 | 68.42 | 21.86 |
| P-ClArC | 20 | 89.97 | 6.21 | 65.42 | 8.09 |
| **Dyn. rectifying** | **1** | **93.65** | **1.34** | 66.77 | **1.61** |
| **Dyn. rectifying** | **20** | **93.61** | **0.26** | **68.84** | **0.12** |

### 消融实验

| 配置 | 说明 |
|------|------|
| Static rectifying (n=1) | 只改最后层，ASR=2.57，OA=92.93 |
| Dynamic rectifying (n=1) | 自动选层+迭代，ASR=1.34，OA=93.65 |
| Patched model (n=20) | 剪枝法，ASR=12.19，OA=89.70 |

**触发器泛化性（不同可见度/位置）**

| 方法 | OA↑ | ASR(0.3)↓ | ASR(0.5)↓ | ASR(0.7)↓ | ASR(1.0)↓ |
|------|-----|-----------|-----------|-----------|-----------|
| Patched | 89.61 | 30.84 | 26.86 | 32.42 | 37.19 |
| **Dyn. rectifying** | **91.21** | **6.84** | **5.17** | **7.65** | **7.91** |

### 关键发现
- 仅用1个清洁样本，动态修复就将CIFAR-10上的ASR从99.94%降到1.34%，同时OA几乎不变
- 动态修复始终优于静态修复（只改最后层），验证了层定位的价值
- 方法对不同可见度和位置的触发器都有很好的泛化性——用一个位置的样本修复，对其他位置也有效
- 与fine-tuning相比，在相同样本量下ASR降低幅度远大于fine-tuning，且不损失OA

## 亮点与洞察
- **1个样本修复后门**：理论上证明了rectification设定的数据高效性（rectifiability + span-aligned control），这不是偶然的好运气而是有结构保证的
- **归因→可编辑性的巧妙映射**：不是直接用归因大小选层，而是将归因投影到rank-one编辑方向再取范数——这个"归因×编辑方向"的交叉信号才是真正衡量"改这层能修好多少"的指标
- **统一处理三类不可靠行为**：后门、虚假相关、特征泄漏用同一个框架处理，说明方法的泛化性好

## 局限性 / 可改进方向
- Rank-one编辑的线性关联假设在复杂的非线性层中可能不够精确
- 需要知道哪个样本被corrupted（即需要一对clean-corrupted pair），在实际中这个先验可能不always available
- 动态框架的迭代次数与层数相关，对于非常深的模型可能需要较多轮次
- 仅在分类任务上验证，对生成任务或回归任务的适用性未探索

## 相关工作与启发
- **vs ROME/MEMIT**: ROME是生成模型的rank-one编辑，本文将其适配到判别模型的行为纠正，关键贡献是rectification设定的理论分析和自动层选择
- **vs P-ClArC/A-ClArC**: 它们通过添加artifact module修补，本文直接编辑权重，不增加推理开销且性能更好
- **vs Fine-tuning**: Fine-tuning需要更多样本且容易损害OA，本文的rank-one编辑更精准高效

## 评分
- 新颖性: ⭐⭐⭐⭐ 归因引导层定位的idea新颖，rectification设定的理论分析有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖三类不可靠行为、多数据集、泛化性实验、消融实验丰富
- 写作质量: ⭐⭐⭐⭐ 论述清晰但理论部分偏长
- 价值: ⭐⭐⭐⭐ 1个样本修复后门的实用价值很高
