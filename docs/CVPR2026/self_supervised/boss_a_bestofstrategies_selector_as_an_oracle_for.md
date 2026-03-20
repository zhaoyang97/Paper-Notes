# BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning

**会议**: CVPR 2026  
**arXiv**: [2603.13109](https://arxiv.org/abs/2603.13109)  
**代码**: [GitHub](https://github.com/dhuseljic/dal-toolbox)  
**领域**: 主动学习 / 数据选择  
**关键词**: 主动学习, Oracle策略, 策略集成, 批次选择, 基础模型  

## 一句话总结
提出BoSS，一种通过集成多个选择策略生成候选批次、冻结backbone仅重训最后一层来快速评估性能增益、然后选取最优批次的可扩展Oracle策略，揭示当前SOTA主动学习策略在大规模多类数据集上距离最优仍有显著差距。

## 背景与动机
主动学习（AL）旨在用最少标注获得最佳模型性能，但实践中没有任何单一选择策略能在所有模型架构、标注预算和数据集上一致获胜（如BADGE在某些评估中最优，Margin在另一些中最优）。为量化这种差距并提供改进方向，需要一个可扩展的Oracle策略作为性能上界参考。现有Oracle方法（SAS：模拟退火搜索、CDO：贪婪逐样本选择）要么搜索步骤过多（SAS默认25k+5k步），要么随批量大小二次扩展（CDO），无法应用到ImageNet等大规模数据集。

## 核心问题
如何设计一个batch-level的Oracle策略，既能利用全量标签信息逼近最优数据选择，又能高效扩展到大规模深度神经网络和复杂数据集？

## 方法详解
### 整体框架
BoSS分三步：(1) 候选批次生成——用10种互补的AL策略各生成10个候选批次（共100个）；(2) 性能评估——冻结预训练backbone，对每个候选批次仅重训最后线性层50个epoch；(3) 选取最优——选择在测试集上zero-one loss最低的批次。

### 关键设计
1. **策略集成的候选批次生成**: 从10种策略中选取候选批次——Random/Margin/CoreSets/BADGE/FastBAIT/TypiClust/AlfaMix/DropQuery及其监督版本（TypiClust*/DropQuery*用真实标签做聚类）。每种策略从随机采样的候选池（最大k_max）中选择，并变化候选池大小以增加多样性。
2. **Selection-via-proxy快速重训**: 冻结特征提取器参数phi，仅重训最后线性层theta，epochs从正式200减到评估用50——大幅降低重训成本同时足以区分候选批次质量。
3. **基于性能的批次评估**: 使用zero-one loss（直接对应准确率）在测试集上评估每个候选批次。Brier score也同样有效，cross-entropy略差。

### 损失函数 / 训练策略
正式训练：冻结backbone + 线性层微调200 epochs，SGD，batch 64，lr=0.01，weight decay 1e-4，cosine annealing。BoSS评估阶段：50 epochs相同设置。默认T=100个候选批次（每策略10个）。

## 实验关键数据
| 数据集 | 指标(AULC) | BoSS | 最佳AL策略 | 差距 |
|--------|-----------|------|-----------|------|
| CIFAR-10 | Low-budget | 0.829 | 0.796 (BAIT) | +0.033 |
| ImageNet | Low-budget | 0.569 | 0.530 (DropQuery) | +0.039 |
| Food101 | Low-budget | 0.643 | 0.576 (DropQuery) | +0.067 |
| ImageNet | Mid-budget | 0.700 | 0.680 (BAIT) | +0.020 |
| CIFAR-100 | Mid-budget | 0.769 | 0.753 (BAIT) | +0.016 |

- 10个图像数据集（CIFAR-10/100, STL-10, Food101, ImageNet等）+ 4个文本数据集
- DINOv2-ViT-S/14 和 SwinV2-B 两种backbone
- ImageNet上BoSS约达到最佳AL策略2倍的准确率提升
- 简单数据集（CIFAR-10）差距小，复杂多类数据集差距显著

### 消融实验要点
- 策略集成 vs 随机候选批次：策略集成显著优于朴素随机生成
- 候选批次数T：每策略10个就够了，>10提升微乎其微
- 重训epochs：50/100/200几乎无差，甚至10也相近；5开始有明显下降
- 损失函数：zero-one 约等于 Brier > cross-entropy
- 批次大小敏感性：4倍batch size仅带来轻微性能下降

## 亮点
- 第一个可扩展到ImageNet的深度AL Oracle策略
- 清晰量化了SOTA策略与Oracle间的差距：大规模多类场景差距最大，说明这是最值得投入的研究方向
- Pick frequency分析揭示了AL过程中的动态特性：早期DropQuery*/TypiClust*主导（代表性），后期无单一策略一致最优
- 设计理念简单优雅：集成+proxy重训+性能选择，易于扩展新策略

## 局限性 / 可改进方向
- Oracle策略依赖全量标签（测试集），不可直接用于实际AL，仅作为评估基准
- 不能将BoSS与实际策略的差距全部归因为"可改进空间"——部分差距来自监督信息本身不可获得
- 多臂赌博机自适应分配策略的候选批次数量是一个有前途的扩展方向
- 文本任务仅在4个数据集上验证，其他领域（检测/分割）未探索

## 与相关工作的对比
- vs CDO（NeurIPS'24）：CDO贪婪逐样本选择，复杂度O(m*b^2)随批量二次增长，b=50时必须减少m至4-3才能匹配BoSS运行时间；BoSS复杂度与b无关
- vs SAS（AISTATS'21）：默认需25000+5000搜索步，降低步数后性能大幅下降；BoSS在对齐运行时间后始终优于SAS
- vs 实际策略：BADGE和BAIT是最接近Oracle的策略，CoreSets始终表现不佳

## 启发与关联
- 集成多策略的思路可以启发设计无需Oracle的自适应AL方法
- Pick frequency分析为"何时该探索何时该利用"提供了实证数据

## 评分
- 新颖性: ⭐⭐⭐ 方法本身是策略集成+暴力搜索，理念简单但有效
- 实验充分度: ⭐⭐⭐⭐⭐ 10个图像+4个文本数据集，2种backbone，大量消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，实验组织有条理，附录详尽
- 价值: ⭐⭐⭐⭐ 为AL社区提供了标准化Oracle评估工具和大量启发性观察
