# Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning

**会议**: CVPR 2026  
**arXiv**: [2603.12816](https://arxiv.org/abs/2603.12816)  
**代码**: 无  
**领域**: 持续学习 / Prompt-based CL  
**关键词**: 域增量学习, prompt池, α-entmax, 伪特征重放, 漂移检测, 不确定性加权  

## 一句话总结
提出Residual SODAP框架，在无任务ID、无数据存储的域增量学习中，联合解决表示适应（α-entmax稀疏prompt选择+残差聚合）和分类器保持（统计伪特征重放+知识蒸馏），在DR、皮肤癌和CORe50三个基准上达到SOTA。

## 背景与动机
现有Prompt-based持续学习(PCL)方法存在两个关键限制：(1)prompt选择方案不够好——Top-k硬选择不可微且表达力有限，Softmax软选择虽然可微但噪声累积（不相关prompt也有非零权重）；(2)忽略了分类器层面的遗忘——现有PCL主要关注prompt/prompt池设计来改善表示适应，但通过cross-composition诊断实验发现，分类器决策边界不稳定才是域增量学习中遗忘的主要来源。

## 核心问题
如何在prompt-based CL框架内同时实现高质量的表示适应和分类器层面的知识保持，在无Task-ID、无过去数据存储的严格约束下缓解灾难性遗忘？

## 方法详解

### 整体框架
四个核心组件在frozen ViT backbone上协同工作：(1)α-entmax稀疏prompt选择与残差聚合，(2)基于统计的伪特征重放做分类器知识保持，(3)基于prompt使用模式的域漂移检测(PUDD)，(4)不确定性加权多目标优化。

### 关键设计
1. **α-entmax残差prompt选择**: 用memory bank增强查询（CLS token + 全局上下文 + memory检索信号），在bottleneck空间通过α-entmax（α=1.5）做稀疏prompt选择——自动将不相关prompt权重设为精确零。Prompt池分为frozen集ℱ和active集𝒜，frozen保持先前知识，active做残差适应（$p_{out} = p_\mathcal{F} + 0.1 \cdot p_\mathcal{A}$）。

2. **统计知识保持**: 每阶段结束后用Welford在线算法存储类别级特征均值和方差（$\mu_c, \sigma_c^2$），下一阶段通过(a)真实特征蒸馏（当前数据过teacher和student head对齐KL散度）和(b)伪特征重放（从$\mathcal{N}(\mu_c, \text{diag}(\sigma_c^2))$采样伪特征蒸馏）双路保持分类器决策边界。

3. **PUDD漂移检测**: 监控prompt选择模式变化来检测域漂移——结合选择熵变化（短期波动刷新的z-score）和使用集变化（当前使用prompt集与滑动窗口历史的IoU）。漂移分数$D$按比例决定prompt池扩展量。

4. **不确定性加权**: 为5个损失项（CE、real蒸馏、pseudo重放、diversity、norm）各学一个log方差$s_i$，自动平衡：$\mathcal{L}_{total} = \sum_i (e^{-s_i}\mathcal{L}_i + s_i)$。

### 损失函数 / 训练策略
5项损失通过不确定性加权自动平衡。辅助损失包括：diversity loss（惩罚高频共激活prompt间的相似性）和norm正则（限制active prompt值只做残差）。AdamW, lr=1e-3, cosine schedule, 100 epochs, early-stop patience 5。

## 实验关键数据
| 基准 | 方法 | AvgACC↑ | AvgF↓ |
|------|------|---------|-------|
| DR | OS-Prompt++ | 0.769 | 0.113 |
| DR | Coda-Prompt | 0.688 | 0.140 |
| DR | **Residual SODAP** | **0.850** | **0.047** |
| Skin Cancer | OS-Prompt++ | 0.725 | 0.063 |
| Skin Cancer | **Residual SODAP** | **0.760** | 0.031 |
| CORe50 (11-stage) | DER++ | 0.994 | 0.061 |
| CORe50 (11-stage) | **Residual SODAP** | **0.995** | **0.003** |

### 消融实验要点
- Query Enhancer去掉后AvgACC降4.2pp——查询增强对可靠prompt选择至关重要
- Diversity loss去掉后AvgACC降3.2pp且AvgF升2.5pp——防止prompt坍塌和保持旧知识都需要它
- 蒸馏+伪重放各贡献1.5~2.2pp准确率提升
- 组件间存在accuracy-forgetting trade-off：某些消融降低遗忘但牺牲准确率，完整模型在trade-off曲线最佳点

## 亮点
- **Backbone×Classifier诊断分析清晰地揭示了PCL中被忽视的分类器级遗忘问题**，是非常好的动机分析
- α-entmax巧妙地解决了Top-k(不可微)和Softmax(噪声累积)之间的困境——精确零权重+可微性兼得
- 统计伪特征重放极其轻量——只需每类存储均值和方差，用高斯采样即可回放
- 不确定性加权免去了5个损失权重的手动调参
- CORe50 11阶段仅0.003遗忘率，展示了在长序列域漂移下的极强稳定性

## 局限性 / 可改进方向
- 仅在域增量(DIL)设置上验证，未扩展到类增量(CIL)
- 高斯假设的伪特征可能在特征分布非高斯时失效
- PUDD的超参（窗口大小、阈值、D_max等）较多，虽不需要手动调损失权重但引入了其他超参
- Prompt池持续扩展（60→84→94），长期部署下参数量会线性增长

## 与相关工作的对比
- **OS-Prompt++**: 同是PCL方法但无分类器保持机制，DR上AvgACC 0.769 vs 0.850
- **Coda-Prompt**: 正交正则化的prompt学习，DR上AvgACC仅0.688
- **DER++**: 需要replay buffer（存储过去数据），即使存数据仍不如本文无数据存储方案
- **Online EWC**: 经典正则化方法，AvgF 0.174远高于本文0.047

## 启发与关联
- "分类器级遗忘"的insight可推广到其他CL方法——不仅是PCL，任何使用共享分类器的CL方法都可能存在这个问题
- α-entmax稀疏选择机制可用于其他需要从大池中选取子集的场景（如MoE路由）
- 统计伪特征重放的思路可用于任何禁止数据存储的隐私敏感场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 分类器保持+prompt适应的联合框架有新意，但各组件单独看并不全新（α-entmax、KD、不确定性加权都是已有技术）
- 实验充分度: ⭐⭐⭐⭐ 三个基准、完整消融、cross-composition诊断、prompt可视化分析全面
- 写作质量: ⭐⭐⭐⭐ 动机分析（Fig.1）有说服力，方法描述详尽且有数学严谨性
- 价值: ⭐⭐⭐⭐ 在无数据存储的医学影像域增量学习这一实用场景有直接价值
