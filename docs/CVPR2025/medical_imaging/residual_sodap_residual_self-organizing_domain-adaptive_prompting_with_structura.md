# Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning

**会议**: CVPR2025  
**arXiv**: [2603.12816](https://arxiv.org/abs/2603.12816)  
**代码**: 待确认  
**领域**: medical_imaging  
**关键词**: continual learning, domain-incremental learning, prompt-based learning, catastrophic forgetting, knowledge distillation, sparse selection

## 一句话总结

针对无任务 ID 和无数据回放的领域增量学习（DIL），提出 Residual SODAP 框架，通过 α-entmax 稀疏 prompt 选择与残差聚合、基于特征统计的伪回放蓏馏、prompt 使用模式漂移检测和不确定性加权，同时解决表示适配和分类器遗忘问题。在 DR、皮肤癌和 CORe50 上均达 SOTA。

## 研究背景与动机

- **持续学习核心挑战**：灾难性遗忘在领域增量学习（DIL）中尤为严重——无任务 ID、不存储历史数据
- **现有 Prompt-based CL 的两个局限**：
  1. **Prompt 选择方案不足**：Top-k 硬选择限制表达力且不可微；Softmax 软选择让不相关 prompt 也施加影响导致噪声累积
  2. **忽视分类器结构**：现有 PCL 方法主要关注 prompt 池设计来适配表示，但分类器层在域偏移下也存在不稳定性（如 cross-composition 诊断实验所示）
- **关键发现**：通过 backbone × classifier 交叉组合分析（参照 Liu et al. 的诊断方法）发现，即使 backbone 表示通过 prompt 适配保持良好，分类器层的决策边界仍会随着域增量训练显著退化
- **出发点**：需要一个框架同时解决 prompt 层的表示适配和分类器层的知识保持

## 方法详解

### 1. α-Entmax 残差 Prompt 选择
- **Query 增强**：融合当前层 CLS token、初始 CLS token（全局上下文）和可学习记忆库的检索信号，通过 MHA + 瓶颈适配器生成增强 query
- **稀疏选择**：用 α-entmax（α=1.5）替代 softmax，可以给低分 prompt 精确赋零权重，兼顾全 prompt 池利用和噪声抑制
- **残差结构**：从 Stage 2 开始，prompt 池分为冻结集 F 和活跃集 A，分别独立做稀疏路由，最终以残差形式组合：p_out = p_F + 0.1·p_A，冻结集保留先验知识，活跃集仅做残差适配
- **辅助损失**：多样性损失（惩罚高频共激活 prompt 间的相似性）+ 范数正则化（约束活跃 prompt 值仅作残差作用）

### 2. 统计知识保持（伪回放蒸馏）
- 每个阶段结束时，用 Welford 在线算法收集逐类特征统计（均值+方差），并冻结当前分类器头作为教师
- 多阶段训练时统计量通过 Welford 公式累积合并，单次遍历即可完成，内存高效
- 下一阶段训练时执行两种蓏馏：
  1. **实特征蓏馏**：当前批次特征通过教师和学生头的 KL 散度对齐（温度 T=2.0）
  2. **伪特征回放**：从存储的类统计中采样伪特征（重参数化技巧），用冻结教师和可训练学生头的 KL 散度对齐
  - 类别均匀采样以缓解少数类欠表示，每批采样 K=B 个伪特征

### 3. Prompt 使用模式漂移检测（PUDD）
- 同时监控两个信号：(a) prompt 选择分布的熵变化（短期波动反映域变化），(b) 使用集合的结构偏移（IoU）
- 两个信号加权合成漂移分数 D_t（α=1.0, β=0.5），跨层跨批次平均后决定 prompt 池扩展规模
- 扩展量与漂移强度成正比：弱偏移仅增少量 prompt（E_min=10），强偏移大幅扩展（E_max=80）
- 扩展后现有活跃 prompt 移入冻结集，新增 prompt 成为新活跃集

### 4. 不确定性加权
- 采用 Kendall 等人的同方差不确定性加权，为每个损失项学习 log 方差 s_i
- 总损失：L_total = Σ(e^{-s_i}·L_i + s_i)，噪声大的损失自动降权

## 实验关键数据

### 基准设置
- 三个 DIL 场景：糖尿病视网膜病变（DR，3个域：APTOS→DDR→DRD）、皮肤癌（3个域：ISIC→HAM→DERM7）、CORe50（通用基准）
- 无数据回放、无任务 ID，全部结果取 3 次独立运行平均
- 评估指标：AvgACC（平均准确率）和 AvgF（平均遗忘量）

### 与 SOTA 对比
| 方法 | DR AvgACC↑ | DR AvgF↓ | Skin AvgACC↑ | Skin AvgF↓ | CORe50 AvgACC↑ | CORe50 AvgF↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| OS-Prompt++ | 0.769 | 0.113 | 0.725 | 0.063 | 0.983 | 0.014 |
| Coda-Prompt | 0.688 | 0.140 | 0.713 | 0.041 | 0.974 | 0.056 |
| DER++ | 0.607 | 0.288 | 0.722 | 0.099 | 0.994 | 0.061 |
| **Residual SODAP** | **0.850** | **0.047** | **0.760** | 0.031 | **0.995** | **0.003** |

- DR 场景：AvgACC 较次优（OS-Prompt++ 0.769）提升 8.1pp，AvgF 从 0.113 降至 0.047
- 皮肤癌场景：AvgACC 0.760 最优，AvgF 0.031（Dual-Prompt 的 0.012 更低但其 AvgACC 仅 0.637，准确率-遗忘权衡不佳）
- CORe50：几乎完美（AvgACC 0.995，AvgF 0.003），证明方法在通用域同样有效

### 消融实验（DR）
- 去除 Query Enhancer：AvgACC 下降 4.2pp（影响最大的单组件）
- 去除 Residual（退化为 SODAP）：AvgACC 下降 1.9pp，AvgF 下降 2.0pp
- 去除伪回放：AvgACC 下降 1.5pp
- 去除蒸馏：也导致性能下降
- PUDD 控制 prompt 池从 60→84→94 动态扩展，扩展后无冗余 prompt

## 亮点

1. **全面设计**：同时解决 prompt 选择噪声、分类器遗忘、域漂移检测三个维度的问题，而非仅解决单一矛盾
2. **α-entmax 稀疏选择**：在保留全 prompt 池可微优化的同时精准抑制无关 prompt，比 Top-k（不可微）和 Softmax（噪声累积）都优
3. **无数据伪回放**：仅存储逐类均值和方差（每类仅 2D 向量，极低存储开销），通过重参数化采样实现知识保持
4. **PUDD 自动扩展**：基于使用模式而非固定规则检测域漂移，扩展量自适应，避免容量浪费或不足
5. **通用性**：在医学影像（DR、皮肤癌）和通用视觉（CORe50）上均达 SOTA，证明方法不限于特定领域

## 局限性

1. 实验场景较短（仅 3 个域），未验证长序列（10+ 域）下的可扩展性，prompt 池可能持续膨胀导致内存和计算负担
2. α 和 λ_r 等超参数（α=1.5，λ_r=0.1）固定设置，未探索不同场景的敏感性分析
3. 基于冻结 ViT backbone（预训练 ImageNet-21K），方法对 backbone 架构和预训练数据的依赖性未探讨
4. 伪回放假设特征分布为对角高斯，对复杂多模态分布可能不够准确，尤其在域间特征重叠时
5. 不确定性加权的 clamp 范围 [-3, 6] 为经验选择，缺乏理论指导
6. PUDD 的超参数（滑动窗口 W=100，D_max=5.0，阈值 τ_s=0.01）未做充分灵敏度分析
7. 与多头分类器方法的本质区别值得更深入讨论——本文方案虽不需要 Task-ID 但需要漂移检测

## 评分
- 新颖性: 4/5 — α-entmax 残差 prompt 选择和 PUDD 漂移检测是有意义的创新点，分类器知识保持填补了 PCL 的盲区
- 实验充分度: 4/5 — 三个基准（2个医学+1个通用）、全面消融和可视化分析，但场景长度有限（仅3域）
- 写作质量: 4/5 — 结构清晰、公式准确，交叉组合诊断分析有说服力，但部分符号较密集
- 价值: 4/5 — 为无数据无 Task-ID 的 DIL 提供了一个系统且实用的解决方案，对医学影像持续学习有直接应用价值

<!-- 实验硬件和超参数详见 Supplementary Material -->
<!-- Backbone: ViT-B/16 预训练于 ImageNet-21K，冻结 backbone 仅训练 prompt 相关参数 -->
<!-- 蒸馏温度 T=2.0，批次大小 B=64，prompt 池初始大小 60 -->
