# Representation Learning for Spatiotemporal Physical Systems

**会议**: CVPR 2026  
**arXiv**: [2603.13227](https://arxiv.org/abs/2603.13227)  
**代码**: 无  
**领域**: 自监督/表示学习  
**关键词**: JEPA, 物理系统, 表示学习, 参数估计, 自监督学习  

## 一句话总结
通过在三个PDE物理系统（活性物质、剪切流、Rayleigh-Bénard对流）上对比JEPA、VideoMAE、MPP和DISCO，发现隐空间预测方法(JEPA)在物理参数估计任务上全面优于像素级预测方法(MAE/自回归模型)，MSE平均改善30-50%。

## 背景与动机
机器学习在时空物理系统上的工作主要聚焦于"下一帧预测"式的代理建模(surrogate modeling)，但这些自回归模型训练昂贵且存在累积误差。更重要的是，科学研究的实际需求往往不是逐帧预测，而是估计系统的物理参数（如Reynolds数、Prandtl数等）。然而，哪种学习范式最能保留物理意义信息，目前少有系统研究。

## 核心问题
通用自监督表示学习方法（JEPA vs MAE）在学习物理相关表示方面谁更有效？为物理建模设计的方法（自回归基础模型、算子学习）在下游科学任务上是否真的优于通用方法？

## 方法详解

### 整体框架
对比四种方法在三个物理系统上预训练后、通过冻结encoder+可训练probe进行参数估计的效果：
- **JEPA**: 隐空间时序预测，ConvNeXt encoder，VICReg损失（方差+不变性+协方差正则化）
- **VideoMAE**: 像素级masked重建，ViT-small/16
- **DISCO**: 基于算子学习的上下文推理，超网络生成轨迹特定演化算子
- **MPP**: 自回归物理基础模型，像素级下一帧预测

### 关键设计
1. **JEPA动力学版本**: 给定$k$帧上下文$x_{t:t+k}$，学习预测下$k$帧$x_{t+k:t+2k}$在隐空间中的表示。encoder $f: \mathcal{X} \to \mathcal{Z}$ 和 predictor $g: \mathcal{Z} \to \mathcal{Z}$，用VICReg损失避免模式坍塌——invariance项对齐预测和目标、variance项保持维度方差、covariance项去相关。

2. **物理参数估计作为评估任务**: 物理参数（如Reynolds数、Rayleigh数）决定系统时间演化行为，因此参数估计误差直接量化了表示中包含多少物理信息。这比下一帧预测误差更能反映"模型是否理解了物理"。

3. **冻结encoder+attentive probe**: 保持预训练encoder权重不变，仅训练probe head，确保评估的是预训练表示质量而非微调能力。

### 损失函数 / 训练策略
- JEPA/VideoMAE各系统单独预训练6 epochs
- MPP用已发布预训练权重+端到端微调（因预训练不含这三个数据集）
- DISCO使用The Well上的预训练权重
- 微调均100 epochs，AdamW + cosine lr

## 实验关键数据
| 方法 | Active Matter MSE↓ | Shear Flow MSE↓ | Rayleigh-Bénard MSE↓ |
|------|-------------------|-----------------|---------------------|
| **JEPA** | **0.079** | **0.38** | 0.13 |
| VideoMAE | 0.160 | 0.67 | 0.18 |
| **DISCO** | **0.057** | **0.13** | **0.01** |
| MPP (full FT) | 0.230 | 0.59 | 0.08 |

- JEPA vs VideoMAE: 活性物质-51%，剪切流-43%，Rayleigh-Bénard -28%
- DISCO（隐空间算子学习）全面最优
- MPP（自回归）尽管端到端微调仍表现不佳

**数据效率**: 在shear flow上，JEPA用10%微调数据(0.57)即超过VideoMAE用100%数据(0.67)。

### 消融实验要点
- 隐空间预测(JEPA, DISCO)普遍优于像素级预测(VideoMAE, MPP)
- 自回归模型(MPP)在非生成任务上表现最差，与NLP中"自回归不如encoder-only"的结论一致
- JEPA数据效率显著优于VideoMAE——50%数据即可达95%性能

## 亮点
- **核心insight深刻**：物理理解≠像素预测，隐空间预测捕获高层动力学特征而非低层视觉细节
- 物理参数估计作为evaluation metric是巧妙的选择——提供了可量化的"物理相关性"指标
- 与NLP领域"BERT vs GPT"的类比有启发性：编码器方法在理解任务上优于生成方法
- 来自Flatiron/Polymathic AI/NYU的强阵容（Yann LeCun等），研究质量可靠

## 局限性 / 可改进方向
- 仅三个物理系统，泛化性有限
- JEPA和VideoMAE使用不同架构(ConvNeXt vs ViT)，架构差异可能是混淆因素
- 仅评估全局参数估计，未探索场重建、异常检测等其他下游任务
- 论文较短（workshop论文级别），实验深度有待加强

## 与相关工作的对比
- **DISCO**: 专门为物理系统设计的算子学习方法，在所有系统上最优，但需要更多物理先验知识
- **MPP**: 自回归物理基础模型，在参数估计上反而最差(0.230 on active matter)，说明像素级生成目标与物理理解目标不一致
- **V-JEPA (Bardes et al.)**: 本文JEPA设计受其启发，但针对物理时序数据做了适配

## 启发与关联
- 对科学ML社区的重要提醒：代理模型(surrogate model)的下一帧预测能力≠对物理的理解能力
- JEPA范式在科学数据上的潜力值得进一步探索——可能是比自回归更好的科学foundation model基础
- 隐空间vs像素空间的对比结论可迁移到医学图像、遥感等领域的表示学习

## 评分
- 新颖性: ⭐⭐⭐⭐ 将JEPA引入物理系统表示学习的视角新颖，核心insight有价值
- 实验充分度: ⭐⭐⭐ 三个系统、四种方法，但架构未对齐、系统数量有限
- 写作质量: ⭐⭐⭐⭐ 简洁有力，核心观点表达清楚
- 价值: ⭐⭐⭐⭐ 对科学ML方向有启发意义，提出了"表示学习 vs 代理建模"的重要对比视角
