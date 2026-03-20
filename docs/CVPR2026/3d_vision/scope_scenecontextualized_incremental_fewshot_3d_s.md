# SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation

**会议**: CVPR 2026  
**arXiv**: [2603.06572](https://arxiv.org/abs/2603.06572)  
**代码**: [github.com/Surrey-UP-Lab/SCOPE](https://github.com/Surrey-UP-Lab/SCOPE)  
**领域**: 语义分割 / 3D点云few-shot  
**关键词**: 3D点云分割, 增量few-shot, 背景挖掘, 原型增强, 即插即用  

## 一句话总结
提出即插即用的背景引导原型增强框架SCOPE，从背景区域挖掘伪实例原型丰富新类原型表示，在ScanNet上5-shot新类IoU达23.86%(vs GW 16.88%，+6.98%)，且几乎无额外计算开销(<1MB, 0.02s)。

## 背景与动机
全监督3D点云分割需要大量标注且标签空间固定，但实际部署中新类别会随时间出现且仅有少量标注。现有范式(few-shot/class-incremental/generalized few-shot)各只解决部分挑战。关键被忽视的线索：新类别经常作为未标注的"背景"出现在基础训练场景中，这些背景区域包含可迁移的物体级结构信息。

## 核心问题
如何在极少样本下学到判别性的新类原型？核心insight：背景区域包含的物体级结构可以丰富稀疏的few-shot原型。

## 方法详解

### 整体框架
三阶段：(1)基础训练——DGCNN编码器+余弦相似度分类器全监督训练；(2)场景上下文化——类无关分割模型(Segment3D)从背景提取伪实例mask，构建Instance Prototype Bank(IPB)；(3)增量类注册——few-shot原型通过CPR检索+APE注意力融合从IPB中增强，无需微调backbone。

### 关键设计
1. **Instance Prototype Bank (IPB)**: 类无关模型预测背景区域的伪实例mask(置信度>τ)，编码器特征经masked average pooling得实例原型，收集到IPB。构建一次，冻结使用，<1MB。

2. **Contextual Prototype Retrieval (CPR)**: 计算few-shot原型与所有IPB原型的余弦相似度，选top-R最相似原型作为上下文池。无需未来类知识。

3. **Attention-Based Prototype Enrichment (APE)**: 交叉注意力(query=few-shot原型, key/value=检索原型)加权融合。最终原型：$\tilde{p}_c = \lambda p_c + (1-\lambda)h_c$。完全无参数，抑制噪声保留可迁移线索。

### 损失函数 / 训练策略
基础阶段标准CE损失。增量阶段完全无训练——backbone冻结，原型解析计算。τ=0.75, R=50, λ=0.5。

## 实验关键数据
**ScanNet (K=5, 5-shot):**

| 方法 | mIoU | mIoU-N | HM |
|------|------|--------|-----|
| GW | 34.27 | 16.88 | 23.94 |
| CAPL | 31.73 | 14.75 | 21.36 |
| **SCOPE** | **36.52** | **23.86** | **30.38** |

**S3DIS (K=5, 5-shot):**

| 方法 | mIoU | mIoU-N | HM |
|------|------|--------|-----|
| GW | 57.71 | 39.42 | 51.29 |
| **SCOPE** | **59.41** | **43.03** | **54.25** |

6阶段长期可扩展性：SCOPE mIoU-N 19.75 vs GW 15.64。

### 消融实验要点
- CPR单独(均值聚合): mIoU-N 16.88→22.12(+5.24)
- +APE: mIoU-N→23.86(再+1.74)，注意力加权有效抑制噪声
- GT mask vs 伪mask差距极小(24.77 vs 23.86)——滤波+APE消除噪声影响
- 即插即用验证：应用到PIFS(3.43→4.93)和CAPL(14.75→18.70)均有效
- 运行时开销可忽略：18.60s vs 基线18.58s

## 亮点
- "背景蕴含未来类信息"的insight新颖且有力
- 完全即插即用、无参数、无需微调，real-world部署友好
- 几乎零额外开销(<1MB内存, 0.02s运行时间)

## 局限性 / 可改进方向
- 依赖类无关分割模型质量(目前仅Segment3D可用)
- 仅室内数据集验证(ScanNet/S3DIS)
- λ=0.5固定权重可能不是所有场景最优

## 与相关工作的对比
- **GW(ICCV23)**: 几何词汇原型学习。ScanNet K=5 mIoU-N 16.88 vs SCOPE 23.86(+41.3%)
- **HIPO(CVPR25)**: 双曲原型嵌入，但远落后于GFS基线。mIoU-N 7.44 vs SCOPE 23.86
- **CAPL(CVPR22)**: 共现先验原型学习。mIoU-N 14.75 vs SCOPE 23.86

## 启发与关联
- "从背景挖掘future class信息"的思路可推广到2D few-shot分割和开放世界检测
- 类无关分割作为中间表示的使用方式有通用价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 背景挖掘思路新颖、insight有力、设计优雅
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、即插即用验证、GT vs pseudo对比、长期可扩展性
- 写作质量: ⭐⭐⭐⭐ 清晰系统
- 价值: ⭐⭐⭐⭐⭐ 对few-shot 3D分割有范式性贡献
