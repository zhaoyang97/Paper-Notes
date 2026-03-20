# COS3D: Collaborative Open-Vocabulary 3D Segmentation

**会议**: NeurIPS 2025  
**arXiv**: [2510.20238](https://arxiv.org/abs/2510.20238)  
**代码**: 无  
**领域**: 3D分割 / 开放词汇  
**关键词**: 开放词汇3D分割, 3D Gaussian Splatting, 协作场, Instance-Language协作, Kernel Regression  

## 一句话总结
提出COS3D协作式开放词汇3D分割框架，在3D Gaussian Splatting中同时维护instance field（学习清晰边界）和language field（学习语义），通过两阶段训练实现Ins2Lang映射，推理时Language→Instance prompt精化实现互补协作，在LeRF数据集上mIoU达50.76%，大幅超越Dr.Splat（43.58%）。

## 研究背景与动机

1. **领域现状**：开放词汇3D分割旨在用文本查询分割3D场景中的任意目标。基于NeRF/3DGS的方法将CLIP/DINO特征蒸馏到3D表示中，通过文本-视觉相似度定位目标。
2. **现有痛点**：（a）纯language field方法（LangSplat等）边界模糊——语义特征在物体边界处过度平滑；（b）class-agnostic分割+后选方法（OpenGaussian等）误差累积——分割错误不可纠正。两类方法各有优势但未结合。
3. **核心idea**：让instance field和language field在训练和推理中协作——训练时Instance→Language映射，推理时Language→Instance精化。

## 方法详解

### 整体框架
基于3D Gaussian Splatting，每个Gaussian额外存储instance feature (16维) + language feature (512维)。两阶段训练：Stage 1用SAM mask做对比学习训练instance field；Stage 2从instance field映射到language field。推理时language field生成相关性图，instance field精化边界。

### 关键设计

1. **Stage 1 - Instance Field Learning**：用SAM mask做InfoNCE对比学习，同一mask内的Gaussian特征拉近，不同的推远。学习到边界清晰的实例特征，仅需16维。
2. **Stage 2 - Ins2Lang Mapping**：两种实现——(a) Shallow MLPs映射（~3min训练）；(b) Kernel Regression（Nadaraya-Watson估计，无需训练）。从30-40min联合训练压缩到<3min。
3. **推理 - Lang2Ins Prompt Refinement**：先用language field生成3D文本相关性图→以高相关区域为prompt在instance field中做余弦相似度扩展→自适应过滤低相关区域。

## 实验关键数据

### 主实验（LeRF Dataset）

| 方法 | mIoU | mAcc |
|------|------|------|
| LangSplat | 9.66 | 12.41 |
| Dr.Splat | 43.58 | 63.87 |
| OpenGaussian | 38.36 | 51.43 |
| InstanceGaussian | 45.30 | 58.44 |
| **COS3D (kernel)** | **50.76** | **72.08** |

### ScanNetv2 (19类)

| 方法 | mIoU | mAcc |
|------|------|------|
| OpenGaussian | 24.73 | 41.54 |
| **COS3D (kernel)** | **32.47** | **49.05** |

### 关键发现
- Kernel regression比MLP映射略优（50.76 vs 49.75 mIoU），且无需训练。
- 总训练时间仅50min，比联合训练（165min）快3.3×。
- Language→Instance精化是关键：去掉后mIoU大幅下降。

## 亮点与洞察
- **Instance和Language的互补协作**：训练时Instance教Language（映射），推理时Language教Instance（提供语义prompt）——优雅的双向协作。
- **Kernel Regression的零训练映射**：不需要训练就能从instance特征预测language特征，高效且有效。

## 局限性 / 可改进方向
- 16维instance feature可能限制了复杂场景的表达能力。
- 依赖SAM的mask质量，SAM失败的场景会影响instance field。

## 评分
- 新颖性: ⭐⭐⭐⭐ Instance-Language协作的双向设计新颖
- 实验充分度: ⭐⭐⭐⭐ LeRF+ScanNet+充分消融
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法简洁
- 价值: ⭐⭐⭐⭐ 为开放词汇3D分割提供了高效新范式
