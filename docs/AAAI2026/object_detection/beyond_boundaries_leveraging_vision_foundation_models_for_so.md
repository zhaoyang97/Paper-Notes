# Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection

**会议**: AAAI 2026  
**arXiv**: [2511.07301](https://arxiv.org/abs/2511.07301)  
**代码**: [https://github.com/HuizaiVictorYao/VFM_SFOD](https://github.com/HuizaiVictorYao/VFM_SFOD)  
**领域**: 目标检测  
**关键词**: 无源域自适应目标检测, 视觉基础模型, DINOv2, Grounding DINO, 伪标签融合  

## 一句话总结
提出利用VFM（DINOv2+Grounding DINO）增强无源域自适应目标检测（SFOD）的框架，通过全局特征对齐(PGFA)、实例级原型对比学习(PIFA)和双源伪标签融合(DEPF)三个模块，在6个跨域检测基准上取得SOTA，例如Cityscapes→Foggy Cityscapes达47.1% mAP（比DRU高3.5%），Sim10k→Cityscapes达67.4% AP（比DRU高8.7%）。

## 背景与动机
无源域自适应目标检测（SFOD）需要在没有源域数据的情况下，将预训练检测器适配到目标域。现有方法仅利用源模型的内部知识（通过teacher-student蒸馏），导致两个问题：(1) 特征迁移能力受限于源模型的语义空间，(2) 伪标签存在偏差，尤其在大域偏移时。VFM（如DINOv2、Grounding DINO）在大规模数据上预训练，拥有强大的泛化能力和丰富的语义先验，但在SFOD中的潜力尚未被充分挖掘。

## 核心问题
如何有效利用VFM的外部知识来同时增强SFOD中的特征迁移能力（transferability）和类别区分能力（discriminability）？既有方法要么需要源数据（违反SFOD约束），要么只关注单一维度，未能充分发挥VFM的潜力。

## 方法详解

### 整体框架
基于Mean Teacher自训练框架（EMA更新teacher），在三个维度引入VFM外部知识：PGFA利用DINOv2进行全局patch级特征对齐；PIFA利用DINOv2构建类原型进行实例级对比学习；DEPF融合Grounding DINO和teacher的预测来生成更可靠的伪标签。推理时不引入额外参数/计算。

### 关键设计
1. **Patch-weighted Global Feature Alignment (PGFA)**: 将student backbone的patch特征与DINOv2对齐，但不是均等对待所有patch——通过计算DINOv2 patch间余弦相似度矩阵，用top-k加权策略给语义一致性强的patch分配更高权重。用加权余弦损失对齐student→DINOv2特征空间。关键insight是不同patch的领域不变性差异很大，语义一致的区域更值得对齐。

2. **Prototype-based Instance Feature Alignment (PIFA)**: 从DINOv2特征上用RoIAlign提取实例特征，按类别计算均值特征，用EMA($\mu$=0.9)维护类原型。然后让student的实例特征通过InfoNCE对比损失与对应类原型对齐。momentum更新保证原型稳定性，对比学习同时提升类间区分度和域不变性。

3. **Dual-source Enhanced Pseudo-label Fusion (DEPF)**: 创新性地融合teacher和Grounding DINO的检测框。不用传统WBF（会因不同源标签冲突而出错），而是丢弃类别标签只用IoU聚类框，在每个cluster内计算各预测的Shannon熵，用反熵权重加权融合框坐标和类别概率。低熵（高确信度）的预测获得更大权重，自然解决了两个源标签冲突的问题。

### 损失函数 / 训练策略
$\mathcal{L}_{tot} = \mathcal{L}_{det} + \lambda(\mathcal{L}_{pgfa} + \mathcal{L}_{pifa})$，$\lambda=1$。用Deformable DETR作为基础检测器，lr=$5 \times 10^{-5}$，batch=8，训练30 epochs。EMA系数0.999，每5 iterations更新teacher。推理时VFM完全不参与，无额外开销。

## 实验关键数据

| 基准(SFOD) | 指标 | 本文 | DRU(prev SOTA) | Source Only |
|--------|------|------|----------|------|
| City→Foggy(跨天气) | mAP | **47.1** | 43.6 | 29.6 |
| City→BDD100K(跨场景) | mAP | **43.0** | 36.6 | 28.3 |
| Sim10k→City(虚拟→真实) | AP(car) | **67.4** | 58.7 | 50.8 |
| KITTI→City(跨场景) | AP(car) | **54.7** | 45.1 | 33.9 |
| City→ACDC Snow | mAP | **47.9** | 37.9 | - |
| City→ACDC Fog | mAP | **54.0** | 45.4 | - |

跨检测器验证：Faster R-CNN +3.4%, RT-DETR +3.3%, YOLOv5 +2.4% mAP。跨backbone验证（Swin-T/S/B/L, ViT-B）均一致提升。

### 消融实验要点
- 三个模块逐步叠加：MT baseline 42.3 → +PGFA 43.4(+1.1) → +PIFA 43.9(+1.6) → +PGFA+PIFA 45.0(+2.7) → +DEPF 45.9(+3.6) → 全部 47.1(+4.8)
- DEPF贡献最大（+3.6），因为伪标签质量直接决定自训练上界
- patch权重(PGFA)和熵权重(DEPF)各贡献+0.6和+0.3 mAP
- VFM backbone选择：DINOv2 ViT-G(47.1) > ViT-L(46.8) > ViT-B(46.7) > Grounding DINO Swin-B(46.2)
- 推理时无额外开销，训练时增加79%时间（主要来自VFM特征提取）

## 亮点
- **将VFM引入SFOD的思路很自然且有效**——VFM的泛化能力恰好弥补了源模型的域偏差
- **DEPF的熵引导融合设计巧妙**——丢弃类别标签只聚类框，用反熵权重融合，优雅解决了多源标签冲突问题
- 推理时完全不增加开销（VFM只在训练时使用），实际部署友好
- 跨5种检测器架构和6种backbone都一致有效，方法通用性极强
- 即使源模型很弱（训练5个epoch），也能通过VFM获得显著提升

## 局限性 / 可改进方向
- 训练时需要额外的VFM推理（+79%训练时间），且需要DINOv2+Grounding DINO两个大模型
- Grounding DINO需要文本提示，假设目标域的类别名称已知
- 在极端域偏移下（如ACDC夜景）提升幅度有限，夜景mAP仅23.0
- 未探索更强的VLM（如Qwen-VL、InternVL）替代Grounding DINO的可能

## 与相关工作的对比
- **vs DRU**: DRU同样使用DETR但只依赖内部知识，本文引入VFM外部知识全面超越（+3.5~+9.6 mAP跨不同基准）
- **vs DINO Teacher(DT)**: DT需要源数据训练DINOv2 labeler（违反source-free约束），本文无需源数据却性能接近（差4.8%但设定更严格）
- **vs CODA**: CODA只利用外部检测增强discriminability忽略transferability，本文同时优化两者

## 启发与关联
- VFM作为"外部知识锚点"的思路可以推广到其他领域适应场景——如分割、深度估计的无源域适应
- 熵引导融合策略可以用于任何需要融合多个检测器输出的场景（如ensemble）
- 与模型压缩的交叉点：VFM知识蒸馏到轻量检测器，同时做域适应+模型小型化

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统性地将VFM引入SFOD，但各技术组件（feature alignment, prototype contrastive, box fusion）本身不算新
- 实验充分度: ⭐⭐⭐⭐⭐ 6个基准+5种检测器+6种backbone+极其详尽的消融和分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验图表丰富，motivation说服力强
- 价值: ⭐⭐⭐⭐ 实用价值高，方法通用性强，但SFOD应用范围较窄
