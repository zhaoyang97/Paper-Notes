# LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models

**会议**: ICCV 2025  
**arXiv**: [2504.14032](https://arxiv.org/abs/2504.14032)  
**代码**: [https://github.com/andrehuang/loftup](https://github.com/andrehuang/loftup)  
**领域**: 自监督学习 / 视觉基础模型 / 特征上采样  
**关键词**: feature upsampling, vision foundation model, coordinate-based, self-distillation, DINOv2, 密集预测  

## 一句话总结
提出LoftUp，通过坐标-cross-attention架构直接将低分辨率VFM特征映射到任意高分辨率，并用class-agnostic mask精炼+自蒸馏构建全分辨率伪GT进行训练，在6个下游任务上平均提升10-20%且在视频目标分割上提升近50%。

## 背景与动机
DINOv2/CLIP等VFM输出的特征图分辨率通常只有输入的1/16（如224输入→16×16特征），这严重限制了需要像素级理解的密集预测任务（分割、深度估计等）。现有方案包括：(1) 增大输入分辨率→计算量暴增；(2) 训练任务特定decoder→缺乏通用性；(3) FeatUp/LiFT等通用特征上采样器→但它们在低分辨率计算损失，无法充分约束高分辨率细节，且存在模糊和artifacts。

## 核心问题
如何设计一个任务无关的、即插即用的VFM特征上采样器，能生成清晰、高质量的全分辨率特征图？核心挑战有二：(1) 上采样器架构——如何避免多层上采样带来的累积模糊？(2) 训练目标——没有高分辨率GT特征，如何构建有效的监督信号？

## 方法详解

### 整体框架
LoftUp是一个轻量级的2层cross-attention transformer（<20%的VFM参数），两阶段训练。Stage 1用SAM的class-agnostic mask精炼双三次上采样特征作为pseudo-GT；Stage 2用自蒸馏（teacher处理高分辨率裁剪、student处理标准分辨率）进一步生成高质量pseudo-GT。训练完成后，LoftUp是即插即用的，无需test-time优化。

### 关键设计
1. **坐标-cross-attention架构**：受3D重建中隐式神经表示（如NeRF）启发，将特征上采样视为"坐标→特征"的映射。高分辨率图像的每个像素坐标（经正弦编码）+RGB值作为query，低分辨率VFM特征作为key/value，通过cross-attention全局交互生成高分辨率特征。这比多层反卷积/双线性上采样更锐利（避免累积模糊），比LIIF的局部MLP更强（全局注意力），且支持任意上采样倍率。

2. **Stage 1: SAM mask精炼的Pseudo-GT**：将低分辨率特征双三次上采样到全分辨率，然后用SAM生成的class-agnostic mask对每个mask区域计算均值特征，与原始特征混合（$F_{\text{Mask-Bicubic}}[m] = \alpha \cdot \overline{F_{\text{Bicubic}}[m]} + (1-\alpha) \cdot F_{\text{Bicubic}}[m]$，α=0.8）。这利用SAM的边界信息使特征在物体内部平滑、在边界处锐利。

3. **Stage 2: 自蒸馏精炼**：teacher和student都从Stage 1的预训练模型初始化。Teacher处理2-4x高分辨率裁剪（更容易的任务→更好的特征），其输出下采样后作为student的监督。Teacher用EMA更新。这进一步减少Stage 1残留的模糊和artifacts，生成更几何一致的pseudo-GT。

### 损失函数 / 训练策略
- Stage 1: L2 loss与mask-refined pseudo-GT
- Stage 2: Affinity matrix loss（优于L2），EMA decay=0.99，每10步更新teacher
- 在SA1B 1M子集上训练，batch size 8，AdamW，Stage 1 lr=1e-3，Stage 2 lr=1e-4

## 实验关键数据
| 任务 | 指标 | Low-res | Bilinear | FeatUp | LiFT | **LoftUp** |
|------|------|---------|----------|--------|------|---------|
| COCO分割 | mIoU | 51.21 | 56.15 | 56.30 | 53.35 | **61.11** |
| Cityscapes分割 | mIoU | 36.54 | 44.79 | 44.19 | 35.80 | **53.10** |
| NAVI深度 | δ3↑ | 89.08 | 87.68 | 88.57 | 88.71 | **91.35** |
| DAVIS VOS | J&F | 52.82 | 54.26 | 55.03 | 41.14 | **67.31** |
| OV分割 | mIoU | 25.70 | 25.78 | 26.61 | 25.96 | **27.82** |
| 交互式分割 | IoU@1 | 55.77 | 55.83 | 56.67 | 31.99 | **65.24** |

- 语义分割：比FeatUp提升+7.3%(COCO)和+15.6%(Cityscapes)
- 视频目标分割：比低分辨率baseline提升**近50%**（J Mean: +39.6%，F Mean: +97.6%）
- 推理速度与bilinear相当（0.089s vs 0.092s），远快于FeatUp-Implicit（54.3s）
- 同样训练目标下，LoftUp架构在所有任务上优于resize-conv/LIIF/FeatUp-JBU
- 适用于不同VFM：DINOv2/CLIP/RADIO上均一致优于所有对比方法

### 消融实验要点
- 正弦位置编码+3x3卷积处理图像输入→最优组合
- 2层cross-attention足够（3层无额外增益）
- Pseudo-GT对比：自蒸馏 > mask-bicubic > per-image优化 > 2x特征
- 训练数据50k→1M性能逐步提升，但收益渐减
- 全分辨率loss是关键——FeatUp/LiFT的低分辨率loss不够约束细节

## 亮点
- **从3D重建借鉴坐标表示**：将NeRF的坐标→颜色映射思想迁移到特征上采样，绕过了传统多层上采样的根本限制
- **两阶段pseudo-GT构建很精巧**：SAM mask提供几何边界→双三次做半径平滑→自蒸馏进一步精炼，逐步提升质量
- **真正的task-agnostic**：在6种差异极大的任务（分割/深度/法线/VOS/OV分割/交互式分割）上一致提升，通用性极强
- **<20%额外参数 + bilinear级推理速度**：plug-and-play的设计极其实用
- **VOS上的惊人提升**：+50%的提升说明高分辨率特征对temporal对应关系极为重要

## 局限性 / 可改进方向
- Stage 1依赖SAM生成mask，增加了训练时的计算依赖
- 仅在DINOv2-S/14上做主实验，更大模型的效果未充分验证
- 尚未在更高分辨率输入（如512/1024）上系统评估
- Cross-attention的全局交互在超高分辨率时可能面临计算瓶颈

## 与相关工作的对比
- **vs. FeatUp**：FeatUp用modified JBU架构+多视角重建任务做低分辨率loss；LoftUp用坐标cross-attention架构+全分辨率self-distilled GT，两个维度都更强
- **vs. LiFT**：LiFT用U-Net做2x上采样+2x特征做GT；LoftUp直接映射到任意分辨率+高质量pseudo-GT
- **vs. LIIF**：LIIF也是坐标方法但仅局部MLP交互；LoftUp的cross-attention实现全局交互

## 启发与关联
- 坐标-based设计使特征上采样器具有分辨率无关的灵活性，这对部署到不同分辨率场景很有价值
- SAM mask + 自蒸馏的pseudo-GT生成方法可以迁移到其他缺乏GT的学习场景
- 与Scaling Language-Free Visual Repr的研究互补——SSL encoder的特征在高分辨率上可能有更大提升空间

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从3D到2D的坐标表示迁移+两阶段pseudo-GT设计，两个关键创新都非常新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 6个异质性下游任务、多种VFM backbone、逐组件消融、伪GT对比、效率分析
- 写作质量: ⭐⭐⭐⭐⭐ 问题分解清晰（架构+训练目标），Table 1/2的pseudo-GT属性对比表格很有教学价值
- 价值: ⭐⭐⭐⭐⭐ 可以作为所有VFM的标配增强模块，对dense prediction社区有很大影响
