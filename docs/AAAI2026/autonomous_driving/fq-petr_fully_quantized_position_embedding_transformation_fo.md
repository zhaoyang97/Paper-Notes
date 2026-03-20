# FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection

**会议**: AAAI 2026  
**arXiv**: [2502.15488](https://arxiv.org/abs/2502.15488)  
**代码**: [https://github.com/JiangYongYu1/FQ-PETR](https://github.com/JiangYongYu1/FQ-PETR)  
**领域**: 目标检测 / 模型压缩  
**关键词**: 3D目标检测, 量化, PETR, 位置编码, 自动驾驶  

## 一句话总结
首次实现PETR系列3D检测器的全INT8量化部署，通过量化友好的LiDAR-ray位置编码(QFPE)解决多模态特征幅度不匹配问题、双查找表(DULUT)高效逼近非线性算子、数值稳定后量化(QANS)避免softmax注意力失真，在PETR/StreamPETR/PETRv2/MV2D上W8A8精度损失<1%且延迟降低75%（3.9×加速）。

## 背景与动机
PETR系列是transformer-based多视图3D检测的主流方法，但部署在自动驾驶边缘设备上面临严重的计算和内存瓶颈。直接对PETR做量化会造成灾难性精度下降（最多20.8% mAP降低），原因是两个独特挑战：(1) 相机射线位置编码（Camera-ray PE）的动态范围（±130）比图像特征（±4）大近两个数量级，fusion后量化时图像特征被压缩到仅3-5个有效整数bin内；(2) inverse-sigmoid等非线性操作导致outlier，softmax输入范围过大造成量化后注意力失真。

## 核心问题
如何设计适合量化的3D位置编码使其幅度与图像特征匹配，同时高效准确地量化模型中的非线性算子（SiLU/GELU/Softmax），实现PETR的全整数推理而不损失精度？

## 方法详解

### 整体框架
三个协同创新：QFPE重新设计位置编码消除幅度失配 → DULUT用双级联线性LUT逼近非线性函数 → QANS在数值稳定化后再做softmax输入量化。三者组合实现首个PETR的全INT8推理框架。

### 关键设计
1. **Quantization-Friendly LiDAR-ray PE (QFPE)**: 两个创新——(a) LiDAR先验引导的单点采样：每个pixel沿深度射线只采样一个3D点（固定深度30m），消除多点插值和inverse-sigmoid变换（幅度放大11.5×的元凶）。(b) 基于锚点的约束嵌入：每个轴设3个锚点embedding，通过凸组合（线性插值）生成位置编码，保证幅度严格有界($\|\mathbf{e}_\alpha\|_\infty \leq \gamma \approx 0.8$)。最终QFPE动态范围±29.7 vs 原始±127.3，缩小4.4×。不仅利于量化，FP32性能也提升（NDS +1.09/PETR, +0.46/StreamPETR）。

2. **Dual-Lookup Table (DULUT)**: 用两个级联的线性LUT替代单个大LUT或需要专用硬件的NN-LUT。第一个LUT是"非线性索引映射器"——在函数曲率高的区域分配更多indices，平坦区域合并；第二个LUT存储实际值做线性插值。通过迭代优化（error-driven split/merge）获得最优分割。INT8下32+32 entries即可达到256-entry单LUT的精度，SiLU/GELU/Softmax均适用。仅需标准LUT单元，无需专用硬件。

3. **Quantization After Numerical Stabilization (QANS)**: 在softmax数值稳定化（减去最大值）之后再做量化，将输入约束到非正范围$[-\beta, 0]$。自适应选择最优截断下界$\beta$（通过候选集搜索最小化softmax分布偏差），$\beta=20$即为无损。避免了直接量化极端值导致的注意力峰值衰减和位置偏移。

### 损失函数 / 训练策略
QFPE需要finetune（与原PETR训练配置一致，24 epochs，4×RTX4090），但DULUT和QANS是PTQ（仅需32张校准图）。总体effort远低于QAT。

## 实验关键数据

| 模型 | 方法 | mAP | NDS | FPS | Memory |
|--------|------|------|------|-----|--------|
| PETR(R50) | FP32 | 31.42 | 36.11 | 7.1 | 4.8GB |
| PETR(R50) | SmoothQuant W8A8 | 20.67 | 29.32 | - | - |
| PETR(R50) | QuaRot W8A8 | 22.81 | 30.00 | - | - |
| FQ-PETR(R50) | FP32 | 31.49 | 37.20 | - | - |
| FQ-PETR(R50) | SQ*+QANS W8A8 | **31.34** | **37.17** | **27.6** | **1.3GB** |
| StreamPETR(V2-99) | FP32 | 49.51 | 58.03 | - | - |
| FQ-StreamPETR | Quarot*+QANS W8A8 | **50.12** | **58.39** | - | - |

SmoothQuant直接用导致mAP大跌10.8%, FQ-PETR恢复到仅降0.08。3.9×加速，75%显存减少。

### 消融实验要点
- inverse-sigmoid是幅度爆炸的根源（放大11.5×），QFPE通过单点采样+锚点插值彻底消除
- 锚点数=3最优，增减都降低性能
- DULUT (32,32)entries ≈ 256-entry线性LUT精度，但entries少4×
- QANS截断$N \geq 20$即无损，$N < 20$因过截断而降低
- I-BERT/I-ViT的多项式逼近精度不如DULUT
- QFPE甚至在FP32下也提升性能（更好的局部相似性）

## 亮点
- **首次实现PETR全INT8量化且精度几乎无损**——直接填补了transformer-based 3D检测量化的空白
- **QFPE的设计极其巧妙**——不仅解决量化问题，FP32性能也提升了，一石二鸟
- **DULUT通用性强**——可以用于任何需要量化非线性函数的场景（SiLU/GELU/Softmax），且在LLM上也验证有效
- **QANS简单有效**——一个直觉清晰的trick（先稳定再量化），但效果显著（无此步骤mAP降10+%）
- 实际部署效果impressive：3.9×加速，75%显存减少，满足车载部署需求

## 局限性 / 可改进方向
- QFPE需要重新训练位置编码模块，虽然cost类似QAT但仍不是纯PTQ
- 仅在nuScenes上验证，未扩展到Waymo/KITTI等其他数据集
- 只做了INT8，未探索更激进的INT4量化
- 未与BEV-based方法（BEVFormer等）的量化作对比

## 与相关工作的对比
- **vs SmoothQuant/QuaRot**: 这些通用量化方法直接应用于PETR导致灾难性降低（10-20% mAP），因为没有处理PE的幅度不匹配
- **vs QD-BEV**: QD-BEV是QAT+蒸馏方法用于BEV检测器，FQ-PETR的QFPE在更少训练epoch(24)下就超越了QAT(36 epoch)
- **vs I-BERT/I-ViT**: 它们用多项式逼近非线性，精度不如DULUT（NDS低1-2%）

## 启发与关联
- DULUT可以直接用于LLM的INT8量化（论文已初步验证），是一个通用工具
- QFPE的"锚点嵌入+凸组合保幅"思路可以推广到其他需要量化的位置编码
- "先稳定再量化"的QANS思路在任何有softmax的模型（VLM、LLM）中都适用

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 三个创新组件各自解决不同问题，QFPE和DULUT都是genuinely novel
- 实验充分度: ⭐⭐⭐⭐⭐ 4个PETR变体+多种backbone/分辨率+极详尽的消融+理论分析
- 写作质量: ⭐⭐⭐⭐⭐ 问题分析透彻，理论推导严谨，图表直观
- 价值: ⭐⭐⭐⭐⭐ 直接解决PETR部署的核心瓶颈，实用价值极高
