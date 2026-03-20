# A2P: From 2D Alignment to 3D Plausibility for Occlusion-Robust Two-Hand Reconstruction

**会议**: CVPR 2026  
**arXiv**: [2503.17788](https://arxiv.org/abs/2503.17788)  
**代码**: [项目页](https://gaogehan.github.io/A2P/)  
**领域**: 人体理解 / 手部重建  
**关键词**: two-hand reconstruction, fusion alignment encoder, penetration-free diffusion, MANO, Sapiens  

## 一句话总结
解耦双手重建为2D结构对齐+3D空间交互对齐：Stage 1用Fusion Alignment Encoder隐式蒸馏Sapiens的关键点/分割/深度三种2D先验(推理时免基础模型)，Stage 2用穿透感知扩散模型+碰撞梯度引导将穿透姿态映射到物理合理配置——InterHand2.6M上MPJPE降至5.36mm(超SOTA 4DHands 2.13mm)，穿透体积降7倍。

## 背景与动机
单目双手重建面临互遮挡和手指穿透两大挑战。现有方法(IntagHand/ACR/4DHands)缺乏显式的对齐机制：2D-3D对应关系模糊导致空间不一致/非自然交互/穿透伪影。基础模型(Sapiens等)的2D先验(关键点/分割/深度)可提供指导，但直接使用太重(1B参数)；扩散模型可建模交互先验，但需准确对齐且容易漂移到不合理状态。

## 核心问题
如何在推理高效的条件下，利用多模态2D先验实现2D结构对齐，并用扩散模型实现3D空间交互的物理合理性（消除穿透）？

## 方法详解

### 整体框架
两阶段Pipeline。Stage 1(2D对齐)：用Sapiens提取关键点/分割/深度先验特征→Projection融合→Fusion Alignment Encoder(ResNet-50)用MSE蒸馏学习融合特征→Transformer Encoder融合图像特征+蒸馏先验→MANO回归。推理时移除所有基础模型。Stage 2(3D交互)：检测双手IoU>0且穿透→将穿透MANO参数作为条件输入穿透感知扩散模型→DDIM去噪+碰撞梯度引导→输出物理合理配置。

### 关键设计
1. **Fusion Alignment Encoder(FAE)**: 训练时用Sapiens(1B)提取3种2D先验特征F_k/F_s/F_d，Projection层融合为F_p。FAE(轻量ResNet-50)用MSE损失学习对齐F_p。推理时只需FAE(52.6M参数)替代Sapiens(1B)，3fps→56fps加速18.7倍，精度仅降0.38mm MRRPE。关键：不提取显式先验预测而是蒸馏隐式特征→避免先验预测误差传播。

2. **穿透感知扩散模型**: Transformer-based架构，MDM风格。训练数据：低性能模型的穿透输出+对GT加噪直到穿透。条件输入=穿透MANO参数，目标输出=干净MANO参数。L2去噪损失。推理只在IoU>0时激活(大部分帧跳过)。

3. **碰撞梯度引导**: 每步DDIM去噪后：(1) 计算双手mesh的Chamfer距离找近邻顶点对；(2) 法向余弦相似度检测穿透(法向量方向相反)；(3) 用GMoF鲁棒碰撞损失L_collision的负梯度更新去噪结果。结合混合距离-方向准则确保只纠正真正穿透而非正常接触。

### 损失函数 / 训练策略
Stage 1: L = L_hand(MANO参数+关节+2.5D坐标L1) + L_prior(MSE蒸馏)。Stage 2: L = L2去噪损失。4×A100，AdamW lr=1e-4，batch 48。

## 实验关键数据

### InterHand2.6M (5fps test)
| 方法 | MRRPE↓ | MPJPE↓ | MPVPE↓ |
|------|--------|--------|--------|
| InterWild | 26.74 | 7.85 | 8.16 |
| InterHandGen | 25.42 | 7.50 | 7.78 |
| 4DHands | 24.58 | 7.49 | 7.72 |
| **Ours** | **21.60** | **5.36** | **5.58** |

### HIC (野外数据，无训练数据)
| 方法 | MRRPE↓ | MPJPE↓ | MPVPE↓ |
|------|--------|--------|--------|
| 4DHands | 25.26 | 9.32 | 9.93 |
| **Ours** | **22.24** | **6.67** | **6.93** |

### 穿透指标
| PenVol↓ | PenDist↓ | ProxRatio↑ |
|---------|----------|------------|
| InterHandGen: 0.76 | 0.04 | 0.97 |
| **Ours: 0.11** | **0.01** | **0.99** |

### 消融实验要点
- **逐步加先验**: 关键点(-1.29 MPJPE) > 分割(-0.29) > 深度(-0.45 MPJPE, -2.14 MRRPE)
- **深度先验对Z维度关键**: MPJPE-Z从4.54→3.37(主要在深度方向)
- **穿透扩散模型**: MPJPE再降0.38，MRRPE降0.78，XY和Z维度都改善
- **FAE效率**: 参数52.6M(vs 1B基础模型)，推理56fps(vs 3fps)，MRRPE仅增0.47

## 亮点 / 我学到了什么
- **训练时用大模型、推理时用蒸馏小模型**: FAE的蒸馏策略是"foundation-level guidance without foundation-level cost"的实用方案
- **条件扩散做"修复"而非"生成"**: 输入穿透姿态→输出合理姿态，比从零生成更稳定
- **碰撞梯度引导的混合距离-方向准则**: 距离近+法向量反向=穿透，距离近+法向量同向=正常接触。避免错误纠正合理的手接触

## 局限性 / 可改进方向
- 运动模糊时2D先验不可靠导致失败
- 未利用时序信息
- 扩散模型推理仍引入额外开销

## 与相关工作的对比
- **vs 4DHands**: 4DHands用RAT+SIR建模双手关系但无显式穿透处理。A2P直接用扩散模型学习穿透→合理的映射
- **vs InterHandGen**: 扩散模型仅做正则化，穿透抑制不充分。A2P显式建模条件去穿透+碰撞梯度引导

## 评分
- 新颖性: ⭐⭐⭐⭐ 2D先验蒸馏+穿透扩散的两阶段解耦设计新颖，碰撞梯度引导巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ InterHand2.6M/HIC/FreiHAND/野外数据，详细消融(先验/扩散/FAE效率)
- 写作质量: ⭐⭐⭐⭐ 动机清晰，pipeline图表信息量大
- 对我的价值: ⭐⭐⭐ 手部重建非核心方向，但FAE蒸馏和条件扩散修复的思路可迁移
