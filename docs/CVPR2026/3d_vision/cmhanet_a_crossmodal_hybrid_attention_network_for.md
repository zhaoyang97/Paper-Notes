# CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration

**会议**: CVPR 2026  
**arXiv**: [2603.12721](https://arxiv.org/abs/2603.12721)  
**代码**: [有](https://github.com/DongXu-Zhang/CMHANet)  
**领域**: 3D视觉 / 点云配准  
**关键词**: 点云配准, 跨模态融合, 混合注意力, RGB-D, 对比学习  

## 一句话总结
提出CMHANet，通过三阶段混合注意力（几何self-attention→图像aggregation-attention→源-目标cross-attention）融合2D图像纹理语义与3D点云几何信息，并引入跨模态对比损失，在3DMatch/3DLoMatch上达到最优配准性能。

## 背景与动机
点云配准是3D视觉基础任务，但现有深度学习方法在真实场景（数据不完整、传感器噪声、低重叠率）中性能显著下降。核心问题是：绝大多数方法仅利用几何信息，忽略了RGB-D传感器已经普遍提供的配对2D图像数据。点云的3D几何精确但缺少纹理描述，图像提供密集的纹理和语义但缺乏3D信息——二者天然互补。已有的多模态融合方法（IMFNet、CMIGNet、PCR-CG）大多使用通用融合机制，缺乏对几何-视觉特征交互的精细建模。

## 核心问题
如何设计精细的跨模态注意力机制，智能地融合3D几何结构与2D视觉语义，提升复杂场景下的点云配准精度和鲁棒性？

## 方法详解

### 整体框架
四阶段pipeline：(1)特征提取与下采样——KPConv-FPN提取点云超级点及特征,ResUNet-50提取图像特征；(2)混合注意力超级点匹配——三种注意力交替N次迭代；(3)密集点对应模块——从粗匹配到精匹配；(4)变换估计——加权SVD计算局部变换，Local-to-Global验证选最佳。

### 关键设计
1. **几何自注意力(Geometric Self-Attention)**: 每个超级点与同一点云内所有超级点交互，Key融合了学习特征和几何位置编码（距离编码+三角角度编码的聚合），使注意力具备空间感知能力。位置编码由正弦函数+MLP生成，最终E_P = E_D·W_D + max_r{E_A·W_A}。
2. **几何聚合注意力(Geometric Aggregation-Attention)**: 跨模态融合的核心。3D超级点生成Query，2D图像patch生成Key/Value，在Q和K中同时注入各自模态的位置编码（3D坐标嵌入和2D像素坐标嵌入），通过独立的W_f和W_g投影到共享语义空间，保证几何一致性。每个3D点选择性吸收最相关的2D视觉线索，通过残差连接更新。
3. **跨模态对比损失(L_cmc)**: 在超级点级别构建3D几何特征和对应图像特征的对比学习——对角线为正样本对，非对角线为负样本。即使batch size=1也有效。与粗匹配损失(overlap-aware circle loss)和精匹配损失联合优化。

### 损失函数 / 训练策略
总损失L = L_c + L_f + λ·L_cmc。L_c为overlap-aware circle loss（粗匹配，重叠>10%为正，无重叠为负）；L_f为点级精匹配的负对数似然损失；L_cmc为跨模态对比损失。Sinkhorn迭代L=50次将相似度矩阵转为双随机矩阵（含learnable dustbin处理outlier）。PyTorch实现，RTX 3090，Adam优化器，50 epochs，lr=1e-4指数衰减0.05/epoch，matching radius τ_a=5cm，λ=0.5。

## 实验关键数据
| 数据集 | 指标 | CMHANet | GeoTransformer | OIF-PCR | CoFiNet |
|---|---|---|---|---|---|
| 3DMatch | RR%(5000) | **92.4** | - | - | 89.3 |
| 3DLoMatch | RR%(5000) | **75.5** | - | - | 67.5 |
| 3DMatch | RRE(°) | **1.764** | 1.772 | - | 2.002 |
| 3DMatch | RTE(m) | **0.060** | 0.061 | - | 0.064 |
| 3DMatch | IR%(250) | **86.2** | - | 67.5 | 52.2 |
| TUM RGB-D(零样本) | RMSE(×10⁻²)均值 | **0.76** | - | - | - |

vs 多模态方法：RR在3DLoMatch上比PCR-CG高+9.2%（75.5 vs 66.3）。
LGR(无RANSAC)配准：91.9%/74.2%，精度接近RANSAC但速度快100倍。

### 消融实验要点
- 去掉Image Module: 所有指标显著下降（3DMatch PIR 83.8→86.8, 3DLoMatch RR 71.9→75.5）——跨模态融合是核心
- 去掉Hybrid Attention: 3DMatch RR 90.5→92.4, 3DLoMatch RR 72.4→75.5——三阶段迭代注意力比直接特征比较好3.1%
- 去掉Aggregation-Attention: 3DLoMatch RR从75.5降至73.6——图像聚合注意力贡献最大
- 去掉对比损失: 3DMatch RR 91.4→92.4——损失函数设计有效
- 图像backbone比较：ResUNet-50 > ResNet-101 ≈ ResNet-34（50在精度和效率间最佳平衡）

## 亮点
- 三种注意力按几何self→跨模态aggregation→源目标cross的顺序交替迭代，设计逻辑清晰
- 在Key中融合特征和几何位置编码的方式比简单拼接更优雅，使注意力具备空间感知
- TUM RGB-D零样本评估RMSE 0.76(×10⁻²)大幅领先Robust ICP的1.69和Teaser++的14.06，泛化性强
- Inlier Ratio提升巨大（如250采样下3DLoMatch从OIF-PCR的33.1%到58.3%），说明特征质量本质性提升

## 局限性 / 可改进方向
- 需要配对的RGB-D输入，纯LiDAR点云场景无法使用
- 图像特征编码增加了推理时间（Model time 0.144s vs CoFiNet 0.115s），但total time几乎持平
- 极低重叠(<10%)或完全无纹理平面场景可能失效（作者承认）
- 室外大规模场景（自动驾驶）的适用性未验证

## 与相关工作的对比
vs IMFNet: 同为多模态方法但CMHANet在3DLoMatch RR上大幅领先（75.5 vs 48.4），核心优势在于混合注意力比IMFNet的简单注意力融合更有效。
vs PCR-CG: 另一多模态方法，CMHANet RR在3DMatch高3%(92.4 vs 89.4)、3DLoMatch高9.2%(75.5 vs 66.3)。
vs GeoTransformer: 单模态SOTA基于Transformer的方法，CMHANet通过多模态引入额外增益(RRE 1.764 vs 1.772)。
vs 传统ICP系列: 零样本TUM上RMSE 0.76远优于ICP(2.8)、Robust ICP(1.69)。

## 启发与关联
- 聚合注意力中3D位置编码和2D位置编码共享语义空间的设计可迁移到其他3D-2D融合任务
- 对比学习在几何匹配中构建正/负样本的方式值得借鉴

## 评分
- 新颖性: ⭐⭐⭐ 三种注意力的组合方式有贡献但每个组件独立来看不算全新
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多估计器、多采样率的全面评估，零样本泛化令人信服
- 写作质量: ⭐⭐⭐ 方法描述详细但related work中引入过多弱相关工作
- 价值: ⭐⭐⭐ 对RGB-D点云配准有实际推动，实验结果solid
