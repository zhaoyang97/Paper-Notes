# MambaSeg: Harnessing Mamba for Accurate and Efficient Image-Event Semantic Segmentation

**会议**: AAAI 2026  
**arXiv**: [2512.24243](https://arxiv.org/abs/2512.24243)  
**代码**: [https://github.com/CQU-UISC/MambaSeg](https://github.com/CQU-UISC/MambaSeg) (有)  
**领域**: 语义分割  
**关键词**: 事件相机, 多模态融合, Mamba/SSM, 语义分割, 时空交互  

## 一句话总结
提出 MambaSeg，用双分支并行 Mamba 编码器分别处理 RGB 图像和事件流，通过空间-时间双维度交互模块 (DDIM) 实现细粒度跨模态融合，在 DDD17 和 DSEC 数据集上以 25.44M 参数取得 77.56%/75.10% mIoU 的 SOTA，效率远优于 Transformer 方案。

## 背景与动机
- **帧相机局限**：传统 RGB 语义分割在高速运动、低光照、高动态范围场景下因运动模糊和延迟严重退化
- **事件相机的互补性**：事件相机具有微秒级时间分辨率和高动态范围，但缺少颜色和纹理信息，单独使用不足以完成密集预测
- **现有融合方法的问题**：(1) Transformer-based 方法（CMX、EISNet）虽有效但计算开销大（自注意力二次复杂度）；(2) 多数方法只做空间融合，忽视了事件流固有的时间动态特性，导致跨模态对齐不充分、语义不一致

## 核心问题
如何在保持低计算开销的前提下，同时沿空间和时间两个维度进行 RGB-Event 跨模态融合，减少跨模态歧义？

## 方法详解

### 整体框架
双分支架构：两条并行的 VMamba-T 编码器（预训练于 ImageNet-1K）分别处理图像和 Voxel Grid 化的事件流。四个尺度的 VSS Block 提取多尺度特征，每个尺度嵌入 DDIM 模块做跨模态交互，融合后的特征反馈回各自编码器用于下一阶段。最后用 SegFormer 的 MLP 解码器生成分割结果。

事件流预处理：原始异步事件 $(x_i, y_i, t_i, p_i)$ 被分割到 $T=10$ 个时间 bin 中积累为 Voxel Grid $E \in \mathbb{R}^{T \times H \times W}$。

### 关键设计
1. **CSIM (Cross-Spatial Interaction Module)**：
   - **跨模态空间注意力**：对事件、图像、及浅层融合特征分别做 AvgPool 和 MaxPool（共 6 张空间图），拼接后经两层卷积+sigmoid 生成三组空间注意力权重，交叉应用到各模态（事件特征乘以图像注意力权重，反之亦然）
   - **SS2D 空间细化**：将拼接特征展开为四个方向序列，分别由独立 S6 Block 处理，捕获多方向长距离依赖后重组为 2D 特征
   - **模态感知残差更新**：分离回各模态后施加空间注意力+残差连接，保留模态特异性

2. **CTIM (Cross-Temporal Interaction Module)**：
   - **跨模态时间注意力**：将事件和图像特征沿时间维度交替插入（interleave），形成 $2T \times H \times W$ 的时间序列，经全局 MaxPool/AvgPool + 1×1 卷积生成时间注意力权重 $W_F^T \in \mathbb{R}^{T \times 1 \times 1}$，同时调制两个模态
   - **双向时间选择性扫描 (BTSS)**：拼接后展平为时间序列，前向和反向各由一个 S6 Block 处理，聚合过去和未来的时间上下文后求和重塑
   - **模态感知残差更新**：同 CSIM，分离+时间注意力+残差

3. **DDIM = CSIM + CTIM**：两个模块串联使用，在每个编码器尺度都进行空间+时间双维度融合

### 损失函数 / 训练策略
- 损失函数：标准交叉熵
- 优化器：AdamW，训练 60 epochs
- DDD17：lr=2e-4, batch_size=12；DSEC：lr=6e-5, batch_size=4
- 数据增强：随机裁剪、水平翻转、随机缩放
- 单卡 RTX 4090D 训练

## 实验关键数据

### 主要对比 (Table 1)

| 方法 | 类型 | Backbone | DDD17 mIoU | DSEC mIoU |
|------|------|----------|-----------|-----------|
| SegFormer | 纯图像 | Transformer | 71.05% | 71.99% |
| EV-SegNet | 纯事件 | CNN | 54.81% | 51.76% |
| CMX | 融合 | Transformer | 71.88% | 72.42% |
| CMNeXt | 融合 | Transformer | 72.67% | 72.54% |
| EISNet | 融合 | Transformer | 75.03% | 73.07% |
| **MambaSeg** | **融合** | **Mamba** | **77.56%** | **75.10%** |

对比 SOTA EISNet：DDD17 +2.53%，DSEC +2.03%。

### 效率对比 (Table 2, DDD17)

| 方法 | 参数量 (M) | MACs (G) | mIoU |
|------|-----------|----------|------|
| CMX | 66.56 | 16.29 | 71.88% |
| EISNet | 34.39 | 17.30 | 75.03% |
| **MambaSeg** | **25.44** | **15.59** | **77.56%** |

参数量仅为 EISNet 的 74%，MACs 也更低，但 mIoU 高 2.53%。

### 消融实验要点
- **CSIM vs CTIM**（Table 4）：baseline 74.38% → +CTIM 76.20% → +CSIM 76.32% → 两者结合 77.56%，说明空间和时间融合互补
- **DDIM vs 其他融合方法**（Table 3）：Element-wise Add 74.38%，FFM 76.44%，MRFM 76.19%，CSF 76.65%，**DDIM 77.56%**
- **CSIM 子模块**（Table 5）：CSA、SS2D、SA 三者缺一不可，完整 CSIM 最优
- **CTIM 子模块**（Table 6）：CTA、BTSS、TA 三者同样互补，完整 CTIM 最优

## 亮点
1. **首次将 Mamba 引入 RGB-Event 多模态融合分割**，利用 SSM 的线性复杂度替代 Transformer 的二次复杂度，效率提升显著
2. **空间-时间双维度融合设计**合理且新颖：CSIM 利用事件的边缘优势+图像的纹理优势做空间互补，CTIM 利用 Mamba 擅长的序列建模做时间对齐，二者互补
3. **定性结果**显示 MambaSeg 在小目标（行人、交通标志）和复杂光照下明显优于 EISNet
4. 消融实验设计很系统，从模块级到子组件级都有详细分析

## 局限性 / 可改进方向
1. **数据集单一**：仅在自动驾驶场景的 DDD17（6类）和 DSEC（11类）上验证，类别少、场景单一，泛化能力未知
2. **事件表示局限**：采用固定时间窗口的 Voxel Grid，可能丢失事件流的精细时间信息；可考虑自适应时间分段或直接处理异步事件
3. **编码器未联合训练**：两条分支用相同的预训练 VMamba-T 初始化，事件分支用 ImageNet 预训练是否最优有待验证
4. **仅使用交叉熵损失**：未探索 Dice Loss、Lovász Loss 等对分割更友好的损失
5. **未在更大规模场景**（如城市场景、室内场景）和更多模态（深度、LiDAR）上验证

## 与相关工作的对比
- **vs CMX/CMNeXt**：同为多模态融合但用 Transformer 做 cross-attention，计算量大（66M/58M 参数）；MambaSeg 用 Mamba 替代，参数量降至 25M 且精度更高
- **vs EISNet**：EISNet 用门控注意力+渐进重校准做自适应对齐，侧重空间；MambaSeg 增加了时间维度融合，且效率更高
- **vs Hybrid-Seg**：CNN+SNN 混合架构，参数效率好但精度差距大（67.31% vs 77.56% on DDD17）
- **vs VM-UNet 等医学 Mamba 分割**：VM-UNet 是单模态 Mamba 分割，MambaSeg 是多模态双分支+跨模态交互

## 启发与关联
- **Mamba 在多模态融合中的潜力**：本文验证了 Mamba 在 RGB-Event 融合中的有效性，可推广到 RGB-Depth、RGB-Thermal、RGB-LiDAR 等其他多模态组合
- **时间维度建模**：CTIM 的双向时间选择性扫描思路可用于视频分割等需要时间建模的任务
- **与 ideas/ 中 Mamba 相关方向的关联**：
  - `ideas/model_compression/20260317_mamba_light_medical_seg.md`：全 Mamba 轻量医学分割方向可参考本文的双分支设计
  - `ideas/video_understanding/20260317_human_video_mamba.md`：Mamba 视频理解方向可借鉴本文的 CTIM 时间融合模块
  - 本文的 DDIM 空间-时间双维度融合思路可推广到更多需要多源异构数据融合的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ Mamba 引入 RGB-Event 融合是新颖的，DDIM 的空间-时间双维度设计有创新，但各子模块（池化+注意力、SS2D）相对常规
- 实验充分度: ⭐⭐⭐⭐ 消融实验非常系统（模块级+子组件级+融合方法对比+效率对比），但仅两个驾驶数据集，缺少其他场景验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表规范，公式推导完整，但 Related Work 偏简略
- 价值: ⭐⭐⭐⭐ 在 RGB-Event 分割这个小方向上有明确推进，效率优势明显，但受限于事件相机的实际应用范围
