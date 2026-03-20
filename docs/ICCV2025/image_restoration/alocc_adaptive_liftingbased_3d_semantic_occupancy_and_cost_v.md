# ALOcc: Adaptive Lifting-Based 3D Semantic Occupancy and Cost Volume-Based Flow Predictions

**会议**: ICCV 2025  
**arXiv**: [2411.07725](https://arxiv.org/abs/2411.07725)  
**代码**: [https://github.com/cdb342/ALOcc](https://github.com/cdb342/ALOcc)  
**领域**: 3D视觉 / 自动驾驶  
**关键词**: 3D Occupancy Prediction, 2D-to-3D View Transformation, Occlusion-Aware Lifting, Semantic Prototype, Occupancy Flow  

## 一句话总结

提出ALOcc框架，通过遮挡感知的自适应提升机制、语义原型对齐和BEV代价体flow预测三个改进，在多个占据预测基准上取得SOTA，同时保持较高推理速度。

## 背景与动机

基于视觉的3D语义占据预测需要将2D图像特征变换到3D空间，这是一个病态问题。现有的两种2D-to-3D变换方法各有缺陷：
- **Depth-based LSS**：显式预测深度概率引导特征传递，但深度目标是δ分布，权重集中在表面点，被遮挡区域权重极低，远距离处空间密度也低。初期不准确的深度估计还会让网络陷入局部最优。
- **2D-3D Cross Attention**：通过交叉注意力被动传递信息，不建模显式结构信息。

此外，语义占据和flow的联合预测会对特征编码施加双重压力（既要编语义又要编运动），场景中的长尾分布问题也影响稀有类别的预测精度。

## 核心问题

1. 如何让2D-to-3D变换不只关注表面、而能覆盖遮挡和稀疏区域？
2. 3D特征的语义质量如何与原始2D信号保持一致，同时应对类别不平衡？
3. 联合预测语义和flow时，如何减轻特征的多任务编码负担？

## 方法详解

### 整体框架

输入N个环视相机图像 → ResNet-50提取2D特征 → **自适应提升**将2D特征变换到3D体素空间 → 3D编码器（结合16帧历史做时序融合）→ 两个解码头分别输出语义占据和occupancy flow。框架是纯卷积架构，无Transformer。

### 关键设计

1. **遮挡感知自适应提升（Occlusion-Aware Adaptive Lifting）**：

   传统LSS用深度概率填充变换矩阵M_T，但深度目标是δ分布——几乎所有权重都在表面上，遮挡区域得不到特征。ALOcc做了三件事来解决：

   - **软填充替换硬取整**：用三线性插值将深度概率扩散到周围8个体素中心，使2D→3D映射对坐标可微。
   - **同物体内遮挡建模**：设计条件概率矩阵 P(o_l|o_d)，将表面深度概率传递到更深位置。物理含义是"如果相机光线到达了深度i的表面，那么深度j(j>i)也可能被同一物体占据"。用因果条件矩阵乘法实现：深度i之前的位置概率为0（因为光线能到达i说明前面都是空的），深度i本身为1，后面的位置用网络 f_h(x, j-i) 预测。
   - **物体间遮挡建模**：为每个点预测多个偏移 (Δu, Δv) 和传递权重 w，将占据概率传播到邻近像素位置，覆盖被前景物体遮挡的背景物体。为节省计算，只对top-k深度概率的点做传播。
   - **深度去噪**：借鉴目标检测中的query denoising思想，训练时用GT深度和预测深度的加权平均，权重按余弦退火从1→0。这样早期训练有GT深度"兜底"，不会因深度不准就陷入局部最优。

2. **语义原型占据头（Semantic Prototype-based Occupancy Head）**：

   为每个语义类别初始化一个可学习的原型向量，这个原型同时作为2D和3D特征的分类权重，天然建立跨维度的语义对齐桥梁。预测时计算体素特征与原型的内积，取最大响应类别。

   为应对长尾问题设计了两种策略：
   - **原型独立训练**：某类不在当前场景GT中时，不训练对应原型，每个类别mask独立预测，避免多数类压制少数类。
   - **不确定性+类先验采样**：用logit map衡量每个体素的不确定性，结合类别先验构成多项分布，从全体素中采样K个难样本，只在这些点上计算loss。

3. **BEV代价体flow预测（BEV Cost Volume-based Flow Head）**：

   直接在体素特征上同时预测语义和flow会造成特征编码冲突。ALOcc的解法是构建一个独立的运动先验：
   - 将0~4m高度的体素特征平均池化到BEV平面，下采样增大感受野
   - 用相机参数将前一帧BEV特征warp到当前帧坐标系
   - 在多个假设偏移点处计算当前帧与warp后前帧特征的余弦相似度，构建代价体
   - 代价体+当前帧体素特征一起送入flow网络，体素特征可以专注语义编码
   - **分类-回归混合预测**：将flow值离散化为bins，预测每个bin的概率，最终 flow = Σ(p_bin × bin_center)。同时用回归损失(L2+cosine)和分类交叉熵监督。

   代价体复用cached的历史帧特征，不需要二次推理。

### 损失函数 / 训练策略

- 总损失 L_total = L_3D + L_2D + L_flow + L_depth
- L_3D / L_2D: 5×Dice Loss + 20×BCE，仅在采样的K个体素(像素)上计算
- L_flow = L_flow_reg (L2 + cosine similarity) + L_flow_cls (bin分类交叉熵)
- AdamW (lr=2e-4), batch=16, 12 epochs (occ) / 18 epochs (occ+flow), CBGS类平衡采样
- 三种模型变体：ALOcc-3D (3D卷积, ch=32) / ALOcc-2D (高度压缩, ch=80) / ALOcc-2D-mini (单目深度+小通道)

## 实验关键数据

| 数据集/设置 | 指标 | 本文 (R50) | 之前SOTA (R50) | 提升 |
|------------|------|-----------|---------------|------|
| Occ3D (w/ mask) | mIoU_m | 45.5 (ALOcc-3D) | 44.5 (COTR) | +1.0 |
| Occ3D (w/ mask) | mIoU_D_m | 39.3 (ALOcc-3D) | 38.6 (COTR) | +0.7 |
| Occ3D (w/o mask) | RayIoU | 38.0 (ALOcc-3D) | 31.6 (P-FlashOcc) | +6.4 |
| Occ3D (w/o mask) | mIoU | 43.7 (ALOcc-3D) | 41.2 (OPUS) | +2.5 |
| OpenOcc | Occ Score | 43.0 (Flow-3D) | 41.0 (F-Occ) | +2.0 |
| OpenOcc | mAVE↓ | 0.481 (Flow-3D) | 0.493 (F-Occ) | -0.012 |

- Swin-Base大模型版本: mIoU_m **50.6**, mIoU_D_m **46.1**
- ALOcc-2D-mini实时版: **30.5 FPS**, mIoU_m 41.4 (FlashOcc: 29.6FPS/32.0, 速度相近但精度高+9.4)
- CVPR24 Occupancy and Flow Competition **第2名**

### 消融实验要点

- 去掉自适应提升(AL): mIoU_m 44.5→43.5 (-1.0), mIoU_D_m 38.5→37.5 (-1.0)
- 去掉语义原型头(SP): mIoU_m 44.5→42.1 (-2.4), 影响更大
- AL和SP都去掉: mIoU_m 41.2 (-3.3)
- 加flow head伤害语义: RayIoU从42.4降到39.7
- BEV代价体(CV)缓解冲突: 加CV后RayIoU 39.7→40.2, flow也改善
- Bin分类(BC)显著提升flow精度: mAVE 0.508→0.464
- 通道数40→80是联合学习的瓶颈，扩大后整体显著提升

## 亮点

- 遮挡感知的理论建模严谨——用贝叶斯条件概率推导"表面→遮挡"的概率传递，像人从局部推断完整物体
- 深度去噪训练策略巧妙，余弦退火从GT平滑过渡到预测深度，避免"鸡生蛋"困境
- 共享语义原型同时约束2D和3D，一个prototype集合搭建跨维度语义桥梁
- BEV cost volume复用历史帧cached特征，不需要二次推理，设计高效
- 三个速度/精度变体从30.5FPS到6FPS，实用性极强

## 局限性 / 可改进方向

- 自适应提升中的偏移传播只用top-k点，极端密集遮挡场景可能不够
- Flow预测仅在BEV平面(X/Y)，高度方向(Z)运动信息被压缩
- 深度去噪依赖LiDAR投影的GT深度，限制纯视觉独立性
- 语义原型是固定类别数量，无法处理开放词汇场景（可结合CLIP扩展）
- 未在Waymo等其他大规模数据集上验证泛化性

## 与相关工作的对比

- **vs FB-Occ**: 同为LSS改进路线, ALOcc额外建模遮挡传播+语义原型, mIoU_m高出+5.7
- **vs COTR**: 性能最接近的竞争者, 但COTR只有0.5FPS, ALOcc-3D 6.0FPS快12倍
- **vs FlashOcc**: 实时方法中速度相近(30.5 vs 29.6 FPS)但mIoU高+9.4
- **vs SparseOcc**: SparseOcc引入RayIoU指标, ALOcc在RayIoU上高+7.1
- **vs LetOccFlow**: flow竞品, ALOcc用BEV cost volume替代自监督flow, OccScore高+6.6

## 启发与关联

- 遮挡感知的概率传递思想可迁移到其他处理遮挡的任务（行人重识别、室内场景补全等）
- 语义原型的2D-3D桥接思路与CLIP跨模态对齐异曲同工，可引入预训练VLM的语义原型实现开放词汇占据预测
- 与ideas/中的[开放词汇3D占据网格预测](../../ideas/3d_vision/20260316_open_vocab_3d_occupancy.md)高度相关——ALOcc的语义原型机制可作为开放词汇扩展的基础架构
- BEV cost volume的思路来自光流估计(RAFT等)，跨领域借鉴的典范

## 评分

- 新颖性: ⭐⭐⭐⭐ （各模块有清晰改进但非范式革命）
- 实验充分度: ⭐⭐⭐⭐⭐ （3个benchmark, 多变体, 详尽消融）
- 写作质量: ⭐⭐⭐⭐ （结构清晰, 公式多但逻辑连贯）
- 价值: ⭐⭐⭐⭐ （速度-精度权衡有实际工程价值）
