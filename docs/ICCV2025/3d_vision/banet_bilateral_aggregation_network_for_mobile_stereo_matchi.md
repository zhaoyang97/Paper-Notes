# BANet: Bilateral Aggregation Network for Mobile Stereo Matching

**会议**: ICCV 2025  
**arXiv**: [2503.03259](https://arxiv.org/abs/2503.03259)  
**代码**: [https://github.com/gangweix/BANet](https://github.com/gangweix/BANet) (有)  
**领域**: 3D视觉 / 立体匹配  
**关键词**: 立体匹配, 移动端部署, 双边聚合, 代价体, 2D卷积  

## 一句话总结

提出双边聚合网络BANet，通过将代价体分离为高频细节体和低频平滑体分别聚合再融合，仅使用2D卷积即可在移动设备上实现实时高精度立体匹配（骁龙8 Gen 3上45ms，KITTI 2015 D1-all=1.83%，比MobileStereoNet-2D精度高35.3%）。

## 背景与动机

- 当前SOTA立体匹配方法依赖昂贵的3D卷积聚合完整代价体，计算和内存开销大，难以在移动设备上部署
- 直接用2D卷积替代3D卷积进行代价聚合会导致**边缘模糊、细节丢失、无纹理区域误匹配**
- AANet使用可变形卷积、HITNet使用迭代warping可以部分缓解问题，但这些复杂操作**不适合移动端部署**
- MobileStereoNet-2D虽然用纯2D卷积但精度严重下降
- **核心洞察**：图像同时包含高频细节区域和低频平滑/无纹理区域，用统一的2D聚合网络难以同时处理好两者

## 核心问题

如何设计一个**仅使用移动端友好操作**（纯2D卷积）的立体匹配网络，同时保持高精度（清晰边缘+细节保留+无纹理区域准确匹配）？

## 方法详解

### 整体框架

四步流程：特征提取（MobileNetV2 backbone）→ 相关代价体构建（1/4分辨率）→ **双边聚合**（核心创新）→ 视差预测（softmax回归 + 超像素上采样）

### 关键设计

1. **双边聚合 (Bilateral Aggregation)**:
   - 使用空间注意力图 $\mathbf{A}$ 将完整相关代价体 $\mathbf{C}_{cor}$ 分离为两部分：
     - 细节代价体：$\mathbf{C}_d = \mathbf{A} \odot \mathbf{C}_{cor}$（高频区域）
     - 平滑代价体：$\mathbf{C}_s = (1-\mathbf{A}) \odot \mathbf{C}_{cor}$（低频区域）
   - 分别用独立的聚合分支 $\mathbf{G}_d$ 和 $\mathbf{G}_s$ 处理（结构相同但不共享权重）
   - 最终融合：$\mathbf{C}_{agg} = \mathbf{A} \odot \mathbf{C}'_d + (1-\mathbf{A}) \odot \mathbf{C}'_s$
   - 每个分支由MobileNetV2 inverted residual blocks构成：1/4分辨率4块、1/8分辨率6块、1/16分辨率8块，扩展因子为4

2. **尺度感知空间注意力 (Scale-aware Spatial Attention, SSA)**:
   - 利用多尺度特征（1/4、1/8、1/16）的感知差异：细尺度感知高频细节，粗尺度感知低频平滑信息
   - 将多尺度特征上采样至1/4分辨率后，各自通过卷积层、拼接、再通过卷积+sigmoid生成注意力图
   - 公式：$\mathbf{S} = Concat[Conv(\mathbf{F}^{up}_{l,16}), Conv(\mathbf{F}^{up}_{l,8}), Conv(\mathbf{F}_{l,4})]$，$\mathbf{A} = \sigma(Conv(\mathbf{S}))$

3. **3D版本扩展 (BANet-3D)**:
   - 将双边聚合概念应用于3D卷积聚合网络
   - 3D聚合网络包含3个下采样块（2个3×3×3 3D卷积）和3个上采样块（4×4×4转置3D卷积 + 2个3×3×3 3D卷积）
   - 在高端GPU上实现所有已发表实时方法中的最高精度

### 损失函数 / 训练策略

- **损失函数**：双层Smooth L1 Loss
  $$\mathcal{L} = \lambda_0 \cdot SmoothL1(\mathbf{d}_0 - \mathbf{d}_{gt}) + \lambda_1 \cdot SmoothL1(\mathbf{d}_1 - \mathbf{d}_{gt})$$
  - $\lambda_0=0.3$（1/4分辨率视差），$\lambda_1=1.0$（全分辨率视差）
- **训练策略**：Scene Flow 200k步（batch 16）→ KITTI混合微调 50k步，裁剪256×512
- **优化器**：AdamW，one-cycle学习率，最大LR=8e-4
- $D_{max}=192$

## 实验关键数据

| 数据集 | 指标 | BANet-2D | BANet-3D | MobileStereoNet-2D | Fast-ACVNet+ | HITNet |
|--------|------|----------|----------|-------------------|--------------|--------|
| Scene Flow | EPE (px) | 0.57 | 0.51 | 1.11 | 0.59 | - |
| Scene Flow | Bad 3.0 (%) | 2.49 | 2.21 | - | 2.70 | - |
| KITTI 2015 | D1-all (%) | **1.83** | **1.77** | 2.83 | 2.01 | 1.98 |
| KITTI 2015 | D1-bg (%) | 1.59 | 1.52 | 2.49 | 1.70 | 1.74 |
| KITTI 2015 | D1-fg (%) | 3.03 | 3.02 | 4.53 | 3.53 | 3.20 |
| KITTI 2012 | 3-noc (%) | 1.38 | 1.27 | - | 1.45 | 1.41 |
| — | MACs (G) | **36** | 78 | 127 | 85 | 47 |

**移动端延迟**（骁龙8 Gen 3，512×512输入）：BANet-2D仅需**45ms**（不到MobileStereoNet-2D的1/3）
- 延迟分解：特征提取16ms + 代价体构建6.5ms + 双边聚合22.5ms

### 消融实验要点

- **双边聚合 (BA) 的效果**：
  - 2D基线 EPE 0.63 → +BA 0.59 → +SSA 0.57（Scene Flow）
  - 3D基线 EPE 0.56 → +BA 0.53 → +SSA 0.51
  - KITTI前景区域D1-fg提升：2D聚合提升17%，3D聚合提升22%
- **反射区域性能**：BA使2D聚合在反射区域提升36.5%，3D聚合提升12.0%
- **通用性验证**：BA插入PSMNet（EPE 1.09→0.77）、GwcNet（0.76→0.67）、Fast-ACVNet+（0.59→0.53），均有显著提升，Fast-ACVNet+的EPE提升10.2%

## 亮点

- **分治思想**巧妙：将代价体按高低频分离再各自聚合的思路简单有效，是对2D聚合本质缺陷的深刻理解
- **极致移动友好**：全部使用2D卷积 + MobileNetV2 blocks，Snapdragon 8 Gen 3实测45ms，真正可部署
- **通用性强**：双边聚合可以即插即用地提升PSMNet、GwcNet、Fast-ACVNet+等已有方法
- **MACs最低**：BANet-2D仅36G MACs，远低于所有对比方法（MobileStereoNet-2D 127G，HITNet 47G）
- 消融实验覆盖2D和3D两种聚合网络，KITTI在线提交和Scene Flow测试集都有完整验证

## 局限性 / 可改进方向

- 注意力图的质量完全依赖多尺度特征的表达能力，在极端场景（全无纹理/极弱光照）下可能失效
- 细节/平滑的二分法较为rigid，实际场景中频率是连续分布的，可以探索更多频段的分治策略
- 仅在KITTI和Scene Flow上验证，缺少ETH3D、Middlebury等更多数据集的丧泛化评估
- 论文提到可扩展到多视角立体和光流，但未给出实验验证
- 超像素上采样模块的移动端延迟未单独报告

## 与相关工作的对比

- **vs MobileStereoNet-2D**：同样纯2D卷积但精度高35.3%，延迟不到1/3，核心差异在于双边聚合解决了2D聚合的边缘模糊问题
- **vs AANet**：AANet用可变形卷积做自适应聚合，不够移动端友好；BANet仅用标准2D卷积+注意力分离
- **vs HITNet**：HITNet用迭代warping恢复细节，计算量更大且不适合移动端；BANet-2D精度更高且MACs更低（36 vs 47G）
- **vs Fast-ACVNet+**：BANet-3D在D1-all上超过11.9%，2D版本MACs仅为其42%
- **vs PSMNet/GwcNet系列**：BA模块可直接插入提升这些经典方法的性能

## 启发与关联

- **分治策略的通用性**：将代价体按空间频率分离的思想可以推广到光流估计、多视角立体、甚至语义分割中的边缘处理
- **与自动驾驶代价体研究的关联**：ideas目录中的"代价体稀疏占据预测"(`ideas/autonomous_driving/20260317_cost_volume_sparse_occ.md`)同样关注代价体的高效利用，BANet的高低频分离思路可启发稀疏体素采样中按区域重要性自适应采样
- **移动端轻量化范式**：用MobileNetV2 inverted residual blocks + 任务特定注意力的组合，是移动端密集预测任务的有效设计模式
- **频率感知**思想与bilateral filtering有异曲同工之妙，说明传统图像处理的先验在深度学习时代仍有重要指导意义

## 评分

- 新颖性: ⭐⭐⭐⭐ 分治思路本身不新，但将其具体化为双边聚合+尺度感知注意力用于移动立体匹配是有巧思的
- 实验充分度: ⭐⭐⭐⭐⭐ Scene Flow+KITTI在线测试、消融全面（2D/3D、通用性、反射区域）、移动端实测
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰、动机明确、图表质量高
- 价值: ⭐⭐⭐⭐ 对移动端立体匹配有实际部署价值，双边聚合的即插即用特性增加了通用性
