# CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation

**会议**: CVPR 2026  
**arXiv**: [2603.12008](https://arxiv.org/abs/2603.12008)  
**代码**: [GitHub](https://github.com/VisionXLab/CrossEarth-SAR)  
**领域**: 遥感 / SAR 基础模型 / 域泛化语义分割  
**关键词**: SAR, 基础模型, 物理引导MoE, 域泛化, 语义分割  

## 一句话总结
提出首个十亿参数级 SAR 视觉基础模型 CrossEarth-SAR，在 DINOv2 基础上引入物理引导的稀疏 MoE 架构（用方向熵、等效视数、局部粗糙度三个 SAR 物理描述符引导路由），配套 200K 级预训练数据集和 22 个子基准，在 20/22 个跨域分割任务上达到 SOTA。

## 背景与动机
SAR 成像全天候全天时的优势使其对地球观测至关重要，但 SAR 面临三重域偏移挑战：(1) 散斑噪声破坏纹理特征；(2) 侧视几何引起严重空间扭曲（叠掩/透视缩短/阴影）；(3) 后向散射特性由表面粗糙度和介电常数决定，导致同类异貌、异类同貌。更严重的是传感器碎片化——不同平台（Sentinel-1, ALOS-2, Capella）、波段（C/L/X）、极化模式（HH/HV/VV/VH）、入射角产生了极端域特异性。现有 SAR 基础模型要么聚焦于目标检测，要么不是为跨域泛化设计。

## 核心问题
如何构建一个具备足够容量吸收 SAR 极端域多样性、同时保持可控推理成本的跨域泛化 SAR 语义分割基础模型？

## 方法详解
### 整体框架
在 DINOv2 ViT backbone 上将每个 block 的 FFN 替换为物理引导的稀疏 MoE，进行 200K 数据上的持续预训练（CPT），然后配合 Mask2Former 解码器在下游冻结 backbone 微调。提供 S/B/L 三个版本。

### 关键设计
1. **SAR 物理描述符**: 对每张输入图像计算三个物理量：(a) 方向熵 $H_{DE}$ — 梯度方向直方图的熵，刻画成像几何特征；(b) 等效视数 ENL = $(\mu/\sigma)^2$ — 刻画散斑强度/雷达系统特性；(c) 局部粗糙度 $R_{LR}$ — 块均值方差，刻画目标散射/纹理变异。三者拼接为 $s \in \mathbb{R}^3$
2. **物理引导路由**: 将 $s$ 沿 token 维度 tile 为 $S \in \mathbb{R}^{B \times N \times 3}$，与 token 嵌入 $Z$ 拼接后送入路由器计算各专家的 softmax 得分，选 top-k 专家激活。每个专家从 DINOv2 FFN 权重初始化
3. **负载均衡**: 引入 $\mathcal{L}_{BC} = \lambda_{BC} \cdot n \cdot \sum_k f_k p_k$（$\lambda_{BC}=0.005$）防止专家坍缩，总目标 $\mathcal{L} = \mathcal{L}_{seg} + \mathcal{L}_{BC}$
4. **CrossEarth-SAR-200K 数据集**: 整合 40K 有标注 + 160K 伪标注（用 CrossEarth 光学模型在配对光学图像上预测后迁移给 SAR），覆盖 109 个地区、6 大洲，达 203,240 张图像

### 损失函数 / 训练策略
CPT 阶段 18 epochs, batch 4, AdamW lr 3e-5, 16 × A100 (80GB)。下游 40k iterations, batch 2, lr 1e-4, backbone 冻结。Earth-Adapter (PEFT) 进一步提升。

## 实验关键数据
| 设置 | 指标(mIoU) | CrossEarth-SAR-L | DINOv2 基线 | 提升 |
|---|---|---|---|---|
| 单域差异 Avg (12项) | mIoU | 62.7 | 55.5 | +7.2 |
| HH2F (极化) | mIoU | 72.3 | 56.8 | +15.5 |
| VV2F (极化) | mIoU | 73.8 | 65.7 | +8.1 |
| 双域差异 Avg (10项) | mIoU | 28.5 | 24.3 | +4.2 |
| 三域差异 Avg (4项) | mIoU | — | — | +3.4~7.2 |

在 22 个基准中 20 个达到 SOTA。CrossEarth-SAR-L* (with PEFT) 在单域差异上平均提升 +7.2 mIoU。

### 消融实验要点
- 仅 40K 真标注 vs 200K 含伪标注：后者平均高 14.3% mIoU，证明伪标注大规模数据有效
- 纯 MoE（无物理引导无负载均衡）已提升 +1.7，加负载均衡 +2.8，加物理描述符 +2.2，二者结合 +3.0
- 6 专家 top-1 最优；增加 top-k 到 2/3 反而下降，200K 数据规模下单专家专化更优
- 三个物理描述符对不同域差异敏感性不同：$H_{DE}$ 对极化和波段、ENL 对复数值、$R_{LR}$ 对区域和平台

## 亮点
- 将 SAR 物理先验（散斑/几何/散射）编码为可微分描述符引导 MoE 路由，物理与学习的优雅结合
- 22 个子基准覆盖 8 种域差异组合，为 SAR 社区建立了首个统一 DG 评测标准
- 可视化显示不同专家自动分工：早层专注散斑统计，中层建模几何纹理，深层做高级语义

## 局限性 / 可改进方向
- 1.3B 参数量大，部署到资源受限的遥感平台存在挑战
- 伪标注质量受 CrossEarth 光学模型限制，mean agreement 仅 75.88%
- 可进一步探索光学-SAR 跨模态联合预训练

## 与相关工作的对比
- vs SARATR-X (90M HiViT)：CrossEarth-SAR-L 在单域差异上高 3.0 mIoU，三域差异高 3.4+
- vs DINOv3 (300M)：CrossEarth-SAR-L 单域差异 +9.7 mIoU
- vs CrossEarth (光学)：本文专为 SAR 设计，填补了 SAR VFM 在 DG 语义分割上的空白
- vs SatMAE/ScaleMAE/MTP：均在 SAR 域上表现显著更差

## 启发与关联
- 物理引导路由机制可推广到其他传感器特异性模态（如红外/多光谱）
- 稀疏 MoE 在域多样性极大的数据上优于密集缩放的经验值得关注

## 评分
- 新颖性: ⭐⭐⭐⭐ 物理描述符引导 MoE 路由创新且有理论依据
- 实验充分度: ⭐⭐⭐⭐⭐ 22 个基准、16 种对比方法、详尽消融和可视化
- 写作质量: ⭐⭐⭐⭐ 结构完整，物理先验解释清晰
- 价值: ⭐⭐⭐ 对遥感/SAR 社区价值高，对通用视觉社区价值中等
