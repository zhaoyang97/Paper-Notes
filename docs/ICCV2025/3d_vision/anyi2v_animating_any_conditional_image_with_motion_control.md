# AnyI2V: Animating Any Conditional Image with Motion Control

**会议**: ICCV 2025  
**arXiv**: [2507.02857](https://arxiv.org/abs/2507.02857)  
**代码**: [https://henghuiding.com/AnyI2V/](https://henghuiding.com/AnyI2V/) (有项目页面)  
**领域**: 视频生成 / 图像到视频(I2V)  
**关键词**: Training-free, Image-to-Video, Motion Control, Diffusion Model, 多模态条件  

## 一句话总结
提出AnyI2V，一个无需训练的框架，可接受任意模态图像（mesh、点云、深度图、骨架等）作为首帧条件，结合用户定义的轨迹实现运动控制的视频生成，在FID/FVD/ObjMC指标上优于现有training-free方法并与训练方法竞争。

## 背景与动机
- **T2V方法**（如AnimateDiff）依赖文本提示，缺乏对空间布局的精确控制
- **I2V方法**（如DragNUWA、DragAnything）依赖真实RGB图像作为首帧，限制了内容的可编辑性
- **ControlNet系方法**虽然可引入图像条件，但需要大量训练数据（尤其是配对数据），不支持mesh/点云等模态，且换backbone需重新训练
- 现有运动控制方法要么缺乏空间布局控制（T2V-based），要么缺乏运动可控性（ControlNet-based），且大多需要训练

**核心痛点**: 没有一个统一的、免训练的框架能同时支持(1)任意模态的首帧空间条件和(2)用户自定义轨迹的运动控制。

## 核心问题
如何在不训练额外模块的情况下，让视频扩散模型接受任意模态的条件图像并按用户指定轨迹生成运动？关键挑战在于：
1. 不同模态图像的特征如何注入而不引入外观偏差
2. 如何在时域上实现跨帧对齐以控制运动
3. 如何处理不规则形状物体的精确运动控制

## 方法详解
### 整体框架
基于3D U-Net的视频扩散模型（默认使用AnimateDiff），整体pipeline分三步：
1. 对条件图像做DDIM反演提取特征（1000步，在t_α=201处提取）
2. 将去偏后的特征注入到生成过程的首帧（结构保持的特征注入）
3. 通过优化latent实现跨帧对齐和轨迹控制（零样本轨迹控制+语义掩码）

### 关键设计
1. **Structure-Preserved Feature Injection（结构保持的特征注入）**:
   - 通过实验分析发现：residual hidden state提供最好的结构控制但会泄漏外观信息；query提供高级语义和实体感知表示；attention map时域一致性差
   - 对residual hidden采用**Patch-wise AdaIN去偏**：将特征分割为不重叠的patch（p=4），在patch级别做AdaIN操作消除外观信息泄漏，同时保留结构信息
   - 注入去偏后的residual hidden + query作为首帧引导
   - 将首帧的K/V复制到后续帧（K_{2:f}=K_1, V_{2:f}=V_1）确保内容一致性

2. **Zero-Shot Trajectory Control（零样本轨迹控制）**:
   - 通过PCA降维分析发现：query在时域上具有强一致性和实体感知能力，是最适合做跨帧对齐的特征
   - 用户定义bounding box指定目标区域和轨迹
   - 通过PCA提取query的前M=64个主成分，对齐后续帧与首帧的特征
   - 优化目标：$$z_t^* = \arg\min_{z_t} \sum_{i=1}^{n}\sum_{j=2}^{f} \mathcal{L}(F_j[\mathcal{B}_j^i], \text{SG}(F_1[\mathcal{B}_1^i]))$$
   - 其中 $F_j = \text{PCA}(\text{Query}_j, M)$，SG为stop-gradient

3. **Semantic Mask Generation（语义掩码生成）**:
   - 解决bounding box对不规则形状物体控制不精确的问题
   - 在首帧bounding box内选取显著点P，计算与各帧特征的余弦相似度
   - 对相似度图做K-Means二聚类生成自适应语义掩码
   - 最终损失函数为掩码约束的MSE：$$\mathcal{L}_j^i = \|M_1^i \odot M_j^i \odot (F_j[\mathcal{B}_j^i] - \text{SG}(F_1[\mathcal{B}_1^i]))\|_2^2$$
   - 动态掩码允许物体自然形变，比静态掩码更灵活

### 损失函数 / 训练策略
- **完全免训练**：无需额外训练任何模块
- DDIM采样25步，每5步优化一次latent（t' ≥ 20），学习率0.01
- 反演阶段约8秒，生成阶段约35秒（A800 GPU，半精度）
- 优化目标为decoder的up_blocks.1的Query 1.1和up_blocks.2的Query 2.0（多分辨率优化）

## 实验关键数据
| 数据集 | 指标 | AnyI2V | DragAnything | FreeTraj | ObjCtrl-2.5D | Baseline |
|--------|------|--------|-------------|----------|-------------|----------|
| VIPSeg+Web | FID↓ | **104.53** | 95.83 | 128.78 | 111.82 | 141.95 |
| VIPSeg+Web | FVD↓ | **569.89** | 556.09 | 672.87 | 605.96 | 970.26 |
| VIPSeg+Web | ObjMC↓ | **16.39** | 13.60 | 24.00 | 23.12 | 38.26 |

注：DragAnything等为需要训练的方法（FID 95.83更优），但AnyI2V在**所有training-free方法中排名第一**，且与训练方法差距很小。特别是ObjMC（运动轨迹精度）上AnyI2V的16.39远优于其他free方法（最好23.12），接近训练方法DragAnything的13.60。

### 消融实验要点
| 配置 | FID↓ | FVD↓ | ObjMC↓ |
|------|------|------|--------|
| w/o K&V consistency | 108.18 | 587.69 | 16.81 |
| w/o PCA Reduction | 105.95 | 585.04 | 17.14 |
| w/ Static Mask | 105.44 | 598.15 | 16.92 |
| w/o Semantic Mask | 105.78 | 579.88 | 17.62 |
| opt. Residual Hidden（替代Query） | 129.40 | 647.52 | 36.23 |
| **Full（AnyI2V）** | **104.53** | **569.89** | **16.39** |

- 优化residual hidden而非query导致ObjMC暴涨至36.23，验证了query作为对齐目标的关键性
- PCA降维维度M=64为最优，过小或过大都降低性能
- 动态语义掩码优于静态掩码，尤其在FVD（时域一致性）上
- 多分辨率优化（Query 1.1 & 2.0）显著优于单分辨率

## 亮点
- **任意模态输入**: 首次支持mesh、点云等ControlNet无法处理的模态作为条件图像
- **混合模态输入**: 可同时使用深度图（背景）+草图（前景）等组合
- **完全免训练**: 切换backbone无需重新训练，已验证在AnimateDiff、LaVie、VideoCrafter2上的泛化能力
- **Patch-wise AdaIN去偏**: 简单有效地解决了residual hidden的外观泄漏问题
- **PCA-based特征分析**: 通过PCA降维可视化深入理解了不同特征的时域特性，发现query是最佳对齐目标
- 支持LoRA和文本编辑，提供丰富的编辑空间

## 局限性 / 可改进方向
- 大范围运动控制精度不足，轨迹过长时效果下降
- 模糊遮挡场景处理不佳，空间关系可能不清晰
- 首帧控制精度不如ControlNet（因为特征注入仅在早期去噪步骤）
- 生成速度（约43秒/视频）仍有优化空间
- 未来可结合轻量级微调进一步提升适应性

## 与相关工作的对比
- **vs DragAnything/DragNUWA**: 这些方法需要训练且只接受RGB图像，AnyI2V免训练且支持任意模态。DragAnything在量化指标上略优（FID 95.83 vs 104.53），但AnyI2V灵活性远超
- **vs ControlNet-based方法**: ControlNet需要每个模态单独训练，无法处理mesh/点云，且不支持运动控制。AnyI2V统一处理所有模态且自带轨迹控制
- **vs ObjCtrl-2.5D/FreeTraj/TrailBlazer**: 同为training-free方法，AnyI2V在所有指标上全面领先（ObjMC 16.39 vs 23.12），且支持更丰富的条件输入

## 启发与关联
- 与 [利用视频生成先验桥接少样本图像修复](../../ideas/image_generation/20260316_video_prior_restoration.md) 相关：AnyI2V的免训练特征注入思路可能启发无需训练的视频先验提取方案
- 与 [概念瓶颈视频世界模型](../../ideas/video_understanding/20260316_concept_bottleneck_world_model.md) 相关：AnyI2V的PCA语义特征分析方法可能用于世界模型中的状态表示学习
- **潜在研究方向**: 将AnyI2V的免训练条件控制思路扩展到DiT架构（如Open-Sora、CogVideoX），可能需要重新分析DiT中哪些特征适合注入和对齐

## 评分
- 新颖性: ⭐⭐⭐⭐ 免训练统一框架和PCA特征分析有新意，但单个技术（AdaIN去偏、latent优化）并非全新
- 实验充分度: ⭐⭐⭐⭐ 消融实验全面（5个变体+超参分析），但缺少用户研究和更多骨干网络的量化评估
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，特征分析可视化直观，motivation论述充分
- 价值: ⭐⭐⭐⭐ 免训练+任意模态的组合有很强实用价值，但在DiT时代的适用性存疑
