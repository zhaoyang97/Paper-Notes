# M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs

**会议**: CVPR 2026  
**arXiv**: [2603.09737](https://arxiv.org/abs/2603.09737)  
**代码**: [https://github.com/qixi7up/M2-Occ](https://github.com/qixi7up/M2-Occ) (有)  
**领域**: 3D视觉 / 自动驾驶 / 语义占据预测  
**关键词**: semantic occupancy prediction, missing camera view, multi-view reconstruction, feature memory, robustness  

## 一句话总结
针对自动驾驶中相机故障导致的不完整输入问题，提出M²-Occ框架，通过多视角掩码重建（MMR）利用相邻相机重叠视场恢复缺失特征，并引入特征记忆模块（FMM）用类级语义原型精化体素表示，在缺失后视摄像头时IoU提升4.93%，不影响全视角性能。

## 背景与动机
3D语义占据预测为自动驾驶提供密集的体素级场景理解，比BEV感知更全面（覆盖任意形状障碍物和细粒度语义）。现有基于多相机的方法（SurroundOcc、TPVFormer等）隐含假设所有6个环视相机都正常工作。但实际部署中，相机故障（镜头遮挡、硬件故障、通信中断）是常见的。初步实验显示，即使SurroundOcc这样的经典模型，丢失单个关键视角就会导致严重性能下降——后视摄像头丢失时IoU从31.45%降至23.94%，产生严重的几何盲区。现有BEV领域有一些鲁棒性工作（M-BEV、MetaBEV、SafeMap），但在3D语义占据预测领域，对传感器失效的鲁棒性研究几乎为空白。

## 核心问题
如何在多相机输入不完整（一个或多个相机完全失效）的情况下，保持3D语义占据预测的几何完整性和语义一致性？这是一个安全关键问题——后视相机失效意味着车辆后方完全"失明"，可能导致严重事故。

## 方法详解
### 整体框架
标准2D→3D范式：多视角图像经共享ResNet-101+FPN提取多尺度2D特征，经空间交叉注意力提升为3D体素表示，最后3D占据头预测体素语义标签。在两个关键位置插入新模块：(1) 特征提取阶段后插入MMR模块恢复缺失视角特征；(2) 体素精化阶段插入FMM模块用语义原型精化。训练时用Random View Masking（RVM）随机丢弃视角模拟故障；测试时mask特定视角评估鲁棒性。

### 关键设计
1. **多视角掩码重建（MMR）**: 利用相邻相机的物理重叠视场来恢复缺失视角特征。三步流程：
   - **视角关系建模**：将6个相机建模为循环图，每个视角$v_i$的邻域为左右相邻相机$\mathcal{N}(v_i) = \{v_{(i-1)\text{mod}N}, v_{(i+1)\text{mod}N}\}$
   - **重叠区域特征聚合**：从左右邻居的特征图中裁剪重叠边界区域（宽度$w_{ov}$对应物理重叠），与可学习mask token拼接：$\mathbf{f}_{ref} = \text{Concat}(\mathbf{f}_{left}[:,-w_{ov}:], \mathbf{e}_{mask}, \mathbf{f}_{right}[:,:w_{ov}])$
   - **Transformer解码重建**：6层Transformer blocks（8头注意力，MLP ratio=4），加可学习位置编码，从粗糙结构先验重建缺失特征 $\hat{\mathbf{f}}_i = \mathcal{D}(\mathbf{f}_{ref} + \mathbf{p}_{pos})$
   - MMR损失仅在被mask的视角上计算MSE：$\mathcal{L}_{MMR} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \|\hat{\mathbf{f}}_i - \mathbf{f}_i^{gt}\|^2$

2. **特征记忆模块（FMM）**: MMR恢复的特征可能模糊或语义模糊，FMM用全局语义原型作为"长期记忆"来精化。两种策略：
   - **Single-Proto策略**：每个语义类维护一个全局质心$\mathbf{m}_k$，用动量移动平均更新：$\mathbf{m}_k^{(t)} = (1-\lambda)\mathbf{m}_k^{(t-1)} + \lambda \cdot \bar{\mathbf{f}}_k$，$\lambda=0.1$。稳定但无法捕捉类内多样性
   - **Multi-Proto策略**：每个类维护$N_p$个子原型$\mathbf{m}_{k,j}$，通过余弦相似度+softmax温度$\tau$计算检索权重$\alpha_{k,j}$。能建模类内差异（如不同车型），但在严重缺失情况下可能因噪声路由导致过度碎片化
   - **记忆增强特征**：用预测的类概率$P(k)$作为门控，聚合加权原型作为残差修正：$\mathbf{x}' = \mathbf{x} + \sum_{k=1}^{K}(P(k)\sum_{j=1}^{N_p}\alpha_{k,j}\mathbf{m}_{k,j})$

### 损失函数 / 训练策略
- 主损失：标准语义占据交叉熵损失 + $\mathcal{L}_{MMR}$
- 基于SurroundOcc官方实现，ResNet-101 + FCOS3D预训练权重
- AdamW优化器，学习率$2 \times 10^{-4}$，weight decay 0.01
- 训练24 epochs，体素大小200×200×16，范围[-50m, 50m]
- 训练时随机丢弃视角（RVM），测试时按特定模式丢弃

## 实验关键数据
| 缺失视角 | 指标 | M²-Occ | SurroundOcc基线 | 提升 |
|--------|------|------|----------|------|
| Front | IoU↑ | 30.40 | 25.03 | **+5.37** |
| Back（安全关键） | IoU↑ | 28.87 | 23.94 | **+4.93** |
| Front Left | IoU↑ | 31.25 | 30.74 | +0.51 |
| Front Right | IoU↑ | 31.17 | 30.56 | +0.61 |
| Back Left | IoU↑ | 31.08 | 30.35 | +0.73 |
| Back Right | IoU↑ | 31.19 | 30.62 | +0.57 |
| 1 View (avg) | IoU↑ | 30.66 | 28.42 | +2.24 |
| 3 Views | IoU↑ | 26.06 | 20.52 | **+5.54** |
| 5 Views | IoU↑ | 18.36 | 13.35 | **+5.01** |

### 消融实验要点
- **MMR单独**：IoU从26.76恢复到28.19（+1.43），主要恢复大尺度空间结构（道路、车辆体积）
- **MMR+FMM(Single-Proto)**：IoU进一步到28.38（+0.19），语义精化有效
- **Multi-Proto vs Single-Proto**：Multi-Proto（IoU 27.76）反而不如Single-Proto（28.38），在缺失视角条件下细粒度原型路由引入噪声
- **计算开销**：额外仅约0.15GB显存（约2.5%），推理时间随缺失视角数线性增加
- **大物体恢复好**：如drive.surf（27.51→35.02，+7.51），但小物体反而下降（pedestrian 12.50→10.51，traffic cone 8.70→5.71）

## 亮点
- **问题定义精准**：首次系统研究3D语义占据预测在相机失效下的鲁棒性，填补了重要空白
- **MMR利用物理重叠很巧妙**：不幻想原始像素，而是在特征空间利用相邻相机的真实重叠区域作为结构先验
- **开销极低**：0.15GB显存增加就获得显著鲁棒性提升，实际部署友好
- **评估协议完善**：覆盖确定性单视角失效和随机多视角丢弃，是好的benchmark设计
- **诚实的分析**：明确指出对小物体的局限性，并给出合理解释

## 局限性 / 可改进方向
- 小物体重建质量下降明显：MMR依赖边界重叠区域，分辨率不足以捕捉远距离小物体的细节
- Multi-Proto策略在缺失条件下不如Single-Proto，说明在不完整观测下原型路由机制需改进
- 仅验证了SurroundOcc作为baseline，是否对其他占据预测方法（如BEVFormer、TPVFormer）通用有待验证
- 未考虑时序信息——相邻帧的时序特征可以补偿当前帧的缺失视角
- MMR推理时间随缺失视角数线性增长，5个视角缺失时推理时间增加约62%
- 未讨论同时多帧连续缺失（非单帧独立缺失）的情况

## 与相关工作的对比
- **vs M-BEV**：M-BEV在BEV空间做整个视角的掩码重建，本文在特征空间做且面向3D占据预测而非BEV检测
- **vs MetaBEV**：MetaBEV处理传感器损坏和模态缺失，但面向BEV联合任务（检测+分割）；M²-Occ专注3D语义占据的体素级重建
- **vs SafeMap/FlexMap**：这些面向BEV地图构建，不涉及密集3D语义占据预测

## 启发与关联
- 与 ideas 中的 [开放词汇3D占据预测](../../ideas/3d_vision/20260316_open_vocab_3d_occupancy.md) 相关——如果结合开放词汇能力+缺失视角鲁棒性，将是更实用的系统
- FMM的语义原型思路可以与LR-SGS的反射率通道结合——用反射率作为额外的类级特征来增强原型
- 时序信息是明显的改进方向——用前几帧的完整观测来补偿当前帧缺失

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次在3D占据预测中系统研究传感器失效鲁棒性，MMR利用物理重叠的设计有针对性
- 实验充分度: ⭐⭐⭐⭐ 覆盖了6个单视角+多视角dropout场景，消融完整，但baseline单一
- 写作质量: ⭐⭐⭐⭐ 问题阐述清晰，实验分析诚实（承认小物体局限），图示直观
- 价值: ⭐⭐⭐⭐ 对自动驾驶安全部署有直接意义，低开销设计有实际应用潜力
