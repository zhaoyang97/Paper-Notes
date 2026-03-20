# SDF-Net: Structure-Aware Disentangled Feature Learning for Optical-SAR Ship Re-identification

**会议**: CVPR 2026  
**arXiv**: [2603.12588](https://arxiv.org/abs/2603.12588)  
**代码**: [github.com/cfrfree/SDF-Net](https://github.com/cfrfree/SDF-Net)  
**领域**: 遥感 / 跨模态检索  
**关键词**: 光学-SAR跨模态, 船舶重识别, 结构感知, 特征解耦, 梯度能量  

## 一句话总结
提出SDF-Net——物理引导的结构感知解耦特征学习网络，通过中间层梯度能量提取几何结构一致性(SCL)和终端层共享/模态专用特征解耦(DFL)+无参数加法融合，在HOSS-ReID上mAP达60.9%(+3.5% vs SOTA TransOSS)。

## 背景与动机
光学-SAR跨模态船舶重识别面临极大的非线性辐射畸变——两种传感机制（被动反射vs主动微波回散射）导致同一船舶在不同模态中的纹理外观完全不同。现有方法依赖隐式统计对齐或生成合成，但忽略了物理先验：船舶是刚体，其几何结构在不同成像模态下是稳定的。

## 核心问题
如何利用模态不变的几何结构构建鲁棒的跨模态船舶身份关联，同时容忍严重的辐射畸变？

## 方法详解

### 整体框架
ViT-B/16双头tokenizer编码器 → 中间层(Block 6)结构感知一致性学习(SCL) → 终端层解耦特征学习(DFL) → 无参数加法残差融合。

### 关键设计
1. **结构感知一致性学习(SCL)**: 从ViT中间层提取梯度能量——计算空间梯度场$\mathbf{G}_x, \mathbf{G}_y$，整合为结构描述子$\mathbf{f}_{struct}$，经Instance Normalization实现尺度不变，跨模态对齐身份级原型。利用了刚体几何稳定性的物理先验。

2. **解耦特征学习(DFL)**: 终端表示分为共享身份特征$\mathbf{f}_{sh}$和模态专用特征$\mathbf{f}_{sp}$，通过平行线性投影+正交约束$\mathcal{L}_{orth} = \mathbb{E}[|\langle \bar{\mathbf{f}}_{sh}, \bar{\mathbf{f}}_{sp} \rangle|]$确保两者独立。

3. **无参数加法残差融合**: $\mathbf{f}_{fuse} = \mathbf{f}_{sh} + \mathbf{f}_{sp}$，模态专用特征作为残差。零额外参数，仅+0.17G FLOPs。

### 损失函数 / 训练策略
$\mathcal{L} = \mathcal{L}_{id} + 10.0 \cdot \mathcal{L}_{orth} + 1.0 \cdot \mathcal{L}_{struct}$，$\mathcal{L}_{id}$含标签平滑CE+加权三元组损失。SGD, batch 32 (8身份×4图像), 100 epochs。

## 实验关键数据
| 协议 | 指标 | SDF-Net | TransOSS | 提升 |
|------|------|---------|----------|------|
| All-to-All | mAP | **60.9%** | 57.4% | +3.5% |
| All-to-All | Rank-1 | **69.9%** | 65.9% | +4.0% |
| SAR-to-Optical | mAP | **46.6%** | 38.7% | **+7.9%** |
| Optical-to-SAR | mAP | **50.0%** | 48.9% | +1.1% |

### 消融实验要点
- SCL单独：SAR-to-Optical mAP 44.5%→46.6%(+2.1%)——几何锚定有效
- DFL单独：All Rank-1 67.6%→69.9%(+2.3%)——身份解耦有效
- 加法融合(60.9%)优于拼接(59.5%)和仅共享特征(59.2%)

## 亮点
- 物理引导设计——利用刚体几何不变性先验，在SAR-to-Optical最难场景+7.9%
- 零额外参数的加法融合，仅+0.17G FLOPs
- 梯度能量提取结构特征的思路可推广到其他跨模态匹配任务

## 局限性 / 可改进方向
- 仅在HOSS-ReID单一数据集验证
- 假设近垂直观测——极端入射角下3D畸变未处理
- 极低分辨率SAR中结构轮廓被散斑淹没时可能失效

## 与相关工作的对比
- **TransOSS(ICCV25)**: ViT基线，57.4% mAP → 本文60.9%
- **D2InterNet(SIGIR25)**: 单模态船舶ReID，50.2% mAP

## 评分
- 新颖性: ⭐⭐⭐⭐ 物理引导的梯度能量结构特征+解耦设计
- 实验充分度: ⭐⭐⭐⭐ 多协议评估+详细消融
- 写作质量: ⭐⭐⭐⭐ 物理动机清晰
- 价值: ⭐⭐⭐⭐ 对跨模态遥感检索有实用价值
