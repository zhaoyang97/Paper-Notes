# VidEoMT: Your ViT is Secretly Also a Video Segmentation Model

**会议**: CVPR 2026  
**arXiv**: [2602.17807](https://arxiv.org/abs/2602.17807)  
**代码**: [https://www.tue-mps.org/videomt/](https://www.tue-mps.org/videomt/)  
**领域**: 语义分割 / 视频理解 / 高效模型  
**关键词**: 视频分割, encoder-only, 查询传播, ViT, DINOv2, 实时  

## 一句话总结
提出encoder-only视频分割模型VidEoMT，通过查询传播和查询融合将分割与时序关联统一在单个ViT编码器中，消除所有专用追踪模块，在YouTube-VIS 2019上达到160 FPS（比CAVIS快10×+），同时AP仅差0.3。

## 背景与动机
现有在线视频分割方法（CAVIS、DVIS++、DVIS-DAQ）遵循"分割器+追踪器"的解耦范式：分割器由ViT+ViT-Adapter+Mask2Former像素解码器+Transformer解码器组成，追踪器由上下文感知特征+重识别层+Transformer追踪块组成。虽然精度高，但架构极其复杂且慢（CAVIS仅15 FPS）。EoMT论文证明了图像分割可以encoder-only完成（无需解码器/像素解码器）。那么视频分割是否也可以？关键额外挑战是时序追踪。

## 核心问题
能否用一个简单的encoder-only ViT同时完成视频分割和时序关联，实现接近SOTA精度但快一个数量级的速度？

## 方法详解

### 整体框架
VidEoMT基于EoMT：将N个可学习查询注入到DINOv2 ViT的最后L2层与patch token联合处理，查询输出直接预测类别和mask。在此基础上，VidEoMT引入两个轻量机制实现时序建模：(1) 查询传播：将前一帧的输出查询作为当前帧的输入；(2) 查询融合：前帧查询经线性变换后与可学习查询逐元素相加。无任何追踪模块。

### 关键设计
1. **渐进式模块移除实验**：从CAVIS出发逐步验证：替换分割器为EoMT（AP -0.8, 速度3×↑）→移除上下文感知特征（AP +0.3, 速度1.7×↑至72 FPS）→移除重识别层（AP -0.4, 速度↑至74 FPS）→移除追踪器（AP -7.6, 速度↑至162 FPS）。关键发现：上下文感知特征和重识别层在DINOv2预训练下是冗余的——DINOv2的特征已经包含了足够的实例判别信息。

2. **查询传播 (Query Propagation)**：$t=0$时用可学习查询$\mathbf{Q}^{lrn}$初始化，$t>0$时用前帧输出查询$\mathbf{Q}_{t-1}^S$替代，注入ViT最后L2层。这使信息跨帧传递，零额外计算。但纯传播会让可学习查询的影响逐渐消失，导致无法检测新出现的物体。

3. **查询融合 (Query Fusion)**：$\mathbf{Q}_t^F = \text{Linear}(\mathbf{Q}_{t-1}^S) + \mathbf{Q}^{lrn}$，前帧查询经单层线性变换后与可学习查询逐元素相加。这确保模型既有前帧时序上下文，又保持对新物体的检测能力。仅引入一个线性层，开销可忽略。

### 损失函数 / 训练策略
使用Mask2Former标准损失（CE分类 + BCE/Dice分割）。两阶段训练：第一阶段COCO+目标视频数据集做图像分割训练；第二阶段引入时序建模微调。VidEoMT需要微调ViT编码器（因为encoder-only），而CAVIS等方法可以冻结。200个查询，D=1024，H100 GPU训练。

## 实验关键数据
| 方法 | Backbone | YT-VIS 2019 AP | FPS | GFLOPs |
|--------|------|------|------|------|
| CAVIS | ViT-L | 68.9 | 15 | 838 |
| DVIS-DAQ | ViT-L | 68.3 | 10 | 851 |
| DVIS++ | ViT-L | 67.7 | 18 | 846 |
| EoMT+CAVIS | ViT-L | 68.1 | 42 | 699 |
| **VidEoMT** | **ViT-L** | **68.6** | **160** | **566** |

视频语义分割VSPW：VidEoMT mIoU 64.9（比DVIS++ 62.8高+2.1），mVC16 95.0，速度73 FPS（DVIS++ 13 FPS）。
视频全景分割VIPSeg：VidEoMT VPQ 55.2 vs CAVIS 56.9，速度75 vs 10 FPS（7.5×加速）。

### 消融实验要点
- **查询融合是关键**：无传播61.3 AP → 查询传播63.9（+2.6）→ 查询融合68.6（+4.7），速度几乎不变
- **模型大小影响**：ViT-L差距仅0.3 AP （vs CAVIS），ViT-S差距2.7 AP，说明大预训练模型是关键
- **预训练质量决定性**：DINOv2下差距0.3 AP，IN21K下差距1.4 AP，IN1K下差距2.7 AP
- **VidEoMT vs EoMT+tracker**：VidEoMT（68.6 AP, 160 FPS）优于EoMT+CAVIS（68.1 AP, 42 FPS）——统一比解耦更好更快
- **查询融合 vs TrackFormer**：融合（68.6 AP, 160 FPS）优于TrackFormer（67.7 AP, 117 FPS）——更简单更快更准

## 亮点
- 10×+加速是game-changing级别的——160 FPS使实时视频分割成为现实
- "VFM预训练的ViT已经隐式学会了追踪"是一个深刻的洞察——DINO训练目标促进跨视图一致性，这正是追踪所需
- 渐进式模块移除实验非常有说服力——每一步都定量验证了"哪些组件是冗余的"
- 查询融合设计极度简洁（一个线性层+元素加法），体现了"简洁即力量"
- FPS的巨大提升主要不是因为FLOPs减少（只减32%），而是因为纯ViT architecture可以更好利用FlashAttention+torch.compile等硬件优化

## 局限性 / 可改进方向
- 对小ViT（ViT-S/B）性能差距更大，说明方法的有效性严重依赖大模型和强预训练
- OVIS上（严重遮挡场景）与CAVIS差距约1.6 AP，极端遮挡场景可能需要更多追踪能力
- 查询融合只用前一帧信息，长期时序建模能力有限
- 训练需要微调整个ViT编码器（CAVIS可以冻结），内存成本更高
- 仅支持online模式，不适用于需要全局时序推理的offline设置

## 与相关工作的对比
- **vs CAVIS (ICCV 2025)**：CAVIS是当前SOTA但仅15 FPS，VidEoMT 160 FPS（10.7×加速），AP仅差0.3——用极少精度换巨大速度
- **vs MinVIS**：MinVIS同样追求简单高效但用了Swin-L+Mask2Former解码器，VidEoMT彻底encoder-only，更快（160 vs 29 FPS）且更准（68.6 vs 61.6 AP）
- **vs EoMT (CVPR 2025 图像分割)**：VidEoMT是EoMT的视频版本，通过查询传播+融合实现了7.3 AP的提升（68.6 vs 61.3）

## 启发与关联
- "强预训练可以消除下游任务特定组件"的论点在越来越多任务上被验证——从图像分割（EoMT）到视频分割（VidEoMT），下一步可能是3D感知、视频生成等
- 查询传播+融合的时序建模方式可以用于其他需要帧间关联的任务——如视频目标检测、动作检测
- 对自动驾驶等实时应用意义重大：160 FPS在各种应用场景下都足够

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次实现encoder-only视频分割，且10×加速是质的飞跃
- 实验充分度: ⭐⭐⭐⭐⭐ 6个基准（4 VIS + 1 VPS + 1 VSS）、渐进移除消融、多预训练/模型尺寸对比、替代方案对比
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑链极其清晰（假设→验证→设计→实验），从CAVIS到VidEoMT的渐进推演很精彩
- 价值: ⭐⭐⭐⭐⭐ 160 FPS的视频分割对工业应用有巨大价值，证明了VFM时代"less is more"
