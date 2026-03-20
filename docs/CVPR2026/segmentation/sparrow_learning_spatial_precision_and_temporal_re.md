# SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs

**会议**: CVPR 2026  
**arXiv**: [2603.12382](https://arxiv.org/abs/2603.12382)  
**代码**: [github.com/RISys-Lab/SPARROW](https://github.com/RISys-Lab/SPARROW)  
**领域**: 视频理解 / 像素级视频MLLM  
**关键词**: 视频MLLM, 像素级定位, 时序一致性, 目标跟踪特征, 双提示定位  

## 一句话总结
SPARROW通过目标特定跟踪特征(TSF)和双提示[BOX]+[SEG]定位机制增强视频MLLM的时空一致性，在MeViS上J&F +8.9、VidSTG上mIoU +5.49，可即插即用到三种backbone上。

## 背景与动机
像素级视频MLLM用静态[SEG] token做逐帧定位，但缺乏时序上下文，导致空间漂移、身份切换和不稳定初始化。从图像级定位扩展到时序一致的视频理解仍是关键挑战。

## 核心问题
静态分割token提供语义但无时序/几何先验，导致跨帧指称不一致和误差传播。

## 方法详解

### 整体框架
基于UniPixel/GLUS/VideoGLaMM三种视频MLLM，加两个即插即用模块+30,646视频的策划数据集。

### 关键设计
1. **TSF(目标特定跟踪特征)**: GroundingDINO检测→CLDTracker传播→K-means选4个多样外观→裁剪特征投影为TSF token。训练时提供时序身份监督，推理时可选。

2. **双提示定位**: [BOX] token条件化Deformable-DETR做粗空间先验；[SEG] token通过SAM2解码器精细化mask。粗到细设计稳定首帧并减少漂移。

3. **两阶段训练**: Stage 1注入TSF到adapters+LoRA；Stage 2训练筛选头做box prompt学习。仅+0.017B参数。

### 损失函数 / 训练策略
策划30,646视频、45,231 Q&A对的训练数据集。

## 实验关键数据
| 基准 | 指标 | 最佳结果 | 增益 |
|------|------|---------|------|
| MeViS val_u | J&F | 57.4 | **+8.9** vs 基线48.5 |
| Ref-DAVIS17 | J&F | 76.8 | **+7.3** vs 基线69.5 |
| VidSTG | I-mIoU | 46.74 | **+5.49** |
| VideoGCG | CLAIR | 33.6 | **+5.4** |

### 消融实验要点
- 仅训练TSF(无测试跟踪): +2.9 J&F；+[BOX]达+7.3——双提示是关键
- 双提示[BOX]+[SEG](72.5) vs 仅[SEG](69.5) vs 仅[BOX](68.2)
- 空间TSF注入(76.8)优于时序(72.5)和双路径(74.9)

## 亮点
- 即插即用跨3种backbone一致有效(+1.8~+8.9 J&F)
- TSF训练时使用、推理时可选的灵活设计
- 粗到细双提示对首帧稳定性和漂移抑制效果显著

## 局限性 / 可改进方向
- 需要GroundingDINO+CLDTracker的前处理pipeline
- 仅测试了3种backbone
- 数据集策划依赖外部模型质量

## 与相关工作的对比
- **UniPixel**: 基线mIoU 41.25 → +SPARROW 46.74
- **VideoGLaMM**: 基线J&F 48.5 → +SPARROW 57.4

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ TSF+双提示的组合对视频MLLM时序一致性贡献显著
- 实验充分度: ⭐⭐⭐⭐⭐ 5个基准、3个backbone、详细消融
- 写作质量: ⭐⭐⭐⭐ 清晰系统
- 价值: ⭐⭐⭐⭐⭐ 即插即用设计有广泛适用价值
