# SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs

**会议**: CVPR 2025  
**arXiv**: [2603.12382](https://arxiv.org/abs/2603.12382)  
**代码**: 待确认  
**领域**: 分割  
**关键词**: 视频分割, 视频MLLM, 时序一致性, 像素级grounding, 双提示机制

## 一句话总结
提出SPARROW框架，通过目标特定跟踪特征(TSF)和双提示(BOX+SEG)机制，解决视频MLLM中时序引用一致性差和首帧初始化不稳定的问题，在6个基准上对3个主流视频MLLM均取得一致提升。

## 研究背景与动机
1. **领域现状**：多模态大语言模型(MLLM)在图像级别的像素grounding已经取得很好的效果，但将其扩展到视频领域面临运动动态、遮挡和时序一致性的挑战。
2. **现有痛点**：现有视频MLLM（如VideoGLaMM、UniPixel、GLUS）依赖静态的[SEG]文本token进行逐帧grounding，只传达"看什么"的语义信息，无法捕捉目标在时间维度上的位置和外观变化。这导致空间漂移（目标分割随时间偏移）、身份切换（同一目标在不同帧被错误关联）和引用不一致（同一语言描述在不同帧指向不同区域）。
3. **核心矛盾**：文本提示是静态的而视频是动态的，模型必须完全依赖视觉线索推断运动和外观变化，缺乏显式时序引用机制。同时首帧初始化不稳定（[SEG]只提供语义线索无空间先验），错误沿时间传播累积。
4. **本文要解决什么？** (i) 时序引用一致性——如何在帧间保持目标身份不漂移；(ii) 首帧grounding鲁棒性——如何给出准确的初始定位避免误差传播。
5. **切入角度**：从跟踪的角度注入时序对齐的目标特征做训练监督，同时用粗到精的双提示稳定初始化。
6. **核心idea一句话**：用离线跟踪生成的目标特征做训练时时序监督(TSF)，配合BOX+SEG双提示做推理时粗到精grounding，实现即插即用的视频MLLM增强。

## 方法详解

### 整体框架
输入为视频 $\mathbf{V} \in \mathbb{R}^{T_v \times H \times W \times C}$ 和文本查询 $Q$。视频经双分支编码器（空间编码器 $\mathcal{F}_g$ + 时序编码器 $\mathcal{F}_h$）提取特征，通过V→L适配器投射到LLM嵌入空间。LLM（LoRA微调）输出[BOX]和[SEG]两个grounding token，分别通过L→V适配器投射回视觉空间，驱动class-agnostic proposer和SAM2解码器生成最终分割mask。全流程即插即用，不修改基础LLM和视觉骨干。

### 关键设计

1. **目标特定跟踪特征 (TSF)**:
   - 做什么：在训练时注入时序对齐的目标级特征，教会模型跨帧保持目标身份
   - 核心思路：给定文本查询，用GroundingDINO在某帧检测目标，CLDTracker跨帧传播得到轨迹框。为减少冗余，在联合视觉-空间特征空间做K-means聚类（$K=4$），取最近质心样本作为代表子集。对区域编码后投射为TSF token拼接到LLM输入中
   - 设计动机：静态[SEG] token无法编码运动信息。TSF提供多样化目标外观表示（不同帧、不同姿态），使模型在训练阶段学会身份持久性。推理时默认不使用TSF即无需外部检测器/跟踪器
   - 配套数据集：整合7个公开数据集共30,646视频序列、45,231个QA对

2. **双提示Grounding (Dual-Prompt)**:
   - 做什么：结合[BOX]空间先验和[SEG]语义grounding，实现粗到精的分割
   - 核心思路：LLM输出[BOX] embedding驱动class-agnostic proposer生成K=300个候选框，通过cross-attention融合语言和视觉特征后打分筛选并回归精化。筛选出的高置信框与[SEG] embedding一起送入SAM2解码器生成精细mask
   - 设计动机：单纯用[SEG]初始化首帧不稳定漂移后难以恢复，[BOX]提供粗粒度几何约束，[SEG]做语义精化。自然支持多实例查询

3. **Class-Agnostic Proposer**:
   - 做什么：在冻结的Hiera特征上生成category-free候选框
   - 核心思路：多尺度Hiera特征构建FPN，送入Deformable-DETR解码器，分类分支替换为单一objectness头。在COCO、Objects365、OpenImages、V3Det上预训练
   - 设计动机：与外部检测器解耦，轻量且不需要类别监督

### 损失函数 / 训练策略
两阶段训练：
- **Stage 1 (TSF注入)**：训练V→L适配器、L→V SEG适配器和LoRA。损失 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{DICE}}$
- **Stage 2 (BOX学习)**：冻结Stage 1，先预训练proposer，再微调Filtration Head和L→V BOX适配器。$\lambda_{\text{cls}}=1.0$，$\lambda_{\text{box}}=2.0$

## 实验关键数据

### 主实验

| 基准 | 指标 | 基线(VideoGLaMM) | +SPARROW | 提升 |
|------|------|-----------------|----------|------|
| MeViS val | J&F | 45.2 | 47.5 | +2.3 |
| MeViS val^u | J&F | 48.5 | 57.4 | **+8.9** |
| Ref-DAVIS17 | J&F | 69.5 | 76.8 | **+7.3** |
| Ref-YTVOS | J&F | 66.8 | 68.9 | +2.1 |
| VidSTG | mIoU | 39.66 | 45.06 | +5.4 |
| VideoGCG | mIoU | 62.34 | 65.59 | +3.25 |

对更强基线也有一致提升：UniPixel上Ref-DAVIS17 +2.2；GLUS上Ref-DAVIS17 +2.6。

### 消融实验（Ref-DAVIS17, VideoGLaMM基线）

| 配置 | J&F | 说明 |
|------|-----|------|
| Baseline | 69.5 | 无TSF无BOX |
| + TSF(train only) | 72.4 (+2.9) | TSF训练监督有效 |
| + BOX only | 72.5 (+3.0) | 双提示有效 |
| + TSF(train) + BOX | **76.8 (+7.3)** | 默认配置，两者互补 |
| + TSF(train+infer) + BOX | 77.7 (+8.2) | 推理时也用TSF最强 |
| [SEG] only推理 | 69.5 | 单提示弱 |
| [BOX]+[SEG]推理 | 72.5 (+3.0) | 双提示协同增益显著 |

### 关键发现
- TSF即使推理时不使用也能带来+2.9提升，说明模型通过训练已内化了时序一致性能力
- BOX+SEG双提示比任一单提示高3个点以上
- 对弱基线(VideoGLaMM)提升最大（MeViS val^u上+8.9），对强基线也有稳定提升
- Filtration Head用[BOX]特征监督比[SEG]特征好1.9个点

## 亮点与洞察
- **即插即用设计**：TSF和Dual-Prompt作为轻量模块可无缝集成到任意视频MLLM，不修改骨干网络，在3个不同架构上都验证了效果
- **训练-推理解耦**：TSF在训练时用伪跟踪信号监督，推理时可以去掉（默认配置），不增加推理开销但保留性能提升
- **粗到精的双提示**：[BOX]→[SEG]的级联设计同时解决了"定位不准"和"边界不清"两个问题，自然支持多实例查询

## 局限性 / 可改进方向
- 依赖proposal recall：小物体、严重遮挡或unseen物体可能被proposer漏掉，后续BOX/SEG无法恢复
- TSF依赖GroundingDINO和CLDTracker的伪跟踪标注，跟踪噪声或ID切换可能引入偏差
- 长序列中早期BOX错误仍可能累积，缺少显式纠错机制

## 相关工作与启发
- **vs VideoGLaMM**: 只用[SEG]做逐帧grounding无时序线索，SPARROW在其上+7.3 J&F
- **vs UniPixel**: 有在线memory但从首帧mask初始化，SPARROW的BOX提供更好的初始化
- **vs GLUS**: 用全局上下文+密集查询帧，但仍是per-frame语义，SPARROW加入显式跟踪特征
- 与 `ideas/segmentation/20260317_video_seg_multi_distill.md` 有关联

## 评分
- 新颖性: ⭐⭐⭐⭐ TSF+双提示的组合在视频MLLM中是新颖的，但各组件单独看不完全新
- 实验充分度: ⭐⭐⭐⭐⭐ 3个基线×6个数据集×详细消融，非常充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ 即插即用设计实用性强，但提升幅度在强基线上有限
