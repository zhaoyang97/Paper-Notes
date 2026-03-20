# VACE: All-in-One Video Creation and Editing

**会议**: ICCV 2025  
**arXiv**: [2503.07598](https://arxiv.org/abs/2503.07598)  
**代码**: [https://ali-vilab.github.io/VACE-Page/](https://ali-vilab.github.io/VACE-Page/)  
**领域**: 视频生成 / 视频编辑 / 统一框架  
**关键词**: video generation, video editing, unified framework, Video Condition Unit, Context Adapter, DiT  

## 一句话总结
提出VACE统一视频生成和编辑框架，通过Video Condition Unit（VCU）将参考图→视频生成、视频→视频编辑、mask视频编辑等多种任务的输入统一为标准接口，配合Context Adapter注入时空条件信息，单一模型在各子任务上达到专用模型水平并支持灵活的任务组合。

## 背景与动机
Diffusion Transformer在高质量图像/视频生成上展现了强大能力。在图像领域，统一生成和编辑的框架已经取得显著进展。但视频领域由于时间和空间动态一致性的内在要求，统一方案更加困难。现有视频模型通常为每种任务（文生视频、图生视频、视频编辑、视频补全等）分别训练独立模型或adapter，资源浪费且难以组合多种能力。

## 核心问题
如何设计一个统一的视频合成框架，让同一个模型同时支持reference-to-video生成、video-to-video编辑、masked video editing等多种任务，且性能不弱于专用模型？

## 方法详解

### 整体框架
VACE基于视频Diffusion Transformer（类似Wan/CogVideoX架构），引入两个核心组件：VCU统一不同任务的条件输入格式，Context Adapter将条件信息注入生成模型。

### 关键设计
1. **Video Condition Unit (VCU)**：将所有视频任务的条件输入组织为统一的时空结构。对于reference-to-video：将参考图扩展到视频长度作为条件；对于video-to-video编辑：将源视频作为条件；对于masked editing：在视频上施加mask区分保留/生成区域。所有条件都被格式化为与目标视频相同维度的tensor（时间×高×宽×通道），使模型只需处理一种统一的输入格式。

2. **Context Adapter**：轻量级模块，将VCU的条件信息注入到DiT的生成过程中。它在时间和空间维度上分别处理条件信号——空间维度捕获每帧的结构/外观信息，时间维度保证跨帧的一致性。通过形式化的时空表示注入，模型可以灵活处理各种不同的视频合成任务。

3. **任务组合灵活性**：因为所有条件都通过VCU标准化，用户可以同时提供多种条件进行组合任务——如reference图+mask实现"在参考风格下的局部编辑"，或depth条件+参考图实现"结构引导的风格化视频生成"。这种组合能力在专用模型方案中极难实现。

### 损失函数 / 训练策略
标准视频扩散loss，训练时随机采样不同任务类型的数据，通过VCU统一处理。

## 实验关键数据
- 在reference-to-video、video-to-video editing、masked video editing等多个子任务上，单一VACE模型性能与各自专用模型**持平**
- 支持多种任务组合应用（reference+mask editing、depth-guided+style等），专用模型无法实现
- 统一框架大幅减少了模型存储和部署成本——一个模型替代多个专用模型

### 消融实验要点
- VCU的统一格式化对多任务性能至关重要——去掉统一格式化导致任务间干扰
- Context Adapter的时空分离设计优于简单拼接——更好地保持时间一致性
- 混合训练策略（随机采样不同任务）比分阶段训练效果更好
- 任务组合在推理时直接可用，无需额外训练

## 亮点
- **真正的All-in-One**：参考图生成、视频编辑、视频补全等多种任务在一个模型中统一
- **VCU的设计哲学**：万物皆tensor——所有条件都标准化为时空tensor，优雅地消除了task-specific adapter的需求
- **任务组合是杀手级功能**：用户可以自由搭配条件实现新的编辑效果，这是专用模型无法做到的
- **与VisualCloze的理念呼应**：VisualCloze用infilling统一图像任务，VACE用VCU统一视频任务——都是"统一切入点"的思路
- 来自阿里巴巴达摩院的工作，与Wan/CogVideoX生态兼容

## 局限性 / 可改进方向
- 统一模型在某些极精细的编辑任务上可能不如深度专用方案
- 视频长度和分辨率受限于DiT backbone的能力
- 组合任务的效果依赖于用户正确指定多种条件，需要一定交互设计
- 未展示与LLM结合做自然语言指令驱动的编辑

## 与相关工作的对比
- **vs. CogVideoX/Wan**：这些是基础视频生成模型，VACE在其基础上统一了生成+编辑
- **vs. VisualCloze**：VisualCloze统一图像任务用visual ICL；VACE统一视频任务用VCU——理念相似但领域不同
- **vs. AnyV2V/InsV2V**：这些是特定的视频编辑方法；VACE包含编辑但不限于编辑
- **vs. SANA-Sprint**：SANA-Sprint加速图像扩散推理；VACE统一视频扩散任务——互补

## 启发与关联
- VCU的条件统一思路可以扩展到3D/4D生成——将点云/mesh条件也格式化为统一tensor
- 与REPA-E结合：如果视频VAE也能端到端训练，VACE的生成质量可能进一步提升
- 任务组合能力使VACE特别适合与LLM agent结合——agent分解用户需求，VACE执行组合任务

## 评分
- 新颖性: ⭐⭐⭐⭐ VCU和Context Adapter设计实用，统一视频任务的框架填补空白
- 实验充分度: ⭐⭐⭐⭐ 多子任务评估+组合应用展示
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，应用场景展示充分
- 价值: ⭐⭐⭐⭐⭐ All-in-One视频合成对产品级应用价值巨大，阿里达摩院出品质量有保障
