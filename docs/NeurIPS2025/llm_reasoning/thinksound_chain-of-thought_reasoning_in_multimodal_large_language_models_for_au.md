# ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing

**会议**: NeurIPS 2025  
**arXiv**: [2506.21448](https://arxiv.org/abs/2506.21448)  
**代码**: https://ThinkSound-Project.github.io  
**领域**: 音频/语音  
**关键词**: 视频转音频, Chain-of-Thought, 音频生成, 多模态推理, Flow Matching

## 一句话总结
提出三阶段交互式视频转音频框架 ThinkSound，通过 MLLM 生成结构化 CoT 推理来指导统一的音频生成基础模型，在 VGGSound 和 MovieGen Audio 基准上达到 SOTA，同时支持对象级精细化和自然语言指令编辑。

## 研究背景与动机

1. **领域现状**：视频转音频（V2A）已从端到端扩散模型（Diff-Foley、FoleyCrafter）发展到多模态条件生成（MMAudio、MultiFoley），质量大幅提升。但当前方法仍是"黑箱"式一步生成，缺乏对视觉内容的深层推理。
2. **现有痛点**：生成逼真音频需要像专业音效师一样推理——判断猫头鹰是在叫还是扑翅、识别树枝摇晃的声音、同步多个音效事件。现有端到端方法压缩了这些推理步骤，常产生泛化的声音或与细微视觉线索不同步。
3. **核心矛盾**：SonicVisionLM 用 MLLM 做字幕再 T2A 生成，但丢失关键视觉细节；DeepSound-V1 引入 CoT 但把过程碎片化为三个独立模型。两者都未充分利用 MLLM 的推理能力来指导统一的音频生成。
4. **本文要解决什么**：如何将 MLLM 的 CoT 推理能力深度整合到 V2A 管线中，实现逐步、交互式的音频生成与编辑？
5. **切入角度**：模仿专业音效师的工作流——先生成整体音景，再精细化特定物体的声音，最后按指令编辑——每一步都由 CoT 推理指导。
6. **核心 idea 一句话**：用 MLLM 生成音频特定的 CoT 推理链，作为结构化条件信号指导统一的 flow matching 音频基础模型完成三阶段音频生成。

## 方法详解

### 整体框架

ThinkSound 由两个主要模块组成：(1) 基于 VideoLLaMA2 微调的 MLLM，负责分析视频/文本输入并生成结构化 CoT 推理；(2) 基于 MM-DiT 的统一音频基础模型，接受 CoT + 视频 + 文本 + 音频上下文的多模态条件，通过 flow matching 生成高保真音频。整个流程分三个阶段：基础 Foley 生成 → 对象级交互精细化 → 指令驱动编辑。

### 关键设计

1. **AudioCoT 数据集**:
   - 做什么：构建大规模多模态 CoT 标注数据集，桥接视觉内容、文本描述和音频合成
   - 核心思路：三阶段自动化管线——(a) 用 VideoLLaMA2 + Qwen2-Audio 提取视觉/音频信息，GPT-4.1-nano 合成 CoT 链；(b) 用 Grounded SAM2 提取 ROI 区域，生成对象级 CoT；(c) 基于四种操作（extension/inpainting/addition/removal）生成编辑 CoT
   - 设计动机：没有大规模音频 CoT 数据就无法训练 MLLM 生成有意义的推理链；现有数据集缺乏结构化推理标注

2. **CoT 推理 MLLM**:
   - 做什么：微调 VideoLLaMA2 使其能生成音频特定的结构化推理
   - 核心思路：在 AudioCoT 上用标准交叉熵 loss 微调，使模型获得三种能力：(a) 音频理解（声学属性、声音传播、时序因果关系），(b) 结构化分解（将复杂音频场景拆分为可操作的步骤），(c) 多模态指令跟随
   - 设计动机：通用 MLLM 缺乏音频生成所需的专门推理能力

3. **统一音频基础模型（CoT-Guided MM-DiT）**:
   - 做什么：从任意输入模态组合生成高保真音频
   - 核心思路：基于 flow matching 训练，关键设计包括——(a) 双路文本编码：MetaCLIP 编码视觉字幕提供场景级上下文，T5-v1-xl 编码 CoT 推理捕捉细粒度时序因果关系；(b) 混合 Transformer：multi-stream blocks（每模态独立参数，共享 attention）+ single-stream blocks；(c) 自适应融合模块：上采样视频特征并通过门控机制与音频 latent 融合；(d) classifier-free guidance dropout（每模态随机 drop，概率 0.2）支持任意输入组合
   - 设计动机：统一架构让三个阶段共享同一个音频生成模型，CoT 作为结构化条件比原始字幕提供更精确的生成指导

4. **Click-Based 交互接口（Stage 2）**:
   - 做什么：用户点击视频中的特定物体来触发对象级音频精细化
   - 核心思路：用 Grounded SAM2 从点击位置生成 ROI，跨帧跟踪，MLLM 针对 ROI 生成专门 CoT，基础模型以已有音频为上下文条件，合成并融入对象特定声音
   - 设计动机：让非技术用户也能进行精细的音频控制

### 训练策略
- VAE：在 Stability AI VAE 基础上训练 50 万步（24×A800），再冻结 encoder 训练 decoder 50 万步
- 基础模型：10 万步（8×A100），batch 256，lr $10^{-4}$
- 任务微调：5 万步（8×A100），分别针对三个阶段

## 实验关键数据

### 主实验（VGGSound V2A 生成）

| 方法 | FD↓ | KL_PaSST↓ | DeSync↓ | CLAP_CoT↑ | MOS-Q↑ | MOS-A↑ |
|------|-----|-----------|---------|-----------|--------|--------|
| MMAudio | 43.26 | 1.65 | 0.44 | 0.40 | 3.84 | 3.97 |
| **ThinkSound** | **34.56** | **1.52** | 0.46 | **0.46** | **4.02** | **4.18** |
| w/o CoT | 39.84 | 1.59 | 0.48 | 0.41 | 3.91 | 4.04 |

### OOD 评估（MovieGen Audio Bench）

| 方法 | CLAP_CoT↑ | DeSync↓ | MOS-Q↑ | MOS-A↑ |
|------|-----------|---------|--------|--------|
| MMAudio | 0.45 | 0.77 | 3.95 | 3.62 |
| MovieGen | 0.47 | 1.00 | 3.98 | 3.70 |
| **ThinkSound** | **0.51** | **0.76** | **4.11** | **3.87** |

### 关键发现
- **CoT 推理的贡献显著**：去掉 CoT 后 FD 从 34.56 升到 39.84（+15%），CLAP_CoT 从 0.46 降到 0.41，确认 CoT 提供了音效事件、时序关系和声学特性的关键信息
- **OOD 泛化强**：在从未见过的 MovieGen 数据上仍然 SOTA，说明 CoT 推理带来了更好的泛化
- **对象级和编辑任务**：ThinkSound 在对象级生成（FD 43.27 vs MMAudio 44.46）和音频编辑（FD 34.78 vs AudioLDM-2 61.28）上都大幅领先
- **推理效率**：生成时间仅 1.07s，快于 MMAudio (3.01s) 和 FoleyCrafter (3.84s)

## 亮点与洞察
- **三阶段交互工作流设计合理**：完美模拟了专业音效师的工作流程（先整体→再局部→再修改），每阶段都有 MLLM 的 CoT 推理桥接用户意图和音频合成
- **AudioCoT 数据集价值大**：自动化 CoT 标注管线可扩展到更多数据源，解决了音频 CoT 训练数据缺失的问题
- **双路文本编码（MetaCLIP + T5）**：场景级全局上下文 + 细粒度 CoT 推理的互补设计，比单一编码器效果好很多

## 局限性 / 可改进方向
- **依赖额外 MLLM 推理**：每次生成需要先跑 MLLM 产生 CoT，增加了系统复杂度（虽然论文说生成时间更短，可能是因为音频生成本身更快）
- **CoT 质量依赖 GPT-4.1-nano**：数据集构建管线依赖闭源模型，CoT 的错误会传播到下游
- **人工评估规模未知**：MOS 分数的评估者数量和详细设置在主文中不够清晰
- **缺乏对话式交互**：三阶段是预定义的线性流程，不支持用户反馈式的迭代调整

## 相关工作与启发
- **vs MMAudio**: MMAudio 也用 flow matching + 多模态条件，但缺乏 CoT 推理；ThinkSound 通过 CoT 分解复杂场景为可管理的声音组件，FD 改善 20%
- **vs SonicVisionLM**: SonicVisionLM 将视频→文字→音频两段式处理，中间丢失视觉细节；ThinkSound 保持视频直接参与条件生成
- **vs DeepSound-V1**: 也用 CoT 但碎片化为三个独立模型；ThinkSound 用统一基础模型覆盖所有阶段

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 CoT 推理引入 V2A 生成的三阶段交互框架设计新颖，AudioCoT 数据集有独立贡献
- 实验充分度: ⭐⭐⭐⭐ 多基准（VGGSound + MovieGen）+ 三个子任务 + 消融实验 + 主客观评估，较为全面
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，图表丰富，方法描述详细
- 价值: ⭐⭐⭐⭐ 展示了 CoT 推理在生成任务（而非纯理解/推理任务）中的价值，开拓了新的应用方向
