# VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning

**会议**: ICCV 2025  
**arXiv**: [2504.07960](https://arxiv.org/abs/2504.07960)  
**代码**: [https://visualcloze.github.io/](https://visualcloze.github.io/)  
**领域**: 图像生成 / 统一框架 / In-Context Learning  
**关键词**: universal image generation, visual in-context learning, image infilling, Graph200K, 多任务统一  

## 一句话总结
提出VisualCloze，将多种图像生成任务（编辑、翻译、超分、风格化等）统一为"视觉完形填空"范式——用视觉示例（而非文本指令）定义任务，通过图像infilling模型实现统一生成，并构建Graph200K数据集增强任务间知识迁移，支持域内任务、未见任务泛化、多任务组合和反向生成。

## 背景与动机
扩散模型在各种图像生成任务上取得了巨大成功，但目前主流做法仍是为每种任务训练专用模型（如ControlNet做边缘引导、IP-Adapter做风格迁移、InstructPix2Pix做编辑）——效率低下且难以扩展到新任务。已有的统一模型尝试用文本指令描述任务（如"将这张图转为深度图"），但面临三个问题：(1) 文本描述任务存在歧义——"style transfer"可能有多种理解；(2) 视觉任务分布稀疏——不同任务类型差异大，难以学到跨任务的可迁移知识；(3) 架构设计缺乏统一性——不同条件类型需要不同的adaptation模块。

## 核心问题
如何设计一个真正统一的图像生成框架，能通过少量视觉示例理解任意任务定义，并在同一架构下执行生成？

## 方法详解

### 整体框架
VisualCloze将所有图像生成任务重新定义为"视觉完形填空"（visual cloze）问题——给定一组输入-输出示例对和一个新的输入图像，模型需要"填空"生成对应的输出。形式上，就是给模型看几个(input, output)对作为context，然后给一个新的input让它预测output。这与图像infilling（补全被mask区域）在数学上等价——把所有图像拼在一个网格中，要生成的output位置作为待补全区域。

### 关键设计
1. **Visual In-Context Learning替代文本指令**：不用文本描述任务类型（容易歧义且泛化差），而是直接展示1-2个(input, output)示例对让模型"看懂"要做什么。这类似于LLM的few-shot prompting，但在视觉域。模型通过观察示例中的转换关系来理解任务定义——无论是边缘→图像、图像→深度图、还是任何自定义变换。

2. **Graph200K数据集**：为解决视觉任务分布稀疏的问题，构建了一个图结构数据集——每张图像作为节点，不同类型的变换（深度、边缘、颜色、风格等）作为边，形成密集的任务关系网络。包含200K+图像和多种变换关系，大幅增加了任务密度和跨任务迁移的机会。

3. **基于Infilling的统一架构**：核心洞察——将所有示例对和待生成图拼成一个大网格，将目标位置mask掉，然后用预训练的图像infilling模型去补全。这使得VisualCloze可以直接利用FLUX等强大infilling模型的生成先验，无需设计任何task-specific适配模块。同一个模型、同一个推理流程就能处理所有任务。

### 损失函数 / 训练策略
标准的扩散/flow matching loss在拼接后的图像网格上训练，仅在target位置计算loss。

## 实验关键数据
- **域内任务**：在多种标准任务（depth-to-image、edge-to-image、colorization等）上与专用模型性能相当
- **未见任务泛化**：在训练时未见过的任务类型上，通过提供1-2个示例即可泛化生成
- **多任务组合**：可以串联多种变换（如深度引导+风格化），无需额外训练
- **反向生成**：给定output→input的示例可以反向执行任务（如从图像→边缘图）
- 相比text-instructed统一模型（如InstructPix2Pix variants），VisualCloze在未见任务上泛化能力显著更强

### 消融实验要点
- Visual ICL vs Text instruction：视觉示例提供的任务定义更精准，泛化性更好
- Graph200K vs 稀疏任务数据：密集图结构显著提升跨任务迁移
- Infilling formulation vs 独立条件注入：infilling统一性更好且可利用预训练先验
- 示例数量：1个示例即可识别任务，2-3个进一步提升

## 亮点
- **视觉完形填空是一个优雅的统一范式**：把所有conditioned generation重新定义为infilling，概念简洁且功能强大
- **Visual ICL比text instruction更适合定义视觉任务**：这是一个重要的实用发现——用户展示"我想要什么样的变换"比用文字描述更直观
- **Graph200K的图结构数据设计**巧妙地解决了任务稀疏问题
- **零shot泛化到未见任务**：通过ICL，模型可以执行训练时从未见过的变换类型——这是genuine zero-shot能力
- **利用infilling先验无需改架构**：直接在FLUX等模型上微调，工程成本低

## 局限性 / 可改进方向
- 生成质量受限于infilling模型的基础能力
- 拼接大网格导致分辨率受限（需要在有限尺寸中放入多张示例+目标图）
- 某些精细控制任务（如精确的几何变换）可能不如专用模型
- Graph200K的构建依赖自动化pipeline，数据质量可能不均

## 与相关工作的对比
- **vs. OmniGen/UniDiffuser**：这些用text instruction定义任务，在未见任务上泛化差；VisualCloze用visual ICL泛化能力显著更强
- **vs. Stable Diffusion models + ControlNet/IP-Adapter**：这些是task-specific adapter；VisualCloze是一个统一模型处理所有任务
- **vs. Prompt Diffusion/InstructAny2Pix**：类似的统一意图但VisualCloze的infilling formulation更优雅

## 启发与关联
- **idea潜力非常大**：Visual ICL范式可以扩展到视频生成——展示几帧视频的变换示例就能定义视频编辑任务
- Graph结构数据集的思路可以迁移到多模态学习——不同模态间的转换关系也可以构建成图
- 与EVEv2的Divide-and-Conquer思路互补——VisualCloze统一任务formulation，EVEv2统一模态formulation

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "视觉完形填空"=infilling的统一范式是真正的创新，Visual ICL for task definition是重要方向
- 实验充分度: ⭐⭐⭐⭐ 域内/未见任务/组合/反向四种场景验证，但量化对比可以更详细
- 写作质量: ⭐⭐⭐⭐ 概念清晰，框架优雅
- 价值: ⭐⭐⭐⭐⭐ 定义了universal image generation的新范式，Graph200K数据集有长期价值
