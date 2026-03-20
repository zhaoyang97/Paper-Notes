# ChartCoder: Advancing Multimodal Large Language Model for Chart-to-Code Generation

**会议**: ACL 2025  
**arXiv**: [2501.06598](https://arxiv.org/abs/2501.06598)  
**代码**: [https://github.com/thunlp/ChartCoder](https://github.com/thunlp/ChartCoder)  
**领域**: 多模态VLM  
**关键词**: 图表理解, 代码生成, Chart-to-Code, Code LLM, 合成数据  

## 一句话总结
提出首个专用chart-to-code MLLM（ChartCoder），以Code LLM为语言骨干+160K大规模图表-代码数据集+Snippet-of-Thought逐步推理方法，7B模型在三个基准上超越所有开源MLLM，接近GPT-4o水平。

## 研究背景与动机
1. **领域现状**：MLLM在图表理解任务（ChartQA等）上已取得良好表现，但主流方法将图表信息用自然语言描述，不可避免地丢失密集信息（数据值、样式细节等）。
2. **现有痛点**：(a) 将图表解析为代码是无损表示，但现有开源MLLM生成的代码可执行率低、图表细节还原差（如InternVL2-8B经常搞错图表类型和坐标尺寸）；(b) 缺乏大规模、多样化的chart-to-code训练数据——最大的ChartLlama仅7.8K，且仅10种图表类型。
3. **核心矛盾**：通用LLM训练语料中代码占比低，导致MLLM在代码生成任务上与chart理解任务不对齐；而直接的代码生成容易忽略关键细节（颜色、数据值、样式参数）。
4. **本文要解决什么？** (a) 如何提高MLLM生成图表代码的可执行率和还原质量？(b) 如何构建大规模高质量chart-to-code训练数据？(c) 如何让模型关注图表中的关键细节？
5. **切入角度**：用Code LLM（DeepSeek Coder）替代通用LLM做语言骨干来天然增强代码能力；用LLM先生成代码再执行的方式批量构造数据（一对一映射）；用Snippet-of-Thought分步生成来强调关键信息。
6. **核心idea一句话**：Code LLM + 160K合成数据 + 逐步推理 = 首个专用chart-to-code MLLM。

## 方法详解

### 整体框架
输入：图表图像。输出：可执行的Python代码（Matplotlib/Seaborn），执行后还原原始图表。模型架构：SigLIP-384视觉编码器 + 两层MLP连接器 + DeepSeek Coder 6.7B语言骨干。采用Any Resolution策略处理高分辨率图表图像。

### 关键设计

1. **Chart2Code-160K数据集构建**:
   - 做什么：构建首个大规模chart-to-code指令微调数据集
   - 核心思路："先生成代码再执行"策略——让LLM生成代码，执行代码渲染图表，形成(图表, 代码)对。具体流程：LLM生成领域关键词 → 生成模拟数据 → 基于27种图表类型的79个模板代码做in-context示例 → 生成代码 → 执行过滤（去除像素异常、刻度异常的图表）→ 得到160K高质量数据对
   - 设计动机：chart-to-code数据有三个特殊要求：一对一映射（一张图对应一段代码）、多样性体现在图表类型而非指令变化、代码必须语法正确可执行。这些要求使得传统数据增强方法不适用

2. **Snippet-of-Thought (SoT)**:
   - 做什么：将直接chart-to-code转换为逐步生成格式
   - 核心思路：模拟人类推理过程分4步：Step 1 — 图表类型和布局（如plt.bar(), plt.subplot()）；Step 2 — 数据和颜色（如data=[10,15], colors=['#FF0000']）；Step 3 — 关键细节（如hatch='/', loc='upper left'）；Step 4 — 完整最终代码。每步包含文本解释+代码片段。从160K中采样50K，用LLM将完整代码分解为这4步（分解而非直接生成，避免中间代码与最终代码不一致）
   - 设计动机：直接代码模板化问题——模板代码相似，仅颜色、数值等细节不同，模型容易忽略这些关键区分信息。分步强调每类信息可以提升模型对细节的关注

3. **Code LLM作为语言骨干**:
   - 做什么：用DeepSeek Coder 6.7B替代通用LLM作为MLLM的语言骨干
   - 核心思路：Code LLM在代码预训练语料上训练，天然具备更强的代码生成能力和语法理解。两阶段训练：(1) chart-to-text对齐（冻结视觉编码器和LLM，仅训练连接器）；(2) chart-to-code指令微调（全模型联合微调）
   - 设计动机：通用LLM代码占比低，导致chart-to-code生成的可执行率和质量不佳

### 损失函数 / 训练策略
两阶段训练：Stage 1对齐阶段使用UniChart、Chart-to-Text、SciCap等图表描述数据+LLaVA预训练数据+Chart2Code-160K（仅训练连接器）；Stage 2指令微调阶段在Chart2Code-160K+SoT数据+ChartQA PoT等上联合微调所有参数。

## 实验关键数据

### 主实验
三个chart-to-code基准对比（开源模型最佳结果加粗）：

| 模型 | 参数 | ChartMimic Exec.Rate | ChartMimic High-Level | Plot2Code Pass Rate | ChartX GPT-score |
|------|------|---------------------|----------------------|--------------------|--------------------|
| GPT-4o | - | 93.2 | 83.5 | 88.6 | - |
| InternVL2-76B | 76B | 83.2 | 62.2 | 85.6 | 1.74 |
| Qwen2-VL-72B | 72B | 73.3 | 50.9 | 72.0 | 1.69 |
| InternVL2-8B | 8.1B | 61.8 | 38.9 | 77.3 | 1.63 |
| TinyChart | 3B | 42.5 | 25.9 | 43.2 | 1.89 |
| **ChartCoder** | **7B** | **91.4** | **74.0** | **87.9** | **2.09** |

### 消融实验（ChartMimic High-Level Score）
ChartMimic细粒度子分数（满分各不同）：

| 模型 | Chart Types(/20) | Layout(/10) | Text(/20) | Data(/20) | Style(/20) | Clarity(/10) |
|------|-----------------|-------------|-----------|-----------|------------|-------------|
| GPT-4o | 18.96 | 9.59 | 17.16 | 15.68 | 14.66 | 8.84 |
| InternVL2-8B | 7.20 | 6.82 | 8.81 | 5.74 | 5.42 | 6.64 |
| **ChartCoder** | **16.83** | **9.13** | **14.77** | **12.41** | **12.68** | **8.29** |

### 关键发现
- **7B模型接近GPT-4o**：ChartCoder在ChartMimic的Exec.Rate（91.4 vs 93.2）和High-Level（74.0 vs 83.5）上非常接近GPT-4o，远超同规模开源模型
- **Code LLM骨干是关键**：对比同架构用通用LLM，Code LLM骨干使可执行率从约60%提升到91.4%，说明代码预训练对chart-to-code至关重要
- **SoT显著提升细节还原**：分步生成在Chart Types、Data、Style等细节相关维度上提升最大
- **超越专业图表模型**：ChartLlama（13B）和ChartVLM（14.3B）在chart-to-code上远不如ChartCoder（7B），说明专用数据和训练策略比模型大小更重要
- **合成图表质量接近真实**：GPT-4o评估的Chart2Code-160K图表质量（77.32分）与ChartMimic真实图表（78.96分）接近

## 亮点与洞察
- **Code LLM做MLLM骨干的开创性探索**：将Code LLM引入多模态场景，这一思路可以扩展到所有需要代码输出的视觉任务（设计稿转代码、截图转HTML等）
- **Snippet-of-Thought的实用性**：将代码分解为类型→数据→细节→完整代码的4步，既提升了模型表现，也增强了输出的可解释性
- **数据构建的巧妙反转**：从"看图生成代码"的训练目标出发，用"先生代码再执行"的方式构建数据，保证了代码-图表的精确对应

## 局限性 / 可改进方向
- **仅限Matplotlib/Seaborn**：生成的代码限制在Python标准绑图库，不支持D3.js、ECharts等Web可视化库
- **27种图表类型仍有限**：未覆盖一些特殊图表（如桑基图、时间轴、地理地图等）
- **无交互式修改能力**：只能一次性生成完整代码，不支持基于用户反馈的迭代修改
- **模板代码相似性**：尽管有SoT，160K数据中代码结构仍可能存在一定程度的同质化
- **可改进方向**：扩展到更多编程语言和可视化库；加入自执行验证的RL训练

## 相关工作与启发
- **vs CoSyn (上一篇)**: CoSyn也用代码生成合成图表数据，但目标是生成训练数据给VLM做图表QA；ChartCoder的目标是让MLLM直接输出代码还原图表。两者互补——CoSyn的数据可以增强ChartCoder的训练
- **vs GPT-4o**: GPT-4o在chart-to-code上目前最强，但ChartCoder仅7B就接近其水平，成本优势明显
- **vs TinyChart**: TinyChart虽然3B很小，但chart-to-code能力极弱（High-Level仅25.9），说明单纯压缩不能解决代码生成问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个专用chart-to-code MLLM，Code LLM骨干和SoT方法有新意
- 实验充分度: ⭐⭐⭐⭐ 三个基准+细粒度分析+消融+数据质量评估
- 写作质量: ⭐⭐⭐⭐ 问题定义清楚，方法描述系统
- 价值: ⭐⭐⭐⭐ 对图表理解和代码生成交叉领域有实际推动作用
