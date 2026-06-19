---
title: >-
  [论文解读] The AI Gap: How Socioeconomic Status Affects Language Technology Interactions
description: >-
  [ACL 2025][LLM 其他][社会经济地位] 通过对1000名不同社会经济地位(SES)用户的大规模调查和6482条真实LLM prompts的分析，揭示了高低SES群体在语言技术使用频率、交互方式和话题选择上存在显著系统性差异，呼吁开发更具包容性的NLP技术以缩小AI鸿沟。 领域现状：随着ChatGPT等大语言模型…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "社会经济地位"
  - "语言技术"
  - "数字鸿沟"
  - "LLM交互"
  - "用户调查"
---

# The AI Gap: How Socioeconomic Status Affects Language Technology Interactions

**会议**: ACL 2025  
**arXiv**: [2505.12158](https://arxiv.org/abs/2505.12158)  
**代码**: [https://huggingface.co/datasets/MilaNLProc/survey-language-technologies](https://huggingface.co/datasets/MilaNLProc/survey-language-technologies)  
**领域**: 其他  
**关键词**: 社会经济地位, 语言技术, 数字鸿沟, LLM交互, 用户调查

## 一句话总结

通过对1000名不同社会经济地位(SES)用户的大规模调查和6482条真实LLM prompts的分析，揭示了高低SES群体在语言技术使用频率、交互方式和话题选择上存在显著系统性差异，呼吁开发更具包容性的NLP技术以缩小AI鸿沟。

## 研究背景与动机

**领域现状**：随着ChatGPT等大语言模型的广泛普及，AI技术正在深刻改变人们与技术交互的方式。然而，不同社会群体对AI工具的采用率和使用方式差异显著，UNESCO和经济学人等机构已注意到这种差距。

**现有痛点**：以往关于SES与语言技术关系的研究依赖代理指标（如教育水平、收入）和合成数据，缺乏直接的用户调查数据。现有的大规模prompt数据集（如ShareGPT、LMSYS-Chat-1M）虽然规模大，但都没有收集用户的社会经济背景信息，无法分析SES对LLM使用的影响。

**核心矛盾**：语言技术看似对所有人开放，但根据技术接受模型(TAM)，技术采用受感知有用性和易用性影响。不同SES群体的数字素养、访问设备和文化资本差异可能导致"数字鸿沟"在AI时代进一步加剧——即AI Gap。

**本文目标** (1) SES如何影响语言技术的采用率和使用场景？(2) 不同SES群体与LLM交互时的语言特征有何差异？(3) 不同群体讨论的话题和对AI系统的感知有何不同？

**切入角度**：作者直接通过Prolific平台招募1000名英美被试，收集他们的社会经济背景信息和与LLM的真实交互prompts，结合定量统计和定性聚类分析来回答上述问题。

**核心 idea**：首次通过大规模用户调查直接研究SES对语言技术使用和LLM交互方式的影响，发现SES差异系统性地体现在使用频率、语言风格和话题选择中。

## 方法详解

### 整体框架

本文的方法框架是一个大规模用户调查研究，包含三个核心部分：(1) 社会人口学信息收集（17个问题，涵盖Macarthur量表自评SES等），(2) 语言技术使用习惯调查（频率、任务类型、使用场景），(3) LLM prompt收集（要求被试提供最近10条与AI聊天机器人的交互记录）。调查在Prolific平台分两阶段进行，第一阶段501人，第二阶段针对性补充380名低和高SES被试。

### 关键设计

1. **SES测量与分组**:

    - 功能：准确衡量被试的社会经济地位
    - 核心思路：采用Macarthur量表（1-10分），让被试自评社会经济地位，然后映射到西方阶层体系——1-3为低、4-7为中、8-10为高。同时收集教育、父母职业、住房等客观指标用于交叉验证。
    - 设计动机：自评SES比客观指标更能反映个人心理感受，研究表明主观阶层认知对行为有重要影响

2. **Prompt语言学分析**:

    - 功能：量化不同SES群体的交互风格差异
    - 核心思路：从多个维度分析6482条真实prompts——长度（词数）、具体性（使用Brysbaert等人的40K单词具体性评分）、拟人化程度（礼貌用语、寒暄语、专业术语vs隐喻语言的使用率），以及训练词袋分类器来验证群体间prompts的可区分性
    - 设计动机：Bernstein的语言编码理论预测高SES群体使用更抽象的语言，本文在LLM交互场景中验证了这一经典假说

3. **话题聚类与定性分析**:

    - 功能：发现不同SES群体关注话题的差异
    - 核心思路：用SentenceTransformer和M3-Embedding编码prompts，通过UMAP+HDBSCAN聚类，再用GPT-4o给聚类分配描述性标签，最后人工评估聚类质量。比较三个SES群体的特有话题和共有话题的不同"框架"
    - 设计动机：相同话题（如金融、求职、食物）在不同SES群体中呈现截然不同的需求框架，这种差异揭示了更深层的社会不平等

### 损失函数 / 训练策略

本文是调查研究，不涉及模型训练。统计检验采用卡方独立性检验和bootstrap重抽样显著性测试，分类器使用词袋+逻辑回归。

## 实验关键数据

### 主实验

| 分析维度 | 指标 | 低SES | 中SES | 高SES | 统计显著性 |
|----------|------|-------|-------|-------|------------|
| Prompt平均长度 | 词数 | 27.0 | 22.3 | 18.4 | p < 0.05 |
| 语言具体性 | 具体性分数(1-5) | 2.66 | 2.63 | 2.57 | p < 0.05 |
| AI聊天机器人使用频率 | 日常使用比例 | 低→高递减 | 中等 | 低→高递增 | χ²=67.79, p<0.001 |
| 设备访问 | 多设备日常使用 | 较少 | 中等 | 较多 | χ²=55.11, p<0.001 |

### 消融实验

| 分析配置 | 关键指标 | 说明 |
|----------|---------|------|
| BoW分类器 | Macro-F1=39.25 | 远超多数类基线25.02，证明不同SES prompts可区分 |
| 专业术语使用率 | 低3.32%/中4.16%/高4.94% | 高SES更多使用专业术语 |
| 拟人化(寒暄语) | 低6.34%/中5.08%/高4.29% | 低SES更倾向将LLM拟人化 |
| 搜索式提问比例 | 低46.6%/中43.5%/高45.4% | 各群体都在用LLM替代搜索引擎 |

### 关键发现

- **使用场景差异显著**：高SES群体更多在工作、学习和技术场景中使用LLM，进行编程、数据分析、写作等高级任务；低SES群体更多用于娱乐和一般性问答
- **同一话题不同框架**：金融话题中，低SES询问省钱方法，高SES询问投资策略；求职中，低SES寻找不需要学历的远程工作，高SES涉及管理岗位的cover letter
- **高SES用户prompts更短但更抽象**：可能因为拥有更丰富的词汇量，能用更少的词表达更精确的需求
- **评估基准偏差风险**：高SES用户常用的任务（摘要、数学题）更容易用ground-truth评估，低SES用户常用的任务更依赖人类偏好评估，可能导致现有评测不公平

## 亮点与洞察

- **首个带SES标注的真实prompt数据集**：填补了现有大规模prompt数据集缺乏社会经济背景信息的空白，为后续研究提供了宝贵资源
- **经典社会语言学理论的新验证场景**：在LLM交互中验证了Bernstein的受限/精细编码理论，说明社会阶层差异在人机交互中同样存在
- **从使用差异到AI Gap的因果推理**：不仅描述了差异，还分析了差异可能如何通过反馈循环加剧数字鸿沟——低SES用户使用简单语言→系统表现差→满意度低→更少使用

## 局限与展望

- **样本局限**：仅限英美Prolific平台用户，众包工人可能比一般人群更熟悉技术，且SES分布偏中低层
- **自报告偏差**：被试自行提供最近10条prompts，可能存在选择性报告；2.5%被试的人口学信息与Prolific档案不一致
- **因果链未验证**：文章推测SES差异会通过NLP系统性能差异加剧不平等，但没有实际测试不同prompt风格在系统中的表现差异
- **缺少纵向分析**：横截面数据无法追踪使用习惯随时间的变化，也无法观察AI技术是否会逐渐缩小差距

## 相关工作与启发

- **vs Cercas Curry et al. (2024a)**：他们用影视作品中的台词作为代理数据研究SES对NLP性能的影响，本文直接收集真实用户数据，更具生态效度
- **vs Daepp and Counts (2024)**：他们分析美国不同地区ChatGPT使用意图的差异，本文在个体层面上收集了更细粒度的SES信息
- **vs Kirk et al. (2024) / ShareGPT数据集**：这些大规模prompt数据集规模更大，但缺乏社会经济背景标注，无法进行本文这样的分析

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统研究SES对LLM交互的影响，开创了一个重要的研究方向
- 实验充分度: ⭐⭐⭐⭐ 1000人调查+6482条真实prompts，多维度分析，统计检验充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分，但部分分析较浅
- 价值: ⭐⭐⭐⭐ 对AI公平性研究有重要启示，提出了实际可操作的改进方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] How Numerical Precision Affects Arithmetical Reasoning Capabilities of LLMs](how_numerical_precision_affects_arithmetical_reasoning_capabilities_of_llms.md)
- [\[ACL 2025\] Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy](mind_your_tone_investigating_how_prompt_politeness_affects_llm_accuracy_short_pa.md)
- [\[ACL 2025\] Retrospective Learning from Interactions](retrospective_learning_from_interactions.md)
- [\[ACL 2026\] Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models](../../ACL2026/llm_nlp/mind_the_gap_how_elicitation_protocols_shape_the_stated-revealed_preference_gap_.md)
- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)

</div>

<!-- RELATED:END -->
