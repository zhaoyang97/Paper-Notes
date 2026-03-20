# BelarusianGLUE: Towards a Natural Language Understanding Benchmark for Belarusian

**会议**: ACL 2025 (Long Paper, acl-long.25)  
**arXiv**: 无（仅ACL Anthology发表）  
**代码**: [https://github.com/maaxap/BelarusianGLUE](https://github.com/maaxap/BelarusianGLUE)  
**数据**: https://hf.co/datasets/maaxap/BelarusianGLUE
**领域**: NLU基准 / 低资源语言 / 多语言评估  
**关键词**: Belarusian, NLU Benchmark, Low-Resource Language, GLUE, Multilingual Evaluation  

## 一句话总结
为白俄罗斯语（Belarusian，东斯拉夫语族）构建了首个NLU benchmark——BelarusianGLUE，包含5个任务约15K条实例，系统评估了BERT系列和LLM的表现，发现简单任务（情感分析）接近人类水平但难任务（Winograd）仍有显著差距，且最优模型类型因任务而异。

## 背景与动机
在多语言大模型时代，评估模型对低资源语言的理解能力仍然是一个挑战。白俄罗斯语虽为东斯拉夫语族语言（与俄语、乌克兰语同族），但长期缺乏专门的NLU评测资源。现有的多语言benchmark（如XGLUE、XTREME）对白俄罗斯语覆盖极为有限，而语言特有的现象（如白俄罗斯语的反身代词 свой vs 物主代词 яго/ягоны 之区分、正字法变体 narkamaŭka vs taraškievica）使得简单翻译英语benchmark并不够用。因此需要专家精心构建的、针对白俄罗斯语特点的NLU benchmark。

## 核心问题
如何为白俄罗斯语这样的低资源语言构建一个高质量、多任务的NLU benchmark？现有的BERT模型和LLM在白俄罗斯语理解上达到什么水平？与人类表现差距有多大？什么类型的模型更适合哪类任务？

## 方法详解

### 整体框架
BelarusianGLUE包含5个二分类NLU任务，总计约15K条实例，所有数据均由语言学背景的白俄罗斯语母语专家标注或审核：

| 任务 | 缩写 | 实例数 | 训练/验证/测试 | 数据来源 |
|------|------|--------|---------------|---------|
| 情感分析 | BeSLS | 2000 | 1500/250/250 | 5个领域的评论（电影、书籍、酒旅、购物、社交） |
| 语言可接受性 | BelaCoLA | 3592 | 1992/300/300 (域内) + 500/500 (域外) | RuCoLA翻译、规范文献、CommonVoice、LM幻觉、机器翻译 |
| 上下文词义消歧 | BeWiC | ~5000+ | 大量/400/400 | 白俄罗斯语解释词典（1977-1984） |
| Winograd模式挑战 | BeWSC | 970 | 570/200/200 | WSC-285翻译+原创 |
| 文本蕴含 | BeRTE-WD | 1800 | 1080/360/360 | Wikidata知识库 |

评估三类主体：
1. **Human baseline** — 白俄罗斯语母语者通过Streamlit UI标注
2. **BERT系列** — mBERT、XLM-RoBERTa、mDeBERTa-v3、HPLT BERT (be) 的fine-tuning
3. **LLM** — 使用lm-evaluation-harness进行zero-shot评估 + Gemma 2 9B fine-tuning

### 关键设计

1. **BeSLS（情感分析）**: 从5个领域（电影、书籍、酒旅、购物、社交）均匀采样，每域平衡正/负类。来源涵盖专业影评（报纸Zviazda、Kultura）、Telegram频道、LiveLib书评、Booking/TripAdvisor旅行评价、Onliner商品评论、Mastodon社交帖文。用Lingua过滤非白俄罗斯语句子，用户名做匿名化处理。数据反映了白俄罗斯语书面变体（官方现代正字法 narkamaŭka、经典正字法 taraškievica、拉丁字母 łacinka）的真实分布。

2. **BelaCoLA（语言可接受性）**: 参照CoLA/RuCoLA/BLiMP设计，但不可接受句的类型更广——不仅包括形态、句法、语义偏差，还包括语用异常、规范性规则违反、语言模型幻觉和机器翻译错误。域外测试集专门包含trigram模型输出、GPT-2（117M）生成文本、NLLB/Google Translate/Belazar的机器翻译结果，这些正是现实中越来越常见的白俄罗斯语文本类型。

3. **BeWiC（上下文词义消歧）**: 基于白俄罗斯语5卷解释词典，利用词典中的例句构建语境对。与原版WiC不同，这里正例/反例的区分基于更强的同形异义词（homonym）标准而非多义词（polysemy），使任务对人类更容易但更适合用词典数据构建。短语级例句被扩展为完整句子，多句例句被精简为单句。

4. **BeWSC（Winograd模式挑战）**: 提供WSC和WNLI两种格式。训练集主要翻译自英语WSC-285，但因白俄罗斯语语法差异（性别语法化、反身代词 свой vs 物主代词）需要大量适应性改编。测试集200条基于白俄罗斯语小说文本原创，故意设计为难以通过选择限制（selectional restrictions）解决。

5. **BeRTE-WD（文本蕴含）**: 创新性地从Wikidata知识库构建。从Wikidata提取有白俄罗斯语标签的实体-属性-值三元组（时间戳、数值、实体三类各200条），由3名专家将三元组转化为自然语言文本并撰写蕴含/非蕴含假设。蕴含类型极其丰富——时间比较、数值推理、约束满足、单位换算、领域知识、世界知识、单调推理、逻辑推论、释义等。

### 评估策略
- **BERT评估**: 标准fine-tuning + 跨语言迁移学习（利用其他语言的类似数据集如MELA、XLWiC、RUSSE、WinoGrande等预训练）+ 层冻结实验
- **LLM评估**: 使用lm-evaluation-harness的zero-shot log probability评估（本地模型）和生成式评估（商业API模型），以及Gemma 2 9B在白俄罗斯语/英语提示下的fine-tuning
- **Human baseline**: 通过定制Streamlit界面收集母语者判断

## 实验关键数据

根据论文描述的关键发现（具体数值来自论文Tables）：

| 任务 | 指标 | 最佳BERT | 最佳LLM | Human |
|------|------|---------|---------|-------|
| BeSLS（情感分析） | Accuracy | 接近人类 | 接近人类 | ~高水平 |
| BelaCoLA（可接受性） | MCC/Accuracy | BERT有竞争力 | 弱于BERT | ~高水平 |
| BeWiC（词义消歧） | Accuracy | 中等 | 中等 | 高水平 |
| BeWSC（Winograd） | Accuracy | 显著低于人类 | 显著低于人类 | 高水平 |
| BeRTE-WD（文本蕴含） | Accuracy | BERT弱 | LLM更好 | 高水平 |

**评估的BERT模型**: mBERT、XLM-RoBERTa-base、mDeBERTa-v3-base、HPLT BERT Belarusian

**核心发现**:
- **情感分析**是最简单的任务，BERT和LLM均接近人类水平
- **Winograd挑战**差距最大，机器与人类表现之间存在显著gap
- **模型选择因任务而异**: BERT在语言可接受性任务上有竞争力，但在文本蕴含任务上表现不佳；LLM在需要世界知识的蕴含任务上更有优势
- 跨语言迁移学习（利用其他语言相似任务的数据）可以提升BERT在BelarusianGLUE上的表现

### 消融实验要点
- **层冻结实验**: 测试了冻结mDeBERTa-v3所有12层encoder只训练分类头 vs 完整fine-tuning的效果差异，探索预训练表示质量
- **跨语言迁移**: 用其他语言类似数据集（如英语WiC/XLWiC、俄语RUSSE用于BeWiC；多语言CoLA如MELA、Dutch CoLA、HuCoLA用于BelaCoLA；英语WinoGrande用于BeWSC）先预训练再在白俄罗斯语数据上fine-tuning
- **提示语言**: Gemma 2 9B fine-tuning比较了白俄罗斯语提示和英语提示的效果

## 亮点
- **高质量专家构建**: 所有数据由具有语言学硕/博学位的白俄罗斯语母语者标注或审核，而非众包或机翻，保证了benchmark质量
- **BeRTE-WD设计巧妙**: 利用Wikidata结构化知识构建蕴含任务，蕴含类型覆盖极广（时间推理、数值推理、世界知识等），形成了一个需要多种推理能力的挑战性任务
- **BelaCoLA的域外测试集**: 用LM幻觉和机器翻译结果作为域外不可接受句，直接对应现实中低资源语言的实际问题——机翻质量差和LM生成错误
- **BeWSC的语言适应性**: 不是简单翻译英语Winograd，而是根据白俄罗斯语语法特点（性别语法化、反身代词系统）进行了深度适应，测试集甚至从白俄罗斯语文学作品原创
- **完整的评估代码和数据开源**: 提供了可直接复现的BERT fine-tuning、LLM评估pipeline

## 局限性
- **规模偏小**: 约15K条实例在NLU benchmark中属于小规模，尤其BeSLS仅2000句、BeWSC仅970条
- **任务类型有限**: 仅5个任务且均为二分类，缺少问答、阅读理解、生成等任务类型
- **训练数据不足**: BeWSC训练集主要翻译而非原创，可能引入翻译偏差；BeWiC训练集中允许句子和目标词重复
- **法律风险**: 由于白俄罗斯当局大量将信息源列为"极端主义材料"，作者无法保证数据集在白俄罗斯境内使用的法律安全性
- **缺乏与更多LLM的对比**: 论文发表时的最新大模型（如GPT-4、Claude等）可能未被完整评估
- **评估指标单一**: 大部分任务仅用Accuracy，BelaCoLA用了MCC和F1，但缺少更细粒度的分析

## 与相关工作的对比
- **vs SuperGLUE/GLUE**: BelarusianGLUE的任务设计直接对标GLUE/SuperGLUE（情感分析↔SST、可接受性↔CoLA、词义消歧↔WiC、Winograd↔WSC、蕴含↔RTE），但针对白俄罗斯语特点做了大量本地化
- **vs RussianSuperGLUE**: 与同为东斯拉夫语的俄语benchmark相比，BelarusianGLUE规模更小但数据质量更高（全部专家标注），BeRTE-WD的Wikidata构建方法比基于CommonGen的SCR任务更具挑战性
- **vs 其他低资源语言benchmark**（如KorNLI、TurkishGLUE等）: BelarusianGLUE的独特之处在于BeRTE-WD的知识库驱动构建和BelaCoLA对LM幻觉/机翻错误的覆盖，更贴近低资源语言面临的实际挑战

## 启发与关联
- **低资源语言benchmark构建方法论**: BeRTE-WD利用Wikidata构建蕴含任务的方法可迁移到其他有Wikidata标签的低资源语言
- **跨语言迁移策略**: 论文展示了如何利用多种语言的类似任务数据为目标语言的BERT微调提供更多训练信号，是低资源场景的实用策略
- **LM幻觉检测**: BelaCoLA的域外设计思路——用LM生成文本测试可接受性——可以作为LM质量评估的间接方法
- **多语言模型评估**: 对于研究多语言LLM在不同语言上的能力差异有参考价值

## 评分
- 新颖性: ⭐⭐⭐ 任务设计遵循经典GLUE范式，核心贡献在于高质量的白俄罗斯语本地化，BeRTE-WD的Wikidata构建方法有一定新意
- 实验充分度: ⭐⭐⭐⭐ 覆盖了BERT、LLM、Human三类评估主体，包含跨语言迁移和层冻结消融实验
- 写作质量: ⭐⭐⭐⭐ 数据构建过程描述详尽，每个任务的数据来源、标注流程、数据切分都交代清楚
- 对我的价值: ⭐⭐ 主要服务于白俄罗斯语NLP社区，benchmark构建方法论和跨语言迁移策略有一定通用参考价值
