<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📖 NLP 理解

**💬 ACL2025** · 共 **18** 篇

**[A Variational Approach for Mitigating Entity Bias in Relation Extraction](a_variational_approach_for_mitigating_entity_bias_in_relation_extraction.md)**

:   将变分信息瓶颈（VIB）应用于关系抽取的实体去偏——将实体映射到概率分布 $\mathcal{N}(\mu,\sigma)$，通过方差控制实体信息的压缩程度（高方差=更多依赖上下文），在 TACRED/REFinD/BioRED 三个领域（通用/金融/生物医学）的域内和域外设置上达到 SOTA，同时方差分析提供可解释性。

**[Adapting Psycholinguistic Research for LLMs: Gender-Inclusive Language in a Coreference Context](adapting_psycholinguistic_research_for_llms_gender-inclusive_language_in_a_coref.md)**

:   将心理语言学方法从法语适配到英语和德语，研究 LLM 如何处理性别包容性语言——发现英语 LLM 基本保持先行词性别一致但内含男性默认偏见（不愿用 they 单数），德语 LLM 男性偏见更强烈（压倒所有性别中性化策略），但德语性别包容形式确实增加了女性/中性性别的出现概率。

**[Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification](analyzing_political_bias_in_llms_via_target-oriented_sentiment_classification.md)**

:   提出基于目标导向情感分类（TSC）不一致性的 LLM 政治偏见分析新方法——在 450 个政治句子中插入 1319 名不同政治光谱/人口特征的政治家名字，用 7 个模型×6 种语言预测情感，定义熵基不一致性指标量化预测变异性，发现所有模型均存在显著偏见（左翼正面/极右翼负面），大模型偏见更强且更一致，用虚构名字替换可部分缓解。

**[ArgHiTZ at ArchEHR-QA 2025: A Two-Step Divide and Conquer Approach to Patient Question Answering for Top Factuality](arghitz_at_archehr-qa_2025_a_two-step_divide_and_conquer_approach_to_patient_que.md)**

:   在 ArchEHR-QA 2025 共享任务中提出两阶段"分治"方法：先用重排序模型从电子健康记录中提取关键句子，再用小型医学 LLM 生成回复，在不使用外部知识的情况下取得事实性排名第一、总分第 8/30 的成绩。

**[AskQE: Question Answering as Automatic Evaluation for Machine Translation](askqe_question_answering_as_automatic_evaluation_for_machine_translation.md)**

:   提出 AskQE——基于问答的机器翻译质量估计框架，通过对源文本生成问题、分别在源文本和回译输出上回答、对比答案差异来检测翻译错误，帮助不懂目标语言的用户判断翻译是否可接受，在 BioMQM 数据集上 Kendall's τ 相关和决策准确率均优于现有 QE 指标。

**[Automatic Generation of Inference Making Questions for Reading Comprehension Assessments](automatic_generation_of_inference_making_questions_for_reading_comprehension_ass.md)**

:   开发了一套阅读理解推理题分类法（代词桥接/文本连接/填补空白），用 GPT-4o few-shot 提示自动生成针对特定推理类型的多项选择题；93.8% 的题目质量合格，但仅 42.6% 准确匹配目标推理类型，说明 LLM 在精确推理能力控制上仍有不足。

**[BelarusianGLUE: Towards a Natural Language Understanding Benchmark for Belarusian](belarusian_glue.md)**

:   为白俄罗斯语（Belarusian，东斯拉夫语族）构建了首个NLU benchmark——BelarusianGLUE，包含5个任务约15K条实例，系统评估了BERT系列和LLM的表现，发现简单任务（情感分析）接近人类水平但难任务（Winograd）仍有显著差距，且最优模型类型因任务而异。

**[BESSTIE: A Benchmark for Sentiment and Sarcasm Classification for Varieties of English](besstie_a_benchmark_for_sentiment_and_sarcasm_classification_for_varieties_of_en.md)**

:   构建 BESSTIE，首个针对英语变体（澳大利亚/印度/英国英语）的情感分析和讽刺检测标注基准，通过 9 个微调 LLM 评估发现模型在印度英语（外圈变体）上表现显著差于内圈变体，跨变体泛化能力也有限。

**[BookCoref: Coreference Resolution at Book Scale](bookcoref_book_scale.md)**

:   提出首个书级别共指消解基准BookCoref，通过角色链接+LLM过滤+窗口扩展的自动标注管线，在50本完整小说上生成高质量银标注数据，平均文档长度超过20万tokens。

**[CaLMQA: Exploring Culturally Specific Long-Form Question Answering across 23 Languages](calmqa_cultural_multilingual_qa.md)**

:   构建了首个多语言长文本问答数据集 CaLMQA（51.7K 问题，23 种语言），通过无翻译方式收集文化特异性问题，发现 LLM 回答文化特异性问题的事实性（45-52%）显著低于文化无关问题（64-71%），低资源语言表现尤其差。

**[Dynamic Order Template Prediction for Generative Aspect-Based Sentiment Analysis](dot_absa_template.md)**

:   提出 Dynamic Order Template（DOT）方法用于生成式方面级情感分析——为每个实例动态创建最优的预测模板顺序（只含必要的视角），在 ASQP 和 ACOS 数据集上提升 F1 的同时显著减少推理时间。

**[Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering](embqa_embedding_odqa.md)**

:   EmbQA 提出嵌入级 ODQA 框架，用轻量线性层和无监督对比学习优化查询表示实现段落重排序，并引入基于序统计量的探索性嵌入扩展候选答案多样性，配合熵选择机制自动选答，在 4 个 ODQA 数据集上以更低计算成本超越 SuRe 等 prompt 级方法。

**[Towards a More Generalized Approach in Open Relation Extraction](generalized_open_relation_extract.md)**

:   提出 MixORE 框架，在更通用的 Open Relation Extraction 设定下（无标注数据同时包含已知和新颖关系，且不做长尾或预分割假设），通过 Semantic Autoencoder 检测新关系 + 开放世界半监督联合学习，在 FewRel/TACRED/Re-TACRED 上全面超越 SOTA。

**[LACA: Improving Cross-lingual Aspect-Based Sentiment Analysis with LLM Data Augmentation](laca_crosslingual_absa.md)**

:   提出 LACA 框架，利用 LLM 为目标语言生成高质量伪标注数据（而非依赖机器翻译），在六种语言上显著提升跨语言 ABSA 性能，在 mBERT 和 XLM-R 上分别平均超过前 SOTA 1.50% 和 2.62%。

**[NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](neusym_rag_pdf_qa.md)**

:   NeuSym-RAG 提出了一个混合神经-符号检索框架，将 PDF 文档通过多视角分块解析同时存入关系数据库和向量库，LLM Agent 通过可执行动作（SQL 查询 + 向量检索 + 查看图片等）迭代式交互检索，在学术论文 QA 上比经典 RAG 提升 17.3%。

**[Exploring Persona Sentiment Sensitivity in Personalized Dialogue Generation](persona_sentiment_dialogue.md)**

:   大规模分析 LLM 对人设情感极性的敏感性，发现负面人设导致过度强调人设属性和对话矛盾、弱/中性人设产生低质量对话，提出结合逐轮生成、人设排序和情感感知提示的对话生成框架来缓解这些问题。

**[ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision](rescore_multihop_qa.md)**

:   提出 ReSCORE，利用 LLM 生成的文档-问题相关性（relevance）和文档-答案一致性（consistency）的联合概率作为伪标签，在迭代 RAG 框架中无监督训练 dense retriever，在三个多跳 QA 数据集上达到 SOTA。

**[YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering](yescieval_llm_judge_science.md)**

:   提出YESciEval框架，结合九维细粒度评估准则和SFT+RL对齐策略来缓解LLM评估者的乐观偏差(optimism bias)，在科学问答场景下构建鲁棒的开源LLM-as-a-Judge系统，无需人类标注和闭源模型。
