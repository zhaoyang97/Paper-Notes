<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**💬 ACL2025** · 共 **10** 篇

**[AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset](afrimed_qa_pan_african.md)**

:   构建了首个大规模泛非洲多专科医学问答基准 AfriMed-QA（15,275 题，来自 16 个国家 60+ 医学院，涵盖 32 个专科），评估 30 个 LLM 发现：大模型在非洲医疗问题上的准确率显著低于 USMLE，生物医学专用 LLM 反而不如通用模型，消费者盲评时更偏好 LLM 回答而非临床医生回答。

**[The Impact of Auxiliary Patient Data on Automated Chest X-Ray Report Generation and How to Incorporate It](auxiliary_patient_data_xray.md)**

:   本文研究如何将急诊科患者数据（生命体征、药物、分诊信息等）整合到多模态语言模型中用于自动胸部X光报告生成，提出将异构表格数据、文本和图像转化为统一嵌入的方法，在MIMIC-CXR + MIMIC-IV-ED数据集上显著提升了报告的诊断准确性，超越了包括CXRMate-RRG24在内的多个基准模型。

**[Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge](biore_llm_judge_evaluation.md)**

:   本文首次系统研究了 LLM-as-Judge 在生物医学关系抽取评估中的表现，发现其准确率通常低于 50%，并提出结构化输出格式（JSON）和域适应技术来提升约 15% 的评估准确率。

**[CheXalign: Preference Fine-tuning in Chest X-ray Interpretation Models without Human Feedback](chexalign_preference_finetuning.md)**

:   CheXalign 提出了一种无需放射科医生反馈的自动化偏好数据生成管线，利用公开数据集中的参考报告和基于参考的评估指标（如 GREEN、BERTScore）构造偏好对，通过 DPO 等直接对齐算法对胸部X光报告生成模型进行偏好微调，在 MIMIC-CXR 上取得 SOTA CheXbert 分数。

**[Aligning AI Research with the Needs of Clinical Coding Workflows: Eight Recommendations Based on US Data Analysis and Critical Review](clinical_coding_eight_recommendations.md)**

:   这篇 position paper 通过对 MIMIC 数据集和现有自动化临床编码研究的深入分析，指出当前评估方法（如仅关注前50个高频编码、使用不恰当指标）与真实临床场景严重脱节，并提出八条具体建议来改进评估方法和研究方向。

**[Online Iterative Self-Alignment for Radiology Report Generation](oisa_radiology_report_gen.md)**

:   提出在线迭代自对齐（OISA）方法用于放射学报告生成——四阶段循环（自生成多样数据→自评估多目标偏好→自对齐多目标优化→自迭代进一步提升），无需额外人工标注即可迭代提升报告质量，在多个评估指标上达到 SOTA。

**[Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications](omni_rag_medical.md)**

:   针对医疗 LLM 需要多类型多结构知识源（教科书/指南/论文/知识图谱等）的特殊需求，提出 MedOmniKB 多源知识库和 Source Planning Optimization 方法——让模型学会"该从哪个源检索什么信息"，优化后的小模型在多源医疗知识利用上达到 SOTA。

**[RADAR: Enhancing Radiology Report Generation with Supplementary Knowledge Injection](radar_radiology_report_gen.md)**

:   提出 Radar 框架，通过"内部知识提取+外部补充知识检索+聚合注入"三步策略增强放射学报告生成——先提取 LLM 已有的与专家分类一致的知识，再检索缺失的补充知识，最终聚合两者生成更准确的放射学报告，在 MIMIC-CXR/CheXpert-Plus/IU X-ray 三个数据集上超越 SOTA。

**[ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents](reflectool_clinical_agent.md)**

:   ReflecTool 提出了一个反思感知的工具增强临床 Agent 框架，通过优化阶段积累成功轨迹和工具级经验，推理阶段检索相似案例并用验证器改进工具使用，在涵盖 18 个任务的 ClinicalAgent Bench 上超越纯 LLM 10+ 分、超越已有 Agent 方法 3 分。

**[Query-driven Document-level Scientific Evidence Extraction from Biomedical Studies](urca_biomedical_evidence_extraction.md)**

:   本文提出 URCA（Uniform Retrieval Clustered Augmentation）框架，通过均匀检索+聚类+知识提取的 RAG 流程，从 RCT 研究全文中自动提取与临床问题相关的科学证据结论，在新构建的 CochraneForest 数据集上比最佳基线提升了 8.81% F1。
