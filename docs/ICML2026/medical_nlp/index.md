---
title: >-
  ICML2026 医疗LLM论文汇总 · 4篇论文解读
description: >-
  4篇ICML2026的医疗 LLM 方向论文解读，涵盖医学影像、对齐/RLHF、对抗鲁棒、域适应、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "医疗 LLM"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "对齐/RLHF"
  - "对抗鲁棒"
  - "域适应"
  - "推理"
item_list:
  - u: "a_machine-learned_comorbidity_index/"
    t: "A Machine-Learned Comorbidity Index"
  - u: "clintutor-r1_advancing_scalable_and_robust_one-to-many_alignment_in_clinical_soc/"
    t: "ClinTutor-R1: Advancing Scalable and Robust One-to-Many Alignment in Clinical Socratic Education"
  - u: "exploring_accurate_and_transparent_domain_adaptation_in_predictive_healthcare_vi/"
    t: "Exploring Accurate and Transparent Domain Adaptation in Predictive Healthcare via Concept-Grounded Orthogonal Inference"
  - u: "medcase-structured_a_text-to-fhir_dataset_for_benchmarking_diagnostic_reasoning_/"
    t: "MedCase-Structured: A Text-to-FHIR Dataset for Benchmarking Diagnostic Reasoning in Clinically Realistic EHR Settings"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🩺 医疗 LLM

**🧪 ICML2026** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (1)](../../CVPR2026/medical_nlp/index.md) · [🔬 ICLR2026 (20)](../../ICLR2026/medical_nlp/index.md) · [💬 ACL2026 (47)](../../ACL2026/medical_nlp/index.md) · [🤖 AAAI2026 (12)](../../AAAI2026/medical_nlp/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/medical_nlp/index.md) · [🧪 ICML2025 (4)](../../ICML2025/medical_nlp/index.md)

🔥 **高频主题：** 医学影像 ×2

**[A Machine-Learned Comorbidity Index](a_machine-learned_comorbidity_index.md)**

:   传统共病评分（Charlson、Elixhauser）是为死亡率手工调权的线性规则，换个临床结局就失准；本文用神经网络把一次住院的 ICD 诊断码压成一个标量分数，并通过**最大化该分数与多个临床结局之间的归一化 HSIC（核依赖）**来训练，使这一个分数能在死亡、再入院、住院时长、ICU 转入等多结局上给出一致的严重度排序，在 MIMIC-III/IV 上的依赖性指标显著超过传统指数与多种机器学习基线。

**[ClinTutor-R1: Advancing Scalable and Robust One-to-Many Alignment in Clinical Socratic Education](clintutor-r1_advancing_scalable_and_robust_one-to-many_alignment_in_clinical_soc.md)**

:   提出 ClinTutor-R1，首个面向临床苏格拉底式教学的一对多对齐视觉语言 Agent，通过多智能体模拟器 ClinEdu 构建 48k 对话数据集 ClinTeach，利用显式心智理论推理和三轴 rubric 强化学习，在学员扩展至 10 人时仍保持教学质量稳定，超越基线模型 20% 并达到 GPT-4o 水平。

**[Exploring Accurate and Transparent Domain Adaptation in Predictive Healthcare via Concept-Grounded Orthogonal Inference](exploring_accurate_and_transparent_domain_adaptation_in_predictive_healthcare_vi.md)**

:   ExtraCare 用一个"字典度量诱导的正交分解"把电子病历（EHR）患者表征拆成「跨域不变的标签信息」和「域特有的协变量残差」，既在两个真实 EHR 数据集上超过现有域适应基线，又能通过稀疏维度消融把每个隐变量映射回具体 ICD 医学概念，告诉临床医生"适应过程中保留了什么、丢掉了什么"。

**[MedCase-Structured: A Text-to-FHIR Dataset for Benchmarking Diagnostic Reasoning in Clinically Realistic EHR Settings](medcase-structured_a_text-to-fhir_dataset_for_benchmarking_diagnostic_reasoning_.md)**

:   作者提出一个把自由文本病例转成符合 HL7 FHIR R4 标准的"分阶段 LLM + 术语接地 + 修复循环"流水线，并据此从 MedCaseReasoning 构造出 1408 条结构化合成病例数据集 MedCase-Structured（成功率 82.5%），实验显示 GPT-5.4 / Gemini-3.1-Pro / Claude-Opus-4.6 在结构化 FHIR 输入上的诊断准确率比纯文本输入一致下降 4–23 个点。
