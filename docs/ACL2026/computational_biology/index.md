---
title: >-
  ACL2026 计算生物论文汇总 · 5篇论文解读
description: >-
  5篇ACL2026的计算生物方向论文解读，涵盖推理、多模态、LLM、医学影像、Agent、生物分子等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "计算生物"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "多模态"
  - "LLM"
  - "医学影像"
  - "Agent"
  - "生物分子"
item_list:
  - u: "aroma_augmented_reasoning_over_a_multimodal_architecture_for_virtual_cell_geneti/"
    t: "AROMA: Augmented Reasoning Over a Multimodal Architecture for Virtual Cell Genetic Perturbation Modeling"
  - u: "biotool_a_comprehensive_tool-calling_dataset_for_enhancing_biomedical_capabiliti/"
    t: "BioTool: A Comprehensive Tool-Calling Dataset for Enhancing Biomedical Capabilities of Large Language Models"
  - u: "chemamp_amplified_chemistry_tools_via_composable_agents/"
    t: "ChemAmp: Amplified Chemistry Tools via Composable Agents"
  - u: "protocycle_reflective_tool-augmented_planning_for_text-guided_protein_design/"
    t: "ProtoCycle: Reflective Tool-Augmented Planning for Text-Guided Protein Design"
  - u: "toxreason_a_benchmark_for_mechanistic_chemical_toxicity_reasoning_via_adverse_ou/"
    t: "ToxReason: A Benchmark for Mechanistic Chemical Toxicity Reasoning via Adverse Outcome Pathway"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧬 计算生物

**💬 ACL2026** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (21)](../../CVPR2026/computational_biology/index.md) · [🔬 ICLR2026 (155)](../../ICLR2026/computational_biology/index.md) · [🧪 ICML2026 (51)](../../ICML2026/computational_biology/index.md) · [🤖 AAAI2026 (20)](../../AAAI2026/computational_biology/index.md) · [🧠 NeurIPS2025 (76)](../../NeurIPS2025/computational_biology/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/computational_biology/index.md)

🔥 **高频主题：** 推理 ×2

**[AROMA: Augmented Reasoning Over a Multimodal Architecture for Virtual Cell Genetic Perturbation Modeling](aroma_augmented_reasoning_over_a_multimodal_architecture_for_virtual_cell_geneti.md)**

:   提出 AROMA 框架，通过整合文本证据、知识图谱拓扑信息和蛋白质序列特征的多模态架构，结合两阶段训练策略（SFT + GRPO），实现了可解释且精确的基因扰动效应预测。

**[BioTool: A Comprehensive Tool-Calling Dataset for Enhancing Biomedical Capabilities of Large Language Models](biotool_a_comprehensive_tool-calling_dataset_for_enhancing_biomedical_capabiliti.md)**

:   BioTool 构建了一个覆盖 NCBI / Ensembl / UniProt 三大生物医学数据库 34 个常用工具、7,040 条经人工核验的「查询–API 调用」对的指令微调数据集，用它微调 4B 量级开源 LLM 后，工具调用质量超过 GPT-5.1 / Gemini-3 Pro / Claude-4.5-Sonnet 等商业大模型 15% 以上。

**[ChemAmp: Amplified Chemistry Tools via Composable Agents](chemamp_amplified_chemistry_tools_via_composable_agents.md)**

:   提出"工具放大"新范式（区别于传统的工具编排），通过 ChemAmp 框架将化学专用工具（UniMol2、Chemformer等）作为可组合积木块动态构建任务专用超级智能体，在分子设计、反应预测等四个核心化学任务上超越专用模型和通用LLM，同时推理token成本减少94%。

**[ProtoCycle: Reflective Tool-Augmented Planning for Text-Guided Protein Design](protocycle_reflective_tool-augmented_planning_for_text-guided_protein_design.md)**

:   ProtoCycle提出一种将LLM作为规划器、结合轻量级工具环境的**反思式智能体框架**，用于文本引导蛋白质序列设计：它用多轮“规划-工具调用-评估-反思”循环替代一次性文本到序列生成，在 Mol-Instructions 上将 ProTrek 提升到 14.681、Retrieval 提升到 0.936，并只用约 2,000 条 SFT 轨迹和在线 RL 达到接近/超过专用蛋白质设计模型的语言对齐效果。

**[ToxReason: A Benchmark for Mechanistic Chemical Toxicity Reasoning via Adverse Outcome Pathway](toxreason_a_benchmark_for_mechanistic_chemical_toxicity_reasoning_via_adverse_ou.md)**

:   本文提出 ToxReason，一个基于不良结局路径 (AOP) 框架的化学毒性机理推理基准，整合药物-靶点实验数据与毒性标签，要求模型从分子起始事件推理到器官级不良结局；通过 GRPO 强化学习训练的 4B 模型在毒性预测（F1 71.4%）和推理质量上均超越 GPT-5 等大模型。
