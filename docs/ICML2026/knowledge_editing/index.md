---
title: >-
  ICML2026 知识编辑论文汇总 · 8篇论文解读
description: >-
  8篇ICML2026的知识编辑方向论文解读，涵盖 LLM、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "知识编辑"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "多模态"
item_list:
  - u: "anyedit_adaptive_long-form_knowledge_editing_via_bayesian_surprise/"
    t: "AnyEdit++: Adaptive Long-Form Knowledge Editing via Bayesian Surprise"
  - u: "crispedit_low-curvature_projections_for_scalable_non-destructive_llm_editing/"
    t: "CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing"
  - u: "do_text_edits_generalize_to_visual_generation_benchmarking_cross-modal_knowledge/"
    t: "Do Text Edits Generalize to Visual Generation? Benchmarking Cross-Modal Knowledge Editing in UMMs"
  - u: "from_backward_spreading_to_forward_replay_revisiting_target_construction_in_llm_/"
    t: "From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing"
  - u: "kore_enhancing_knowledge_injection_for_large_multimodal_models_via_knowledge-ori/"
    t: "KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Controls"
  - u: "reverse-engineering_model_editing_on_language_models/"
    t: "Reverse-Engineering Model Editing on Language Models"
  - u: "revisiting_parameter-based_knowledge_editing_in_large_language_models_theoretica/"
    t: "Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence"
  - u: "the_labyrinth_and_the_thread_rethinking_regularizations_in_sequential_knowledge_/"
    t: "The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models"
item_total: 8
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✏️ 知识编辑

**🧪 ICML2026** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/knowledge_editing/index.md) · [🔬 ICLR2026 (15)](../../ICLR2026/knowledge_editing/index.md) · [💬 ACL2026 (10)](../../ACL2026/knowledge_editing/index.md) · [🤖 AAAI2026 (4)](../../AAAI2026/knowledge_editing/index.md) · [🧠 NeurIPS2025 (6)](../../NeurIPS2025/knowledge_editing/index.md) · [🧪 ICML2025 (2)](../../ICML2025/knowledge_editing/index.md)

🔥 **高频主题：** LLM ×4

**[AnyEdit++: Adaptive Long-Form Knowledge Editing via Bayesian Surprise](anyedit_adaptive_long-form_knowledge_editing_via_bayesian_surprise.md)**

:   AnyEdit++ 用 token 级 Bayesian Surprise 找到长文本中的语义转折点，把 AnyEdit 的固定窗口切分改成结构感知的 Bayes-Chunk，并在数学、代码、新闻、诗歌等长文本知识编辑任务上稳定提升 BLEU 与 BERT Score。

**[CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing](crispedit_low-curvature_projections_for_scalable_non-destructive_llm_editing.md)**

:   把 LLM 编辑写成"最小化编辑损失 s.t. 能力损失不变"的约束优化, 用 Bregman 散度等价转化为 Gauss-Newton Hessian 的低曲率子空间投影, 再借 K-FAC + 一个无需显式构造投影矩阵的 Kronecker 特征基技巧, 让 3000 条编辑在 A40 上 6 分钟跑完, 同时把 LLaMA-3-8B 的 MMLU/IFEval/ARC-C/TruthfulQA/GSM8K 平均掉点压到 < 1%, 显著优于 AlphaEdit / MEMIT / 微调。

**[Do Text Edits Generalize to Visual Generation? Benchmarking Cross-Modal Knowledge Editing in UMMs](do_text_edits_generalize_to_visual_generation_benchmarking_cross-modal_knowledge.md)**

:   本文提出 UniKE——首个面向统一多模态模型 (UMM) 的"跨模态知识编辑"基准（2,971 个编辑主体、5,535 条 VQA 可验证实例），系统性地揭示了"文本侧编辑成功率 ~92% 但图像生成 VQA 仅 ~18.5%"的模态鸿沟，并通过"推理增强参数编辑"协议把 VQA 准确率最多拉高 18.6 个百分点，进一步用条件通路上的余弦漂移指标将根因定位到 LLM-to-DiT 投影瓶颈。

**[From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing](from_backward_spreading_to_forward_replay_revisiting_target_construction_in_llm_.md)**

:   本文系统剖析了 locate-then-edit 编辑中 backward spreading 为什么能 work 又为什么 work 得不彻底，并提出 forward replay：把第一决定层作为优化变量、再通过标准前向传播得到后续各层 target，无需额外算力就能在 MEMIT/RECT/PRUNE/AlphaEdit 之上一致涨点。

**[KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Controls](kore_enhancing_knowledge_injection_for_large_multimodal_models_via_knowledge-ori.md)**

:   KORE 通过两阶段"知识导向控制"为 LMM 注入新知识 — 一边把单条事实自动扩成结构化的多轮对话+指令任务（提升泛化），一边用先前知识的协方差矩阵零空间初始化 LoRA 适配器（最小化对旧能力的干扰），在 LLaVA-v1.5 / Qwen2.5-VL 上同时实现强适配和强保留。

**[Reverse-Engineering Model Editing on Language Models](reverse-engineering_model_editing_on_language_models.md)**

:   论文揭示 locate-then-edit 类知识编辑方法（ROME/MEMIT/AlphaEdit）的参数更新矩阵会通过其行空间泄露"被编辑主语"的指纹，并提出两阶段攻击 KSTER（先用 SVD 恢复主语，再用前后模型的熵差恢复 prompt），同时给出基于"语义诱饵"注入的子空间伪装防御方案。

**[Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence](revisiting_parameter-based_knowledge_editing_in_large_language_models_theoretica.md)**

:   本文从"维度坍塌"假设出发，证明参数级知识编辑会沿低奇异值方向被放大并随序列编辑线性累积，进而在多模型、多数据集、多评测维度上系统性地拖垮 LLM 核心能力，并指出一个简单的检索式基线 SCR 在所有设定下都优于现有参数编辑方法。

**[The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models](the_labyrinth_and_the_thread_rethinking_regularizations_in_sequential_knowledge_.md)**

:   本文从优化角度证明：序列编辑（SE）之所以稳定，本质是"累积更新等价于一次性编辑（OTE）的解"，而 AlphaEdit 的零空间投影、PRUNE/RECT 的后处理正则等花哨机制并非关键——只要保证 OTE-SE 对齐，去掉这些正则也能在 4 个主流 LLM 上稳定完成 2000 步序列编辑。
