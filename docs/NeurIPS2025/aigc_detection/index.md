---
title: >-
  NeurIPS2025 AIGC检测论文汇总 · 9篇论文解读
description: >-
  9篇NeurIPS2025的 AIGC 检测方向论文解读，涵盖 LLM、Agent、对抗鲁棒、翻译、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "AIGC 检测"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "Agent"
  - "对抗鲁棒"
  - "翻译"
  - "推理"
item_list:
  - u: "asciibench_evaluating_language-model-based_understanding_of_visually-oriented_te/"
    t: "ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text"
  - u: "can_llms_write_faithfully_an_agent-based_evaluation_of_llm-generated_islamic_con/"
    t: "Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content"
  - u: "classical_planning_with_llm-generated_heuristics_challenging_the_state_of_the_ar/"
    t: "Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code"
  - u: "clawscreativity_detection_for_llm-generated_solutions_using_attention_window_of_/"
    t: "CLAWS: Creativity Detection for LLM-Generated Solutions Using Attention Window of Sections"
  - u: "duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_/"
    t: "DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code"
  - u: "jutters/"
    t: "\"Jutters\""
  - u: "qimeng-neucomback_self-evolving_translation_from_ir_to_assembly_code/"
    t: "QiMeng-NeuComBack: Self-Evolving Translation from IR to Assembly Code"
  - u: "reasoning_compiler_llm-guided_optimizations_for_efficient_model_serving/"
    t: "Reasoning Compiler: LLM-Guided Optimizations for Efficient Model Serving"
  - u: "synthesizing_performance_constraints_for_evaluating_and_improving_code_efficienc/"
    t: "Synthesizing Performance Constraints for Evaluating and Improving Code Efficiency"
item_total: 9
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔎 AIGC 检测

**🧠 NeurIPS2025** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (10)](../../CVPR2026/aigc_detection/index.md) · [🔬 ICLR2026 (30)](../../ICLR2026/aigc_detection/index.md) · [💬 ACL2026 (17)](../../ACL2026/aigc_detection/index.md) · [🧪 ICML2026 (11)](../../ICML2026/aigc_detection/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/aigc_detection/index.md) · [💬 ACL2025 (15)](../../ACL2025/aigc_detection/index.md)

🔥 **高频主题：** LLM ×4 · Agent ×2

**[ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text](asciibench_evaluating_language-model-based_understanding_of_visually-oriented_te.md)**

:   提出 ASCIIBench，首个公开可用的 ASCII 艺术理解与生成基准（5,315 张图像，752 类），系统评估发现视觉模态显著优于文本模态，多模态融合反而不帮忙，且 CLIP 对 ASCII 结构的表征能力存在根本性瓶颈——只有内部一致性高的类别才能被有效区分。

**[Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content](can_llms_write_faithfully_an_agent-based_evaluation_of_llm-generated_islamic_con.md)**

:   提出双Agent（定量+定性）评估框架，从神学准确性、引用完整性和文体恰当性三个维度系统评估 GPT-4o、Ansari AI 和 Fanar 在伊斯兰内容生成任务上的忠实度，发现即使最优模型也在引用可靠性上存在显著不足。

**[Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code](classical_planning_with_llm-generated_heuristics_challenging_the_state_of_the_ar.md)**

:   提出让 LLM **生成域相关启发式函数的 Python 代码**（而非直接生成计划），通过 $n$ 次采样获得候选启发式池并在训练集上选优，将最优启发式注入 Python 规划器 Pyperplan 配合 GBFS 使用，在 IPC 2023 基准 8 个域上以纯 Python 实现超越了所有 C++ Fast Downward 传统启发式，且与 SOTA 学习型规划器 $h^{\mathrm{WLF}}_{\mathrm{GPR}}$ 持平，同时保证所有找到的计划 100% 正确。

**[CLAWS: Creativity Detection for LLM-Generated Solutions Using Attention Window of Sections](clawscreativity_detection_for_llm-generated_solutions_using_attention_window_of_.md)**

:   提出 CLAWS，通过分析 LLM 在生成数学解答时对不同 prompt 区段的注意力权重分布，无需人工评估即可将生成内容分类为"创造性"、"典型"或"幻觉"三类。

**[DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code](duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)**

:   提出 DuoLens，一种基于 CodeBERT + CodeBERTa 双编码器融合的 AI 生成内容检测框架，在多语言文本（8 种语言）和源代码（7 种编程语言）检测上以极低计算成本（延迟降低 8-12×，VRAM 降低 3-5×）实现 AUROC 0.97-0.99，远超 GPT-4o 等大模型。

**["Jutters"](jutters.md)**

:   通过荷兰传统"jutters"（海岸拾荒者）的隐喻，构建了一个融合真实海滩碎片与AI生成图像/视频的沉浸式装置艺术，引导参观者以拾荒者心态反思如何对待AI生成内容。

**[QiMeng-NeuComBack: Self-Evolving Translation from IR to Assembly Code](qimeng-neucomback_self-evolving_translation_from_ir_to_assembly_code.md)**

:   提出NeuComBack基准数据集用于评估IR到汇编的神经编译任务，并设计自进化提示优化方法，通过从LLM自调试轨迹中学习来迭代改进编译提示，使正确率从44%提升到64%，且87.5%的正确程序性能超越clang-O3。

**[Reasoning Compiler: LLM-Guided Optimizations for Efficient Model Serving](reasoning_compiler_llm-guided_optimizations_for_efficient_model_serving.md)**

:   提出 Reasoning Compiler，将编译器优化建模为序列决策过程，用 LLM 作为上下文感知提案引擎 + MCTS 平衡探索/利用，在 5 个代表性 benchmark 和 5 个硬件平台上实现平均 5.0× 加速且采样效率比 TVM 进化搜索提升 10.8×。

**[Synthesizing Performance Constraints for Evaluating and Improving Code Efficiency](synthesizing_performance_constraints_for_evaluating_and_improving_code_efficienc.md)**

:   提出Wedge框架——通过LLM合成性能刻画约束（performance-characterizing constraints）指导约束感知模糊测试，生成能暴露代码性能瓶颈的压力测试输入，构建PerfForge基准，使LLM代码优化器（如Effi-Learner）多减24% CPU指令。
