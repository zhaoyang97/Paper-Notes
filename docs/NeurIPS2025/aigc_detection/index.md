<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔎 AIGC 检测

**🧠 NeurIPS2025** · 共 **6** 篇

**[ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text](asciibench_evaluating_language-model-based_understanding_of_visually-oriented_te.md)**

:   提出 ASCIIBench，首个用于评估 LLM 对 ASCII 艺术的生成和分类能力的基准数据集（5,315 张 ASCII 图像，752 类），发现当前 LLM 在需要空间/位置推理的 ASCII 任务上仍有显著局限，且 CLIP 嵌入在大多数 ASCII 类别上的区分能力接近随机水平。

**[Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code](classical_planning_with_llm-generated_heuristics_challenging_the_state_of_the_ar.md)**

:   让 LLM 为经典规划问题生成 Python 启发式函数代码，从 n 个候选中选最优，在 IPC 2023 基准上用纯 Python 规划器超越了 C++ 实现的 SOTA 启发式（如 hFF），且保证所有计划正确。

**[DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code](duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)**

:   提出 DuoLens，一种基于 CodeBERT + CodeBERTa 双编码器融合的 AI 生成内容检测框架，在多语言文本（8 种语言）和源代码（7 种编程语言）检测上以极低计算成本（延迟降低 8-12×，VRAM 降低 3-5×）实现 AUROC 0.97-0.99，远超 GPT-4o 等大模型。

**["Jutters"](jutters.md)**

:   通过荷兰传统"jutters"（海岸拾荒者）的隐喻，构建了一个融合真实海滩碎片与AI生成图像/视频的沉浸式装置艺术，引导参观者以拾荒者心态反思如何对待AI生成内容。

**[Reasoning Compiler: LLM-Guided Optimizations for Efficient Model Serving](reasoning_compiler_llm-guided_optimizations_for_efficient_model_serving.md)**

:   提出 Reasoning Compiler，将编译器优化建模为序列决策过程，用 LLM 作为上下文感知提案引擎 + MCTS 平衡探索/利用，在 5 个代表性 benchmark 和 5 个硬件平台上实现平均 5.0× 加速且采样效率比 TVM 进化搜索提升 10.8×。

**[Synthesizing Performance Constraints for Evaluating and Improving Code Efficiency](synthesizing_performance_constraints_for_evaluating_and_improving_code_efficienc.md)**

:   提出Wedge框架——通过LLM合成性能刻画约束（performance-characterizing constraints）指导约束感知模糊测试，生成能暴露代码性能瓶颈的压力测试输入，构建PerfForge基准，使LLM代码优化器（如Effi-Learner）多减24% CPU指令。
