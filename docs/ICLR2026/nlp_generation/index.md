<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✍️ 文本生成

**🔬 ICLR2026** · 共 **4** 篇

**[AP-OOD: Attention Pooling for Out-of-Distribution Detection](ap-ood_attention_pooling_for_out-of-distribution_detection.md)**

:   提出AP-OOD，将Mahalanobis距离的均值池化替换为可学习的注意力池化，解决了均值池化丢失token级异常信息的问题，在文本OOD检测中将XSUM摘要的FPR95从27.84%降至4.67%，支持无监督到半监督的平滑过渡。

**[Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](paper2code_automating_code_generation_from_scientific_papers_in_machine_learning.md)**

:   提出 PaperCoder——一个多智能体 LLM 框架，通过规划（Planning）、分析（Analysis）、生成（Coding）三阶段流水线，将机器学习论文自动转化为可运行的代码仓库，其中 88% 的生成仓库被论文作者评为最佳，且在 PaperBench 基准上大幅超越基线。

**[Sharing State Between Prompts and Programs](sharing_state_between_prompts_and_programs.md)**

:   提出共享程序状态（shared program state）抽象，让 prompt 直接读写程序变量、操作堆对象和控制程序流程，实现为 Nightjar 系统（Python + prompt 混合编程），在保持或提升准确率（+4-19%）的同时减少 39.6% 代码量。

**[ShieldedCode: Learning Robust Representations for Virtual Machine Protected Code](shieldedcode_learning_robust_representations_for_virtual_machine_protected_code.md)**

:   提出 ShieldedCode——首个保护感知的代码表征学习框架，通过层次依赖建模（指令内/前序/跨指令三层）和联合功能感知+保护感知对比学习，使 LLM 能够生成、比较和推理虚拟机保护代码，在 VM 代码生成（Pass@1 26.95% vs. GPT-4o 22.58%）和二进制相似性检测上均超越现有方法。
