<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI 安全

**🤖 AAAI2026** · 共 **10** 篇

**[Alternative Fairness and Accuracy Optimization in Criminal Justice](alternative_fairness_and_accuracy_optimization_in_criminal_j.md)**

:   本文系统综述了算法公平性的三大维度（群体公平、个体公平、过程公平），提出了一种基于容差约束的改进群体公平性优化公式，并构建了面向公共决策系统的"公平三支柱"部署框架。

**[An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses](an_improved_privacy_and_utility_analysis_of_differentially_p.md)**

:   在仅假设损失函数L-光滑（不需要凸性）的条件下，为DPSGD推导出了更紧的闭式RDP隐私界，并首次在有界域场景下给出了完整的收敛性/效用分析，揭示了较小的参数域直径可以同时改善隐私和效用。

**[An Information Theoretic Evaluation Metric for Strong Unlearning](an_information_theoretic_evaluation_metric_for_strong_unlear.md)**

:   提出 Information Difference Index (IDI)，一种基于信息论的白盒评估指标，通过度量中间层特征与遗忘标签之间的互信息来衡量机器遗忘的彻底程度，揭示了现有黑盒指标（MIA、JSD等）无法捕捉的中间层残留信息问题，并提出 COLA 方法在特征层面消除残余信息。

**[An Information Theoretic Evaluation Metric for Strong Unlearning](an_information_theoretic_evaluation_metric_for_strong_unlearning.md)**

:   揭示现有黑盒遗忘评估指标（MIA/JSD等）的根本缺陷——仅修改最后一层即可满足所有黑盒指标但中间层完整保留遗忘数据信息，提出IDI白盒指标通过InfoNCE估计各层与遗忘标签的互信息差异来量化遗忘效果，并提出COLA方法在CIFAR-10/100和ImageNet-1K上实现接近Retrain的IDI得分。

**[An LLM-Based Simulation Framework for Embodied Conversational Agents in Psychological Counseling](an_llm-based_simulation_framework_for_embodied_conversationa.md)**

:   提出 ECAs 框架，基于认知行为治疗(CBT)等心理学理论，利用 LLM 将真实咨询案例扩展为具身认知记忆空间，模拟心理咨询中来访者的完整认知过程，生成高保真度的咨询对话数据，在专家评估和自动评估中均显著优于基线。

**[Angular Gradient Sign Method: Uncovering Vulnerabilities in Hyperbolic Networks](angular_gradient_sign_method_uncovering_vulnerabilities_in_h.md)**

:   提出Angular Gradient Sign Method (AGSM)，将双曲空间中的梯度分解为径向（层次深度）和角度（语义）分量，仅沿角度方向施加扰动来生成对抗样本，在图像分类和跨模态检索任务上比标准FGSM/PGD多降低5-13%的准确率。

**[Argumentative Debates for Transparent Bias Detection (ABIDE)](argumentative_debates_for_transparent_bias_detection_technic.md)**

:   提出ABIDE框架，将偏见检测过程结构化为基于量化二极论辩框架（QBAF）的辩论：通过邻域级局部统计公平性（neighbourhood-based local statistical parity）生成偏见论据，利用批判性问题（critical questions）作为攻击机制挑战不可靠论据，在合成/真实/LLM模型上均优于IRB基线。

**[AUVIC: Adversarial Unlearning of Visual Concepts for Multi-modal Large Language Models](auvic_adversarial_unlearning_of_visual_concepts_for_multi-mo.md)**

:   提出AUVIC框架，通过对抗性扰动生成器 + 动态锚点保留机制，在MLLM中精确遗忘目标视觉概念（如特定人脸），同时避免对语义相似概念的附带遗忘，并构建了首个面向群体场景视觉概念遗忘的评测基准VCUBench。

**[DeepTracer: Tracing Stolen Model via Deep Coupled Watermarks](deeptracer_tracing_stolen_model_via_deep_coupled_watermarks.md)**

:   提出DeepTracer鲁棒水印框架，通过自适应源类选择（K-Means聚类覆盖特征空间）+ 同类耦合损失（拉近水印样本与目标类在输出空间的距离）+ 两阶段关键样本过滤，使水印任务与主任务深度耦合，在6种模型窃取攻击（含hard-label和data-free）下水印成功率平均达77-100%，远超现有方法。

**[Truth, Justice, and Secrecy: Cake Cutting Under Privacy Constraints](truth_justice_and_secrecy_cake_cutting_under_privacy_constraints.md)**

:   首个隐私保护蛋糕切割协议，在保持无嫉妒性和策略防谋性的同时，通过秘密共享和安全多方计算（MPC）技术确保参与者的估值函数不被泄露。
