---
title: >-
  ICML2026 AIGC检测论文汇总 · 11篇论文解读
description: >-
  11篇ICML2026的 AIGC 检测方向论文解读，涵盖 LLM、对抗鲁棒、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "AIGC 检测"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对抗鲁棒"
  - "多模态"
item_list:
  - u: "autobaxbuilder_bootstrapping_code_security_benchmarking/"
    t: "AutoBaxBuilder: Bootstrapping Code Security Benchmarking"
  - u: "black-box_detection_of_llm-generated_text_using_generalized_jensen-shannon_diver/"
    t: "Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence"
  - u: "core_conflict-oriented_reasoning_for_general_multimodal_manipulation_detection/"
    t: "CORE: Conflict-Oriented Reasoning for General Multimodal Manipulation Detection"
  - u: "deep_residual_injection_for_full-spectrum_forensic_signal_perception_in_multimod/"
    t: "Deep Residual Injection for Full-Spectrum Forensic Signal Perception in Multimodal Large Language Models"
  - u: "dissect_and_prune_enhancing_robustness_in_ai-generated_image_detection/"
    t: "Dissect and Prune: Enhancing Robustness in AI-Generated Image Detection"
  - u: "distributional_open-ended_evaluation_of_llm_cultural_value_alignment_based_on_va/"
    t: "Distributional Open-Ended Evaluation of LLM Cultural Value Alignment Based on Value Codebook"
  - u: "feature-augmented_transformers_for_robust_ai-text_detection_across_domains_and_g/"
    t: "Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators"
  - u: "forensicconcept_transferable_forensic_concepts_for_aigi_detection/"
    t: "ForensicConcept: Transferable Forensic Concepts for AIGI Detection"
  - u: "generating_robust_portfolios_of_optimization_models_using_large_language_models/"
    t: "Generating Robust Portfolios of Optimization Models using Large Language Models"
  - u: "llm_self-recognition_steering_and_retrieving_activation_signatures/"
    t: "LLM Self-Recognition: Steering and Retrieving Activation Signatures"
  - u: "on_the_salience_of_low-probability_tokens_for_ai-generated_text_detection_a_mult/"
    t: "On the Salience of Low-Probability Tokens for AI-Generated Text Detection: A Multiscale Uncertainty Perspective"
item_total: 11
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔎 AIGC 检测

**🧪 ICML2026** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (10)](../../CVPR2026/aigc_detection/index.md) · [🔬 ICLR2026 (30)](../../ICLR2026/aigc_detection/index.md) · [💬 ACL2026 (17)](../../ACL2026/aigc_detection/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/aigc_detection/index.md) · [🧠 NeurIPS2025 (9)](../../NeurIPS2025/aigc_detection/index.md) · [💬 ACL2025 (15)](../../ACL2025/aigc_detection/index.md)

🔥 **高频主题：** LLM ×5 · 对抗鲁棒 ×3 · 多模态 ×2

**[AutoBaxBuilder: Bootstrapping Code Security Benchmarking](autobaxbuilder_bootstrapping_code_security_benchmarking.md)**

:   AUTOBAXBUILDER用LLM代理流水线自动生成Web后端安全评测场景、功能测试和端到端安全测试，把人工构建BAXBENCH式任务的成本降低约12倍，并构建出40个新场景的AUTOBAXBENCH来评估当代代码模型的正确性与安全性差距。

**[Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence](black-box_detection_of_llm-generated_text_using_generalized_jensen-shannon_diver.md)**

:   SurpMark 把"AI 文本检测"重构成似然无关假设检验：用代理 LM 算 token surprisal 后 k-means 离散成 k 个状态，估计一阶 Markov 转移矩阵，再用广义 Jensen-Shannon 散度（GJS）和预先建好的"人写 / 机写"参考转移矩阵比较，单次前向就给出黑盒、无需重训、无需 per-instance 重采样的判别分数。

**[CORE: Conflict-Oriented Reasoning for General Multimodal Manipulation Detection](core_conflict-oriented_reasoning_for_general_multimodal_manipulation_detection.md)**

:   作者把"多模态假新闻检测"重新定义为"显式捕获模态间或与世界知识之间的冲突"任务，构建了带细粒度冲突标注的 14k 语料 CAC，并提出 CORE 框架通过冲突感知训练（CPT）重塑 MLLM 的概念边界，使其在 DGM4、MDSM、MMFakeBench、NewsCLIPpings 四个数据集上以 100–750 个样本就大幅超过专用 SOTA。

**[Deep Residual Injection for Full-Spectrum Forensic Signal Perception in Multimodal Large Language Models](deep_residual_injection_for_full-spectrum_forensic_signal_perception_in_multimod.md)**

:   本文发现：把 MLLM 直接微调去学生成器留下的低级伪影，会破坏它早期形成的语义表征（灾难性遗忘）；于是提出 Deep-VRM——冻结早中层保住语义，只在 LLM 深层用一条 LoRA 旁路把伪影特征"残差注入"进去，让同一个 MLLM 不依赖任何外部专家检测器就拿下大多数 AIGI 基准的 SOTA。

**[Dissect and Prune: Enhancing Robustness in AI-Generated Image Detection](dissect_and_prune_enhancing_robustness_in_ai-generated_image_detection.md)**

:   针对现有 AI 生成图像（AIGI）检测器"看起来准、其实只会把图判成真"的预测不对称问题，本文提出 DEAR：用 inpainting 图像当探针、按通道激活与生成区域的对齐度（RAD）做"解剖"，再把两端极值通道双侧剪掉、只重训线性分类头，让检测器丢掉脆弱的捷径特征，在未见生成器与后处理下显著更鲁棒。

**[Distributional Open-Ended Evaluation of LLM Cultural Value Alignment Based on Value Codebook](distributional_open-ended_evaluation_of_llm_cultural_value_alignment_based_on_va.md)**

:   DOVE 用率失真变分优化从 1 万篇人类文本中自动构造紧凑的"价值码本"，再用不平衡最优传输度量人类与 LLM 长文本在价值空间上的分布差异，从而在 12 个 LLM 上把"评测—下游任务"相关性从基线 ≤24% 拉到 31.56%。

**[Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators](feature-augmented_transformers_for_robust_ai-text_detection_across_domains_and_g.md)**

:   本文在「单阈值固定协议」下系统暴露 AI 文本检测器在跨数据集/跨生成器 shift 下的脆弱性，并提出把可学注意力加权的手工语言特征与 transformer [CLS] 表征融合，配合 DeBERTa-v3 backbone，在 M4 多域多生成器基准上达到 85.9% balanced accuracy，比强 zero-shot 基线（Fast-DetectGPT、RADAR、Log-Rank）高最多 +7.22。

**[ForensicConcept: Transferable Forensic Concepts for AIGI Detection](forensicconcept_transferable_forensic_concepts_for_aigi_detection.md)**

:   针对 AI 生成图像（AIGI）检测器"在训练分布内很准、换个生成器就崩"且完全黑箱的问题，本文把检测器依赖的弥散证据显式抽成一本"取证概念码本"，再用扩散特征（CleanDIFT）作外部生成痕迹参照、用邻域结构一致性指标 CKNNA 度量骨干网证据与扩散痕迹的几何对齐度，并通过把扩散码本注入目标骨干网实现跨生成器迁移；GenImage 平均准确率 92.0%，且 CKNNA 越高迁移收益越大。

**[Generating Robust Portfolios of Optimization Models using Large Language Models](generating_robust_portfolios_of_optimization_models_using_large_language_models.md)**

:   本文提出一个轻量、无需训练的算法：用同一个 LLM 同时扮演"随机生成器"和"打分评审"两个角色，把生成概率前缀和达到 $1-\alpha$ 的候选优化模型打包成 portfolio，从理论上证明只要"生成器"或"评审"任一与人类偏好对齐，portfolio 就一定包含高质量优化模型，并在 NL4LP 上用 GPT 验证 portfolio 在最差情况下也稳定优于随机采样。

**[LLM Self-Recognition: Steering and Retrieving Activation Signatures](llm_self-recognition_steering_and_retrieving_activation_signatures.md)**

:   这篇论文不在 token 层加水印，而是在生成时往 LLM 残差流注入一个随机稀疏的转向向量，让模型自带可检测的"激活签名"，之后把文本回喂同一模型、从激活里用余弦相似度或轻量分类器把签名捞回来，在多种检测设定下达到 98% 以上准确率且几乎不损文本质量。

**[On the Salience of Low-Probability Tokens for AI-Generated Text Detection: A Multiscale Uncertainty Perspective](on_the_salience_of_low-probability_tokens_for_ai-generated_text_detection_a_mult.md)**

:   针对零样本 AI 生成文本检测里"高频 boilerplate 稀释信号"和"单点概率脆弱"两大痼疾，作者提出 Uncertainty / Uncertainty++ 检测器：只在每段文本底部 $\rho$ 分位的低概率 token 上聚合 log-prob，并叠加同一组位置上的 Rényi 熵作为分布形状信号，再在 12 个生成器、7 个数据集上把平均 AUROC 从 Lastde 的 86.49 推到 88.74，且在改写 / 改解码这类扰动下显著更稳。
