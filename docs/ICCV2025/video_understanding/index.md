---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎬 视频理解

**📹 ICCV2025** · 共 **3** 篇

**[4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](4dbench_benchmarking_multimodal_large_language_models_for_4d.md)**

:   提出 4D-Bench，首个评估多模态大语言模型对4D物体（具有时间演化的3D物体）理解能力的基准，包含4D物体问答（751 QA对）和4D物体描述（580物体×5标注）两大任务，发现即使SOTA的GPT-4o也仅达63%准确率（人类91%），揭示了MLLM在多视角时空理解上的巨大差距。

**[DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization](dollar_fewstep_video_generation_via_distillation_and_latent.md)**

:   结合变分分数蒸馏（VSD）和一致性蒸馏实现few-step视频生成，同时提出潜空间奖励模型微调方法进一步优化生成质量，4步生成的10秒视频（128帧@12FPS）在VBench上达82.57分超越teacher模型和Gen-3/Kling等基线，1步蒸馏实现278.6倍加速。

**[VACE: All-in-One Video Creation and Editing](vace_allinone_video_creation_and_editing.md)**

:   提出VACE统一视频生成和编辑框架，通过Video Condition Unit（VCU）将参考图→视频生成、视频→视频编辑、mask视频编辑等多种任务的输入统一为标准接口，配合Context Adapter注入时空条件信息，单一模型在各子任务上达到专用模型水平并支持灵活的任务组合。
