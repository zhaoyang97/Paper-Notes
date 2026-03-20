<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM 推理

**📷 CVPR2026** · 共 **3** 篇

**[FaceCoT: Chain-of-Thought Reasoning in MLLMs for Face Anti-Spoofing](facecot_cot_reasoning_face_anti_spoofing.md)**

:   构建了首个面向人脸反欺骗（FAS）的大规模 VQA 数据集 FaceCoT（108 万样本，覆盖 14 种攻击类型），包含六层级 CoT 推理标注（从全局描述到局部推理到最终结论）；同时提出 CoT-Enhanced Progressive Learning (CEPL) 两阶段训练策略，在 11 个基准数据集上平均 AUC 提升 4.06%、HTER 降低 5.00%，超越所有 SOTA 方法。

**[Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](red_rationale_enhanced_decoding_cot.md)**

:   发现现有 LVLM 在多模态 CoT 推理中会忽略生成的 rationale 内容（图像 token 主导注意力），提出 Rationale-Enhanced Decoding (RED)——将 CoT 重新表述为 KL 约束的 rationale 条件对数似然奖励最大化问题，最优解为将图像条件分布 $p(y|x,q)$ 和 rationale 条件分布 $p(y|r,q)^\lambda$ 相乘，无需训练即可显著提升多个基准上的推理性能。

**[VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models](visref_visual_refocusing_test_time_scaling.md)**

:   发现多模态推理模型在延长推理时会逐渐丢失对视觉token的注意力，提出VisRef在推理过程中主动重新注入与当前推理上下文语义相关的视觉token核心子集，在固定计算预算下比现有方法提升最高6.4%。
