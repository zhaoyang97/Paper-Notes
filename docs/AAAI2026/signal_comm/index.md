<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📡 信号/通信

**🤖 AAAI2026** · 共 **5** 篇

**[Beyond Perplexity: Let the Reader Select Retrieval Summaries via Spectrum Projection Score](beyond_perplexity_let_the_reader_select_retrieval_summaries_via_spectrum_project.md)**

:   提出 Spectrum Projection Score (SPS) 这一无需训练的指标，通过衡量摘要 token 嵌入与 reader LLM 主子空间的对齐程度来评估检索摘要质量，替代传统困惑度指标。结合 xCompress 推理时控制器，在 5 个 QA 数据集上显著优于基于困惑度的方法（HotpotQA EM +3.6）。

**[GateRA: Token-Aware Modulation for Parameter-Efficient Fine-Tuning](gatera_token-aware_modulation_for_parameter-efficient_fine-tuning.md)**

:   提出 GateRA，在 PEFT 方法（LoRA/DoRA/HiRA）中引入轻量级 token 感知门控模块，通过 sigmoid 门控动态调整每个 token 的适配强度——对分布内/简单 token 抑制更新以保留预训练知识，对挑战性 token 放大适配。结合熵正则化促进近二值门控决策，在常识推理（+1.1%）、对话和数学推理上一致优于 HiRA。

**[Task Aware Modulation Using Representation Learning For Upsaling Of Terrestrial ](task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)**

:   提出 TAM-RL 框架，将陆地碳通量升尺度问题建模为零样本回归迁移学习任务，用 BiLSTM 任务编码器+FiLM 调制结合碳平衡方程知识引导损失，在 150+ 通量塔站点上将 GPP RMSE 降低 9.6%、NEE R² 提升 43.8%（相较 FLUXCOM-X-BASE）。

**[Text-Guided Channel Perturbation And Pretrained Knowledge Integration For Unifie](text-guided_channel_perturbation_and_pretrained_knowledge_integration_for_unifie.md)**

:   提出 UP-Fusion 统一多模态图像融合框架，通过语义感知通道剪枝 (SCPM)、几何仿射调制 (GAM) 和 CLIP 文本引导通道扰动 (TCPM) 三个模块，用单组权重（仅在红外-可见光数据上训练）同时处理 IVIF 和医学图像融合，在两类任务上均达到 SOTA。

**[Toward Gaze Target Detection in Young Autistic Children](toward_gaze_target_detection_of_young_autistic_children.md)**

:   针对自闭症儿童注视目标检测中面部注视（6.6%）严重不足的类别不平衡问题，提出 Socially Aware Coarse-to-Fine (SACF) 框架，用微调的 Qwen2.5-VL 作为社交上下文感知门控，将输入路由到社交感知/社交无关两个专家模型，在首创的 AGT 数据集上显著提升了面部注视检测性能（Face L2 在 Sharingan 上降低 13.9%, F1 从 0.753 提升至 0.761）。
