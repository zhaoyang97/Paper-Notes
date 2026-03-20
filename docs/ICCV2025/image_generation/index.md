<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎨 图像生成

**📹 ICCV2025** · 共 **6** 篇

**[Aether: Geometric-Aware Unified World Modeling](aether_geometricaware_unified_world_modeling.md)**

:   提出Aether统一框架，通过任务交错特征学习联合优化4D动态重建、动作条件视频预测和目标条件视觉规划三个核心能力，实现geometry-aware的世界建模，纯合成数据训练即可零样本泛化到真实世界。

**[Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences](cycle_consistency_as_reward_learning_imagetext_alignment_wit.md)**

:   提出CycleReward，利用cycle consistency作为自监督信号替代人工偏好标注——将caption用T2I模型重建为图像再比较相似度来排序，构建866K偏好对数据集CyclePrefDB，训练的奖励模型在detailed captioning上比HPSv2/PickScore/ImageReward高6%+，且DPO训练后提升VLM在多个VL任务上的性能，无需任何人工标注。

**[Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation](dense2moe_restructuring_diffusion_transformer_to_moe_for_eff.md)**

:   首次将预训练的dense DiT（如FLUX.1）转换为Mixture-of-Experts结构实现结构化稀疏推理，通过Taylor度量专家初始化+知识蒸馏+Mixture-of-Blocks进一步稀疏化，在激活参数减少60%的同时保持原始生成质量，全面超越剪枝方法。

**[REPA-E: Unlocking VAE for End-to-End Tuning of Latent Diffusion Transformers](repae_unlocking_vae_for_endtoend_tuning_of_latent_diffusion.md)**

:   回答了"潜空间扩散模型能否与VAE端到端联合训练"的基础问题——发现标准扩散loss无法端到端训练但表示对齐（REPA）loss可以，提出REPA-E实现VAE+DiT联合训练，训练速度比REPA快17倍、比vanilla快45倍，在ImageNet 256×256上达到1.12 FID（w/ CFG）的新SOTA。

**[SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](sanasprint_onestep_diffusion_with_continuoustime_consistency.md)**

:   将预训练的SANA flow matching模型通过无损数学变换转化为TrigFlow，结合连续时间一致性蒸馏（sCM）和潜空间对抗蒸馏（LADD）的混合策略，实现统一的1-4步自适应高质量图像生成，1步生成1024×1024图像仅需0.1s（H100），以7.59 FID和0.74 GenEval超越FLUX-schnell且速度快10倍。

**[VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning](visualcloze_a_universal_image_generation_framework_via_visua.md)**

:   提出VisualCloze，将多种图像生成任务（编辑、翻译、超分、风格化等）统一为"视觉完形填空"范式——用视觉示例（而非文本指令）定义任务，通过图像infilling模型实现统一生成，并构建Graph200K数据集增强任务间知识迁移，支持域内任务、未见任务泛化、多任务组合和反向生成。
