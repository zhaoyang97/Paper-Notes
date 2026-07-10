---
title: >-
  ECCV2026 AI安全论文汇总 · 7篇论文解读
description: >-
  7篇ECCV2026的 AI 安全方向论文解读，涵盖对抗鲁棒、少样本学习、个性化生成、视频生成、自监督学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2026"
  - "AI 安全"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
  - "少样本学习"
  - "个性化生成"
  - "视频生成"
  - "自监督学习"
item_list:
  - u: "a_classifier-agnostic_zero-shot_adversarial_attack_detection_via_clip/"
    t: "A Classifier-Agnostic Zero-Shot Adversarial Attack Detection via CLIP"
  - u: "exploiting_local_flatness_for_efficient_out_of_distribution_detection/"
    t: "Exploiting Local Flatness for Efficient Out-of-Distribution Detection"
  - u: "gradiaguard_a_gradient-based_shield_against_vjpeg-adversarial_attacks/"
    t: "Beyond IID: How General Are Tabular Foundation Models, Really?"
  - u: "improving_adversarial_robustness_via_activation_amplification_and_attenuation/"
    t: "Improving Adversarial Robustness via Activation Amplification and Attenuation"
  - u: "ireu_identity-related_encoder-only_unlearning_for_customized_portrait_generation/"
    t: "IREU: Identity-Related Encoder-Only Unlearning for Customized Portrait Generation"
  - u: "moiré_video_authentication_a_physical_signature_against_ai_video_generation/"
    t: "Moiré Video Authentication: A Physical Signature Against AI Video Generation"
  - u: "protofair_fair_self-supervised_contrastive_learning_via_pseudo-counterfactual_pa/"
    t: "ProtoFair: Fair Self-Supervised Contrastive Learning via Pseudo-Counterfactual Pairs"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI 安全

**🎞️ ECCV2026** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (145)](../../CVPR2026/ai_safety/index.md) · [🔬 ICLR2026 (139)](../../ICLR2026/ai_safety/index.md) · [💬 ACL2026 (5)](../../ACL2026/ai_safety/index.md) · [🧪 ICML2026 (114)](../../ICML2026/ai_safety/index.md) · [🤖 AAAI2026 (45)](../../AAAI2026/ai_safety/index.md) · [🧠 NeurIPS2025 (73)](../../NeurIPS2025/ai_safety/index.md)

🔥 **高频主题：** 对抗鲁棒 ×2

**[A Classifier-Agnostic Zero-Shot Adversarial Attack Detection via CLIP](a_classifier-agnostic_zero-shot_adversarial_attack_detection_via_clip.md)**

:   提出 A4D，一个完全黑盒、零样本的对抗攻击检测框架——利用 CLIP 对微小扰动的敏感性，通过将图像嵌入与一组精心设计的文本提示词比较余弦相似度，再经 PCA 聚合为单一检测分数，无需知道攻击类型或分类器结构即可检测对抗样本，在多个攻击/数据集/分类器上达到 SOTA。

**[Exploiting Local Flatness for Efficient Out-of-Distribution Detection](exploiting_local_flatness_for_efficient_out_of_distribution_detection.md)**

:   本文首次系统分析了 OOD 样本与 ID 样本在损失景观曲率上的差异，发现 OOD 输入的 Hessian 曲率更大且随分布偏移加剧而增大；基于此提出轻量级 OOD 检测器 Fold，用特征空间 Hessian 替代昂贵的参数空间曲率近似，配合部分特征归一化增强 ID-OOD 可分性，并用自监督 logit 遮蔽（AutoFold）自动标定归一化参数，在多个 benchmark 上平均 AUROC 提升 1.63%、FPR95 降低 2.30%，计算开销仅与单次前向传播相当。

**[Beyond IID: How General Are Tabular Foundation Models, Really?](gradiaguard_a_gradient-based_shield_against_vjpeg-adversarial_attacks.md)**

:   本文构建了首个统一、跨学科的表格数据基准 BeyondArena（含 142 个数据集、覆盖 IID/时序/分组任务），系统评测 11 个主流表格基础模型后发现：当前模型仅在中小规模 IID 数据上领先，而在非 IID、大规模、高维度场景中传统树模型和深度学习模型仍占主导。

**[Improving Adversarial Robustness via Activation Amplification and Attenuation](improving_adversarial_robustness_via_activation_amplification_and_attenuation.md)**

:   本文提出 A3（Activation Amplification and Attenuation），一个轻量的可学习激活缩放模块，通过同一组参数实现激活放大与衰减两种模式——训练时利用放大模式的降级预测作为负参考构造对比/排名损失，推理时仅用衰减模式即可在不引入显著计算开销的前提下提升对抗鲁棒性。

**[IREU: Identity-Related Encoder-Only Unlearning for Customized Portrait Generation](ireu_identity-related_encoder-only_unlearning_for_customized_portrait_generation.md)**

:   IREU 首次针对定制人像生成（CPG）提出身份遗忘问题，通过 Face-Swap 定位嵌入空间中的身份相关维度、仅在这些维度上做特征扰动，实现只更新图像编码器即可去除目标身份生成能力、同时保持其他身份生成保真度的遗忘管线，且遗忘后的编码器可零微调迁移到不同 CPG 生成器。

**[Moiré Video Authentication: A Physical Signature Against AI Video Generation](moiré_video_authentication_a_physical_signature_against_ai_video_generation.md)**

:   本文提出利用莫尔干涉条纹作为物理签名来认证视频真伪：将紧凑的双层光栅结构放置在拍摄场景中，通过计算条纹相位变化与光栅图像平移位移之间的 Pearson 相关系数区分真实视频与 AI 生成视频——真实视频中两者由光学几何定律严格耦合（相关系数均值 0.87），而当前最强视频生成模型（Veo 3.1、Grok Imagine、LTX-2）在最优配置下也无法精确复现这种耦合（相关系数均值 0.57），Cohen's d 达 1.71，差异高度显著。

**[ProtoFair: Fair Self-Supervised Contrastive Learning via Pseudo-Counterfactual Pairs](protofair_fair_self-supervised_contrastive_learning_via_pseudo-counterfactual_pa.md)**

:   ProtoFair 提出一种即插即用的公平性正则项，在不修改现有自监督对比学习目标的前提下，通过动量更新的无监督聚类原型发现「同语义内容但不同敏感组」的伪反事实对，在嵌入空间中拉近这些跨组样本，迫使编码器学习对敏感属性不变的表征；在 CelebA、UTKFace 和 NIH Chest X-rays 上搭配 SimCLR / SupCon / BarlowTwins / BYOL 均能显著降低 Equalized Odds 同时保持竞争力准确率。
