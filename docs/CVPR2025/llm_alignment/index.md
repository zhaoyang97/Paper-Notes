---
title: >-
  CVPR2025 对齐/RLHF论文汇总 · 5篇论文解读
description: >-
  5篇CVPR2025的对齐 / RLHF 方向论文解读，涵盖对齐/RLHF、LLM、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "对齐 / RLHF"
  - "论文解读"
  - "论文笔记"
  - "对齐/RLHF"
  - "LLM"
  - "多模态"
item_list:
  - u: "bases_of_steerable_kernels_for_equivariant_cnns_from_2d_rotations_to_the_lorentz/"
    t: "Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group"
  - u: "cad-llama_leveraging_large_language_models_for_computer-aided_design_parametric_/"
    t: "CAD-Llama: Leveraging Large Language Models for Computer-Aided Design Parametric 3D Model Generation"
  - u: "continual_sft_matches_multimodal_rlhf_with_negative_supervision/"
    t: "Continual SFT Matches Multimodal RLHF with Negative Supervision"
  - u: "do_we_really_need_curated_malicious_data_for_safety_alignment_in_multi-modal_lar/"
    t: "Do We Really Need Curated Malicious Data for Safety Alignment in Multi-Modal LLMs?"
  - u: "jailbreaking_the_non-transferable_barrier_via_test-time_data_disguising/"
    t: "Jailbreaking the Non-Transferable Barrier via Test-Time Data Disguising"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐 / RLHF

**📷 CVPR2025** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (12)](../../CVPR2026/llm_alignment/index.md) · [🔬 ICLR2026 (102)](../../ICLR2026/llm_alignment/index.md) · [💬 ACL2026 (38)](../../ACL2026/llm_alignment/index.md) · [🧪 ICML2026 (37)](../../ICML2026/llm_alignment/index.md) · [🤖 AAAI2026 (17)](../../AAAI2026/llm_alignment/index.md) · [🧠 NeurIPS2025 (36)](../../NeurIPS2025/llm_alignment/index.md)

🔥 **高频主题：** 对齐/RLHF ×2

**[Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group](bases_of_steerable_kernels_for_equivariant_cnns_from_2d_rotations_to_the_lorentz.md)**

:   提出一种求解可转向等变 CNN 核约束方程的替代方法，通过在不动点处求解更简单的不变性条件再"转向"到任意点，绕过了计算 Clebsch-Gordan 系数的需要，为 SO(2)、O(2)、SO(3)、O(3) 及 Lorentz 群给出了显式的核基底公式。

**[CAD-Llama: Leveraging Large Language Models for Computer-Aided Design Parametric 3D Model Generation](cad-llama_leveraging_large_language_models_for_computer-aided_design_parametric_.md)**

:   本文提出 CAD-Llama 框架，通过层次化标注管线将 3D CAD 模型转化为富含语义描述的 Python 风格代码（SPCC），再用自适应预训练和指令微调将 LLaMA3-8B 转化为参数化 CAD 模型生成器，在 text-to-CAD 任务上精度超出先前方法约 14%，并支持补全、添加、删除等多种 CAD 编辑任务。

**[Continual SFT Matches Multimodal RLHF with Negative Supervision](continual_sft_matches_multimodal_rlhf_with_negative_supervision.md)**

:   通过梯度分析发现多模态 RLHF 相比持续 SFT 的核心优势在于 rejected response 中的负监督信号，据此提出 nSFT 方法，用 LLM 从拒绝回复中提取错误信息并构造纠正性对话数据，仅用 SFT loss 就能匹配甚至超越 DPO/PPO 等 RLHF 方法，且只需 1 个模型，显存效率大幅提升。

**[Do We Really Need Curated Malicious Data for Safety Alignment in Multi-Modal LLMs?](do_we_really_need_curated_malicious_data_for_safety_alignment_in_multi-modal_lar.md)**

:   探讨多模态大语言模型安全对齐是否真正需要精心策划的恶意数据，发现利用现有良性数据并结合简单的安全微调策略即可实现有效的安全对齐，大幅降低了安全对齐的数据成本。

**[Jailbreaking the Non-Transferable Barrier via Test-Time Data Disguising](jailbreaking_the_non-transferable_barrier_via_test-time_data_disguising.md)**

:   提出 JailNTL，首个针对 Non-Transferable Learning (NTL) 模型的黑盒攻击方法，通过测试时数据伪装将未授权域的数据"变装"为授权域的数据，仅用 1% 授权样本即可将未授权域准确率提升最高 55.7%，无需修改模型。
