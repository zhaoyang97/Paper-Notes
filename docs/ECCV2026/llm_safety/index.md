---
title: >-
  ECCV2026 LLM安全论文汇总 · 5篇论文解读
description: >-
  5篇ECCV2026的 LLM 安全方向论文解读，涵盖多模态、对抗鲁棒、少样本学习、推理、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2026"
  - "LLM 安全"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "对抗鲁棒"
  - "少样本学习"
  - "推理"
  - "LLM"
item_list:
  - u: "learning_to_compose_revisiting_proxy_task_design_for_zero-shot_composed_image_re/"
    t: "Learning to Compose: Revisiting Proxy Task Design for Zero-Shot Composed Image Retrieval"
  - u: "neural_gate_mitigating_privacy_risks_in_lvlms_via_neuron-level_gradient_gating/"
    t: "Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating"
  - u: "reshift_aha-moment-driven_reasoning-level_backdoor_attacks_on_vision-language_mo/"
    t: "ReShift: Aha-Moment-Driven Reasoning-Level Backdoor Attacks on Vision-Language Models"
  - u: "slowba_an_efficiency_backdoor_attack_towards_vlm-based_gui_agents/"
    t: "SlowBA: An efficiency backdoor attack towards VLM-based GUI agents"
  - u: "towards_benign_memory_forgetting_for_selective_multimodal_large_language_model_u/"
    t: "Towards Benign Memory Forgetting for Selective Multimodal Large Language Model Unlearning"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔒 LLM 安全

**🎞️ ECCV2026** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (12)](../../CVPR2026/llm_safety/index.md) · [🔬 ICLR2026 (184)](../../ICLR2026/llm_safety/index.md) · [💬 ACL2026 (115)](../../ACL2026/llm_safety/index.md) · [🤖 AAAI2026 (41)](../../AAAI2026/llm_safety/index.md) · [🧠 NeurIPS2025 (81)](../../NeurIPS2025/llm_safety/index.md) · [📹 ICCV2025 (10)](../../ICCV2025/llm_safety/index.md)

🔥 **高频主题：** 多模态 ×3 · 对抗鲁棒 ×2

**[Learning to Compose: Revisiting Proxy Task Design for Zero-Shot Composed Image Retrieval](learning_to_compose_revisiting_proxy_task_design_for_zero-shot_composed_image_re.md)**

:   FoCo（Focus-then-Complete）将零样本组合图像检索（ZS-CIR）中的视觉-文本组合建模为可学习的"先聚焦后补全"两阶段过程，通过文本锚定的视觉聚合和上下文条件化的语义补全两个代理任务联合训练，配合跨实例对比损失防止捷径学习，在四个 ZS-CIR 基准上全面超越已有方法，且推理时不依赖 LLM。

**[Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating](neural_gate_mitigating_privacy_risks_in_lvlms_via_neuron-level_gradient_gating.md)**

:   本文提出 Neural Gate，一种神经元级梯度门控的 LVLM 隐私风险缓解方法。核心思路是先通过可学习向量在隐私主体特征上测量各神经元对隐私目标的贡献一致性，将神经元分为强激活、弱激活、非激活三类，再在模型编辑时仅对强激活隐私神经元的梯度做参数更新，其余神经元梯度被截断。在 MiniGPT 和 LLaVA 上，Neural Gate 在敏感查询拒绝率超过 94% 的同时几乎不损失通用任务性能，且在分布外隐私类别上展现出强泛化能力。

**[ReShift: Aha-Moment-Driven Reasoning-Level Backdoor Attacks on Vision-Language Models](reshift_aha-moment-driven_reasoning-level_backdoor_attacks_on_vision-language_mo.md)**

:   ReShift提出一种面向视觉语言模型的推理层后门攻击方法，利用强化学习激发的"啊哈时刻"（aha moment）认知行为，在推理轨迹中诱导可控转向，在保持推理逻辑连贯性的同时将预测结果重定向到预设目标答案，比传统输出层后门攻击更难被检测。

**[SlowBA: An efficiency backdoor attack towards VLM-based GUI agents](slowba_an_efficiency_backdoor_attack_towards_vlm-based_gui_agents.md)**

:   SlowBA 首次提出针对 VLM-based GUI agent 的效率后门攻击：通过两阶段奖励级后门注入（RBI），在 SFT 中教会 agent 生成超长回复、再在 RL 中用触发感知的奖励函数区分带触发器/干净输入，使 agent 在遇到弹窗类触发器时产生极高延迟响应，而任务准确率基本不变。

**[Towards Benign Memory Forgetting for Selective Multimodal Large Language Model Unlearning](towards_benign_memory_forgetting_for_selective_multimodal_large_language_model_u.md)**

:   提出"良性记忆遗忘"新范式与 S-MLLMUn Bench 评测基准，并设计 SMFA（Sculpted Memory Forgetting Adapter）框架，通过保留锚点引导的参数掩码精确抹除 MLLM 中的隐私敏感知识，同时不损伤模型的基础视觉理解能力。
