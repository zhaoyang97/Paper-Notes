---
title: >-
  [论文解读] AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting
description: >-
  [ECCV2024][多模态VLM][MLLM安全] 提出AdaShield框架，通过精心设计的静态防御提示(AdaShield-S)和基于LLM的自适应迭代优化框架(AdaShield-A)，在不微调MLLM或训练额外模块的前提下，有效防御结构化越狱攻击，将攻击成功率从75%以上降至15%以下并保持正常任务性能。
tags:
  - "ECCV2024"
  - "多模态VLM"
  - "MLLM安全"
  - "越狱攻击防御"
  - "自适应提示"
  - "结构化攻击"
  - "黑盒防御"
---

# AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting

**会议**: ECCV2024  
**arXiv**: [2403.09513](https://arxiv.org/abs/2403.09513)  
**代码**: [https://github.com/rain305f/AdaShield](https://github.com/rain305f/AdaShield)  
**领域**: AI安全 / 多模态大模型防御  
**关键词**: [MLLM安全, 越狱攻击防御, 自适应提示, 结构化攻击, 黑盒防御]

## 一句话总结

提出AdaShield框架，通过精心设计的静态防御提示(AdaShield-S)和基于LLM的自适应迭代优化框架(AdaShield-A)，在不微调MLLM或训练额外模块的前提下，有效防御结构化越狱攻击，将攻击成功率从75%以上降至15%以下并保持正常任务性能。

## 研究背景与动机

**领域现状** 多模态大语言模型（MLLM）在视觉-语言推理方面取得了显著进展，但安全性问题日益突出。越狱攻击分为两类：扰动型攻击（在图像中添加对抗扰动）和结构化攻击（将恶意内容通过排版或文字嵌入图像）。前者已有成熟对策（如图像净化、对抗训练），后者因嵌入了具有语义意义的结构化信息，传统对抗防御方法几乎无效。

**现有痛点** 训练时对齐方法（如DRESS）需要大量高质量数据和计算资源。后处理过滤方法（如MLLMP）需要额外训练有害内容检测器，且推理时间开销显著（16.03秒 vs 正常9.40秒）。FigStep提出的简单防御提示效果有限，且缺乏场景适应性。此外，MLLMP的有害检测器泛化性差——在QR数据集的色情场景中仅4.34%准确率。

**核心矛盾** 有效防御需要对每种攻击场景定制安全规则，但手动设计无法覆盖所有场景；同时防御不能过度，否则会拒绝正常查询（过度防御问题）。

**本文目标** 如何自动、自适应地为MLLM生成防御提示，在无需微调的前提下同时实现高防御率和低过度防御。

**切入角度** 利用LLM自身能力作为防御提示生成器，通过与目标MLLM的对话式迭代优化自动生产场景化防御提示。

**核心 idea** 让LLM"守护者"通过观察目标模型的越狱失败案例，迭代学习生成针对不同攻击场景的防御提示，形成可检索的防御提示池。

## 方法详解

### 整体框架

AdaShield分为两个版本：AdaShield-S基于4个设计直觉手工设计一个通用静态防御提示 $P_s$，前置到模型输入中；AdaShield-A引入"守护者"LLM $D$ 与目标MLLM $M$ 的协作迭代框架，自动优化防御提示并生成多样化防御提示池 $\mathcal{P}$。推理时根据输入查询的CLIP嵌入相似度检索最匹配的防御提示。

### 关键设计

1. **基于4个直觉的静态防御提示设计(AdaShield-S)**:

    - 功能：手工设计一个涵盖完整检查流程的通用防御提示 $P_s$
    - 核心思路：融合4个设计直觉——(1)彻底检查图像内容是否含恶意文字/物品；(2)链式思维逐步检查指令是否有害；(3)明确指定应对恶意查询的响应方式（如"I am sorry"）；(4)包含处理良性查询的指令以避免过度防御。消融实验验证缺少任一直觉都导致ASR上升，其中直觉3最关键（$P_c$缺少明确响应指令导致CogVLM ASR从16%飙升至75%）
    - 设计动机：结构化攻击的核心在于通过图像绕过安全对齐，因此防御必须引导模型主动审查图像中的文本/语义内容

2. **自适应防御提示自动优化框架(AdaShield-A)**:

    - 功能：自动为不同攻击场景生成定制化防御提示池，推理时自适应检索最佳提示
    - 核心思路：训练阶段5步流程——(1)收集少量恶意样本喂入目标模型 $M$ 获取越狱响应；(2)守护者 $D$（Vicuna-13B）基于失败提示和越狱响应自动生成改进提示；(3)关键词匹配判断新响应是否仍为越狱；(4)若失败则迭代优化；(5)在验证集上筛选泛化性好的提示（ASR阈值 $\alpha=0.8$），用GPT-4改写增加多样性。推理时通过CLIP文本+图像嵌入拼接后计算余弦相似度检索：$Q_{\text{best}}, P_{\text{best}} = \{Q_i, P_i | \arg\max_i \cos(z_t, z_i) \text{ and } \max \cos(z_t, z_i) > \beta\}$，当最大相似度低于阈值 $\beta=0.7$ 时判定为良性查询不使用防御提示
    - 设计动机：单一通用提示无法覆盖法律、经济、医疗等复杂攻击场景，自动化框架可扩展到任意场景且适用于黑盒模型（MLMaaS）

### 损失函数 / 训练策略

AdaShield不涉及模型参数训练。防御提示优化通过对话式迭代完成：守护者 $D$ 接收系统提示+失败案例生成新提示，目标模型 $M$ 评估防御效果。关键超参数：验证集ASR阈值 $\alpha=0.8$，检索相似度阈值 $\beta=0.7$。评估指标包括关键词ASR和GPT Recheck ASR两种。

## 实验关键数据

### 主实验

| 模型 | 方法 | QR ASR↓ | QR Recheck↓ | FigStep ASR↓ | FigStep Recheck↓ | MM-Vet Total↑ |
|------|------|---------|-------------|-------------|-----------------|-------------|
| LLaVA-1.5 | Vanilla | 75.75 | 67.71 | 70.47 | 87.21 | 36.8 |
| LLaVA-1.5 | FSD | 69.50 | 59.38 | 64.88 | 80.93 | 33.1 |
| LLaVA-1.5 | MLLMP | 77.96 | 64.69 | 73.72 | 76.51 | 36.3 |
| LLaVA-1.5 | AdaShield-S | 24.43 | 20.61 | 26.05 | 35.58 | 35.2 |
| LLaVA-1.5 | AdaShield-A | **15.22** | **15.43** | **10.47** | **22.33** | 36.3 |
| CogVLM | AdaShield-A | **1.37** | **1.43** | **0.00** | **0.00** | **51.0** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 随机检索 vs AdaShield-A检索 | 18.2% vs 15.2% (LLaVA QR ASR) | 自适应检索优于随机选择 |
| $P_s$ vs $P_v$ (去除良性指令) | 35.2 vs 29.8 (LLaVA MM-Vet) | 验证Intuition 4防止过度防御 |
| $P_a$ (无图像检查) | 39.6-53.6% ASR | 验证Intuition 1图像检查关键性 |
| $P_c$ (无明确响应指令) | 62-81% ASR | 验证Intuition 3响应指令最关键 |

### 关键发现

- AdaShield-A在CogVLM上实现FigStep零ASR，同时MM-Vet分数（51.0）超过Vanilla（50.0）
- 推理时间大幅优化：AdaShield-A在有害查询上仅需1.46秒 vs MLLMP的16.03秒
- AdaShield-A在未见场景下仍有良好泛化：Easy训练→Hard测试ASR仅27.38%
- 过度防御问题通过相似度阈值 $\beta$ 有效缓解，低相似度查询不附加防御提示

## 亮点与洞察

- 思路简洁优雅：用LLM自身能力来保护LLM，defense prompt作为"安全护盾"前置于输入
- 四大设计直觉经过严谨消融验证，每个直觉的贡献明确可量化
- 完全不需要微调、不需要训练额外模块、适用于黑盒API服务，实际部署门槛极低
- 自适应检索机制兼顾了防御效果（匹配最相关场景的提示）和正常使用（低相似度不触发防御）

## 局限与展望

- 仅评估FigStep和QR两种结构化攻击，对更复杂的混合攻击未验证
- 防御提示池基于有限训练样本构建，覆盖度受限
- 依赖关键词匹配判断越狱是否成功，可能被更微妙的有害回复绕过
- CLIP嵌入检索可能对语义相近但意图不同的查询产生误匹配
- 在MiniGPT-v2上AdaShield-S导致正常任务性能骤降（36.8→1.4），过度防御问题在弱模型上更严重

## 相关工作与启发

本文在MLLM安全领域开辟了"prompt-based defense"这一新路线，与fine-tuning路线（DRESS）和post-hoc路线（MLLMP）形成三足鼎立。对后续工作的启发：(1)防御提示的自动优化可以扩展到对抗型攻击；(2)守护者-目标双模型协作模式可用于更广泛的安全对齐场景；(3)检索增强的防御策略有望与RAG技术融合。

## 评分

- 新颖性: ⭐⭐⭐⭐ 自适应防御提示概念新颖，守护者-目标协作框架有开创性
- 实验充分度: ⭐⭐⭐⭐ 多模型多攻击全面评估，消融实验验证每个设计选择
- 写作质量: ⭐⭐⭐⭐ 动机清晰，直觉驱动的方法论述有说服力
- 价值: ⭐⭐⭐⭐⭐ 零成本部署、黑盒兼容、效果显著，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Attention Prompting on Image for Large Vision-Language Models](attention_prompting_on_image_for_large_visionlanguage_models.md)
- [\[ACL 2025\] Adaptive Linguistic Prompting (ALP) Enhances Phishing Webpage Detection in Multimodal Large Language Models](../../ACL2025/multimodal_vlm/adaptive_linguistic_prompting_alp_enhances_phishing_webpage_detection_in_multimo.md)
- [\[ECCV 2024\] FreeMotion: MoCap-Free Human Motion Synthesis with Multimodal Large Language Models](freemotion_mocap-free_human_motion_synthesis_with_multimodal_large_language_mode.md)
- [\[ECCV 2024\] Meta-Prompting for Automating Zero-Shot Visual Recognition with LLMs](meta-prompting_for_automating_zero-shot_visual_recognition_with_llms.md)
- [\[AAAI 2026\] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)

</div>

<!-- RELATED:END -->
