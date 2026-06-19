---
title: >-
  [论文解读] Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models
description: >-
  [ICML 2025][LLM安全][MLLM隐私] 揭示多模态大语言模型（MLLM）在微调过程中会不经意地记忆与训练任务完全无关的私密内容（如随机水印），这种记忆源于 mini-batch 内的虚假相关性，并提出基于层级探针的检测框架证明模型内部表示已编码此类信息，即使模型输出不直接显示。 领域现状：多模态大语言模型（ML…
tags:
  - "ICML 2025"
  - "LLM安全"
  - "MLLM隐私"
  - "无意记忆"
  - "任务无关内容"
  - "水印探测"
  - "mini-batch虚假相关"
  - "层级探测"
---

# Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models

**会议**: ICML 2025  
**arXiv**: [2503.01208](https://arxiv.org/abs/2503.01208)  
**代码**: [https://github.com/illusionhi/ProbingPrivacy](https://github.com/illusionhi/ProbingPrivacy)  
**领域**: AI安全  
**关键词**: MLLM隐私, 无意记忆, 任务无关内容, 水印探测, mini-batch虚假相关, 层级探测

## 一句话总结
揭示多模态大语言模型（MLLM）在微调过程中会不经意地记忆与训练任务完全无关的私密内容（如随机水印），这种记忆源于 mini-batch 内的虚假相关性，并提出基于层级探针的检测框架证明模型内部表示已编码此类信息，即使模型输出不直接显示。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLM）如 LLaVA、InternVL 等在 VQA 等视觉语言任务中表现出色，但其训练数据不可避免地包含隐私敏感信息。现有隐私泄露研究（模型提取攻击、成员推理攻击 MIA）主要关注与训练目标自然对齐的敏感内容。

**现有痛点**：先前工作聚焦于"任务相关"的隐私内容——文本模态中的个人信息通过 next-token prediction 自然被记忆，视觉模态中的私人图像特征也与主任务紧密关联。但一个被忽视的问题是：与训练目标完全无关的隐私内容是否也会被模型记忆？

**核心矛盾**：从全局训练目标来看，任务无关内容不应影响模型学习；但在 partial mini-batch 训练动态中，这些内容可能与 VQA 的输出产生虚假相关性，导致模型通过正常的梯度更新将其编码到参数中——这构成了一种全新的、难以预见的隐私风险。

**本文目标**：回答三个核心问题——(RQ1) 随机任务无关内容是否影响微调动态？(RQ2) 模型是否在内部表示中编码了这些无关内容？(RQ3) 这种编码在不同层中如何分布？

**切入角度**：以可控的方式注入完全随机的水印（task-irrelevant watermark）作为模拟的"私密内容"，系统性地研究其被记忆的机制和可检测性。

**核心 idea**：设计水印注入实验 + 层级线性探测框架，证明即使内容与任务完全无关，MLLM 也会在其隐状态中编码这些信息。

## 方法详解

### 整体框架

论文提出了一个两阶段的实验与探测框架：

- **阶段一：受控水印注入 (Controlled Watermark Injection)**：在 VQA 微调数据集的图像上，以不同概率（如 10%、30%、50%、100%）嵌入随机生成的任务无关水印内容。水印文本是随机生成的字符串，与 VQA 的问答对完全无关。然后正常执行 VQA 微调训练。
- **阶段二：隐私探测 (Privacy Probing)**：训练完成后，使用线性探针（linear probe）对模型每一层的隐状态进行探测，判断模型是否已在内部表示中编码了水印内容。同时分析不同水印注入概率对训练损失、收敛行为、下游任务性能的影响。

关键洞察：在一个 mini-batch 中，如果某些图像包含相同的水印而其他图像不包含，那么水印的有无可能与该 batch 中特定 VQA 问答对形成虚假相关（spurious correlation），驱动模型在梯度下降过程中不自觉地将水印信息纳入参数。

### 关键设计

1. **水印注入策略 (Watermark Injection)**:

    - 功能：在 VQA 图像上叠加随机文本水印，模拟任务无关的私密内容
    - 核心思路：使用随机生成的文本字符串作为水印，以不同概率 $p \in \{0.1, 0.3, 0.5, 1.0\}$ 嵌入到微调图像中。水印内容完全随机，确保与 Q-A 对无语义关联
    - 设计动机：现实中用户相册中的私人照片可能包含各种背景信息（如路牌、证件号、文件片段），这些内容对 VQA 任务来说是噪声，但可能被模型编码

2. **Mini-batch 虚假相关性分析**:

    - 功能：分析为什么全局无关的内容在 mini-batch 尺度上变得"相关"
    - 核心思路：在一个 mini-batch 内，含有相同水印的样本子集共享一个额外的视觉特征（水印），SGD 的梯度更新倾向于将这个特征与该子集样本的答案关联起来。虽然跨 batch 后这种关联在统计上是虚假的，但在有限次训练中模型并不能自动消除这种记忆
    - 设计动机：揭示 mini-batch SGD 的固有局限性——全局梯度的无偏估计不等于模型不会记忆局部信息

3. **层级线性探针 (Layer-wise Linear Probing)**:

    - 功能：对训练完成的 MLLM 的每一层隐状态训练线性分类器，探测水印是否可从内部表示中恢复
    - 核心思路：令 $h_l$ 为第 $l$ 层的隐状态表示，训练线性分类器 $f_l(h_l) = W_l h_l + b_l$ 来预测输入图像是否包含水印（或水印内容），如果特定层的分类准确率显著高于随机水平，则证明该层已编码水印信息
    - 设计动机：直接 prompt 模型可能无法引出水印信息（模型不会在输出中展示水印），但线性探针可以揭示隐藏在表示空间中的信息编码

4. **训练行为差异分析**:

    - 功能：对比有/无水印条件下模型的训练损失曲线和收敛特性
    - 核心思路：观测注入不同比例水印后训练 loss 的变化趋势、收敛速度差异、验证集性能波动
    - 设计动机：如果任务无关内容真的影响了训练动态，那么训练 loss 和性能指标应当呈现可辨识的差异模式

## 实验关键数据

### 训练动态分析（RQ1）

不同水印注入比例下 MLLM 的微调训练行为差异：

| 水印注入概率 | 训练 Loss 变化趋势 | 下游 VQA 性能影响 | 收敛行为 |
|------------|------------------|-----------------|---------|
| 0%（基线） | 正常下降，平稳收敛 | 基线水平 | 标准收敛 |
| 10% | 轻微波动 | 几乎无影响 | 近正常 |
| 30% | 明显不同训练模式 | 轻微下降 | 出现差异 |
| 50% | 显著不同训练曲线 | 可观测下降 | 明显延迟 |
| 100% | 训练动态大幅改变 | 性能显著受损 | 收敛困难 |

论文关键发现：即使仅有 10%-30% 的图像含有水印，模型训练动态已经产生可测量的偏移，说明"任务无关内容不影响训练"的直觉是错误的。

### 层级探针检测结果（RQ2 & RQ3）

层级线性探针在不同条件下检测水印编码的准确率表现：

| 模型层级 | 含水印图探针准确率 | 无水印图探针准确率 | 差异解读 |
|---------|-----------------|-----------------|---------|
| 浅层（前 1/4） | 中等 | ~随机水平 | 浅层开始出现编码迹象 |
| 中间层（1/4-1/2） | 较高 | ~随机水平 | 水印信息编码增强 |
| 深层（1/2-3/4） | 最高 | ~随机水平 | 编码峰值区域 |
| 输出层附近 | 有所下降 | ~随机水平 | 信息被输出投射部分过滤 |

关键结论：
- 即使水印内容不影响模型的 VQA 输出，线性探针仍能以显著高于随机的准确率从模型中间层恢复水印信息
- MLLM 在遇到之前"见过"的任务无关知识时，触发了与未见过内容截然不同的表示模式
- 这证明了"不泄露 ≠ 未记忆"——模型可能在输出层不显示私密信息，但已将其编码在内部表示中

## 亮点与洞察

- **颠覆直觉的发现**：任务无关内容也会被记忆，这比任务相关隐私泄露更隐蔽、更难防御。传统的隐私保护策略（如输出过滤）无法解决内部表示的记忆问题
- **Mini-batch 虚假相关性机制**的阐释非常深刻：将隐私风险与随机梯度下降的内在性质联系起来，揭示了一个系统性的、与训练方法本身绑定的漏洞
- **层级探针方法论**为隐私审计提供了新工具：可以作为模型部署前的隐私诊断手段，定量评估模型是否编码了不应有的信息
- **问题定义的新颖性**：首次系统研究"task-irrelevant privacy memorization"，将隐私研究从"模型能否被攻击提取数据"扩展到"模型是否无意中编码了完全无关的数据"

## 局限与展望

- **水印作为隐私代理的生态效度**：随机文本水印与真实世界隐私信息（如人脸、证件号）的特征分布不同，实际场景中不经意记忆的程度可能有差异
- **缓解策略缺失**：论文侧重于发现和分析问题，但未提出有效的防御或缓解方案（如差分隐私微调、表示正则化等）
- **模型范围有限**：需要在更多模型架构（如 QWen-VL、GPT-4o 等闭源模型）和更大参数规模上验证普遍性
- **水印注入方式单一**：仅考虑文本水印叠加在图像上，未探索其他类型的任务无关内容（如背景物体、噪声模式、元数据嵌入）
- **训练轮次影响**：未深入分析不同训练 epoch 数量对记忆程度的影响，也未探讨遗忘动态（训练后继续微调是否能消除记忆）
- **线性探针的上限问题**：线性探针只能检测线性可分的编码模式，非线性编码的隐私信息可能被遗漏

## 相关工作与启发

- **vs 成员推理攻击 (MIA)**：传统 MIA 关注"数据是否在训练集中"，本文关注"训练中的任务无关信息是否被编码"——问题更上游，也更基本
- **vs Carlini et al. 数据提取攻击**：Carlini 等人的工作证明 LLM 可以逐字复述训练数据，但那些数据与训练目标（next-token prediction）直接对齐；本文的水印内容则完全不对齐，记忆机制不同
- **vs 差分隐私训练**：DP-SGD 通过裁剪梯度和加噪声来防止隐私泄露，但对 mini-batch 内虚假相关性导致的记忆，目前尚不清楚 DP-SGD 的保护效果是否足够
- **vs 模型水印/后门攻击**：后门攻击是攻击者有意注入 trigger 并关联特定输出，本文的 setting 是"无意中"编码无关内容——机制相似但意图和场景不同
- **启发**：可结合本文发现，设计面向 MLLM 微调的隐私审计工具——在模型部署前用层级探针扫描是否存在不当记忆

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性地研究 MLLM 中任务无关隐私记忆问题，问题定义新颖
- 实验充分度: ⭐⭐⭐ 缓存内容有限，但实验设计（水印注入 + 层级探针）逻辑清晰、可控性强
- 写作质量: ⭐⭐⭐⭐ 问题动机阐述清晰，Figure 1 有效对比了已有工作与本文的区别
- 价值: ⭐⭐⭐⭐ 揭示了一个被忽视的隐私风险来源，对 MLLM 安全部署有重要警示价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models](../../ACL2025/llm_safety/private_memorization_editing_turning_memorization_into_a_defense_to_strengthen_d.md)
- [\[ICML 2025\] Cut out and Replay: A Simple yet Versatile Strategy for Multi-Label Online Continual Learning](cut_out_and_replay_a_simple_yet_versatile_strategy_for_multi-label_online_contin.md)
- [\[ICLR 2026\] Doxing via the Lens: Revealing Location-related Privacy Leakage on Multi-modal Large Reasoning Models](../../ICLR2026/llm_safety/doxing_via_the_lens_revealing_location-related_privacy_leakage_in_vlms.md)
- [\[AAAI 2026\] AUVIC: Adversarial Unlearning of Visual Concepts for Multi-modal Large Language Models](../../AAAI2026/llm_safety/auvic_adversarial_unlearning_of_visual_concepts_for_multi-mo.md)
- [\[ACL 2025\] Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport](../../ACL2025/llm_safety/opt-out_investigating_entity-level_unlearning_for_large_language_models_via_opti.md)

</div>

<!-- RELATED:END -->
