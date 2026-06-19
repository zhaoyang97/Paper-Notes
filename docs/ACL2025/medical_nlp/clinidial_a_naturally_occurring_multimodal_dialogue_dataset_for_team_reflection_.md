---
title: >-
  [论文解读] CliniDial: A Naturally Occurring Multimodal Dialogue Dataset for Team Reflection in Action During Clinical Operation
description: >-
  [ACL 2025][医疗NLP][临床对话数据集] 构建了 CliniDial 数据集，收集自模拟临床手术中的自然对话，包含音频转录、双角度视频和患者生理信号等多模态数据，标注了团队反思行为编码，揭示了现有 LLM 在处理标签不均衡、自然对话交互和领域多模态数据方面的显著不足。 领域现状： 临床手术中团队协作直接影响手术结…
tags:
  - "ACL 2025"
  - "医疗NLP"
  - "临床对话数据集"
  - "多模态对话"
  - "团队协作分析"
  - "行为编码"
---

# CliniDial: A Naturally Occurring Multimodal Dialogue Dataset for Team Reflection in Action During Clinical Operation

**会议**: ACL 2025  
**arXiv**: [2506.12936](https://arxiv.org/abs/2506.12936)  
**代码**: [https://github.com/MichiganNLP/CliniDial](https://github.com/MichiganNLP/CliniDial)  
**领域**: 医疗NLP  
**关键词**: 临床对话数据集, 多模态对话, 团队协作分析, 行为编码, 医疗NLP

## 一句话总结

构建了 CliniDial 数据集，收集自模拟临床手术中的自然对话，包含音频转录、双角度视频和患者生理信号等多模态数据，标注了团队反思行为编码，揭示了现有 LLM 在处理标签不均衡、自然对话交互和领域多模态数据方面的显著不足。

## 研究背景与动机

**领域现状**: 临床手术中团队协作直接影响手术结果。美国每年约 25 万例可预防死亡与团队沟通不畅有关。协作不足可导致比预期多 58% 的死亡率。理解手术中的团队交互模式对提升医疗安全至关重要。

**现有痛点**: 
   - 现有 NLP 对话数据集大多在受控环境中收集（如 MultiWOZ 通常仅 30 轮），无法反映真实临床操作中的复杂交互
   - 现有多模态数据集主要关注视觉+文本，缺少生理信号（vital signs）这一关键模态
   - 临床操作数据具有严重的标签不均衡、超长对话和领域特异性，现有方法应对困难

**核心矛盾**: 真实临床环境中的团队交互数据极难获取（伦理和法律限制），但现有模拟/常规对话数据集又不能反映手术中的实际交互模式。

**本文目标**: 构建一个来自模拟临床手术的自然多模态对话数据集，并系统评估现有 LLM 在处理这类真实世界数据时的能力边界。

**切入角度**: 从模拟恶性高热（MH）手术场景中收集数据，关注麻醉师、支持人员和外科医生三种角色的团队交互行为。

**核心 idea**: 真实临床操作的多模态对话具有独特特征（标签不均衡、超长自然交互、领域多模态），现有 LLM 远未能有效处理。

## 方法详解

### 整体框架

CliniDial 不是提出新方法，而是：
1. 构建数据集：模拟手术中的多模态对话数据
2. 标注行为编码：Seek / Evaluate / Implement / Plan + None
3. 三个案例研究：分别考察标签不均衡、对话上下文和多模态的影响

### 关键设计

1. **数据收集与场景设计**: 

    - 22 个手术模拟 session，每个 session 约 19 分钟
    - 场景：36 岁女性微创手术中出现恶性高热（MH）并发症
    - 角色：外科医生（confederate）、麻醉师（trainee，主决策者）、支持人员
    - 总计 6.5k 轮对话、49.9k 词
    - 平均每段对话 311 轮（远超常规对话数据集的 30 轮）

2. **多模态数据**: 

    - **音频 + 转录**: 各角色的对话内容
    - **双角度视频**: 两个摄像头从不同角度记录手术过程
    - **9 种生理信号**: 来自患者人体模型的模拟生理数据（心率、血氧、呼气末 CO2 等）
    - 所有模态有时间戳对齐

3. **行为编码标注**: 

    - 基于 Schmutz et al. (2021) 的团队反思行为框架
    - 四种标签 + None：Seek（寻求信息）、Evaluate（评估状况）、Implement（执行操作）、Plan（制定计划）
    - 10 折交叉验证：17/2/3 sessions 划分 train/val/test

4. **角色特异性分析**: 

    - 外科医生最多"Seek"标签（30.4%）——依赖团队信息
    - 支持人员最多"Implement"标签（13.7%）——执行辅助操作
    - 词汇特征：外科医生和学员常说"thank you"，支持人员常说"alright"

### 损失函数/训练策略

标注数据用于评估，不提出新训练方法。评估方法包括：
- BERT_base 微调
- LLM few-shot prompting（Llama 3 8B/70B, GPT-4/4o）

## 实验关键数据

### 案例1：标签不均衡

| 方法 | Macro F1 | Micro F1 |
|------|----------|----------|
| BERT_base (fine-tuned) | 48.6 | **66.6** |
| Llama 3 8B (1-shot) | 37.0 | - |
| Llama 3 70B (5-shot) | 48.2 | - |
| GPT-4 (5-shot) | 47.0 | - |
| GPT-4o (5-shot) | **51.1** | - |

### 案例2：对话上下文

| 模型 | Context=1 | Context=3 | Context=5 |
|------|-----------|-----------|-----------|
| GPT-4o (1-shot) Macro F1 | 47.3 | **49.8** | 48.5 |
| GPT-4o (1-shot) Micro F1 | 55.0 | **58.0** | 56.0 |
| Llama 70B (1-shot) Macro F1 | **46.0** | 40.9 | 36.4 |

### 案例3：多模态

| 输入模态 | Macro F1 | Micro F1 |
|----------|----------|----------|
| Text only (T) | **48.2** | - |
| T + Vision (直接) | 46.8 | - |
| T + Physiology (直接) | 44.9 | - |
| T + Vision (文本化) | 46.9 | - |
| T + Physiology (文本化) | 42.9 | - |

### 标签分布统计

| 标签 | None | Seek | Evaluate | Implement | Plan | 总计 |
|------|------|------|----------|-----------|------|------|
| 总数 (k) | 3.7 | 1.3 | 0.8 | 0.6 | 0.3 | 6.9 |
| 占比 | 53.6% | 18.8% | 11.6% | 8.7% | 4.3% | - |

### 关键发现

- **BERT 微调的 Macro/Micro F1 差距巨大**（48.6 vs 66.6），说明模型严重偏向多数类
- **LLM 不受标签不均衡影响**: Few-shot LLM 的 Macro/Micro F1 差距更小，但整体性能有限
- **Llama 3 8B 增加 shot 数反而性能下降**（37.0→32.7），小模型被长输入分散注意力
- **上下文有助于 GPT-4o 但伤害 Llama 70B**: GPT-4o 的 128K 上下文窗口更能利用对话信息，Llama 3 的 8K 窗口在输入约 1000 tokens 时就开始性能退化
- **多模态输入反而降低性能**: 直接输入视频帧和生理信号截图都导致 F1 下降，GPT-4o 读取生理信号时出现幻觉（把 Et 值 64 误判为心率）
- **最佳 Macro F1 仅 51.1%**，说明现有方法远不能满足临床需求

## 亮点与洞察

- **首个包含生理信号的对话数据集**: CliniDial 引入了心率、血氧、呼气末 CO2 等 9 种生理信号，为多模态融合提供了全新挑战
- **自然涌现的对话**: 参与者不是被要求"生成对话"，而是在执行临床操作中自然产生的交流
- **角色动力学分析**: 通过词频和标签分布揭示了手术团队中不同角色的交互模式
- **LLM 的领域局限性**: GPT-4o 在生理信号上的幻觉案例极具启发——AI 需要领域专业知识才能准确解读临床数据
- **实际影响**: 数据集直接关联到医疗安全——理解团队交互可以改进手术培训

## 局限与展望

- 数据来自模拟手术而非真实手术，对话可能缺乏真实手术中的紧迫感
- 数据量相对有限（22 个 session），对大规模预训练可能不够
- 视频数据因伦理考虑不会公开，限制了视觉模态的研究
- 未探索音频特征（语调、语速、情感）的作用
- 未测试 Chain-of-Thought 等高级提示策略
- 标注框架仅4种行为类别，可能不足以捕获所有有意义的团队交互模式

## 相关工作与启发

- **MultiWOZ** (Budzianowski et al., 2018): 任务导向对话标准数据集，但规模和自然度有限
- **Ego4D** (Grauman et al., 2022): 第一人称视角视频理解数据集，启发了多视角数据收集
- **TeamSTEPPS**: 医疗团队培训框架，提供了行为编码的理论基础
- CliniDial 可作为 NLP + 医疗协作研究的种子数据集
- 启示：真实世界数据的多模态融合远非简单拼接，需要领域知识引导

## 评分

- **新颖性**: ⭐⭐⭐⭐ (首个临床手术多模态对话数据集，引入生理信号这一全新模态)
- **实验充分度**: ⭐⭐⭐⭐ (三个案例研究系统覆盖数据集三大特征，多种模型对比)
- **写作质量**: ⭐⭐⭐⭐ (结构清晰，数据分析充分，定性案例有说服力)
- **价值**: ⭐⭐⭐⭐ (填补临床操作对话数据集空白，为 LLM 在高风险领域的评估提供基准)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents](reflectool_clinical_agent.md)
- [\[NeurIPS 2025\] Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](../../NeurIPS2025/medical_nlp/time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)
- [\[ACL 2025\] VITAL: A New Dataset for Benchmarking Pluralistic Alignment in Healthcare](vital_pluralistic_alignment_healthcare.md)
- [\[ACL 2025\] Enhancing Medical Dialogue Generation through Knowledge Refinement and Dynamic Prompt Adjustment](enhancing_medical_dialogue_generation_through_knowledge_refinement_and_dynamic_p.md)
- [\[ACL 2025\] Adaptive-VP: A Framework for LLM-Based Virtual Patients that Adapts to Trainees' Dialogue to Facilitate Nurse Communication Training](adaptive-vp_a_framework_for_llm-based_virtual_patients_that_adapts_to_trainees_d.md)

</div>

<!-- RELATED:END -->
