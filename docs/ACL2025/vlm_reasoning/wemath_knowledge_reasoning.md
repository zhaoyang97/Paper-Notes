---
title: >-
  [论文解读] We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?
description: >-
  [ACL 2025][多模态VLM][视觉数学推理] 本文提出We-Math基准，包含6.5K视觉数学问题和67个层次化知识概念，通过将复合问题分解为子问题引入四维评估指标（知识不足IK、泛化不足IG、完全掌握CM、机械记忆RM），首次从知识掌握角度系统评估LMM的数学推理过程而非仅关注最终结果。 1. 领域现状：LMM在视…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "视觉数学推理"
  - "知识概念"
  - "推理评估"
  - "多模态"
  - "基准测试"
---

# We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?

**会议**: ACL 2025  
**arXiv**: [2407.01284](https://arxiv.org/abs/2407.01284)  
**代码**: [https://github.com/We-Math/We-Math](https://github.com/We-Math/We-Math)  
**领域**: 多模态VLM  
**关键词**: 视觉数学推理, 知识概念, 推理评估, 多模态, 基准测试

## 一句话总结
本文提出We-Math基准，包含6.5K视觉数学问题和67个层次化知识概念，通过将复合问题分解为子问题引入四维评估指标（知识不足IK、泛化不足IG、完全掌握CM、机械记忆RM），首次从知识掌握角度系统评估LMM的数学推理过程而非仅关注最终结果。

## 研究背景与动机
1. **领域现状**：LMM在视觉数学推理上取得进展，但现有基准（MathVista、MathVerse等）仅关注最终答案的对错，忽略了推理过程中的知识掌握情况。
2. **现有痛点**：(1)仅看结果会产生反直觉结论（如LMM在大学题上比小学题表现好）；(2)正确答案不一定反映真正推理能力（可能是机械记忆）；(3)错误答案不一定意味着完全缺乏基础知识。
3. **核心矛盾**：人类通过逐步掌握和泛化知识概念来解决数学问题，但现有评估无法区分模型是"真会"还是"瞎蒙"。
4. **本文目标**：设计一个基于知识概念的评估基准，能揭示LMM数学推理中的内在问题。
5. **切入角度**：将复合问题分解为基于单一知识概念的子问题，通过对比子问题和原问题的答对情况来判断模型的真实推理能力。
6. **核心idea**：知识概念层次化 + 问题分解 + 四维评估指标。

## 方法详解

### 整体框架
数学教科书知识体系 → 5大类12典型问题67知识概念 → 收集6.5K问题 → 标注知识概念和解题步数 → 人工分解1.5K多步问题为子问题 → LMM同时回答子问题和原问题 → 四维指标评估 → 知识概念增强(KCA)策略。

5大类分别为：平面图形（PF）、立体图形（SF）、变换与运动（TMF）、位置与方向（PD）、测量（Mem）。每个终端知识概念包含10-40个样本保证均衡，如"角度与长度"（AL）、"单位理解与换算"（UCU）、"坐标与位置对应"（CCP）等。

### 关键设计

1. **层次化知识结构**:

    - 功能：确保评估覆盖数学推理的各个基础层面。
    - 核心思路：将数学问题按教科书知识体系组织为5层：平面图形、立体图形、变换与运动、位置与方向、测量。每层分解为12个典型问题、67个终端知识概念，每个概念包含10-40个样本保证均衡。
    - 设计动机：现有基准的分类不够系统，导致评估不全面。

2. **知识驱动的问题分解与四维指标**:

    - 功能：从推理过程而非仅最终结果评估LMM的数学能力。
    - 核心思路：对含$k$个知识概念的复合问题，分解为$k$个单概念子问题。让LMM同时回答所有子问题和原问题，然后分类为四种情况：IK（子问题错+原问题错=知识不足）、IG（子问题对+原问题错=泛化不足）、CM（子问题对+原问题对=完全掌握）、RM（子问题错+原问题对=机械记忆）。能力层次：IK < IG < CM。
    - 设计动机：仅看对错无法区分"知识不足"和"泛化不足"，但两者需要不同的改进策略。

3. **知识概念增强策略（KCA）**:

    - 功能：通过补充知识概念描述来缓解LMM的知识不足问题。
    - 核心思路：从Wikipedia和教科书为67个知识概念构建描述，在推理时作为额外知识输入LMM。
    - 设计动机：如果IK是主要问题，直接补充知识应该能改善。

### 损失函数 / 训练策略
We-Math是评估基准，不涉及训练。评估了17个LMM（4个闭源+13个开源），包括GPT-4o、GPT-4V、Gemini 1.5 Pro、Qwen-VL-Max、LLaVA-NeXT-110B/70B、DeepSeek-VL等。使用testmini子集（1740样本：1215个一步题、360个两步题、165个三步题）加速评估，所有问题标准化为选择题格式并额外加入"不确定"选项防止LMM从选项推导答案。问题分解由专家标注员完成，交叉验证保证质量。

## 实验关键数据

### 主实验

| 模型 | S1准确率 | S2准确率 | S3准确率 | 最弱领域 |
|------|---------|---------|---------|----------|
| GPT-4o | 72.84% | 58.06% | 43.64% | 角度与长度(39.12%) |
| GPT-4V | 65.51% | 49.17% | 38.18% | 角度与长度(38.42%) |
| Gemini 1.5 Pro | 56.13% | 51.39% | 33.94% | 角度与长度(31.23%) |
| LLaVA-NeXT-110B | 较高 | 中等 | 中等 | 细粒度测量 |
| DeepSeek-VL-1.3B | 低 | 很低 | 很低 | 多数领域 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 无KCA | 基线 | 标准推理 |
| +KCA | IK显著减少 | 知识增强有效 |
| 仅1步问题 | 最高准确率 | 步数越多越难 |
| 3步问题 | 最低准确率 | 知识概念数与难度正相关 |

### 关键发现
- 解题步数与性能负相关——多步问题显著更难，说明知识概念组合是核心挑战。
- GPT-4o首次从IK阶段进入IG阶段，是第一个走向知识泛化的LMM。
- 多数LMM存在严重的RM问题——能答对复合问题却答不对子问题，暗示可能并非真正推理。
- KCA策略有效减少IK问题，但对IG和RM帮助有限。
- LMM在"角度与长度"等细粒度视觉测量上表现最差。

## 亮点与洞察
- **四维评估的深刻性**：RM（机械记忆）的发现令人警醒——模型可能通过模式匹配而非真正推理来"蒙"对答案。
- **知识概念驱动的评估范式**：不评最终答案而评知识掌握，更接近人类教育评估体系。
- **可迁移到其他推理评估**：四维指标框架可推广到代码推理、科学推理等任何可分解为子知识的推理任务。
- **IK→IG阶段跃迁的标志意义**：GPT-4o首次从知识不足阶段进入泛化不足阶段，说明大规模预训练确实能解决基础知识缺乏问题，但知识组合与泛化仍是核心瓶颈。
- **反直觉现象的系统解释**：现有基准中"大学题比小学题简单"的反直觉结论，在We-Math框架下得到了合理解释——大学题涉及更少知识概念组合，因此IG问题更轻。
- **评估中的额外发现**：GPT-4o在"单位理解与换算"（UCU）上高达86.61%，但在"角度与长度"（AL）上仅39.12%，反映细粒度视觉测量是所有LMM的共同短板。闭源模型整体显著优于开源模型，但参数量大的开源模型（如LLaVA-NeXT-110B）已接近闭源GPT-4V水平。

## 局限与展望
- 问题分解依赖人工专家，成本高且可能存在主观性。
- 仅覆盖基础视觉数学（5大类12典型问题），未涉及高等数学（微积分、线性代数等）。
- RM的成因尚未深入分析（是数据泄露还是捷径学习？）。
- 知识概念增强（KCA）策略效果有限——对IK有效但对IG和RM基本无效，说明简单补充知识描述无法替代真正的推理能力训练。
- 67个知识概念之间的依赖关系未被建模，未来可探索知识图谱驱动的评估。
- 评估局限于选择题和填空题，未涉及开放式数学证明题。
- 部分领域样本量不均衡：UCU高达86.61%的准确率可能与更简单的问题设计有关，而AL仅39.12%反映了视觉测量本身的难度。

## 相关工作与启发
- **vs MathVista**: MathVista仅评估最终结果，We-Math评估推理过程和知识掌握。MathVista中"大学题比小学题简单"的反直觉结论在We-Math中得到解释。
- **vs MathVerse**: MathVerse尝试评估推理路径但基于参考答案，We-Math基于知识概念分解更系统。
- **vs G-LLaVA**: G-LLaVA-13B在某些专业领域表现突出但RM比例高达约36%，说明其可能依赖训练数据中的模式而非真正推理。
- **与教育评估的联系**：IK/IG/CM/RM的四维指标可类比为教育心理学中的知识诊断框架，为AI评估提供了教育学理论基础。

## 评分

### 实现细节
使用bert-base-uncased进行embedding，所有实验在NVIDIA A100上运行。评估采用正则匹配LMM预测并计算准确率。
- 新颖性: ⭐⭐⭐⭐⭐ 四维评估指标和知识驱动分解都很创新
- 实验充分度: ⭐⭐⭐⭐⭐ 17个LMM全面评估，发现深刻
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，框架系统
- 价值: ⭐⭐⭐⭐⭐ 改变了数学推理评估的范式，testmini子集便于快速评估

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?](../../ECCV2024/multimodal_vlm/mathverse_does_your_multi-modal_llm_truly_see_the_diagrams_in_visual_math_proble.md)
- [\[ACL 2025\] The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights](the_role_of_visual_modality_in_multimodal_mathematical_reasoning_challenges_and_.md)
- [\[ACL 2025\] MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning](mathcoder-vl_bridging_vision_and_code_for_enhanced_multimodal_mathematical_reaso.md)
- [\[CVPR 2025\] MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts](../../CVPR2025/multimodal_vlm/mv-math_evaluating_multimodal_math_reasoning_in_multi-visual_contexts.md)
- [\[ACL 2025\] OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference](omnialign-v_towards_enhanced_alignment_of_mllms_with_human_preference.md)

</div>

<!-- RELATED:END -->
