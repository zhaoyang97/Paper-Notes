---
title: >-
  [论文解读] CADReview: Automatically Reviewing CAD Programs with Error Detection and Correction
description: >-
  [ACL 2025][CAD 程序审查] 提出 CAD 程序审查任务及 ReCAD 框架，基于参考图像自动检测 CAD 程序中的错误并生成修正反馈，构建了包含 20K+ 样本（8 类错误）的 CADReview 数据集。 计算机辅助设计（CAD）在工业设计中至关重要，设计师需要反复审查 3D 原型与设计图之间的一致性…
tags:
  - "ACL 2025"
  - "CAD 程序审查"
  - "3D 建模"
  - "多模态大语言模型"
  - "错误检测与纠正"
  - "几何组件识别"
---

# CADReview: Automatically Reviewing CAD Programs with Error Detection and Correction

**会议**: ACL 2025  
**arXiv**: [2505.22304](https://arxiv.org/abs/2505.22304)  
**代码**: [有](https://cgl-pro.github.io/cadreview)  
**领域**: 其他  
**关键词**: CAD 程序审查, 3D 建模, 多模态大语言模型, 错误检测与纠正, 几何组件识别

## 一句话总结

提出 CAD 程序审查任务及 ReCAD 框架，基于参考图像自动检测 CAD 程序中的错误并生成修正反馈，构建了包含 20K+ 样本（8 类错误）的 CADReview 数据集。

## 研究背景与动机

计算机辅助设计（CAD）在工业设计中至关重要，设计师需要反复审查 3D 原型与设计图之间的一致性，这一过程极其耗时。现有 AI 方法主要聚焦于 CAD 程序生成，忽略了生成后的审查与修正环节。

核心挑战：

**几何组件对齐困难**：3D 对象由多个组件组成，每个对应程序中特定代码块（含子程序、控制流、布尔运算），模型需要将视觉组件与代码块关联

**隐藏内部组件**：3D 对象可能包含不可直接视觉检查的内部结构（如内六角孔），需要程序化分析

**几何操作映射**：精确纠正需要将几何操作（旋转、平移）映射到对应的代码修改

**MLLM 表现不佳**：GPT-4o 等先进模型在 CAD 审查上效果有限，主要因为无法有效对齐代码与视觉信息

## 方法详解

### 整体框架

ReCAD 框架由两个 MLLM 模块组成：
- **反馈生成器 $\phi_1$**：输入参考图像、CAD 程序和渲染图像，输出错误描述反馈
- **代码编辑器 $\phi_2$**：利用反馈指导程序错误纠正

训练采用两阶段 SFT + RL 精炼策略。两个模块共享 MLLM 骨干网络（支持 Qwen2-VL 和 LLaVA-OV）。

### 关键设计

#### 几何组件识别（GCR）机制

为增强反馈生成器识别视觉和代码组件的能力，设计序贯训练流程：

**CAD 字幕**：使用 Text2CAD 数据集的图像-文本对训练视觉-语言投影器，让 LLM 学会识别几何组件（如"带中心孔的圆形 CAD 模型"）。仅训练投影器，冻结视觉编码器和 LLM。

**CAD 定位**：三种对齐任务：
- 语义匹配：根据语义标签预测对应代码块
- 坐标匹配：根据渲染图像坐标查询对应代码块
- 坐标定位：给定代码块确定其在图像中的坐标

使用 CADTalk 数据集和 GPT-4o 增强数据构建训练数据。在字幕训练基础上初始化，联合微调视觉编码器和 LLM。定位准确率达到约 90%，且不影响字幕性能。

#### 空间几何操作（SGO）机制

增强代码编辑器理解空间关系和几何变换的能力：

- 将 CAD 程序的代码块随机掩盖 30% 让模型预测（代码补全任务）
- 发现交叉熵损失在语法结构相似的代码上快速收敛但无法反映操作数值差异
- 将空间位置值量化为 8 位（最大 256），并对数值 token 加倍损失权重

$$\mathcal{L}_{sgo} = w_i \cdot \mathcal{L}_i, \quad w_i = \begin{cases} 2, & \text{if } y_i \in \mathbb{R} \\ 1, & \text{otherwise} \end{cases}$$

#### SFT 训练

反馈生成：冻结 $\phi_1$，用 LoRA（rank=8）在 LLM 上微调，使用交叉熵损失预测真实反馈。
代码编辑：$\phi_2$ 输入参考图像、程序、渲染图像和生成的反馈，用 LoRA（rank=64）微调 LLM。

### 损失函数 / 训练策略

#### RL 精炼（DPO）

设计两个奖励函数精炼反馈生成器：

**错误诊断奖励** $\mathcal{V}_d$：反馈中代码块和错误类型均正确则为 1，否则为 0。

**视觉相似性奖励** $\mathcal{V}_v$：用 $\phi_1$ 的视觉编码器提取参考图像和编辑后渲染图像的特征，计算余弦相似度。

训练流程：随机选 2000 样本，用 top-p 采样（T=0.8, p=0.9）生成 K 个反馈，根据两个奖励构建偏好对，保留诊断奖励为 1 且视觉奖励差超 0.25 的对，用 DPO 优化。

## 实验关键数据

### 主实验

数据集：CADReview 包含 17,334 训练 / 2,000 验证 / 1,615 测试样本，含人工编写和机器生成两类程序。

| 方法 | Machine-Acc↑ | Machine-CD↓ | Machine-JSD↓ | Human-Acc↑ | Human-CD↓ | Human-JSD↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Claude 3.5 | 32.19 | 4.06 | 36.16 | 21.51 | 9.03 | 79.63 |
| Gemini 2.0 | 36.83 | 3.96 | 11.49 | 23.36 | 5.97 | 55.85 |
| GPT-4o | 41.54 | 4.34 | 12.07 | 31.84 | 5.52 | 48.70 |
| Llama 3.2† | 56.23 | 2.71 | 5.44 | 51.24 | 5.82 | 36.18 |
| **ReCAD-LA†** | **73.11** | 1.45 | 3.42 | 59.15 | 4.76 | 33.09 |
| **ReCAD-QW†** | 71.83 | **1.43** | **2.80** | **63.60** | **4.42** | **30.87** |

（CD, JSD 均乘以 $10^3$；†表示在 CADReview 上微调）

| 方法 | Machine-IR↓ | Human-IR↓ |
|------|:---:|:---:|
| GPT-4o | 28.43 | 18.57 |
| ReCAD-LA | 0.00 | 13.74 |
| ReCAD-QW | 0.00 | 10.59 |

### 消融实验

| 消融组件 | Machine-Acc | Machine-CD | Human-Acc | Human-CD |
|----------|:---:|:---:|:---:|:---:|
| w/o GCR | 67.43 | 2.08 | 51.75 | 5.86 |
| w/o SGO | 70.55 | 1.93 | 58.79 | 6.84 |
| w/o Feedback | — | 1.84 | — | 5.71 |
| w/o Reward | 71.16 | 1.57 | 58.01 | 5.27 |
| **Full ReCAD-LA** | **73.11** | **1.45** | **59.15** | **4.76** |

### 关键发现

1. **GPT-4o 准确率仅 41.5%/31.8%**：闭源 MLLM 难以对齐 CAD 程序与视觉信息，且无效程序率超 25%
2. **ReCAD 大幅超越同规模微调模型**：比 Llama 3.2（同规模但无 GCR/SGO）准确率高 12-17%
3. **GCR 是关键**：移除后性能全面下降，且无反馈版本甚至优于无 GCR 版，说明错误的组件对齐产生的反馈反而有害
4. **SGO 对复杂程序更重要**：在人工编写程序上的影响比机器生成程序更大
5. **人工程序比机器程序更难**：所有方法在人工程序上性能下降，因含更复杂的控制流和嵌套结构

## 亮点与洞察

1. **新任务定义**：首次将代码审查概念引入 CAD 领域，连接了 NLP 和 3D 建模
2. **序贯训练策略**：字幕→定位→反馈生成的逐步训练让模型渐进式学习从视觉到代码的对齐能力
3. **数值损失加权**：识别到 CAD 代码中数值 token 的重要性并加倍权重，解决了语法快速收敛但数值不准确的问题
4. **反馈循环设计**：反馈生成器和代码编辑器的双模块设计让纠正过程可解释，DPO 精炼确保两者协同
5. **8 类错误分类**：系统性地定义了 CAD 程序错误类型，为后续研究提供了标准化框架

## 局限与展望

1. 未考虑最优代码编辑方案（同一几何组件可有多种代码实现）
2. 目前仅支持 OpenSCAD 语言，虽声称框架可迁移但未验证
3. 错误类型为人工预定义，未覆盖所有可能的设计偏差
4. 每次仅处理一个错误，实际场景可能有多个同时出现的错误

## 相关工作与启发

- **CAD 程序生成**：DeepCAD、Text2CAD 等聚焦生成，本文补充了审查维度
- **代码编辑**：CoffeeGym、CodeAgent 等用于通用代码，本文迁移到几何代码领域
- **MLLM 应用**：展示了 MLLM 在特定领域（CAD）经专门训练后可大幅超越通用闭源模型
- 启发：其他领域特定代码（如分子描述、电路设计）也可借鉴此审查框架

## 评分

- **新颖性**: ★★★★☆ — 首个 CAD 程序审查任务，任务定义和数据集均为新贡献
- **技术深度**: ★★★★☆ — GCR/SGO 机制设计精巧，DPO 精炼有理有据
- **实验充分性**: ★★★★☆ — 多基线（含 GPT-4o/Claude/Gemini）+ 消融 + 案例分析
- **实用性**: ★★★★☆ — 对 CAD 设计工作流有直接应用价值
- **写作质量**: ★★★★☆ — 图示丰富，框架描述清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LegalReasoner: Step-wised Verification-Correction for Legal Judgment Reasoning](legalreasoner_step-wised_verification-correction_for_legal_judgment_reasoning.md)
- [\[ACL 2025\] Zero-Shot Conversational Stance Detection: Dataset and Approaches](zero-shot_conversational_stance_detection_dataset_and_approaches.md)
- [\[ACL 2025\] Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection](explaining_matters_leveraging_definitions_and_semantic_expansion_for_sexism_dete.md)
- [\[ICML 2025\] How Do Transformers Learn Variable Binding in Symbolic Programs?](../../ICML2025/others/how_do_transformers_learn_variable_binding_in_symbolic_programs.md)
- [\[ACL 2025\] SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection](seoe_semantic_eval.md)

</div>

<!-- RELATED:END -->
