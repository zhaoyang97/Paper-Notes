# The Impact of Auxiliary Patient Data on Automated Chest X-Ray Report Generation and How to Incorporate It

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2406.13181](https://arxiv.org/abs/2406.13181)  
**代码**: [GitHub (匿名)](https://anonymous.4open.science/r/anon-D83E) | [HuggingFace Model](https://huggingface.co/aehrc/cxrmate)  
**领域**: 医学影像 / 放射报告生成  
**关键词**: 胸部X光报告生成, 多模态语言模型, 辅助患者数据, 电子健康记录, 强化学习  

## 一句话总结
本文研究如何将急诊科患者数据（生命体征、药物、分诊信息等）整合到多模态语言模型中用于自动胸部X光报告生成，提出将异构表格数据、文本和图像转化为统一嵌入的方法，在MIMIC-CXR + MIMIC-IV-ED数据集上显著提升了报告的诊断准确性，超越了包括CXRMate-RRG24在内的多个基准模型。

## 背景与动机
- 胸部X光（CXR）报告生成是一个重要的医学AI任务，但现有方法主要依赖CXR图像和有限的放射学数据（如indication部分）
- 真实临床场景中，放射科医生在解读影像时会参考患者的临床信息（生命体征、用药史、主诉等），这些信息可以显著提高诊断准确性
- 然而，现有的CXR报告生成模型几乎没有利用急诊科（ED）的患者记录数据
- 电子健康记录（EHR）系统整合到放射学工作流的趋势日益明显，但如何将异构的患者数据源转化为语言模型可用的嵌入，缺乏系统性研究

## 核心问题
1. 哪些辅助患者数据源能有效提升CXR报告生成的诊断准确性？
2. 如何将异构数据类型（数值型、分类型、文本型、时间序列、图像）转化为多模态语言模型的统一嵌入表示？

## 方法详解

### 整体框架
模型基于CXRMate-RRG24架构，采用UniFormer作为图像编码器、Llama作为解码器。将来自MIMIC-CXR和MIMIC-IV-ED的多种患者数据转化为嵌入（patient data embeddings），作为prompt输入解码器，生成放射报告的findings和impression部分。每个嵌入由四部分求和构成：患者数据嵌入 + 来源嵌入 + 位置嵌入 + 时间差嵌入。

### 关键设计
1. **时间差嵌入（Time Delta Embeddings）**: 通过 $D = 1/\Delta + 1$ 映射事件与检查的时间差，再经FNN（SiLU激活）投射到解码器隐藏维度，使模型关注时间上更近的事件。位置嵌入按时间差排序，利用旋转位置编码（RoPE）使时间最近的数据获得更高注意力权重。

2. **表格数据嵌入（Grouped Embeddings）**: 将表格中的数值列和分类列按时间差分组，形成特征向量（数值直接放置，分类激活为1），通过FNN转化为嵌入。高基数列（如药物名称）则通过tokenizer和token embeddings处理为文本嵌入。对比实验表明分组嵌入（Grouped Embeddings）优于分离嵌入和值转文本方法。

3. **报告部分嵌入**: 利用放射报告的indication（检查原因）、history（病史）和comparison（对比检查）部分作为额外输入。history部分首次被研究用于CXR报告生成。

4. **三阶段训练**:
   - 阶段1: 仅用图像在MIMIC-CXR上进行Teacher Forcing训练
   - 阶段2: 在MIMIC-CXR + MIMIC-IV-ED联合数据集上用多源数据进行TF训练（冻结图像编码器）
   - 阶段3: 使用SCST强化学习，以CXR-BERT + BERTScore + ARN复合奖励优化

5. **ARN指标**: 提出Absence of Repeated N-grams（ARN）指标衡量生成文本的重复率，并将其纳入RL奖励函数以减少重复生成。

6. **分段奖励（Reward per Section）**: 分别为findings和impression计算奖励（$\alpha_1=0.75, \alpha_2=0.25$），避免findings部分主导impression的优化。

## 实验关键数据

### 数据源消融（Table 1, findings + impression）
| 数据源配置 | RG | CX | CB | G |
|-----------|-----|-----|-----|-----|
| Images only | 24.54 | 30.10 | 59.25 | 35.16 |
| + triage | 24.59 | 31.33 | 62.79 | 35.78 |
| + reconciled medicines | 25.10 | 32.05 | 64.70 | 36.32 |
| + indication | 25.01 | 32.78 | 65.49 | 35.88 |
| + history | 24.88 | 31.66 | 63.91 | 35.76 |
| **effective sources (h=0)** | **25.52** | **32.49** | **65.93** | **36.26** |

### 基准对比（Table 2, findings only）
| 模型 | RG | CX | CB | G | BS | B4 |
|------|-----|-----|-----|-----|-----|-----|
| CXRMate | 26.5 | 33.9 | 71.3 | 40.3 | 30.5 | 7.5 |
| CXRMate-RRG24 | 28.9 | 31.2 | 58.2 | 40.2 | 31.0 | 6.6 |
| **本文 + RL + ARN** | **30.2** | **33.6** | **78.0** | **40.7** | **37.3** | **7.6** |

本文模型在训练样本仅76,398 exams的情况下，显著超越使用550,395 exams的CXRMate-RRG24。

### 消融实验要点
- **有效数据源**: triage、reconciled medicines、indication、history四个数据源各自显著提升性能，组合后效果更优
- **无效数据源**: ED stays表、metadata表、administered medicines未带来显著提升
- **先验检查**: 使用1-2个先验检查（prior exams）可提升性能，但3个反而下降，可能因attention dilution
- **组合effective sources + prior exams反而下降**: 输入过多导致注意力稀释
- **表格嵌入方法对比**: Grouped embeddings > Values-to-text > Separate embeddings（RG分别为31.69/30.70/25.28）
- **ARN奖励**: 有效减少重复（ARN从93.5提升至99.3），但其他指标略有下降

## 亮点
- 首次系统研究急诊科患者数据对CXR报告生成的影响，挖掘了triage、药物、病史等多个新的有效数据源
- 提出将异构数据（数值、分类、文本、时间序列、图像）转化为统一嵌入的通用框架
- 用更少的训练数据（76K exams vs 550K）超越了SOTA模型，说明辅助数据的价值
- 案例分析详尽：通过TP/FP/TN/FN四类案例深入分析了辅助数据如何影响模型预测
- 提出ARN指标和分段奖励机制，解决了RL训练中的文本重复问题
- 首次将放射报告的history部分用于CXR报告生成，发现其与indication同等重要

## 局限性 / 可改进方向
- **单一数据源偏差**: 数据仅来自Beth Israel Deaconess Medical Center，泛化性有待验证
- **缺乏放射科医生的主观评估**: 目前仅使用自动指标评估
- **Attention Dilution**: 模型架构限制——当输入过多（如多个prior exams + 所有有效源）时，自注意力权重被稀释，性能反而下降
- **证据平衡问题**: 模型有时无法正确平衡辅助数据与影像证据，导致假阳性（被辅助数据误导）或假阴性（未能利用辅助数据证据）
- **模型可解释性不足**: 多模态语言模型的决策过程仍是黑盒
- **未来方向**: 使用更大的LLM解码器（更强的推理能力）、探索层次化注意力机制、扩展到多机构数据集

## 与相关工作的对比
- **vs CXRMate/CXRMate-RRG24**: 本文基于CXRMate-RRG24架构扩展，加入辅助患者数据后以更少训练数据超越了原模型
- **vs 仅用indication的方法** (Nguyen et al., 2023): 本文发现history部分同样重要，且多源数据组合效果更优
- **vs 仅用prior exams的方法** (Wu et al., 2022): 本文发现在结合多源数据时，prior exams反而有害，attention dilution是关键瓶颈
- **vs 多模态EHR模型** (MeTra, ETHOS): 这些工作用于预测任务（ICU存活率等），本文将多模态EHR数据应用于报告生成这一语言生成任务
- **vs CXR-LLaVA, MedXChat, RaDialog**: 本文在多个指标上显著优于这些基于LLM的方法

## 启发与关联
- **数据驱动 vs 模型驱动**: 本文强有力地证明了"更丰富的数据"比"更大的模型/更多的训练样本"更重要——76K exams + 辅助数据 > 550K exams仅用图像
- **异构数据融合的通用范式**: 将数值/分类/文本/时间/图像统一转化为嵌入的框架，可推广到其他多模态医学AI任务
- **Attention Dilution是多源输入方法的核心瓶颈**: 未来可探索选择性注意力、门控机制或层次化编码来缓解
- **辅助数据的双刃剑效应**: 辅助数据可以提供支持性证据（TP），但也可能产生混淆性证据（FP），如何让模型学会"鉴别性推理"是关键课题

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统研究ED患者数据对CXR报告生成的影响，嵌入方法有一定创新，但整体架构基于已有工作
- 实验充分度: ⭐⭐⭐⭐⭐ 消融实验非常全面（数据源、嵌入方法、RL奖励），10次训练运行的统计显著性检验，详尽的错误分析
- 写作质量: ⭐⭐⭐⭐⭐ 论文结构清晰，图表设计优秀，案例分析深入，局限性讨论诚恳充分
- 对我的价值: ⭐⭐⭐⭐ 多模态数据融合的方法论有参考价值，attention dilution的发现对多源输入方法有普遍启发
