# PunchBench: Benchmarking MLLMs in Multimodal Punchline Comprehension

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2412.11906](https://arxiv.org/abs/2412.11906)  
**代码**: [https://github.com/OuyangKun10/PunchBench](https://github.com/OuyangKun10/PunchBench)  
**领域**: 多模态VLM  
**关键词**: 梗图理解, 多模态幽默, 讽刺理解, MLLM评测, Chain-of-Question  

## 一句话总结
提出PunchBench（6,000图文对、54,000 QA对），通过同义/反义标题替换消除文本捷径、多种问题格式与两层任务（感知+推理）全面评测MLLM的多模态梗图理解能力，并设计SC-CoQ由简到繁的提问策略显著提升表现，但GPT-4o仍远低于人类98%+水平。

## 背景与动机
多模态梗图（幽默或讽刺的图文对）在社交媒体上广泛传播，但现有MLLM梗图理解评测存在三大缺陷：(1) 文本捷径——模型可仅凭标题中的偏差词（如"enjoy"vs"disgusting"）或文本内部语义冲突就判断是否是梗，无需真正理解图文关系；(2) 问题格式单一——仅用Yes/No评测，无法揭示不同格式下的表现差异；(3) 领域狭窄——仅关注漫画或仅关注讽刺，无法覆盖真实世界多样化的梗图类型。

## 核心问题
如何设计一个既能消除文本捷径、又能通过多种问题格式和领域覆盖全面评测MLLM梗图理解能力的基准？以及如何通过推理策略提升MLLM在该任务上的表现？

## 方法详解

### 整体框架
PunchBench构建流程：(1) 从先有数据集（MTSD、MORE、HUB）和社交平台（X、Instagram、YouTube、CartoonMovement、CartoonStock）收集图文对并人工标注；(2) 用GPT-3.5生成同义和反义标题消除捷径；(3) 构建两层任务的多格式指令；(4) 100条/格式的人工质检。最终6,000对图文（3,000含梗+3,000不含梗），54,000 QA对。

### 关键设计
1. **同义/反义标题消除捷径**：用GPT-3.5对原始标题做词替换（同义词→同义标题，反义词→反义标题）。对含内部语义冲突的标题（如"我很开心！今天真恶心的雨天"），分离两部分分别做替换和反转。同义标题应与原始标题共享同一label（是否含梗），反义标题则相反。这迫使模型真正理解图文关系而非利用文本bias。

2. **两层任务×多格式评测**：浅层Task——Punchline Perception（判断图文对是否含梗），含Yes/No QA、Matching QA（选正确标题）、Multi-option QA（四选一描述）三种格式；深层Task——Punchline Reasoning（解释为什么含梗），含Yes/No QA（判断解释是否正确）、Matching QA（选正确解释）、Generation QA（自由生成解释）。难度递增：Yes/No < Matching < Multi-option/Generation。

3. **SC-CoQ (Simple-to-Complex Chain-of-Question)**：受由简到繁解题启发，在推理时先让模型回答简单问题再回答复杂问题。Intra-task SC-CoQ：同任务内按<Yes/No → Matching → Multi-option/Generation>顺序链式提问；Inter-task SC-CoQ：跨任务用同格式问题串联（先感知再推理或反之）。前一个问题的回答被concatenate到下一个问题的prompt中。

### 损失函数 / 训练策略
纯推理评测，无训练。对比了zero-shot、3-shot ICL、CoT和SC-CoQ四种策略。Generation QA用GPT-3.5自动评分。

## 实验关键数据

**Punchline Perception (Accuracy %)**：
| 模型 | Yes/No (zero) | Yes/No (SC-CoQ) | Matching (zero) | Multi-option (zero) |
|------|------|------|------|------|
| GPT-4o | 77.5 | **80.7** | 64.2 | 50.8 |
| GPT-4V | 75.0 | 78.1 | 62.1 | 48.1 |
| Aria (3.5B×8 MoE) | 72.1 | - | 61.8 | 47.9 |
| Qwen2-VL-72B | 73.7 | - | 60.2 | 48.8 |
| LLaVA-7B | 62.7 | - | 54.2 | 36.4 |
| **Human** | **98.3** | - | **97.7** | **90.7** |

**Punchline Reasoning (Accuracy %)**：
| 模型 | Yes/No (SC-CoQ) | Generation (SC-CoQ) |
|------|------|------|
| GPT-4o | 77.4 | **50.1** |
| Qwen2-VL-72B | 74.9 | 48.0 |
| **Human** | **96.0** | **100.0** |

### 消融实验要点
- **SC-CoQ一致优于3-shot和CoT**：在所有模型和所有问题格式上，SC-CoQ都是最优策略（P<0.01）。CoT在Yes/No QA上有时掉点。
- **同义/反义标题导致显著掉点**：即使GPT-4o替换标题后也掉3-5个点，说明模型依赖文本捷径。但SC-CoQ在三种标题下都能提升，跨标题提升更均匀。
- **模型规模效应明显**：Qwen2-VL-2B全面落后（仅33-57%），72B版接近GPT-4o水平。
- **感知 vs 推理**：所有模型在Reasoning上都比Perception低5-10个点，说明"为什么好笑"比"是否好笑"难得多。

## 亮点
- 同义/反义标题替换是消除evaluation shortcut的优雅方案，可迁移到其他multimodal benchmark
- SC-CoQ策略简单有效，利用了问题格式间的内在难度梯度，是一种新颖的prompt engineering方法
- 覆盖幽默+讽刺、4种领域、6种问题格式，评测维度远超先前工作
- 54,000 QA对的规模在punchline领域最大

## 局限性 / 可改进方向
- 仅关注静态图文对，未覆盖视频中的动态梗（如短视频的节奏感、反转）
- SC-CoQ依赖预定义的问题难度排序，不同模型对不同格式的真实难度可能不同
- 人类baseline样本偏小（每格式100条），可能高估人类在大规模数据上的表现
- 未探索用训练数据fine-tune提升MLLM梗图理解

## 与相关工作的对比
- **vs MORE (Desai et al. 2022)**：MORE仅做讽刺解释，单一问题格式（3,510对）；PunchBench覆盖幽默+讽刺，6种格式，54,000 QA对。
- **vs HUB (Hessel et al. 2023)**：HUB仅做漫画幽默理解（704对）；PunchBench跨4种领域，规模大8.5倍。
- **vs MTSD (Castro et al. 2019)**：MTSD仅做讽刺分类；PunchBench增加了推理层级和shortcut消除。

## 启发与关联
- 同义/反义替换消除shortcut的方法可应用于VQA、视觉推理等其他multimodal benchmark的robustness评测
- SC-CoQ的"先简后难"链式提问可以应用到数学推理、代码生成等领域
- 梗图理解需要文化背景知识+图文关系推理+隐含语义理解，是评测"真正理解"的好任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 同义/反义标题消除shortcut和SC-CoQ策略新颖，但benchmark本身是增量贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 10个模型、4种推理策略、6种格式、3种标题条件，维度非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，不过部分公式符号可以简化
- 价值: ⭐⭐⭐⭐ 揭示了MLLM在高级语义理解上的短板，SC-CoQ实用性强
