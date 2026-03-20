# MMhops-R1: Multimodal Multi-hop Reasoning

**会议**: AAAI 2026  
**arXiv**: [2512.13573](https://arxiv.org/abs/2512.13573)  
**代码**: [https://github.com/taoszhang/MMhops-R1](https://github.com/taoszhang/MMhops-R1)  
**领域**: 多模态VLM / 知识增强推理  
**关键词**: 多模态多跳推理, 强化学习, 检索增强生成, 知识库VQA, 动态规划  

## 一句话总结
提出了 MMhops 基准（31K 样本、3-4 跳推理深度）和 MMhops-R1 框架，通过强化学习训练 MLLM 自主规划推理路径、动态调用图像/文本检索器，实现多模态多跳推理，7B 模型超越 72B 基线和现有 mRAG 方法。

## 背景与动机
现有 MLLM 的推理能力主要集中在单步推理（空间推理、数学推理等），但现实世界的复杂问题往往需要多步整合多模态信息和外部知识。现有知识库 VQA 数据集（OK-VQA、INFOSEEK 等）通常只需要"一步视觉识别 + 一步知识检索"的浅层推理，即使 E-VQA 扩展到了两跳，也仅限于文本模态且推理路径长度固定。这限制了模型多模态多跳推理能力的评估和训练。同时，现有多模态 RAG 方法采用静态流程，无法根据问题复杂度动态调整推理深度。

## 核心问题
如何让多模态大模型具备动态多跳推理能力——即根据问题复杂度自主决定何时调用图像检索、何时调用文本检索、何时生成答案，而不是依赖预定义的固定推理流程？这个问题重要在于：真实世界的视觉问答往往需要模型从图像出发，经过多步外部知识检索和跨模态信息整合才能得到答案。

## 方法详解

### 整体框架
MMhops-R1 包含两部分：(1) MMhops 数据集的构建；(2) 基于 RL 的动态多模态检索增强生成框架。

输入：一组图像 + 自然语言问题。模型在每个时间步先进行思考（thinking），然后从三个动作中选择一个：调用图像检索器（image_search）、调用文本检索器（text_search）、或生成最终答案（answer）。检索结果作为观察（observation）反馈给模型，循环交互直到生成答案或达到最大轮次（T=4）。

### 关键设计
1. **MMhops 数据集构建**：包含 Bridging（85%）和 Comparison（15%）两种推理类型。Bridging 通过迭代扩展法构建——从单跳 VQA 出发，以答案作为桥接实体，在其 Wikipedia 页面上生成新的子问题，再将子问题合并为多跳问题。Comparison 则从 Wikipedia 中找到语义相似的实体对，基于可量化属性生成比较问题，并用图像替换实体名。所有样本都需要 3-4 步推理，且必须使用外部知识。

2. **动态动作空间与多步检索交互**：模型的动作空间为 {think, image_search, text_search, answer}，其中 image_search 通过 CLIP 检索相关图像信息，text_search 通过 E5 模型检索 top-3 相关段落。每个动作都用特定 XML 标签格式化，如果格式错误会收到惩罚信号。这种设计让模型学会自主规划推理路径，而非依赖手动设计的流程。

3. **复合奖励函数设计**：总奖励 R = α·R_outcome + β·R_format + γ·R_action。R_outcome 是答案正确性的二值奖励；R_format 奖励格式正确；R_action 是工具使用奖励，关键设计是它被 outcome 和 format 门控——只有答案正确且格式规范时，工具使用才获得奖励，防止模型"乱调工具但答案错误"的行为。

### 损失函数 / 训练策略
- 基于 DAPO 的策略优化目标，使用 clip 操作和动态采样策略（每组 G=8 个响应至少包含一个正确样本）
- Loss masking：外部检索返回的 token 不参与梯度计算，只优化模型自身生成的推理和动作 token
- 骨干模型为 Qwen2.5-VL-7B-Instruct，训练 1 个 epoch，lr=1e-6
- 知识库：100K Wikipedia 文章，每篇配一张图像

## 实验关键数据
| 数据集 | 指标 | MMhops-R1 (7B) | Qwen2.5-VL-72B | GPT-4o | Gemini-2.5-Pro |
|--------|------|------|----------|------|------|
| MMhops Bridging | Overall | **51.35** | 34.39 | 36.62 | 53.98 |
| MMhops Comparison | Overall | **22.01** | 7.59 | 8.76 | 29.39 |
| INFOSEEK | Overall | **33.2** | - | - | - |
| E-VQA 2-hop | Acc | **23.3** | - | - | 22.8 (PaLM) |

### 消融实验要点
- 去掉 R_action 后 Bridging Overall 从 51.35 降到 47.57，Comparison 从 22.01 降到 20.62
- 去掉 R_format 后 Comparison 暴跌到 14.42，说明格式约束对跨图像推理特别重要
- 同时去掉 R_action 和 R_format，Bridging 降到 41.75，Comparison 降到 13.03
- 交互轮次从 2 增到 4 时性能持续提升（Overall 从 39.93 到 51.35），5 轮无显著收益但增加开销
- 4 步推理最适合 MMhops 数据集

## 亮点
- **RL 驱动的动态多模态 RAG**：第一个将 RL 用于多模态多跳推理的工作，让模型自主学习何时/如何检索，而非靠 prompt engineering
- **门控式工具使用奖励**：R_action 被 R_outcome 和 R_format 门控，优雅地避免了"乱调工具"的问题
- **7B 超 72B**：通过 RL 训练，7B 模型在 Bridging 上超过 72B 直接推理 16.96%，充分说明推理能力可通过训练策略弥补模型规模差距
- **数据集构建方法可推广**：迭代扩展法可以系统地将单跳数据集扩展为多跳数据集

## 局限性 / 可改进方向
- 知识库仅限 100K Wikipedia 文章，覆盖范围有限，扩大知识库可能进一步提升性能
- Comparison 任务整体准确率仍然较低（最好的 Gemini-2.5-Pro 也只有 29.39%），跨图像推理仍是重大挑战
- 只在 Qwen2.5-VL-7B 上验证，未测试更大或不同架构的模型
- 最大交互轮次固定为 4，未探索动态终止策略
- 数据集构建依赖 GPT-4o，可能引入偏差

## 与相关工作的对比
- **vs Search-R1/ReSearch**：这些方法将 RL 用于文本多跳 RAG，MMhops-R1 将此范式扩展到多模态，增加了图像检索动作
- **vs OmniSearch (Gemini)**：OmniSearch 依赖手动 prompt 或 SFT 设计规划流程，缺乏自主学习策略的能力；MMhops-R1 的 RL 训练让 7B 模型超过 GPT-4o + OmniSearch
- **vs 传统 mRAG（Wiki-LLaVA, EchoSight）**：传统方法为单跳设计，静态流程无法处理多跳问题

## 启发与关联
- 动态 mRAG + RL 的范式可以迁移到其他需要多步外部交互的任务（如复杂代码生成、科学推理）
- 门控奖励设计（工具使用奖励被答案正确性门控）是一个通用的 RL reward shaping 技巧
- 与 `ideas/multimodal_vlm/20260318_vhd_adaptive_visual_reinjection.md` 中的视觉信息动态注入思想有潜在关联
- 与 `ideas/llm_nlp/20260317_retrieval_reasoning_decoupled_lct.md` 中检索与推理解耦的思路互补

## 评分
- 新颖性: ⭐⭐⭐⭐ 第一个将 RL 应用于多模态多跳 RAG，数据集构建方法系统且可扩展
- 实验充分度: ⭐⭐⭐⭐ 多类 baseline 对比完整，消融实验覆盖主要设计，跨数据集泛化验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，数据集构建流程描述详细
- 价值: ⭐⭐⭐⭐ 提出了重要的 benchmark 和有效的方法，对多模态推理研究有推动作用
