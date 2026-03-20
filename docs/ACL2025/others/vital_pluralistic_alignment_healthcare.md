# VITAL: A New Dataset for Benchmarking Pluralistic Alignment in Healthcare

**会议**: ACL 2025  
**arXiv**: [2502.13775](https://arxiv.org/abs/2502.13775)  
**代码**: [https://github.com/anudeex/VITAL.git](https://github.com/anudeex/VITAL.git)  
**领域**: 医学图像 / LLM 对齐  
**关键词**: pluralistic alignment, healthcare, LLM benchmark, value diversity, dataset  

## 一句话总结
本文构建了首个面向医疗健康领域的多元化对齐（pluralistic alignment）基准数据集 VITAL，包含 13.1K 价值观情境和 5.4K 多选题，并通过对 8 个 LLM 的广泛评估表明，现有多元化对齐技术（尤其是 ModPlural）在医疗场景下表现不佳，简单的 prompting 反而效果更好。

## 研究背景与动机

1. **领域现状**：LLM 对齐技术（如 RLHF、DPO）日益成熟，但通常建模的是"平均"偏好，忽略了不同文化、人口、社区之间价值观的多样性。Sorensen et al. (2024) 提出了多元化对齐框架，定义三种模式：Overton（覆盖所有多样视角）、Steerable（按用户指定属性调控）、Distributional（匹配现实世界分布）。Feng et al. (2024) 提出了 ModPlular 多 LLM 协作方案。
2. **现有痛点**：(1) 现有对齐数据集（OpinionQA、GlobalOpinionQA 等）没有专注于医疗健康领域；(2) 医疗场景中的多元性尤为关键——文化、宗教、个人价值观都会影响健康决策；(3) 不知道现有多元对齐技术在医疗领域是否有效。
3. **核心矛盾**：通用多元化对齐技术可能无法迁移到特定领域。医疗场景的错误对齐可能导致有害的健康建议或信念同质化。
4. **本文要解决什么？** (1) 构建首个医疗健康多元化对齐基准；(2) 系统评估现有技术在该基准上的表现；(3) 探索改进方向。
5. **切入角度**：从医疗健康这个高敏感多争议领域入手，利用调查、民意测验和道德困境场景构建多元化数据集。
6. **核心idea一句话**：医疗健康领域需要专门的多元化对齐评估基准和方法，现有通用方案在此领域效果有限。

## 方法详解

### 整体框架
构建 VITAL 数据集 → 使用四种对齐技术（Vanilla、Prompting、MoE、ModPlular）在 8 个 LLM 上评估 → 分析三种多元化模式（Overton、Steerable、Distributional）的表现。

### 关键设计

1. **数据集构建（VITAL）**:
   - 做什么：构建涵盖 13.1K 价值观情境 + 5.4K 多选题的医疗健康多元化对齐基准
   - 核心思路：从多个调查和道德数据集（OpinionQA、GlobalOpinionQA、MoralChoice 等）中收集多选题，使用 FLAN-T5 的 few-shot 分类过滤出与健康相关、具有多元观点、需要行动的样本
   - 数据分布：Overton 模式 1,649 文本样本，Steerable 模式 15,340 样本（文本+QA），Distributional 模式 1,857 QA 样本
   - 质量验证：人工标注验证 10% 样本，80% 被确认为健康相关（Fleiss' Kappa: 0.49）

2. **评估技术**:
   - **Vanilla**：直接使用 LLM 无特殊指令
   - **Prompting**：在提示中加入多元化指令
   - **MoE**：主 LLM 作为路由器选择最合适的社区 LLM（perspective/culture LLM），将社区 LLM 响应提供给主 LLM 生成最终答案
   - **ModPlular**：主 LLM 与多个社区 LLM 协作——Overton 模式下拼接社区消息做多文档摘要，Steerable 模式下选择最相关社区 LLM，Distributional 模式下聚合社区概率分布

3. **评估指标**:
   - Overton：NLI 模型计算值覆盖率（句子级别 entailment），加上人工评估和 GPT-as-Judge
   - Steerable：准确率（最终回复是否保持指定的引导属性）
   - Distributional：Jensen-Shannon 距离（越低越好，表示与真实分布越接近）

### LLM 代理实验
探索用轻量级 LLM agents（基于 Mistral-7B 的角色扮演 agents）替代微调社区 LLM。构建健康专用 agent 池，由 GPT-4o 选择最相关的 6 个 agent。6 个 agents 的 NLI 覆盖率为 44.16%（vs 原始社区 LLM 的 47.84%），10 个 agents 提升至 49.37%。

## 实验关键数据

### 主实验：Overton 模式覆盖率（%）

| 方法 | LLaMA2-7B | Gemma-7B | Qwen2.5-7B | LLaMA3-8B | ChatGPT | 平均 |
|------|-----------|----------|------------|-----------|---------|------|
| Aligned Vanilla | 20.76 | 38.60 | 32.41 | 18.93 | 26.70 | 26.10 |
| + Prompting | 22.88 | **40.61** | **34.42** | 27.41 | **32.22** | **30.46** |
| + MoE | 19.58 | 26.00 | 28.14 | 24.70 | 18.84 | 22.79 |
| + ModPlural | 15.38 | 22.18 | 22.30 | 24.51 | 18.06 | 20.09 |

### 消融实验：社区 LLM 来源对比

| 配置 | LLaMA2-7B | LLaMA3-8B | Gemma-7B |
|------|-----------|-----------|----------|
| Perspective 社区 LLM | 15.15 | 23.82 | 22.37 |
| Culture 社区 LLM | 17.61 | 25.11 | 22.45 |
| 健康专用 LLM 作主模型 | 12.00 (ModPlural) | - | - |
| 6个 LLM Agents 替代社区 LLM | 44.16 (NLI) | - | - |
| 10个 LLM Agents | 49.37 (NLI) | - | - |

### 关键发现
- **Prompting > ModPlural**：在所有 8 个模型和 3 种对齐模式中，简单的 prompting 一致优于更复杂的 ModPlural 多 LLM 协作方案，最大差距达 55.5%
- **模型规模无关**：增大模型规模并未带来一致的性能提升
- **NLI 评估偏差**：Overton 覆盖率与回复句子数量正相关，ModPlural 的摘要倾向将多个论点压缩到一个句子导致 NLI 分数偏低
- **直接替换专用 LLM 无效**：用健康专用 LLM（mental-llama2-7b）作主模型未带来显著提升，说明简单"打补丁"不够
- **Distributional 模式相对接近**：ModPlural 在分布模式下表现较好，与其他方法差距较小

## 亮点与洞察
- **反直觉发现**：越复杂的多 LLM 协作（ModPlular, MoE）在医疗场景反而不如简单 prompting，说明通用多元化方案在特定领域可能失灵。这提醒我们 domain-specific 才是关键
- **Agent 替代社区 LLM 的可行性**：10 个轻量级 agents 超过了微调社区 LLM 的覆盖率，且无需昂贵微调和动态可扩展，是值得探索的方向
- **数据集设计**：将文本情境和 QA 问题结合，覆盖三种多元化模式的基准构建思路可迁移到其他敏感领域（法律、教育等）

## 局限性 / 可改进方向
- 数据构建依赖 FLAN-T5 过滤，人工验证仅覆盖 10% 且一致性中等（Kappa 0.49），数据质量可能有噪声
- Overton 评估使用 NLI 模型在句子级别判断 entailment，存在句子数量偏差和语义压缩问题
- 未评估最新 LLM（如 GPT-4、LLaMA3-70B 等更强模型）
- 未提出新的对齐方法，仅做了基准评估

## 相关工作与启发
- **vs ModPlural (Feng et al., 2024)**: ModPlular 是当前 SOTA 多元化对齐方法，本文揭示其在医疗领域的不足，但 ModPlural 在原始通用领域表现较好
- **vs OpinionQA (Santurkar et al., 2023)**: OpinionQA 是广泛使用的对齐评估数据集但不专注健康，VITAL 填补了这一空白
- **vs MoralChoice (Liu et al., 2024)**: MoralChoice 提供道德场景但非健康特定，VITAL 从中筛选并扩展了健康相关子集

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个医疗健康多元化对齐基准，填补了重要空白
- 实验充分度: ⭐⭐⭐⭐ 8 个模型 × 4 种方法 × 3 种模式，评估全面；但缺少新方法的提出
- 写作质量: ⭐⭐⭐⭐ 结构清晰，分析详细
- 价值: ⭐⭐⭐⭐ 数据集和反直觉发现对社区有重要参考价值
