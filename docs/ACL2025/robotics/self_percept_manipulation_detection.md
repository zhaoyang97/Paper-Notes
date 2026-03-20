# SELF-PERCEPT: Introspection Improves LLMs' Detection of Multi-Person Mental Manipulation in Conversations

**会议**: ACL 2025  
**arXiv**: [2505.20679](https://arxiv.org/abs/2505.20679)  
**代码**: [https://github.com/danushkhanna/self-percept](https://github.com/danushkhanna/self-percept)  
**领域**: LLM/NLP  
**关键词**: mental manipulation detection, multi-turn dialogue, multi-person conversation, prompting framework, Self-Perception Theory  

## 一句话总结

提出 SELF-PERCEPT 两阶段 prompting 框架，借鉴心理学自我知觉理论（Self-Perception Theory），引导 LLM 先观察对话参与者的行为线索再推断内在态度，显著提升多人多轮对话中心理操纵的检测效果。

## 研究背景与动机

1. **心理操纵检测的重要性**：心理操纵（mental manipulation）是一种隐蔽的人际交往中的滥用行为，通过欺骗性手段控制他人思想和情感以谋取私利，对受害者心理健康构成严重威胁。
2. **现有数据集的局限**：此前代表性数据集 MentalManip 基于电影台本（Cornell Movie Corpus），仅覆盖**两人对话**，且分布不均衡，难以反映真实世界中多人博弈的复杂操纵场景。
3. **多人多轮场景的挑战**：现实中操纵行为往往发生在群体协商中（如团队会议、社交场合），涉及多参与者和多轮交互，LLM 在此类场景下的检测性能严重不足。
4. **传统 prompting 方法的缺陷**：Zero-shot、Few-shot、CoT 等方法聚焦于逐步推理，难以捕捉言行不一致等隐性操纵信号，尤其在区分善意说服与恶意操纵时表现不佳。
5. **心理学理论的启发**：自我知觉理论（SPT）提出个体通过观察自身行为来推断内在态度，这一认知机制可迁移至 LLM，使其具备"先观察行为、再推断意图"的分析能力。
6. **研究目标**：构建更贴近真实场景的多人操纵检测数据集，并设计受心理学启发的 prompting 框架，提升 LLM 对复杂对话中操纵行为的识别精度。

## 方法详解

### 整体框架

SELF-PERCEPT 是一个两阶段 prompting 框架，模拟人类"行为观察→自我推断"的认知过程。与 CoT 侧重逐步逻辑推理不同，SELF-PERCEPT 显式地将行为线索提取与态度推断解耦，使 LLM 能更好地处理多人对话中的复杂社交动态。

### 模块一：MultiManip 数据集构建

- **数据来源**：从真人秀节目 *Survivor* 的 Fandom 公开转录文本中提取多人多轮对话，竞技性质决定了丰富的操纵行为样本。
- **规模与分布**：共 220 段对话，**操纵/非操纵均衡分布**，覆盖 11 种操纵技术（如指责 accusation、羞辱 shaming、否认 denial、假装无辜 feigning innocence 等）。
- **标注流程**：5 名标注者回答两个问题——$\mathcal{Q}_1$（是否含操纵，二分类）和 $\mathcal{Q}_2$（操纵类型，多标签）。采用多数投票聚合，Fleiss' Kappa = 0.429（中等一致性），反映任务的内在挑战性。
- **预处理**：使用 Llama-3.1-70B 进行初步过滤，GPT-4o/Llama-3.1-8B 交叉验证，人工最终校验，多模型策略缓解 LLM 偏置。

### 模块二：Stage 1 — Self-Percept（自我知觉/行为观察）

- 输入为完整多人对话，LLM 被要求全面观察和分析每位参与者的**言语线索**（verbal cues）和**非言语线索**（non-verbal cues）。
- 重点识别**言行不一致**：例如某人口头同意但语气中带有叹息（sigh），可能暗示被动攻击意图。
- 输出为结构化的行为观察列表，记录潜在矛盾和可疑操纵信号，作为下一阶段推理的基础。

### 模块三：Stage 2 — Self-Inference（自我推断/态度推理）

- 基于 Stage 1 的行为观察结果，LLM 推断每位参与者的**内在态度与信念**。
- 特别关注是否存在操纵行为，并按预定义的 11 种操纵类型进行分类。
- 输出简洁的推断结论，旨在捕捉人际动态的本质。

### 评估策略

- 在 MultiManip（本文数据集）和 MentalManip（已有数据集）上评估。
- 模型：GPT-4o、Llama-3.1-8B。
- Baseline：Zero-Shot、Few-Shot、Chain-of-Thought。
- 指标：Accuracy、Precision ($P$)、Recall ($R$)、Macro $F_1$。

## 实验

### 表1：MultiManip 数据集上的多标签操纵检测

| 模型 | Prompting | Acc. | $P$ | $R$ | $F_1$ |
|------|-----------|------|-----|-----|-------|
| GPT-4o | Zero-Shot | 0.27 | 0.20 | 0.31 | 0.16 |
| GPT-4o | Few-Shot | 0.39 | 0.19 | 0.21 | 0.22 |
| GPT-4o | CoT | 0.34 | 0.21 | 0.32 | 0.34 |
| GPT-4o | **SELF-PERCEPT** | **0.42** | **0.31** | 0.20 | **0.37** |
| Llama-3.1-8B | Zero-Shot | 0.11 | 0.09 | 0.37 | 0.29 |
| Llama-3.1-8B | Few-Shot | 0.22 | 0.17 | 0.36 | 0.13 |
| Llama-3.1-8B | CoT | 0.28 | 0.23 | 0.26 | 0.10 |
| Llama-3.1-8B | **SELF-PERCEPT** | **0.30** | 0.17 | 0.26 | **0.34** |

### 表2：MentalManip 数据集上的多标签操纵检测

| 模型 | Prompting | Acc. | $P$ | $R$ | $F_1$ |
|------|-----------|------|-----|-----|-------|
| GPT-4o | Zero-Shot | 0.11 | 0.30 | 0.62 | 0.38 |
| GPT-4o | Few-Shot | 0.22 | 0.39 | 0.53 | 0.39 |
| GPT-4o | CoT | 0.35 | 0.37 | 0.56 | 0.43 |
| GPT-4o | **SELF-PERCEPT** | **0.45** | 0.34 | 0.55 | **0.47** |
| Llama-3.1-8B | Zero-Shot | 0.02 | 0.11 | 0.56 | 0.17 |
| Llama-3.1-8B | Few-Shot | 0.04 | 0.07 | 0.35 | 0.11 |
| Llama-3.1-8B | CoT | 0.19 | 0.14 | 0.38 | 0.18 |
| Llama-3.1-8B | **SELF-PERCEPT** | **0.23** | **0.21** | 0.32 | **0.19** |

### 关键发现

1. **SELF-PERCEPT 一致性优势**：在两个数据集、两个模型上，SELF-PERCEPT 在 Accuracy 和 $F_1$ 上均取得最优，展现出跨数据集和跨模型的稳健性。
2. **精确率-召回率权衡**：SELF-PERCEPT 的 Recall 略低于 Zero-Shot/CoT，但 Precision 显著提升（GPT-4o 在 MultiManip 上 $P$ = 0.31 vs CoT 的 0.21），说明行为观察阶段有效减少了误报。
3. **SHAP 可解释性分析**：Stage 1 正确捕获"anxious""situation""teamwork"等心理压力和说服意图词汇（负 SHAP 值→判定为操纵），而 CoT 过度依赖中性词如"game""desire"导致误判为非操纵。
4. **绝对性能仍有限**：最高 $F_1$ 仅 0.47（GPT-4o + SELF-PERCEPT 在 MentalManip 上），表明多人多轮操纵检测仍是极具挑战性的任务。

## 亮点

- **心理学理论驱动的 prompting 设计**：将自我知觉理论转化为可操作的两阶段提示流程，提供了"领域理论 → NLP 方法"的优雅迁移范式。
- **行为观察与意图推断解耦**：解决了 CoT 在处理隐性社交信号时"一步到位"的推理瓶颈，通过显式中间表示提升可解释性。
- **MultiManip 数据集的实际价值**：基于真人秀而非虚构剧本，多人多轮设计更贴近现实操纵场景，填补了该领域数据空白。
- **SHAP 可视化增强可信度**：通过词级归因对比展示 SELF-PERCEPT 与 CoT 的注意力差异，直观证明方法有效性。

## 局限

- **数据集规模极小**：仅 220 样本，统计检验力不足，性能提升（如 +5% $F_1$）的显著性难以严格验证。
- **领域单一**：数据仅来源于 *Survivor* 真人秀，竞技性对话的操纵模式可能不代表日常场景（如职场 PUA、亲密关系操纵）。
- **缺乏微调实验**：框架仅在 inference-time prompting 层面验证，未探索将 SELF-PERCEPT 的行为分析能力蒸馏到小模型中的可行性。
- **评估指标有限**：仅报告标准分类指标，未评估操纵**类型**级别的细粒度性能（如哪些操纵技术更容易/更难检测）。
- **Recall 损失**：Precision 提升以 Recall 下降为代价，对于安全场景中"宁可错杀不可放过"的需求不够理想。

## 相关工作

- **心理操纵检测**：MentalManip (Wan et al., 2024) 首个心理操纵数据集但限于两人对话；Ma et al. (2024) 用 intent-aware prompting 做二分类但忽略多人场景。
- **有毒语言检测**：DeTexD (Yavnyi et al., 2023)、troll tweets 检测 (Miao et al., 2020) 关注显式毒性，对隐性操纵覆盖不足。
- **多轮对话理解**：Li et al. (2022)、Yang et al. (2022) 推进了多轮对话建模，但未专门针对操纵行为。
- **LLM prompting**：Chain-of-Thought (Wei et al., 2022) 是里程碑式工作，SELF-PERCEPT 在此基础上引入心理学先验实现任务特化。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 自我知觉理论到 prompting 的迁移思路新颖，两阶段行为-推断解耦设计有独创性
- **技术质量**: ⭐⭐⭐ — 方法清晰但本质为 prompt engineering，无模型层面创新；数据集规模偏小
- **实用价值**: ⭐⭐⭐⭐ — 心理操纵检测具有重要社会安全意义，框架可直接应用于在线平台内容审核
- **写作质量**: ⭐⭐⭐⭐ — 动机阐述充分，心理学理论引入自然；实验分析含 SHAP 可视化增强说服力
