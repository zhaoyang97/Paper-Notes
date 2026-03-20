# Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System

**会议**: AAAI 2026  
**arXiv**: [2508.06059](https://arxiv.org/abs/2508.06059)  
**代码**: [https://trustworthycomp.github.io/Fact2Fiction/](https://trustworthycomp.github.io/Fact2Fiction/)  
**领域**: AI安全 / LLM Agent / 事实核查攻击  
**关键词**: 投毒攻击, Agent事实核查, 声明分解, 对抗性证据, 知识库污染  

## 一句话总结
提出 Fact2Fiction，首个针对 Agent 化事实核查系统（如 DEFAME、InFact）的投毒攻击框架：通过 Planner Agent 模拟声明分解生成子问题，利用系统的 justification 反向工程关键推理点来制作定向恶意证据，并按重要性分配投毒预算，在仅 1% 投毒率下比 SOTA PoisonedRAG 高 8.9%-21.2% 的攻击成功率。

## 研究背景与动机

1. **领域现状**：SOTA 事实核查系统（InFact、DEFAME）已从简单 RAG 进化为 Agent 范式——用 LLM Agent 将复杂声明分解为子声明，逐个检索证据并验证，再汇聚结果给出最终判定+justification（解释）。这种分解策略本身就构成了对传统投毒攻击的天然防御。
2. **现有痛点**：现有投毒攻击（如 PoisonedRAG）只针对主声明制作恶意证据。当 Agent 系统把声明拆成多个具体子问题时，这些泛泛的恶意证据既不容易被针对子问题的检索器检索到，即使被检索到也不够具体来误导子声明验证。例如 PoisonedRAG 在 DEFAME 上的 ASR 从 Simple 系统的 57.4% 降到 42.4%。
3. **核心矛盾**：Agent 化事实核查系统提升了准确性，但其 justification（透明解释）暴露了推理逻辑的关键证据和推理模式——这形成了"透明性-安全性"的根本矛盾。
4. **本文要解决什么**：如何设计针对 Agent 化事实核查系统的有效投毒攻击？需要解决：(a) 恶意证据与子声明的语义匹配；(b) 利用 justification 做定向攻击；(c) 有限预算下的最优分配。
5. **切入角度**：反向工程 Agent 系统的两个特性——声明分解策略和 justification 输出。用两个协作 Agent（Planner + Executor）模拟分解过程并利用透明解释来制作针对性恶意内容。
6. **核心 idea 一句话**：模拟 Agent 事实核查的声明分解策略 + 利用 justification 反向制作定向恶意证据 + 按子声明重要性分配投毒预算。

## 方法详解

### 整体框架
Fact2Fiction 由两个 LLM Agent 组成：(1) **Planner** 做攻击规划（声明分解→答案规划→预算规划→查询规划）；(2) **Executor** 执行攻击（生成恶意证据并注入知识库）。攻击前先查询目标系统获取初始 verdict 和 justification。

### 关键设计

1. **Sub-question Decomposition（子问题分解）**:
   - 做什么：模拟 Agent 事实核查系统的分解策略，将目标声明拆为最多 10 个子问题
   - 核心思路：让 Planner 角色扮演事实核查员，对声明 $c_i$ 生成代理子问题集 $\mathcal{Q}_i = \{q_1, \ldots, q_{l_i}\}$。虽然不知道目标系统的确切分解方式，但实验证明代理分解对不同分解策略（InFact 的显式分解 vs DEFAME 的隐式动态分解）都有效
   - 设计动机：针对子问题制作的恶意证据比针对主声明的更容易被检索，且更有针对性

2. **Answer Planning（答案规划）— 利用 Justification**:
   - 做什么：为每个子问题生成定向对抗性答案，直接反驳 justification 中的关键推理
   - 核心思路：给定 justification $j_i$，为每个 $q_k$ 生成答案 $a_k$ 使其精确反驳 justification 中的关键论点。例如 justification 说"该法案保护个人种植权"，则对抗性答案要说"该法案对食品分享和交易施加严格登记要求"——不是模糊地反对，而是针锋相对地反驳具体论点
   - 设计动机：justification 暴露了系统做出正确判断的关键证据和推理路径，攻击者可以精确地打击这些关键支撑点

3. **Budget Planning（预算规划）**:
   - 做什么：在子问题之间最优分配有限的投毒预算
   - 核心思路：Planner 为每对 $(q_k, a_k)$ 基于其对 justification 的重要性打分 $w_k$（0-10），按比例分配预算：$m_k = \lceil m \cdot w_k / \sum_s w_s \rceil$。优先投入更多恶意证据到系统推理中最关键的子声明
   - 设计动机：低预算场景下（1% 投毒率），均匀分配会浪费资源在不重要的子问题上。消融实验证明预算规划在 1% 率下贡献 7.8% ASR

4. **Query Planning（查询规划）**:
   - 做什么：为每个子问题生成代理搜索查询，与恶意证据拼接提升可检索性
   - 核心思路：$e_{k,h} = s_p \oplus \tilde{e}_{k,h}$，将查询与恶意证据拼接，使其在语义搜索中更容易被检索到（因为系统检索时也会用类似的查询）
   - 设计动机：恶意证据必须先被检索到才能生效。针对子问题的查询比主声明更精确匹配

## 实验关键数据

### 主实验（1% 投毒率，~8 条恶意证据/声明）

| 攻击方法 | DEFAME ASR | InFact ASR | Simple ASR |
|---------|-----------|-----------|-----------|
| Naive | 17.8% | 14.6% | 24.2% |
| Prompt Injection | 19.3% | 16.1% | 34.7% |
| PoisonedRAG | 33.5% | 35.8% | 42.4% |
| **Fact2Fiction** | **42.4%** | **46.0%** | **43.4%** |

### 攻击效率对比

| 指标 | PoisonedRAG | Fact2Fiction | 说明 |
|------|-----------|-------------|------|
| 达到~42% ASR (DEFAME) | 需 8% 投毒率 | 仅需 1% | Fact2Fiction 效率高 8 倍 |
| 达到~45% ASR (InFact) | 需 8% 投毒率 | 仅需 1% | Fact2Fiction 效率高 8-16 倍 |

### 消融实验（1% 投毒率，DEFAME）

| 配置 | ASR | 说明 |
|------|-----|------|
| Fact2Fiction (完整) | 42.4% | 最优 |
| w/o Answer Planning | 40.9% | 不用 justification 制作答案 |
| w/o Budget Planning | 34.6% | 均匀分配预算，-7.8% |
| w/o Query Planning | 39.4% | 不针对子问题做查询拼接 |

### 关键发现
- **Agent 化分解确实提供天然防御**：PoisonedRAG 在 Simple 上 ASR 57.4%，但在 DEFAME/InFact 上降到 42-45%。但 Fact2Fiction 重新突破了这道防线
- **Justification 是双刃剑（透明性-安全性矛盾）**：Answer Planning（利用 justification）在低预算下贡献最高 12.4% ASR 提升
- **证据质量比可检索性更重要**：在 1% 投毒率下，Fact2Fiction 的 SIR（64.8%）略低于 PoisonedRAG（65.6%），但 ASR（42.4%）远高（33.5%），说明定向制作的恶意证据质量更高
- **攻击存在饱和点**：Naive/Prompt Injection 在 2% 率就饱和；PoisonedRAG 在 4-8% 饱和；Fact2Fiction 持续到 8% 仍在增长——因为其定向策略能更有效利用额外预算
- **现有防御（改述/恶意检测/PPL 检测）均无效**：Fact2Fiction 生成的恶意证据与清洁证据 PPL 分布重叠度高，无法通过统计方法检测

## 亮点与洞察
- **"透明性-安全性矛盾"**是一个深刻的发现——事实核查系统越透明（提供越详细的 justification），越容易被定向攻击。这对 AI 可解释性/可信赖研究有重要启示
- **模拟分解+利用 justification 的攻击策略**巧妙地利用了 Agent 系统的两个特性来反制它：分解策略被用来制作匹配的恶意内容，透明解释被用来找出攻击的最佳着力点
- **预算规划的有效性**揭示了一个重要的攻防不对称——攻击者可以聚焦资源在最关键的子声明上，而防御者必须保护所有子声明
- **从攻击角度为防御提供了建设性建议**：论文明确指出未来应研究"迫使攻击在低 ASR 就达到饱和"的鲁棒事实核查系统

## 局限性 / 可改进方向
- 攻击需要先查询目标系统获取 justification，在受限场景（如速率限制）下可能不可行
- 只测试了 AVeriTeC 数据集（500 条声明），缺少更大规模的验证
- 只用 GPT-4o-mini 作为主要 LLM backbone，虽然附录扩展了 Gemini/DeepSeek 但范围仍有限
- 论文聚焦于攻击，未提出有效的防御方案——只证明了现有防御不行
- 子问题分解依赖 LLM 的质量，如果 Planner 生成的子问题与目标系统的分解差异太大，效果可能下降

## 相关工作与启发
- **vs PoisonedRAG**：PoisonedRAG 只针对主声明制作泛泛的恶意证据，Fact2Fiction 针对子声明做定向攻击，ASR 高 8.9%-21.2%，且效率高 8-16 倍
- **vs Prompt Injection**：Prompt Injection 在 Agent 系统上几乎无效（ASR ~19%），因为恶意指令与正常证据语义差异太大，无法被检索
- **对事实核查系统设计的启示**：(a) justification 应限制透露推理细节；(b) 子声明验证应增加冗余和交叉验证；(c) 检索应增加对抗性鲁棒性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个针对 Agent 事实核查系统的攻击框架，"透明性-安全性矛盾"是深刻的洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 3 个受害系统 × 4 种投毒率 × 4 个基线 + 完整消融 + 3 种防御 + 跨检索器/LLM 鲁棒性验证
- 写作质量: ⭐⭐⭐⭐⭐ 威胁模型定义清晰，EQ 驱动的实验设计很好，示例生动且直观
- 价值: ⭐⭐⭐⭐⭐ 对AI安全和事实核查社区都有重要意义，揭示了 Agent 系统新的攻击面
