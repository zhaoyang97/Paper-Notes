---
title: >-
  [论文解读] LLMs Get Lost In Multi-Turn Conversation
description: >-
  [ICLR 2026][LLM评测][多轮对话] 本文通过"指令分片 + 模拟对话"的大规模实验（20 万+ 模拟对话、15 个 LLM）证明：所有顶尖 LLM 在多轮欠定对话中相比单轮完整指令平均掉点 39%，而这种退化主要不是能力下降，而是**可靠性崩溃**——模型一旦在某轮走错就"迷失"且无法恢复。
tags:
  - "ICLR 2026"
  - "LLM评测"
  - "多轮对话"
  - "欠定指令"
  - "可靠性"
  - "模拟评测"
  - "Lost-in-Conversation"
---

# LLMs Get Lost In Multi-Turn Conversation

**会议**: ICLR 2026  
**arXiv**: [2505.06120](https://arxiv.org/abs/2505.06120)  
**OpenReview**: [https://openreview.net/forum?id=VKGTGGcwl6](https://openreview.net/forum?id=VKGTGGcwl6)  
**代码**: 待确认  
**领域**: LLM 评测 / 多轮对话  
**关键词**: 多轮对话、欠定指令、可靠性、模拟评测、Lost-in-Conversation  

## 一句话总结
本文通过"指令分片 + 模拟对话"的大规模实验（20 万+ 模拟对话、15 个 LLM）证明：所有顶尖 LLM 在多轮欠定对话中相比单轮完整指令平均掉点 39%，而这种退化主要不是能力下降，而是**可靠性崩溃**——模型一旦在某轮走错就"迷失"且无法恢复。

## 研究背景与动机
**领域现状**：LLM 本质是对话接口，用户常常一开始说不清需求，要靠多轮交互逐步澄清。可现实是，LLM 评测几乎都停留在"单轮、完整指令"设定下，与真实使用场景脱节。

**现有痛点**：已有的多轮评测大多把对话当作 **episodic（情景式）**——每一轮都是一个能被孤立打分的自包含子任务。这种设计回避了人类对话最核心的特征：**欠定（underspecification）**，即信息分散在多轮里、需要模型跨轮融合零碎线索。

**核心矛盾**：单轮 episodic 评测高估了模型能力。一旦要把分散在各轮的线索拼起来完成同一个任务，模型表现会急剧且一致地下滑，而这种下滑在传统评测里完全看不到。

**本文目标**：构造一个能把现有单轮高质量 benchmark 转成多轮欠定对话的公平评测环境，让单轮与多轮跑在**同一组任务**上，从而精确测量从单轮到多轮的退化幅度，并解释退化的成因。

**核心 idea**：**【指令分片 + 受控模拟】** 把一条完整指令切成多个"分片（shard）"，每轮只透露至多一个分片，强制信息逐步揭示；用 LLM 扮演 user / assistant / system 三角色跑模拟，并把退化拆解成 **aptitude（能力）** 与 **unreliability（不可靠性）** 两个可量化维度。

## 方法详解

### 整体框架
方法分两步：先把单轮完整指令经"分片流程"转成一组分片，再把这组分片喂进"分片模拟环境"跑多轮对话。模拟环境是一个三角色循环——被测 assistant 自由作答，user（GPT-4o-mini）持有完整指令并决定每轮揭示哪个分片，system 负责给 assistant 回复打标签（七种策略之一）和打分。当 assistant 给出"答案尝试"时抽取答案片段送任务评估器，对话最终得分取所有轮次得分的最大值，直到答对或分片用尽。

```mermaid
flowchart LR
    A[单轮完整指令] -->|分片流程<br/>切分/改写/校验/人工| B[分片集合 shards]
    B --> C{模拟环境}
    C --> U[User: 每轮揭示≤1分片]
    U --> M[Assistant 被测模型]
    M --> S[System: 策略分类+答案抽取+打分]
    S -->|未答完且有分片| U
    S -->|答对/分片用尽| E[最终分=各轮最大值]
```

### 关键设计

**1. 指令分片（Sharding）：把"一句话说全"拆成"逐渐说清"**　分片的目标是让一组小指令**联合起来**等价于原始完整指令，但信息被显式打散到各分片。为保证公平，作者定义了一套分片必须满足的性质（信息保全、首轮即有清晰意图、顺序不敏感等），并用"切分→改写→自动校验→人工逐条审查"的半自动流程生成（每 100 条约耗 3 小时人工）。人工环节会合并/拆分/重排分片，确保每个分片都是"用户在一轮里自然会说的一个信息单元"，而非对抗性的刻意切割。这一步是整套评测可信度的根基——若分片本身漏了信息，退化就成了假象。

**2. 三角色模拟与得分机制：给被测模型留足"翻盘"空间**　user 模拟器拿到完整指令和对话历史，能**根据上下文挑选并轻度改写**下一个最贴合当前交流的分片（比如 assistant 提了澄清问题，就回相关的分片而非按固定顺序放），比模板/随机更接近真人。assistant 在首轮只拿到最小上下文（如工具列表），从不被告知"对话会欠定/多轮"，以测量其默认行为。值得注意的是，一段 $N$ 分片的对话里 assistant 最多有 $N$ 次答案尝试、取最优计分，这反而让分片设定**比单轮更占便宜**（单轮只允许一次答案尝试）——即便如此多轮仍大幅落后，说明退化是真实的。

**3. 五种模拟类型：隔离"多轮欠定"这一个变量**　基于同一组分片，作者设计五种信息揭示节奏来做对照。单轮组有 FULL（原始完整指令，基线）和 CONCAT（所有分片拼成一条 bullet 列表，一轮给出——去掉欠定但保留分片改写，用来排除"是改写本身导致掉点"的可能）；多轮组有 SHARDED（核心欠定设定）、RECAP（SHARDED 后加一轮复述全部分片，测 agent 式收尾能否补救）、SNOWBALL（每轮揭示新分片同时复述所有旧分片，测持续提醒能否减负）。CONCAT 表现达到 FULL 的 95.1%，干净地证明掉点**来自多轮欠定本身**而非信息丢失或改写。

**4. Aptitude / Unreliability 双指标：把"掉点"拆成"变笨"还是"变飘"**　对每条指令跑 $N=10$ 次模拟得到分数集合 $S=\{S_i\}$，定义三个指标：平均表现 $P=\frac{1}{N}\sum_i S_i$、能力 $A^{90}=\text{percentile}_{90}(S)$（最好 10% 的发挥，衡量"上限"）、不可靠性 $U^{90}_{10}=\text{percentile}_{90}(S)-\text{percentile}_{10}(S)$（90 与 10 分位之差，衡量"波动"）。可靠性即 $R^{90}_{10}=100-U^{90}_{10}$。这套设计的精妙之处在于：同样从 90% 掉到 60%，可能是上限塌了、也可能是发挥时好时坏，两个指标把它们彻底分开，从而能定位退化的真正病灶。

## 实验关键数据

### 主实验表格
15 个 LLM × 6 任务（Code/Database/Actions/Math/Data-to-Text/Summary）× 3 设定，共 20 万+ 模拟对话、成本约 \$5,000。各设定平均表现（相对 FULL 的退化）：

| 设定 | 平均表现 | 相对 FULL 退化 | 说明 |
|------|---------|---------------|------|
| FULL（单轮完整） | ~90% | — | 基线 |
| CONCAT（单轮拼接） | ~95.1%×FULL | ≈ -5% | 排除改写/信息丢失干扰 |
| SHARDED（多轮欠定） | ~65% | **-39%** | 所有模型在所有任务上一致掉点 |

代表性模型 SHARDED 退化幅度（强弱模型一样惨）：

| 模型 | FULL→SHARDED 退化 |
|------|------|
| GPT-4.1 | -32% 量级 |
| Gemini 2.5 Pro | -30~40% |
| Claude 3.7 Sonnet | -30~40% |
| o3 / Deepseek-R1（推理模型） | 与非推理模型同样退化 |
| Llama3.1-8B / Phi-4（小模型） | -30~40% |

### 消融实验表格

| 实验 | 操作 | 结论 |
|------|------|------|
| Aptitude vs Reliability | 拆解 $A$ 与 $U$ | 多轮下能力仅降 **16%**，不可靠性暴增 **+112%**（翻倍多）；最好与最差run差 ~50 分 |
| 渐进分片（1→8 shards） | 固定复杂度只变分片粒度 | **只要≥2轮就迷失**；唯一有效提升可靠性的办法是 1-shard 一次说全 |
| RECAP / SNOWBALL（agent式） | 复述/滚雪球补救 | 优于 SHARDED 但仍达不到 FULL；SNOWBALL 实测 +15~20% |
| 温度消融（T=1/0.5/0） | 降温提升可靠性？ | 单轮有效，多轮无效；即便 T=0 多轮不可靠性仍达 30% |
| System prompt 提示 | 告知"对话会多轮欠定" | 仅 +1%，无实质帮助 |

### 关键发现
- **能力强≠迷失少**：单轮里能力越强越可靠（GPT-4.1、Gemini 2.5 Pro 不可靠性最低），但进入多轮后**所有模型不可靠性都拉平到很高**，再强也救不回来。
- **退化主因是不可靠性而非能力**：这是对 "Lost in Conversation" 现象的核心refinement。
- **四大根因**：模型倾向于（1）过早抛出完整答案并对欠定细节乱做假设、（2）过度依赖此前（错误的）答案尝试导致回答"臃肿"、（3）过度关注首尾轮出现"中间轮丢失"、（4）回答过于冗长引入更多假设干扰对用户话语的注意。推理模型回答平均长 33%，假设更多，反而更易乱。

## 亮点与洞察
- **范式价值**：第一个用"同一组任务、公平对照"精确量化单轮→多轮退化的工作，把"模型在真实对话里没那么好用"这一直觉变成可复现的 39% 数字。
- **Aptitude/Unreliability 二分**是真正的概念贡献——它解释了为什么"换更强的模型"解决不了多轮问题：病根在可靠性，而社区一直只优化能力。
- **可操作建议落到四类人群**：给 LLM builder（联合优化能力与可靠性，目标 $U<15$ 且 T=1）、agent builder（别指望框架记忆补救，模型要原生支持多轮）、NLP 研究者（为易退化任务发布分片变体）、普通用户（"超时就重开""重试前先汇总"）。

## 局限与展望
- **模拟非真人**：依赖 LLM 模拟 user，分片结构偏窄、末轮必定信息完整，缺少真人对话中的术语误解、放弃、无解目标等动态——作者明确指出实测退化很可能是真实世界的**低估**。
- **任务受限**：只覆盖有分析型解答的任务，未验证创意写作等开放式任务是否同样迷失。
- **仅英文文本**：未涉及其他语言与多模态。
- **复现性受限**：多数实验用闭源 API 模型，未来模型退役后难以精确重跑；概率性本身也带来方差。

## 相关工作与启发
- **对比 episodic 多轮评测**（MT-Bench 等）：本文论证 episodic 框架因可孤立打分而系统性高估能力，欠定才是缺失的关键维度。
- **用户模拟谱系**：从模板、固定标注到真人，本文选 LLM 模拟以平衡多样性与可控性，并强调它只是"探测 LLM 行为的探针"，非"建模真人"。
- **启发**：(1) 评测应把"可靠性/方差"作为一等公民，而非只报平均分；(2) agent 框架的"记忆外置"不是银弹，多轮鲁棒性需要进模型本身；(3) "分片化"可作为通用工具，把任意单轮 benchmark 升级成多轮压力测试。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把"欠定多轮"做成公平可量化的评测范式，Aptitude/Unreliability 二分是真正的概念创新。
- 实验充分度: ⭐⭐⭐⭐⭐ 15 个 LLM × 6 任务 × 多设定、20 万+ 对话，并配齐渐进分片/温度/system prompt/agent 式补救等消融。
- 写作质量: ⭐⭐⭐⭐⭐ 论证链条干净（CONCAT 控制变量尤其漂亮），结论可操作，"Lost in Conversation"叙事强。
- 价值: ⭐⭐⭐⭐⭐ 揭示了与真实使用严重脱节的评测盲区，对 LLM/agent/用户三方都有直接指导意义，影响面大。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Multi-turn Evaluation of Anthropomorphic Behaviours in Large Language Models](multi-turn_evaluation_of_anthropomorphic_behaviours_in_large_language_models.md)
- [\[ICLR 2026\] When LLMs Get Significantly Worse: A Statistical Approach to Detect Model Degradations](when_llms_get_significantly_worse_a_statistical_approach_to_detect_model_degrada.md)
- [\[ACL 2026\] Evaluating Temporal Consistency in Multi-Turn Language Models](../../ACL2026/llm_evaluation/evaluating_temporal_consistency_in_multi-turn_language_models.md)
- [\[ACL 2026\] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](../../ACL2026/llm_evaluation/beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)
- [\[ACL 2025\] StructFlowBench: A Structured Flow Benchmark for Multi-turn Instruction Following](../../ACL2025/llm_evaluation/structflowbench_a_structured_flow_benchmark_for_multi-turn_instruction_following.md)

</div>

<!-- RELATED:END -->
