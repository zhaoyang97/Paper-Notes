# Trade-offs in Large Reasoning Models: An Empirical Analysis of Deliberative and Adaptive Reasoning over Foundational Capabilities

**会议**: AAAI 2026  
**arXiv**: [2503.17979](https://arxiv.org/abs/2503.17979)  
**代码**: https://github.com/SCIR-SC-Qiaoban-Team/FreeEvalLM  
**领域**: LLM推理  
**关键词**: Large Reasoning Models, 推理-基础能力权衡, 自适应推理, Zero-Thinking, 安全性

## 一句话总结
系统评估了LRM（如DeepSeek-R1、QwQ、OpenThinker等）在获取深度推理能力后对基础能力（helpfulness和harmlessness）的负面影响，发现deliberative reasoning显著降低指令遵循和安全性能力，并提出Zero-Thinking、Less-Thinking、Summary-Thinking等自适应推理模式可有效缓解这些缺陷。

## 研究背景与动机
1. **领域现状**：以OpenAI o1/o3和DeepSeek-R1为代表的Large Reasoning Models (LRMs) 通过长链思维推理在数学、编程等专业推理任务上取得显著进步。社区对LRM的关注主要集中在推理任务的性能、效率和鲁棒性上。
2. **现有痛点**：LRM在获取deliberative reasoning能力的同时，对其基础能力（general task性能、instruction-following、safety）的影响几乎未被系统研究。认知科学表明人类推理能力与整体认知功能紧密关联，类比到LRM也应如此。
3. **核心矛盾**：通过蒸馏或强化学习获取推理能力的过程，可能以牺牲模型原有的helpfulness和harmlessness为代价——这种trade-off在实际部署中至关重要但尚未被量化。
4. **本文要解决什么？**
   - RQ1: LRM获取推理能力后，哪些基础能力受损最严重？
   - RQ2: 推理时的inference-time compute如何影响LRM在通用任务上的表现？
   - 能否通过控制推理模式来缓解基础能力下降？
5. **切入角度**：通过在LRM的thinking过程中插入特殊token（如`<think></think>`），手动控制推理深度，实现Zero/Less/Summary三种自适应推理模式
6. **核心idea一句话**：LRM的deliberative reasoning在提升专业推理的同时显著损害基础能力，而自适应推理模式（动态分配inference-time compute）是未来LRM发展的关键方向

## 方法详解

### 整体框架
本文是一项系统性的实证研究，不提出新模型，而是：
- **评估对象**：3个模型家族（DeepSeek/Qwen/LLaMA）× 多个规模（7B~671B）的LRM及其对应chat版本
- **评估维度**：Helpfulness（MMLU-Pro、Live-Bench、IFEval、MT-Bench）+ Harmlessness（StrongReject、WildJailbreak）
- **控制变量**：通过不同推理模式（Deliberative/Zero/Less/Summary-Thinking）系统对比

### 关键设计

1. **基础能力评估框架**:
   - 做什么：全面评估LRM在获取推理能力后的基础性能变化
   - 核心思路：将每个蒸馏得到的LRM与其source chat model直接对比（如OpenThinker-7B vs Qwen2.5-7B-Instruct），在6个benchmark上测量性能差异。对RL训练的LRM（如QwQ-32B）则通过修改推理模式间接推断能力变化
   - 设计动机：之前无人系统对比LRM和chat model在非推理任务上的差异——只看推理任务会得到"LRM全面更强"的错误结论

2. **自适应推理模式**:
   - 做什么：通过控制LRM的思考过程长度/内容来调节inference-time compute
   - 核心思路：
     - **Zero-Thinking**：在输入后直接追加`</think>`，强制跳过推理过程
     - **Less-Thinking**：在推理过程的$p\%$处插入`</think>`，提前终止思考（$p$ = 10%, 20%, 50%, 60%, 80%, 90%）
     - **Summary-Thinking**：用GPT-4o将完整推理过程压缩为摘要，插入`<think></think>`之间
     - **Summary-Thinking-Plus**：保留原始推理的第一句话 + 压缩摘要，因为首句模式对准确性有重要影响
   - 设计动机：不同任务对推理深度的需求不同。通用任务和安全任务可能不需要长链推理，过度推理反而有害

3. **Thought安全性分析**:
   - 做什么：分析LRM的thinking过程本身是否安全
   - 核心思路：将LRM的输出按2×2矩阵分类——thought安全/不安全 × response安全/不安全，用GPT-4o做安全判断
   - 设计动机：发现LRM在面对恶意查询时，即使最终response拒绝了请求，其thinking过程中仍可能包含大量不安全内容。这是一个被忽视的安全风险

### 损失函数 / 训练策略
本文为纯评估工作，不涉及训练。所有LRM使用vLLM在8×A100上推理，解码参数和prompt格式严格遵循官方配置。

## 实验关键数据

### 主实验
蒸馏LRM vs 对应Chat模型的基础能力对比：

| 模型 | MMLU-Pro | Live-Bench | IFEval | MT-Bench | StrongReject↑ | WildJailbreak↑ |
|------|----------|------------|--------|----------|--------------|----------------|
| Qwen2.5-7B-Instruct | 54.44 | 36.34 | 67.84 | 7.94 | 95.21 | 10.70 |
| OpenThinker-7B | 39.04↓ | 20.81↓ | 34.20↓ | 7.33↓ | 37.29↓ | 12.45 |
| Qwen2.5-32B-Instruct | 67.07 | 53.85 | 77.26 | 8.32 | 95.00 | 13.30 |
| s1.1-32B | 43.77↓ | 34.42↓ | 37.34↓ | 7.98↓ | 49.38↓ | 4.90↓ |
| Llama-3.3-70B-Instruct | 70.54 | 60.30 | 89.83 | 8.11 | 95.63 | 19.50 |
| R1-Distill-Llama-70B | 71.57 | 54.09↓ | 76.89↓ | 8.03 | 89.17↓ | 28.25 |

s1.1-32B在IFEval上下降47.38%的同时推理成本增加250%！

### 消融实验（自适应推理模式）

| 模型+模式 | IFEval | StrongReject↑ | WildJailbreak↑ |
|-----------|--------|--------------|----------------|
| s1.1-32B (原始) | 37.34 | 49.38 | 4.90 |
| + Zero-Thinking | 42.33 | 64.79 | 11.15 |
| + Summary-Thinking | **54.16** | 53.96 | 4.70 |
| QwQ-32B (原始) | 75.60 | 95.00 | 10.65 |
| + Zero-Thinking | 64.51 | **98.33** | **59.65** |
| + Summary-Thinking | 77.26 | 93.33 | 11.60 |
| R1-Distill-Llama-70B (原始) | 75.60 | 89.17 | 28.25 |
| + Zero-Thinking | 63.22 | **99.17** | **89.10** |

### 关键发现
- **Thought越长，性能越差**：在Win/Lose分析中，性能下降的样本的thought长度显著大于性能提升的样本，说明过度推理实际上损害通用任务表现
- **Zero-Thinking大幅提升安全性**：跳过推理时，LRM的jailbreak抵抗力可从28.25飙升到89.10（R1-Distill-Llama-70B），甚至远超原始chat模型
- **Summary-Thinking大幅提升指令遵循**：s1.1-32B在IFEval上从37.34→54.16（+45%），QwQ上也有提升
- **Unsafe thought是unsafe response的主因**：80%+的R1-Distill-Llama-70B的思考过程包含不安全内容，即使最终response看似safe
- **Less-Thinking的最优比例因任务而异**：不存在一个universal的thinking ratio，不同benchmark在不同比例下达到最优

## 亮点与洞察
- **首次系统量化LRM的推理-基础能力trade-off**：之前社区过度关注LRM在推理benchmark上的进步，忽视了其在实际部署中必须具备的基础能力。这篇论文填补了这个重要gap
- **"Thought也不安全"的发现非常有价值**：即使response被safety filter过滤了，thinking过程中的unsafe content本身就是安全风险（可能被攻击者观察到或利用）。这对LRM safety alignment方向有重要启示
- **自适应推理的简单有效性**：仅通过插入`</think>` token就能实现丰富的推理模式控制，方法极其简单但效果显著，为未来"think when needed"的adaptive inference提供了实践基础

## 局限性 / 可改进方向
- **评估范围有限**：只覆盖了general tasks、instruction following和safety，未涉及multi-modal、open-ended对话等更贴近真实应用的场景
- **模型覆盖不够**：仅评估了特定checkpoint，对MoE架构或多模态LLM的适用性未知
- **自适应推理是手动控制的**：Zero/Less/Summary-Thinking需要预先指定模式，未提出自动判断何时需要推理、推理多深的方法
- **改进方向**：训练LRM自动判断输入难度，动态分配inference-time compute；在蒸馏时混合long/short CoT数据；在RL中将reasoning长度作为reward因子

## 相关工作与启发
- **vs 推理效率研究（o1-preview分析等）**：之前工作关注"推理任务上的over-thinking"，本文扩展到"非推理任务上推理的全面负面影响"，视角更广
- **vs SafeChain等safety研究**：SafeChain关注如何在CoT中嵌入安全推理，本文则揭示CoT本身就是unsafe的来源——两者互补
- **对LRM部署的启示**：实际部署LRM时不能"一刀切"地使用deliberative reasoning，需要根据query类型自适应选择推理深度

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统研究LRM推理能力与基础能力的trade-off，视角新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 3个模型家族、多种规模、6个benchmark、4种推理模式、细粒度分析、case study，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，发现有说服力，case study很好地support了结论
- 价值: ⭐⭐⭐⭐⭐ 对LRM社区有重要警示价值——推理能力的提升可能以基础能力为代价，adaptive reasoning是刚需
