# LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts

**会议**: ICLR 2026 Oral  
**arXiv**: [2510.19363](https://arxiv.org/abs/2510.19363)  
**代码**: 有（附补充材料提供训练代码和 KeyChain 数据合成代码）  
**领域**: LLM推理 / 长上下文推理  
**关键词**: long-context reasoning, reinforcement learning, GRPO, multi-hop QA, emergent reasoning patterns  

## 一句话总结
提出 LoongRL，通过构建 KeyChain 合成数据进行强化学习训练，使 LLM 涌现出 plan–retrieve–reason–recheck 的长上下文推理模式，仅在 16K 上下文上训练即可泛化到 128K，14B 模型达到 74.2 分接近 o3-mini (74.5) 和 DeepSeek-R1 (74.9)。

## 研究背景与动机

1. **领域现状**：当前 LLM 推理方面的进展（如 DeepSeek-R1、o1）主要集中在短上下文推理任务（数学、代码），通过 RL 引导模型产生更长的 chain-of-thought 和自我反思。对于长上下文推理——需要在数千 token 的外部输入中检索并整合信息——则基本未被探索。

2. **现有痛点**：(a) 现有长上下文模型虽然支持长窗口（128K+），但主要只擅长 **检索**，在需要 **推理** 的场景下表现差；(b) 用于 RL 训练的高难度长上下文数据极度稀缺，答案形式多样导致验证困难；(c) 将 RL rollout 从短文本（<1K）扩展到 128K 上下文计算成本极高；(d) 仅在长上下文数据上训练会退化短上下文能力。

3. **核心矛盾**：长上下文推理需要独特的思维模式（先规划→再检索→再推理→再验证），但这种模式无法通过简单的 SFT 或 prompting 获得，需要 RL 来探索和激励。然而，适合 RL 训练的数据不存在——必须足够难以触发推理，必须需要从长上下文中检索信息，且答案必须可验证。

4. **本文要解决什么？** (a) 如何设计高质量 RL 训练数据来激励长上下文推理？ (b) 如何在短上下文上训练但泛化到超长上下文？ (c) 如何保持短上下文能力不退化？

5. **切入角度**：作者观察到，如果 RL 数据本身就需要"追踪线索链→找到真正问题→检索→推理"这种多步操作，模型就会涌现出结构化的长上下文推理模式。这个模式一旦学会，就可以泛化到任意长度。

6. **核心idea一句话**：通过在短多跳 QA 中插入 UUID 链条隐藏真实问题（KeyChain），构造高难度 RL 训练数据，使 LLM 涌现 plan-retrieve-reason-recheck 推理模式并泛化到 128K 上下文。

## 方法详解

### 整体框架
LoongRL 的 pipeline：(1) 从现有多跳 QA 数据集出发 → (2) 通过 KeyChain 方法将其变为高难度长上下文问题 → (3) 用 GRPO 算法进行多阶段 RL 训练 → (4) 混合数学和检索数据保持短上下文能力。输入是 ~16K token 的长文本 + 问题，输出是包含推理过程的回答。

### 关键设计

1. **KeyChain 数据构造**
   - 做什么：将简单的短文本多跳 QA 转变为高难度长上下文推理任务
   - 核心思路：首先从 HotpotQA、MuSiQue、2WikiMultiHopQA 中筛选中等难度的 QA 对（277K→72K，过滤方式：用 Qwen2.5-32B 回答 8 次，保留 pass rate 在 0-1 之间的），然后 (a) 插入无关文档扩展到 ~16K token，(b) 插入多条 UUID key-value 链，其中一条链最终指向原始问题 $o\_q_i$，其余链指向干扰问题。每个 key 是 32 字符 UUID（0-9, A-F），value 包含下一个 key 或最终问题。模型必须从初始 key 追踪正确链条 → 找到隐藏的真实问题 → 在长上下文中检索相关信息 → 推理得出答案
   - 设计动机：仅靠增加干扰文档，难度提升有限，模型仍可直接检索。KeyChain 的妙处在于迫使模型先"找到问题是什么"，这个预处理步骤天然引导模型形成规划-检索-推理的结构化思维

2. **Two-way Substring Exact Match 奖励验证器**
   - 做什么：为 RL 提供可靠的 binary 奖励信号
   - 核心思路：$r_i = 1$ 当且仅当 $a \subseteq y_{\text{ans}} \lor y_{\text{ans}} \subseteq a$，即预测答案包含 ground truth，或 ground truth 包含预测答案（双向子串匹配）。模型被要求将最终答案放在 `\boxed{}` 中以便提取
   - 设计动机：通用 QA 答案形式多样（不像数学有唯一解），strict exact match 过于严格会惩罚正确但格式不同的答案，F1 score 和 LLM-as-a-judge 效果不佳且后者还需额外模型开销。双向子串匹配在宽容度和准确性间取得好平衡

3. **三阶段多课程 RL 训练**
   - 做什么：渐进式提升任务难度，避免一开始就给太难的数据导致训练不稳定
   - 核心思路：
     - **Warm-up**（42 steps, 仅 7B 需要）: 在无 KeyChain 的数据上训练（标准多跳 QA + 检索 + 数学），提升基础能力
     - **Stage I**（168 steps）: 加入 KeyChain 数据，鼓励模型学习 plan-retrieve-reason-recheck 模式
     - **Stage II**（~120-150 steps）: 对每个样本生成 8 条 rollout，丢弃全部答对的样本（~60-70%），仅在剩余困难样本上继续训练，避免在已掌握问题上过拟合
   - 设计动机：小模型（7B）初始能力弱，直接上 KeyChain 会导致所有 rollout 都失败（reward 全为 0），无法产生有效梯度；大模型（14B）可跳过 warm-up

4. **混合数据配方**
   - 做什么：平衡长上下文推理和短上下文通用能力
   - 核心思路：训练数据包含 7,500 KeyChain QA + 7,500 标准多跳 QA + 1,024 needle retrieval + 5,000 数学题（DAPO + MATH），全部限制在 ~16K 上下文长度内
   - 设计动机：纯长上下文训练会退化短上下文能力（R1-distill 系列和 QwenLong-L1 均有此问题），加入数学数据可保持通用推理

### 损失函数 / 训练策略
采用 GRPO (Group Relative Policy Optimization)，group size $G=8$，学习率 $1 \times 10^{-6}$，cosine decay，KL penalty $\beta=0.001$，去掉熵损失项（避免训练不稳定）。最大输出长度 4,096 token，推理时温度 0.6, top-p 0.95。

## 实验关键数据

### 主实验

| 模型 | LongBench v1 Avg | HotpotQA | 2Wiki | MuSiQue | NarrativeQA | QASPER |
|------|------------------|----------|-------|---------|-------------|--------|
| o3-mini | 74.5 | 83.0 | 89.0 | 64.0 | 60.7 | 60.5 |
| DeepSeek-R1 | 74.9 | 82.7 | 91.3 | 72.2 | 66.9 | 61.4 |
| QwenLong-L1-32B | 70.1 | 80.7 | 89.1 | 65.2 | 58.6 | 56.7 |
| Qwen2.5-7B-Instruct | 48.9 | 69.5 | 50.5 | 34.0 | 44.5 | 46.0 |
| **LoongRL-7B** | **72.4** | 83.1 | 91.1 | 65.6 | 58.4 | 63.6 |
| Qwen2.5-14B-Instruct | 53.1 | 74.0 | 60.5 | 36.5 | 48.5 | 46.0 |
| **LoongRL-14B** | **74.2** | 82.2 | 93.3 | 67.5 | 63.4 | 64.5 |

### 消融实验

| 配置 | LongBench v1 Avg | 说明 |
|------|------------------|------|
| Qwen2.5-7B-Instruct | 48.9 | 基线 |
| LoongRL-7B (no KeyChain) | 66.2 | 用等量普通多跳 QA 替代 KeyChain |
| LoongRL-7B (full) | 72.4 | 完整模型，KeyChain 贡献 +6.2% |

| 奖励验证器 | Avg | 说明 |
|-----------|-----|------|
| F1 score | 65.1 | 不够精确 |
| LLM-as-a-judge | 65.2 | 需额外模型且效果差 |
| Exact match | 69.2 | 过于严格 |
| Two-way Substring (ours) | **72.4** | 最优 |

### 关键发现
- **KeyChain 是核心贡献**：去掉 KeyChain 后性能从 72.4 降至 66.2，差距巨大。KeyChain 训练的模型会涌现显式的 plan 步骤和 recheck 行为，而无 KeyChain 模型的推理和检索混在一起，缺乏规划
- **16K 训练泛化到 128K**：LoongRL-7B 在 RULER 128K 上达到 76.8（基线 69.4），LoongRL-14B 达到 79.9（基线 73.6）。NarrativeQA 32K-64K 区间分别提升 +14.8% 和 +16.0%
- **奖励验证器对比**：双向子串匹配显著优于 F1、LLM judge 和 exact match，说明对 QA 这类开放式答案，恰当的验证松弛很重要
- **Needle-in-a-haystack 完美通过**：LoongRL-7B 在所有深度和长度上达到 100% 准确率，甚至基线 Qwen2.5-7B 和 QwenLong-L1-32B 都无法完全通过
- **短上下文能力保持**：MMLU 提升 +2.8%/+1.1%，IFEval 仅微降 -0.3%/-2.6%，数学保持稳定

## 亮点与洞察
- **KeyChain 数据构造思路极其巧妙**：通过在数据层面引入"先找问题再答题"的结构，自然引导模型涌现规划能力。这个思路可以迁移到其他需要结构化推理的任务——不是直接教模型怎么推理，而是设计数据让模型必须推理才能答题
- **短训练长泛化**：仅 16K 训练泛化到 128K 是一个重要发现，说明推理模式一旦学会就是长度无关的。这大幅降低了长上下文 RL 训练的成本（否则 128K rollout 的 GPU 成本不可承受）
- **双向子串匹配**：简单但有效的奖励设计，解决了 QA 答案多样性问题，避免了 LLM-as-a-judge 的额外开销和 reward hacking 风险。可直接复用到其他开放式 QA 的 RL 训练
- **多阶段课程学习**：warm-up → KeyChain → hard-mining 的策略非常实用。特别是 Stage II 的 hard-mining（丢弃全答对样本）策略，避免在简单样本上浪费计算

## 局限性 / 可改进方向
- **仅评估 QA 类型任务**：LongBench 和 NarrativeQA 主要是抽取式/生成式 QA，未测试长文档摘要、跨文档推理等其他长上下文任务类型
- **训练长度固定 16K**：虽然 16K→128K 泛化效果好，但对于更长上下文（256K, 1M）是否仍有效未被验证
- **KeyChain 合成较人工**：UUID 链条是完全人工构造的，与真实世界的"信息追踪"任务存在分布差距。能否设计更自然的 KeyChain 变体？
- **仅用 Qwen 系列实验**：未在 LLaMA、Mistral 等其他架构上验证泛化性
- **涌现模式的分析不够深入**：plan-retrieve-reason-recheck 模式是否总是稳定出现？failure case 长什么样？

## 相关工作与启发
- **vs QwenLong-L1**：QwenLong-L1 在 R1-distill-Qwen-32B 上用 60K 上下文 RL 训练，仅提升 +4.6%。LoongRL 用 16K 训练在 7B 模型上就超越它 +2.3%。关键差异在于 KeyChain 数据的质量
- **vs R1-Distill 系列**：R1 蒸馏在长上下文上效果差甚至退化（7B 退化 -17.7%），因为蒸馏的长 CoT 数据主要针对短上下文推理，未覆盖长上下文特有的检索-推理模式
- **与 [VHD-Guided Adaptive Visual Re-injection](../../ideas/multimodal_vlm/20260318_vhd_adaptive_visual_reinjection.md) 存在关联**：LoongRL 证明了"涌现长上下文推理模式"的可行性，类似思路可应用到多模态长链推理中

## 评分
- 新颖性: ⭐⭐⭐⭐ KeyChain 数据构造非常有创意，但底层 RL 框架（GRPO）本身并非新方法
- 实验充分度: ⭐⭐⭐⭐⭐ 对比全面（含 o3-mini、R1），消融充分（KeyChain、验证器、训练策略），还有 128K 泛化和 NIAH 测试
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，方法描述流畅，图表直观，附录详尽含训练轨迹对比
- 价值: ⭐⭐⭐⭐⭐ 为长上下文 LLM 推理提供了高效且可复现的方案，KeyChain 数据构造可被广泛复用
