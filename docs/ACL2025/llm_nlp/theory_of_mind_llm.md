# Theory of Mind in Large Language Models: Assessment and Enhancement

**会议**: ACL 2025  
**arXiv**: [2505.00026](https://arxiv.org/abs/2505.00026)  
**代码**: 无  
**领域**: LLM / NLP  
**关键词**: Theory of Mind, ToM, benchmark survey, belief reasoning, LLM cognitive abilities

## 一句话总结
系统综述了 LLM 的心智理论（ToM）能力的评估基准（10+ story-based benchmarks）和增强策略（prompt-only 和 fine-tuning 两类方法），指出当前 LLM 在 ToM 推理上仍有显著不足，并提出未来方向。

## 研究背景与动机
1. **领域现状**：心智理论（Theory of Mind）是人类社会智能的基石，指理解他人信念、意图、情感等心理状态的能力。随着 LLM 日益融入日常生活，评估其 ToM 能力至关重要。
2. **现有痛点**：关于 LLM 是否真正具备 ToM 存在争议——部分研究认为 LLM 展现了 ToM 迹象（Kosinski 2024），但更多研究表明这些能力是表面的、不稳定的（Shapira et al. 2024; Ullman 2023）。
3. **核心矛盾**：2023-2024 年涌现了大量 ToM 评测基准和增强方法，但缺乏统一的综述将其系统梳理。之前的综述（Ma et al. 2023b）仅覆盖到 2023 年的基准，且未涵盖增强策略。
4. **本文要解决什么**：(1) 系统梳理近两年的 story-based ToM 基准及多模态基准；(2) 分类总结提升 LLM ToM 能力的策略；(3) 指出未来发展方向。
5. **切入角度**：基于 ATOMS 框架（7 种心理状态：beliefs, intentions, desires, emotions, knowledge, percepts, non-literal communication）对基准和方法进行统一对比分析。
6. **核心 idea 一句话**：首个系统覆盖 LLM ToM 评估和增强两方面的综述，揭示当前研究偏重于 belief 推理，其他心理状态研究严重不足。

## 方法详解

### 整体框架
本文是一篇综述，按"评估→增强→未来方向"组织。评估部分覆盖 10+ 个 story-based 基准（文本 + 多模态），按 ATOMS 7 种心理状态分类对比；增强部分分为纯 prompt 方法和结合额外技术（如 fine-tuning）的方法两类。

### 关键设计

1. **评估基准分析**:
   - 做什么：系统对比 ToMi、HI-TOM、TOMBENCH、BigToM、OpenToM 等 10+ 基准在心理状态覆盖、推理阶数（first/second-order）、数据格式等维度
   - 核心发现：绝大多数基准集中在 belief 推理上，TOMBENCH 是覆盖面最广的（5/7 种心理状态）；多模态基准（MMToM-QA、MuMA-ToM）仅限于家庭场景的合成视频
   - 关键问题：现有基准多为"被动"评估（LLM 作为旁观者），缺乏主动决策场景的评测

2. **纯 Prompt 增强策略**:
   - 做什么：梳理 4 种仅通过 prompt 工程提升 ToM 的方法
   - SymbolicToM：为每个角色构建 belief graph，推理时检索相关信念子图作为 prompt，理论上可处理任意阶 belief 问题，但内存随阶数指数增长
   - SimToM：受"模拟理论"启发，两阶段框架——先做 perspective taking（过滤角色可知信息），再基于过滤后的故事回答问题
   - PercepToM：三阶段流程——识别信息感知者→提取目标角色可感知的信息→基于此回答
   - TimeToM：引入时间线，构建"时间信念状态链"（TBSC），将高阶推理通过时间信念交集转化为一阶推理
   - 设计动机：所有方法的共同思路是先识别目标角色的感知/知识范围，再基于受限信息回答，核心挑战是准确构建角色视角

3. **结合额外技术的增强策略**:
   - 做什么：梳理 3 种引入微调或符号推理的方法
   - ToM-LM：用 LLM 做语义解析，将 ToM 问题转为符号形式后用模型检查器（SMCDEL）验证，增加可解释性
   - BIP-ALM：贝叶斯逆向规划 + LLM，从视频和文本提取符号化信息，fine-tune LLM 预测动作似然
   - LIMP：多智能体场景，使用 VLM 提取视频信息 + LLM 提取文本信息，通过逆向多智能体规划推理心理状态
   - 设计动机：符号推理增加可验证性，但依赖 fine-tuning 数据，且目前仅在多选题设置下验证

### ToM 能力的关键评测结论
- 大多数评测表明 LLM 仍然缺乏稳健的 ToM 能力
- Belief 是研究最多的心理状态，其他心理状态（intentions, desires, emotions 等）严重不足
- 高阶 ToM 推理（second-order 及以上）对 LLM 是显著更难的挑战
- 多模态 ToM 评测刚刚起步，仅限于合成的家庭场景

## 实验关键数据

### 主实验 — 基准覆盖对比

| 基准 | Beliefs | Intentions | Desires | Emotions | Knowledge | Percepts | Non-literal |
|------|---------|-----------|---------|----------|-----------|---------|-------------|
| ToMi | ✓ | | | | | | |
| HI-TOM | ✓ | | | | | | |
| TOMBENCH | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ |
| BigToM | ✓ | ✓ | ✓ | | | | |
| OpenToM | ✓ | | ✓ | | ✓ | | |
| SimpleToM | ✓ | | ✓ | | ✓ | ✓ | |

### 增强方法对比

| 方法 | 类型 | 支持阶数 | 需要微调 | 适用场景 |
|------|------|---------|---------|---------|
| SymbolicToM | Prompt-only | 任意阶 | 否 | 文本 ToM |
| SimToM | Prompt-only | 一阶为主 | 否 | 文本 ToM |
| TimeToM | Prompt-only | 高阶 | 否 | 文本 ToM |
| ToM-LM | 额外技术 | 一阶 | 是 | 符号化 ToM |
| BIP-ALM | 额外技术 | 一阶 | 是 | 多模态 ToM |
| LIMP | 额外技术 | 二阶 | 否 | 多智能体 |

### 关键发现
- 所有 prompt-only 方法都采用流水线架构，存在错误传播风险
- SymbolicToM 内存随阶数指数增长，TimeToM 的 TBSC 构建准确率是瓶颈
- 符号化方法（ToM-LM）增加了透明度和可验证性，但需要逻辑专业知识来准备训练数据
- VLM 在动作识别中的幻觉问题是 LIMP 的主要错误来源

## 亮点与洞察
- **ATOMS 框架作为统一分析视角**：用 7 种心理状态维度对比所有基准和方法，清晰揭示了研究偏向——belief 占绝对主导，其他心理状态几乎空白，这为后续研究指明了方向。
- **"高阶推理→一阶推理"的转化思路**（TimeToM）非常巧妙：通过时间信念交集，将 n 阶推理分解为多个 1 阶推理的组合，是可复用的 trick。
- **被动 vs 主动评测**的区分很有价值：当前所有基准都是被动的（LLM 当观察者），但真正的 ToM 需要 LLM 作为主动代理在社交场景中做决策。

## 局限性 / 可改进方向
- 综述未包含实验对比各增强方法在统一基准上的性能（因为各方法使用不同基准和模型）
- 对 belief 以外的心理状态（如 emotions、intentions）的增强策略讨论较少
- 多模态 ToM 目前仅限于 VirtualHome 合成视频，与现实场景差距大
- 未讨论 LLM 的 ToM 能力与模型规模的关系（scaling law 视角）

## 相关工作与启发
- **vs Ma et al. (2023b) 综述**：之前的综述仅覆盖 2023 年之前的基准，且不涉及增强策略。本文覆盖 2023-2024 新基准 + 增强方法 + 多模态，更全面
- **与 LLM Agent 研究的关联**：ToM 是 LLM 作为社交代理的核心能力，当前 agent 研究主要关注工具使用和规划，ToM 维度被忽视
- **与 Alignment 研究的关联**：理解 LLM 对用户意图的推理能力（ToM 的一种形式）对改善对齐质量有直接帮助

## 评分
- 新颖性: ⭐⭐⭐ 综述类文章，内容组织有新意但无技术创新
- 实验充分度: ⭐⭐⭐ 系统梳理了各方法但缺乏统一实验对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，表格对比直观，未来方向有价值
- 价值: ⭐⭐⭐⭐ 对 ToM 研究者是很好的入门和参考资源
