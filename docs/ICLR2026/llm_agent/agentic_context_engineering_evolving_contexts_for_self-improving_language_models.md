# Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models

**会议**: ICLR 2026  
**arXiv**: [2510.04618](https://arxiv.org/abs/2510.04618)  
**代码**: [https://github.com/ace-agent/ace](https://github.com/ace-agent/ace)  
**领域**: Agent  
**关键词**: context engineering, self-improving agent, prompt optimization, evolving memory, playbook  

## 一句话总结
提出 ACE（Agentic Context Engineering）框架，将 context 视为不断演化的"策略手册"（playbook），通过 Generator-Reflector-Curator 三角色分工和增量式 delta 更新来持续积累和精炼策略，解决了现有 prompt 优化中的简洁偏差和上下文坍塌问题，在 agent 任务上平均提升 10.6%、金融任务提升 8.6%，且自适应延迟降低 86.9%。

## 研究背景与动机
1. **领域现状**：Context adaptation（通过修改 LLM 输入而非权重来改进性能）已成为构建可伸缩 AI 系统的核心范式。现有方法包括 prompt optimization（GEPA、MIPROv2）、test-time memory（Dynamic Cheatsheet）等。
2. **现有痛点**：(1) **简洁偏差（brevity bias）**：多数 prompt 优化器追求简洁通用的指令，压缩掉了领域特定的策略、工具使用指南和常见失败模式；(2) **上下文坍塌（context collapse）**：单体重写方式在迭代过程中逐渐退化为更短、信息更少的摘要——实验中观察到 context 从 18282 token 突然坍塌到 122 token，性能随之骤降。
3. **核心矛盾**：agent 和知识密集应用需要**全面详尽的**领域知识，但现有方法却在压缩知识。LLM 与人不同——人受益于简洁概括，LLM 反而在详尽 context 下表现更好。
4. **本文要解决什么？** 如何构建一种 context 适配方法，既能持续积累知识又不会坍塌退化？
5. **切入角度**：将 context 视为"evolving playbook"而非"optimized prompt"，用结构化的增量更新替代整体重写。
6. **核心idea一句话**：context 应该是持续增长和精炼的策略手册，而非压缩后的简洁指令。

## 方法详解

### 整体框架
ACE 由三个角色组成：Generator（生成推理轨迹）→ Reflector（从轨迹中提取教训和洞见）→ Curator（将教训整合为结构化的 delta 更新，merge 到现有 context 中）。支持离线（system prompt 优化）和在线（test-time memory）两种模式。

### 关键设计

1. **三角色分工（Generator → Reflector → Curator）**:
   - 做什么：将 context 构建的不同职责解耦到专门的角色
   - 核心思路：Generator 用当前 context 解决新问题，产生执行轨迹；Reflector 分析轨迹，提取具体的成功策略和失败教训（可多轮迭代精炼洞察）；Curator 将洞察转化为结构化 bullet 并 merge 到 context 中
   - 设计动机：避免把所有职责堆到单一模型上造成瓶颈。消融实验显示单独的 Reflector 角色是性能提升的关键来源

2. **增量式 Delta 更新（替代整体重写）**:
   - 做什么：用局部化的 bullet 增删改替代 context 的整体重写
   - 核心思路：context 表示为 bullet 集合，每个 bullet 有唯一 ID + 有用/有害计数 + 内容。每次适配只生成小量 delta（新 bullet 或已有 bullet 的修改），通过轻量非 LLM 逻辑确定性 merge，可并行处理
   - 设计动机：彻底解决 context collapse——因为从不执行全文重写，知识只能添加或局部修改，不会被意外压缩掉

3. **Grow-and-Refine 机制**:
   - 做什么：平衡 context 的持续增长和冗余控制
   - 核心思路：新 bullet 追加到 context，已有 bullet 原地更新（如增加计数器）。通过语义嵌入对比做去重（de-duplication），可以在每次 delta 后主动执行或在超出 context window 时懒惰执行
   - 设计动机：确保 context 的规模是可控的，不会无限增长

### 损失函数 / 训练策略
无需训练模型权重。ACE 是纯 context adaptation 方法。离线模式下在训练集上多 epoch 迭代构建 context；在线模式下在测试时逐样本更新。关键超参：Reflector 最大精炼轮数 5，离线最大 epoch 5，batch size 1。值得注意的是无需标注也能工作——利用执行反馈（如代码执行成功/失败）作为自然信号。

## 实验关键数据

### 主实验（AppWorld Agent Benchmark）

| 方法 | 需要标注 | Test-Normal TGC | Test-Challenge TGC | Average |
|------|---------|----------------|-------------------|---------|
| ReAct baseline | - | 63.7 | 41.5 | 42.4 |
| + ICL | ✓ | 64.3 | 46.0 | 46.0 |
| + GEPA | ✓ | 64.9 | 46.0 | 46.4 |
| **+ ACE (有标注)** | ✓ | **76.2** | **57.3** | **59.4** |
| + ACE (无标注) | ✗ | 75.0 | 54.4 | 57.2 |
| + DC (online) | ✗ | 65.5 | 52.3 | 51.9 |
| **+ ACE (online)** | ✗ | **69.6** | **66.0** | **59.5** |

### 消融实验（金融 benchmark）

| 方法 | FiNER Acc | Formula Acc | Average |
|------|-----------|-------------|---------|
| Base LLM | 70.7 | 67.5 | 69.1 |
| GEPA | 73.5 | 71.5 | 72.5 |
| **ACE** | **78.3** | **85.5** | **81.9** |

### 关键发现
- ACE 在 AppWorld 上平均提升 17%（offline 有标注），在排行榜上用开源模型 DeepSeek-V3.1 达到了 GPT-4.1 驱动的 IBM CUGA（排行榜第一）的平均水平，且在 harder test-challenge split 上超过了它
- **无标注也很强**：ACE 在无标注模式下仍提升 14.8%，利用执行反馈即可自我改进
- 金融任务上 ACE 比 GEPA 高 9.4%（72.5→81.9），暴力积累领域知识的策略在知识密集型任务上优势明显
- 适配延迟降低 86.9%：增量 delta 更新比整体重写快得多
- 消融实验确认 Reflector 角色和多 epoch 精炼各自贡献了显著提升

## 亮点与洞察
- **"playbook 而非 prompt"的理念转变**：context 不应被压缩，而应被持续充实。这与 RAG、long-context 等趋势一致，为 context engineering 提供了清晰的设计哲学
- **增量 delta 更新是关键创新**：彻底杜绝了 context collapse，且可并行 merge，是一个简单但极其有效的工程设计
- **无监督自改进能力**：仅靠执行反馈就能构建有效 context，为真正的 self-improving agent 铺平道路
- **三角色分工模式可复用**：Generator-Reflector-Curator 的模式可以迁移到其他需要从经验中学习的 LLM 系统

## 局限性 / 可改进方向
- 随着 bullet 数量增长，context 可能超出 context window，需要更智能的检索或压缩策略
- 去重依赖语义嵌入的质量，相似但不完全重复的 bullet 可能积累
- Generator/Reflector/Curator 强制使用同一模型，限制了利用不同大小模型优化成本的灵活性
- 在线模式下的顺序依赖（先见到的样本影响后续 context）是否引入偏差未深入分析

## 相关工作与启发
- **vs Dynamic Cheatsheet**: ACE 建立在 DC 之上但解决了其 context collapse 问题，引入了 Reflector 和 delta 更新机制
- **vs GEPA**: GEPA 是 prompt 优化器（追求简洁 prompt），ACE 是 context 工程（追求全面 playbook），理念不同，ACE 在 agent 和金融任务上都显著优于 GEPA
- **vs TextGrad**: TextGrad 使用梯度式文本反馈优化 prompt，ACE 用结构化 bullet 积累策略，避免了重写带来的信息损失

## 补充讨论

### 为什么 Context Engineering 比 Prompt Engineering 更重要？
Prompt Engineering 是静态的——一旦写好 system prompt 就固定不变。Context Engineering 是动态的——根据 agent 的运行经验持续演化上下文，更符合真实 agent 在复杂环境中的需求。Playbook 的 delta 更新机制是这一理念的具体实现。

## 评分
- 新颖性: ⭐⭐⭐⭐ "evolving playbook" 理念和 delta 更新设计有实际创新
- 实验充分度: ⭐⭐⭐⭐⭐ 两类 benchmark、多基线、消融完善、排行榜对比
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，概念有说服力，叙事流畅
- 价值: ⭐⭐⭐⭐⭐ context engineering 方向的重要工作，实用性极强
