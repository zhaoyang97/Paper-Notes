---
title: >-
  [论文解读] T2Agent: A Tool-augmented Multimodal Misinformation Detection Agent with Monte Carlo Tree Search
description: >-
  [AAAI 2026 Oral][社会计算][多模态虚假信息检测] 提出 T2Agent，一个集成可扩展工具集与蒙特卡洛树搜索（MCTS）的虚假信息检测智能体，通过多源验证机制将检测任务分解为针对不同伪造源的子任务，在 MMfakebench 上以 GPT-4o 为骨干将基线 MMDAgent 的准确率提升 28.7%，达到新 SOTA。
tags:
  - "AAAI 2026 Oral"
  - "社会计算"
  - "多模态虚假信息检测"
  - "蒙特卡洛树搜索"
  - "工具增强智能体"
  - "多源验证"
  - "训练无关检测"
---

# T2Agent: A Tool-augmented Multimodal Misinformation Detection Agent with Monte Carlo Tree Search

**会议**: AAAI 2026 Oral  
**arXiv**: [2505.19768](https://arxiv.org/abs/2505.19768)  
**代码**: [github.com/cuixing100876/T2Agent](https://github.com/cuixing100876/T2Agent)  
**领域**: 社会计算  
**关键词**: 多模态虚假信息检测, 蒙特卡洛树搜索, 工具增强智能体, 多源验证, 训练无关检测

## 一句话总结

提出 T2Agent，一个集成可扩展工具集与蒙特卡洛树搜索（MCTS）的虚假信息检测智能体，通过多源验证机制将检测任务分解为针对不同伪造源的子任务，在 MMfakebench 上以 GPT-4o 为骨干将基线 MMDAgent 的准确率提升 28.7%，达到新 SOTA。

## 研究背景与动机

### 多模态虚假信息的威胁

AIGC 技术的飞速发展降低了生成复杂多模态虚假信息的门槛，对信息完整性、公共治理和社会福祉构成严重威胁。开发有效的多模态虚假信息检测方法已成为技术和社会的迫切需求。

### 现有方法的两大瓶颈

**伪造源多样性要求定制化工具**：不同基准关注不同类型的伪造——AMG 考虑时间一致性，MMfakebench 包含反事实虚假信息。但现有 LLM 方法依赖固定且有限的工具集，无法灵活应对。

**真实世界虚假信息往往来自混合伪造源**：例如文本不准确、图像篡改或跨模态不一致同时存在。可靠检测需要同时具备：
   - **深度利用（Exploitation）**：为每个伪造源收集充分证据
   - **自适应探索（Exploration）**：在多个潜在伪造源之间灵活切换
   
   现有方法（如 MMDAgent）采用固定的静态工作流，无法在探索与利用之间取得平衡。

### 与 MMDAgent 的对比

MMDAgent 采用固定的顺序验证流程——按预定义路径逐步检查，容易在早期阶段产生错误传播。T2Agent 则借助 MCTS 动态规划验证路径，在多个伪造源之间自适应地平衡探索与利用。

## 方法详解

### 整体框架

T2Agent 由两个核心组件构成：
1. **多源验证 MCTS**：将检测任务分解为多个子任务（对应不同伪造源），通过树搜索动态收集证据
2. **可扩展工具集**：模块化工具以标准化模板描述，支持即插即用集成

### 关键设计

#### 1. **多源验证蒙特卡洛树搜索（MV-MCTS）**

**初始化**：
- 根节点代表整体任务（判断新闻真伪）
- 第一层子节点对应不同伪造源子任务（如文本失实、视觉失实、跨模态不一致）
- 使用 LVLM 分析输入内容，估计每个子任务的概率权重

**选择（Selection）**：改进的 UCT 公式：

$$UCT(s_t) = \frac{V(s_t)}{N(s_t) + 1} + C\sqrt{\frac{\ln(N(s) + 1)}{N(s_t) + 1}}$$

关键改进：在 $N(s_t)$ 上加偏置项 1，使得未访问节点（$N(s_t) = 0$）也能计算 UCT 值，避免给新节点分配任意大的奖励以减少不必要的资源浪费。

**扩展与模拟**：
- 在选中节点处，LVLM 生成思考→选择工具→执行动作→获取观测
- 利用失败轨迹作为记忆，避免重复错误

**评估——双重评分机制**：

1. **推理轨迹分数** $S_t^T$：评估从根节点到当前节点的推理路径质量和连贯性
$$S_t^T = LLM(\{s_i, a_i\}_{i=0...t})$$

2. **置信度分数** $S_t^C$：评估收集到的证据质量和内部一致性
$$S_t^C = LLM(\{o_i\}_{i=0...t}, c)$$

节点值的综合评估：$V(s_t) = \alpha S_t^T + (1 - \alpha) S_t^C$

**回溯传播**：
$$V(s_t) \leftarrow \frac{V(s_t) \cdot N(s_t) + V(s)}{N(s_t) + 1}$$

**剪枝策略**：如果某子任务节点返回高置信度的"真实"结果，则剪枝其子节点，模拟人类专家的工作方式——一旦某来源/模态被确认可靠，就转向验证其他不确定的部分。

#### 2. **协同决策（Decision Making）**

- **早停机制**：若有子任务高置信度判定为假，直接输出"假新闻"
- **概率融合**：无高置信度假信号时，聚合所有子任务结果：

$$p(\text{fake}^i) = \begin{cases} S^{C,i} & \text{if answer}(i) = \text{fake} \\ 1 - S^{C,i} & \text{if answer}(i) = \text{real} \end{cases}$$

$$p(\text{real}) = \prod_{i=1}^{n} (1 - p(\text{fake}^i))^{(1/n)}$$

最终判断：$\text{answer} = \arg\max(p(\text{real}), \{p(\text{fake}^i)\}_{i=1}^{n})$

#### 3. **可扩展工具集**

每个工具封装在**工具卡片**中，抽象其功能、输入输出格式和调用方法：

| 工具类别 | 具体工具 | 功能 |
|---------|---------|------|
| 网络搜索 | Google Search API, Wikipedia API | 检索外部知识 |
| 时间检测 | TinEye 反向图像搜索 | 追溯图片最早出现时间 |
| 伪造检测 | PSCC-NET | 图像篡改检测 |
| 反事实检测 | LLaVA-34B | 检测图像中逻辑/物理上不合理的元素 |
| 图像理解 | 与骨干模型对齐 | 详细的视觉内容问答解释 |
| 实体识别 | 百度实体识别 API | 识别图像中的关键实体 |

**贪心搜索工具选择**：
1. 预定义最小默认工具集 $D_{\text{base}}$
2. 逐个评估候选工具的增量准确率 $\Delta_{d_i} = \text{Acc}(D_{\text{base}} \cup \{d_i\}) - \text{Acc}(D_{\text{base}})$
3. 仅纳入 $\Delta_{d_i} > 0$ 的工具

与 OctoTools 的区别：OctoTools 独立评估每个工具，本文采用增量贪心方式评估组合效果（F1: 0.568 vs 0.550）。

### 损失函数 / 训练策略

T2Agent 是**训练无关（Training-free）**的检测方法——不需要额外训练，仅通过推理时的 MCTS 搜索和工具调用实现检测。推理时的关键超参数包括：
- 模拟次数 $K$
- 搜索深度限制 $d$
- 探索权重 $C$
- 轨迹与置信度的权衡系数 $\alpha$

## 实验关键数据

### 主实验

**MMfakebench 结果**（4类：真实、文本失实TVD、视觉失实VVD、跨模态不一致CCD）：

| 方法 | 骨干模型 | F1 | 准确率 |
|------|---------|-----|-------|
| Standard Prompt | GPT-4o | 0.492 | 0.609 |
| MMD-agent | GPT-4o | 0.614 | 0.616 |
| LRQ-FACT | GPT-4o | 0.716 | 0.708 |
| **T2Agent** | **GPT-4o** | **0.759** | **0.753** |
| MMD-agent | GPT-4o-mini | 0.478 | 0.485 |
| **T2Agent** | **GPT-4o-mini** | **0.631** | **0.629** |
| MMD-agent | GPT-4.1-nano | 0.398 | 0.424 |
| **T2Agent** | **GPT-4.1-nano** | **0.568** | **0.569** |

关键发现：使用 GPT-4o-mini 的 T2Agent（F1=0.631）甚至超过使用 GPT-4o 的 MMD-agent（F1=0.614），且成本仅 129.4$ vs 344.4$。

**AMG 结果**（5类伪造源）：

| 骨干模型 | 方法 | F1 | 准确率 |
|---------|------|-----|-------|
| GPT-4o | MMD-agent | 0.365 | 0.306 |
| GPT-4o | **T2Agent** | **0.510** | **0.579** |
| GPT-4o-mini | MMD-agent | 0.360 | 0.227 |
| GPT-4o-mini | **T2Agent** | **0.499** | **0.538** |

F1 提升幅度达 38.6%-39.7%，归因于 MCTS 避免了 MMD-Agent 顺序决策的错误传播。

### 消融实验

**各模块贡献（MMfakebench, GPT-4.1-nano）**：

| 方法 | F1 | 准确率 | 说明 |
|------|-----|-------|------|
| MMD-agent (baseline) | 0.398 | 0.424 | 固定工作流 |
| +TOOLs（仅工具） | 0.413 | 0.459 | 工具提升有限 |
| +MV_MCTS（仅树搜索） | 0.535 | 0.534 | MCTS 是核心（+34.4% F1） |
| **MV_MCTS + TOOLs（完整）** | **0.568** | **0.569** | 工具进一步提升 +6.2% |

**成本分析（MMfakebench, USD）**：

| 模型 | MMD-agent | T2Agent |
|------|-----------|---------|
| GPT-4o | 344.4 | 1637.1 |
| GPT-4o-mini | 14.3 | 129.4 |
| GPT-4.1-nano | 9.5 | 76.2 |

T2Agent 的计算成本更高，但在更便宜的模型上能超越更贵模型的基线，整体性价比更优。

### 关键发现

1. **MCTS 是性能提升的核心**：仅 MCTS 就带来 F1 +34.4% 的提升，证明动态验证远优于静态工作流
2. **工具选择比工具数量更重要**：贪心选择的 3 个工具优于 OctoTools 的 4 个工具（F1: 0.568 vs 0.550）
3. **轻量模型+T2Agent ≥ 重量模型+基线**：GPT-4o-mini 上的 T2Agent 超过 GPT-4o 上的 MMD-agent
4. **多源验证避免错误传播**：MMD-Agent 的顺序策略在 AMG（5 类伪造源）上严重退化

## 亮点与洞察

1. **MCTS 在信息验证中的创新应用**：将传统 MCTS 从单目标扩展到多源验证，引入子任务节点、剪枝策略和双重评分，优雅地平衡了探索与利用
2. **模拟人类专家的决策过程**：先评估最可能的伪造源，确认可靠后剪枝转向未验证的部分——这正是验证记者的工作方式
3. **训练无关优势**：无需收集标注数据和训练检测模型，可直接适应新出现的伪造类型
4. **工具集的标准化设计**：工具卡片抽象使得新工具的集成工作量最小化

## 局限与展望

1. **计算开销显著**：树搜索机制使得成本比基线高 3-8 倍
2. **依赖商业 LLM API**：需要 GPT-4o 等模型的 API 访问，成本不透明
3. **开源工具链安全风险**：论文注意到了这一点但未深入解决——需要实施最小权限原则和工具调用白名单
4. **未来方向**：引入高效剪枝策略、混合轻量专家模型引导搜索、探索与训练方法的结合

## 相关工作与启发

- **MMDAgent**：静态管道 + GPT-4o，是本文的直接基线，缺乏灵活性
- **LRQ-FACT**：通过生成问题检索证据，但工作流仍然固定
- **MGCA**：端到端训练的多视图特征方法，与 T2Agent 的训练无关路线形成互补
- **AlphaGo / AlphaZero**：MCTS 的经典应用，启发了本文对 MCTS 的扩展
- 对信息验证领域的启示：动态推理智能体是应对混合伪造源虚假信息的有前途方向

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — MCTS + 多源验证 + 可扩展工具集的创新组合
- 实验充分度: ⭐⭐⭐⭐ — 覆盖两个基准和多个骨干模型，含详细消融
- 写作质量: ⭐⭐⭐⭐ — 框架描述清晰，但部分细节需查阅附录
- 价值: ⭐⭐⭐⭐⭐ — 面向真实场景的训练无关检测方案，高度实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] MM-StanceDet: Retrieval-Augmented Multi-modal Multi-agent Stance Detection](../../ACL2026/social_computing/mm-stancedet_retrieval-augmented_multi-modal_multi-agent_stance_detection.md)
- [\[ACL 2026\] Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation](../../ACL2026/social_computing/beyond_the_crowd_llm-augmented_community_notes_for_governing_health_misinformati.md)
- [\[AAAI 2026\] Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference](reasoning_about_the_unsaid_misinformation_detection_with_omission-aware_graph_in.md)
- [\[ACL 2026\] Probing Multimodal Large Language Models on Cognitive Biases in Chinese Short-Video Misinformation](../../ACL2026/social_computing/probing_multimodal_large_language_models_on_cognitive_biases_in_chinese_short-vi.md)
- [\[ACL 2025\] BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](../../ACL2025/social_computing/biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)

</div>

<!-- RELATED:END -->
