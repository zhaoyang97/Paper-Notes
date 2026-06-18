---
title: >-
  [论文解读] VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search
description: >-
  [ACL 2025][多模态VLM][多模态推理] 本文提出VisuoThink框架，通过视觉-文本交织推理和预测性前瞻树搜索，在推理过程中动态整合视觉辅助信息并探索多条推理路径，无需微调即可在几何和空间推理任务上实现SOTA性能（Geomverse-109上Accuracy@1最高达48.5%，相比最优基线提升21.8%）。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "多模态推理"
  - "树搜索"
  - "视觉-文本交织思考"
  - "测试时缩放"
  - "几何推理"
---

# VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search

**会议**: ACL 2025  
**arXiv**: [2504.09130](https://arxiv.org/abs/2504.09130)  
**代码**: [GitHub](https://github.com/ekonwang/VisuoThink)  
**领域**: 多模态VLM / LLM推理  
**关键词**: 多模态推理, 树搜索, 视觉-文本交织思考, 测试时缩放, 几何推理

## 一句话总结

本文提出VisuoThink框架，通过视觉-文本交织推理和预测性前瞻树搜索，在推理过程中动态整合视觉辅助信息并探索多条推理路径，无需微调即可在几何和空间推理任务上实现SOTA性能（Geomverse-109上Accuracy@1最高达48.5%，相比最优基线提升21.8%）。

## 研究背景与动机

**领域现状**：大视觉语言模型（LVLM）在多种任务上取得显著进展，随着o1系列模型的成功，研究者开始探索将"慢思考"（slow thinking）机制引入LVLM。已有方法通过阶段式推理、课程学习、树搜索数据生成等方式扩展深度思考能力。

**现有痛点**：现有扩展方法存在两个核心问题：（1）将视觉信息仅视为静态输入，推理过程完全依赖文本链条——形成"视觉盲区"，忽略了推理过程中视觉信息的潜力；（2）虽然VisualSketchpad和VoT等方法已尝试融入视觉辅助，但仅限于单步辅助或简化的视觉提示（如emoji），缺乏人类慢思考中那种多步视觉-文本交织推理过程。

**核心矛盾**：人类解决复杂几何问题时会反复画辅助线、可视化中间步骤，并探索不同推理路径。但现有LVLM推理方法既没有多步视觉辅助，也没有系统的搜索机制来探索不同推理路径。

**本文目标** (1) 如何在推理过程中实现多步视觉-文本交织推理；(2) 如何通过搜索策略实现推理的测试时缩放（test-time scaling）。

**切入角度**：受人类认知中慢思考机制启发，将视觉工具调用与树搜索算法结合，在每一步推理中既能生成可靠的视觉辅助，又能通过前瞻搜索优化推理路径选择。

**核心 idea**：将多模态推理建模为"视觉-文本交织扩展 + 前瞻模拟 + 自投票选择"的树搜索过程，实现LVLM的多模态慢思考。

## 方法详解

### 整体框架

VisuoThink框架包含三个核心阶段的循环：（1）视觉-文本交织扩展：生成k个候选推理路径，每条路径包含Thought-Action-Observation循环；（2）前瞻模拟：对每个候选节点模拟完整推理到最终结果；（3）选择：通过自投票机制选择最有前途的路径。最终形成一棵推理树，沿最优路径得到答案。

### 关键设计

1. **视觉-文本交织思考 (Vision-Text Interleaved Thinking)**:

    - 功能：在推理的每一步动态融合视觉和文本信息
    - 核心思路：采用类似ReAct的Thought-Action-Observation迭代循环：(1) Thought阶段：模型基于当前视觉信息进行文本推理，规划下一步需要什么视觉辅助；(2) Action阶段：调用外部工具（如Python matplotlib）生成或修改视觉信息（画辅助线、标注关键特征等）；(3) Observation阶段：处理工具返回的视觉反馈，整合到下一步推理中。关键区别于VisualSketchpad的是逐步构建而非一次性生成所有视觉辅助。
    - 设计动机：逐步视觉构建自然地与搜索技术结合，使搜索算法可以在每一步评估和优化视觉推理路径

2. **预测性前瞻搜索 (Predictive Rollout Search)**:

    - 功能：通过模拟未来推理结果来评估当前候选路径的潜力
    - 核心思路：在每个推理步骤，模型采样k个候选节点 $S_t = \{s_t^1, ..., s_t^k\}$。对每个候选节点执行rollout模拟——沿单条路径进行视觉-文本交织推理直到得到最终结果 $r_t^i$。然后通过自投票机制 $\mathbf{Select}(S_t) = \arg\max_{s_t^i \in S_t} \mathbf{Vote}(A_{t-1}, s_t^i, r_t^i)$ 选择最优候选。LVLM自身作为启发式函数，综合考虑历史上下文、候选节点和模拟结果进行投票。
    - 设计动机：视觉推理通常需要多步才能得出结论，单步评估不足以判断路径潜力。通过前瞻模拟，模型可以"看到"每条路径的可能结局，从而做出更明智的选择

3. **几何问题求解的两阶段框架**:

    - 功能：针对几何问题设计专用的视觉构建与代数计算流程
    - 核心思路：Phase I（视觉构建）：模型逐步生成由几何约束定义的辅助线（连接点、作垂线/平行线等），以AUX-END token标记结束；Phase II（代数计算）：将几何关系转化为可解方程，通过Python代码执行精确计算，缓解LVLM数值推理的幻觉问题
    - 设计动机：几何问题不能仅依赖视觉构建或模型内在能力，需要精确数值计算工具来保证结果准确性

### 损失函数 / 训练策略

VisuoThink是一个推理时框架，**不需要任何微调**。直接使用现有的SOTA闭源模型（GPT-4o、Claude-3.5-sonnet）和开源模型（Qwen2-VL-72B-Instruct）进行推理。搜索超参数包括树宽度k和最大推理步数τ。

## 实验关键数据

### 主实验

| 数据集 | 方法 | GPT-4o | Claude-3.5 | Qwen2-VL-72B |
|-------|------|--------|------------|--------------|
| Geomverse-109 | CoT | 11.1 | 14.4 | 5.6 |
| Geomverse-109 | VisualSketchpad+Eq | 13.3 | 17.8 | 11.1 |
| Geomverse-109 | VisuoThink (w/o rollout) | 24.4 | 26.7 | 19.0 |
| Geomverse-109 | **VisuoThink** | **28.9** | **27.8** | **25.6** |
| Visual Nav (level-3) | CoT | 18.8 | 37.5 | 6.7 |
| Visual Nav (level-3) | VoT+Executor | 62.5 | 68.8 | 25.0 |
| Visual Nav (level-3) | **VisuoThink** | **93.8** | **93.8** | **81.3** |
| Visual Tiling (level-2) | CoT | 0.8 | 0.8 | 0.0 |
| Visual Tiling (level-2) | **VisuoThink** | **51.2** | **84.0** | **20.2** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| VisuoThink w/o rollout vs w/ rollout | 几何任务平均+4.1% | 前瞻搜索在几何任务中提升适中 |
| VisuoThink w/o rollout vs w/ rollout | 空间推理平均+34.7% | 空间推理任务中提升巨大 |
| 推理步数10→20→40 | +54.1% → +6.5% (GPT-4o) | 更多步数有帮助但边际递减 |
| 树宽度k=1→3→5→7 | 倒U型趋势 | 过大的树宽度反而因节点评估噪声降低性能 |

### 关键发现

- VisuoThink在所有评估模型和任务上均显著优于基线，Geomverse-109上相比CoT和VisualSketchpad平均提升17.1%和16.7%
- 空间推理任务中前瞻搜索的增益远大于几何任务，可能因为空间推理提供了更强的监督信号（如智能体最终位置等可视化状态）
- 增加推理步数10→20有大幅提升，但20→40边际收益急剧下降，说明仅增加试错机会无法解决最困难的样本
- 树宽度存在最优值（通常为3-5），过大会因模型对子节点评估的固有误差导致混淆，这是一个反直觉但重要的发现

## 亮点与洞察

- 首次将多模态树搜索引入LVLM推理，实现了视觉和语言推理路径的联合探索
- 无需微调即可在推理时大幅提升性能，充分利用了现有模型的能力上限
- 前瞻搜索中强/弱监督的差异分析与DeepSeek-R1的结果遥相呼应，暗示outcome-based supervision的重要性
- 树宽度的倒U型趋势揭示了"更多搜索不一定更好"的原则，对测试时缩放研究有普遍启示

## 局限与展望

- 计算开销大：预测前瞻搜索引入显著的计算负担，不适合实时应用
- 依赖外部工具交互，在某些部署环境中可能需要额外适配
- 受基础VLM推理能力的天花板限制，无法克服模型的根本局限
- 仅在几何和空间推理上评估，在更广泛的视觉推理任务（如物理世界理解、图表推理）上的表现未知
- 自投票选择机制依赖模型自身的判断，可能在某些情况下不可靠

## 相关工作与启发

- VisualSketchpad (Hu et al., 2024) 和VoT (Wu et al., 2024) 是视觉辅助推理的先驱工作，但局限于单步辅助
- o1系列模型和DeepSeek-R1在文本慢思考方面的成功为多模态扩展提供了动机
- MCTS在AlphaZero和LLM decoding中的成功应用启发了预测性前瞻搜索的设计
- 该框架可能与训练时方法（如RLHF）结合，进一步增强LVLM的多模态推理能力

## 评分

- **新颖性**: 8/10 — 多模态树搜索是一个新颖且有价值的范式
- **技术深度**: 7/10 — 框架设计清晰但各组件相对标准（ReAct+MCTS变体）
- **实验充分性**: 8/10 — 多模型多任务评估，消融分析深入
- **写作质量**: 8/10 — 结构清晰，可视化效果好
- **应用价值**: 7/10 — 计算成本限制了直接应用，但思路有启发性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism](vrest_tree_search_vlm_reasoning.md)
- [\[ACL 2025\] Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](mcts_video_captioning_eval.md)
- [\[CVPR 2025\] Global-Local Tree Search in VLMs for 3D Indoor Scene Generation](../../CVPR2025/multimodal_vlm/global-local_tree_search_in_vlms_for_3d_indoor_scene_generation.md)
- [\[ICML 2025\] Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger](../../ICML2025/multimodal_vlm/re-ranking_reasoning_context_with_tree_search_makes_large_vision-language_models.md)
- [\[ACL 2026\] Tree-of-Evidence: Efficient "System 2" Search for Faithful Multimodal Grounding](../../ACL2026/multimodal_vlm/tree-of-evidence_efficient_34system_234_search_for_faithful_multimodal_grounding.md)

</div>

<!-- RELATED:END -->
