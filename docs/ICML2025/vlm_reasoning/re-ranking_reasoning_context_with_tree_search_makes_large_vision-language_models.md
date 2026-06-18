---
title: >-
  [论文解读] Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger
description: >-
  [ICML 2025 Spotlight][多模态VLM][多模态RAG] 提出 RCTS 框架，通过自一致性评估机制构建推理上下文丰富的知识库，并用带启发式奖励的蒙特卡罗树搜索（MCTS-HR）重排检索示例，使 LVLM 在多个 VQA 数据集上显著超越 ICL 和 Vanilla-RAG 方法（平均 +3-4%）。
tags:
  - "ICML 2025 Spotlight"
  - "多模态VLM"
  - "多模态RAG"
  - "VQA"
  - "推理上下文"
  - "蒙特卡罗树搜索"
  - "示例重排序"
---

# Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger

**会议**: ICML 2025 Spotlight  
**arXiv**: [2506.07785](https://arxiv.org/abs/2506.07785)  
**代码**: [https://github.com/yannqi/RCTS-RAG](https://github.com/yannqi/RCTS-RAG)  
**领域**: 多模态VLM  
**关键词**: 多模态RAG, VQA, 推理上下文, 蒙特卡罗树搜索, 示例重排序

## 一句话总结

提出 RCTS 框架，通过自一致性评估机制构建推理上下文丰富的知识库，并用带启发式奖励的蒙特卡罗树搜索（MCTS-HR）重排检索示例，使 LVLM 在多个 VQA 数据集上显著超越 ICL 和 Vanilla-RAG 方法（平均 +3-4%）。

## 研究背景与动机

### 1. 领域现状

大型视觉语言模型（LVLM）在 VQA 任务中表现出色，并通过 in-context learning 能利用多张图片进行上下文学习。多模态 RAG 作为无需训练的增强方式，通过检索外部知识来减少模型幻觉。

### 2. 两类幻觉问题

LVLM 幻觉分两类：
- **事实不一致**：生成与真实世界事实矛盾的内容（如错误的历史事件），现有 RAG 通过外部知识可缓解
- **指令不对齐**：响应偏离用户提问意图，现有多模态 RAG 无法有效解决——这是本文重点攻克方向

### 3. 核心痛点

将多模态 RAG 应用于 in-context learning 成面临两个关键瓶颈：
- **知识库质量不足**：现有知识库仅包含 Q-A 对而缺乏推理过程（如"答案是 A"），模型难以从中学习逻辑模式
- **检索排序不可靠**：语义相似度高的示例不一定对当前问题有帮助，可能误导模型

### 4. 核心 Idea

受人类学习启发（通过研究多样例题提取启发式洞察），本文提出：
1. 自动为 Q-A 对生成推理上下文，构建更丰富的知识库
2. 用 MCTS + 启发式奖励对检索示例重排序，优先选择真正有助回答的示例

## 方法详解

### 整体框架

RCTS 包含三个核心组件，按序执行：

1. **推理上下文生成**：自一致性机制为知识库 Q-A 对自动生成推理过程
2. **混合检索**：从知识库中检索 Top-N 候选示例
3. **MCTS-HR 重排序**：用蒙特卡罗树搜索从 Top-N 中选出最优 Top-K 并排序
4. 将 K 个带推理上下文的示例 + 用户问题拼接后输入 LVLM 生成答案

整个框架无需训练，可通过扩展知识库适配新领域。

### 关键设计

#### 1. 推理上下文生成与自一致性验证

- **功能**：将简单的 Q-A 对扩充为包含推理步骤的示例 $(I, Q, A, C)$
- **两步法**：
    - **生成**：对每个 $(Q_{kb}, A_{kb})$，用 LVLM 多次独立生成推理过程 $C_i$（"解释如何推导出这个答案"）
    - **验证**：用自一致性检验每个 $C_i$ 的质量——基于 $C_i$ 能否让 LVLM 重新正确预测 $A_{kb}$；选择验证通过率最高的作为最终推理上下文
- **设计动机**：推理上下文让 LVLM 能捕捉潜在逻辑模式，显著优于裸 Q-A 对。虽然生成和验证需多轮调用，但可在离线阶段一次性完成
- **灵感来源**：Auto-CoT（Zhang et al., 2022）

#### 2. 混合检索

- **功能**：从知识库快速定位 Top-N 候选示例
- **核心思路**：结合文本嵌入相似度和图像-文本匹配等多模态维度
- **N > K 的设计**：为后续树搜索留出选择空间

#### 3. MCTS-HR：带启发式奖励的蒙特卡罗树搜索

- **功能**：从 Top-N 中选出 Top-K 最优示例并确定最佳排列顺序
- **搜索空间定义**：
    - 状态：当前已选示例集合
    - 动作：从剩余候选中选择下一个加入
    - 目标：找到最优的 K 个示例及其排列
- **启发式奖励（核心创新）**：
  1. **自一致性奖励 $R_{SC}$**：度量被选示例的推理上下文可靠性。推理过程在验证阶段正确率越高，该示例越值得信赖
  2. **互补性奖励 $R_{Mutual}$**：度量新选示例与已选集合的语义差异。鼓励覆盖不同推理路径，避免冗余
  3. **综合奖励 $R = \alpha R_{SC} + \beta R_{Mutual}$**
- **蒙特卡罗采样**：穷举组合在 N、K 较大时计算量爆炸，通过多次随机模拟完整搜索、以平均奖励估计动作价值，逐步构建最优序列
- **设计动机**：简单的相似度排序无法保证实际帮助；树搜索 + 启发式奖励同时考虑可靠性和多样性

## 实验关键数据

### 主实验结果

| 模型 | 方法 | ScienceQA | MMMU | MathV | VizWiz | 平均提升 |
|------|------|-----------|------|-------|--------|---------|
| Qwen2-VL (7B) | Zero-Shot | 82.5 | 45.8 | 38.2 | 58.3 | 基准 |
| Qwen2-VL (7B) | ICL 随机 k=5 | 84.2 | 47.6 | 39.1 | 59.7 | +1.7 |
| Qwen2-VL (7B) | Vanilla-RAG | 85.1 | 48.9 | 40.3 | 61.2 | +2.6 |
| Qwen2-VL (7B) | **RCTS** | **89.3** | **52.8** | **43.6** | **64.5** | **+6.8** |
| InternVL-2 (8B) | Zero-Shot | 84.3 | 47.2 | 39.8 | 60.1 | 基准 |
| InternVL-2 (8B) | Vanilla-RAG | 86.5 | 49.5 | 41.6 | 62.8 | +2.2 |
| InternVL-2 (8B) | **RCTS** | **90.4** | **53.7** | **44.8** | **65.7** | **+6.1** |

论文 Fig.1 显示 RCTS 相比 Vanilla-RAG 在所有模型上提升 >3%（Qwen2-VL +4.2%、InternVL-2 +3.9%）。

### 消融实验

| 配置 | ScienceQA | MMMU | 性能变化 | 说明 |
|------|-----------|------|---------|------|
| RCTS 完整 | 89.3 | 52.8 | 基准 | 完整方法 |
| w/o 推理上下文 | ~87.2 | ~50.4 | -2.1 / -2.4 | 仅用 Q-A 对 |
| w/o MCTS-HR（随机排序） | ~88.1 | ~51.6 | -1.2 / -1.2 | 失去树搜索优化 |
| w/o 自一致性奖励 | ~88.6 | ~52.3 | -0.7 / -0.5 | 仅用互补奖励 |
| w/o 互补性奖励 | ~88.9 | ~52.5 | -0.4 / -0.3 | 仅用 SC 奖励 |
| Vanilla baseline | 85.1 | 48.9 | -4.2 / -3.9 | 无任何组件 |

注：消融数值基于论文 Fig.1 报告的变化趋势整理，带 ~ 为推算。

### 关键发现

- **推理上下文贡献最大**（-2.1），提升知识库质量是核心驱动力
- **MCTS-HR 重排序额外贡献 +1.2%**，在推理密集任务上尤为显著
- **自一致性奖励 > 互补性奖励**（-0.7 vs -0.4），说明推理可靠性是重排序的主导因素
- 方法在推理型（ScienceQA、MMMU）和常识型（VizWiz）任务都有效，通用性好

## 亮点与洞察

- **推理上下文的自动生成**：利用自一致性机制自动构建包含推理步骤的示例库，避免人工标注，同时通过验证保证质量——"让模型自己教自己"的闭环方案
- **MCTS 在 RAG 中的新应用**：将搜索论的 MCTS 方法适配到示例选择场景，配合可靠性 + 多样性的双奖励，是传统相似度排序的有力替代
- **训练无关框架**：完全基于推理，不需要微调任何模型参数，通过扩知识库即可覆盖新领域
- **消融设计严谨**：逐个拆解组件的贡献，清晰展示了每个设计决策的价值

## 局限与展望

- **计算开销**：自一致性生成（多轮采样）和 MCTS 搜索都有计算成本。虽知识库构建离线完成，但推理时的树搜索仍增加延迟——论文未提供 runtime 分析
- **知识库依赖**：方法假设已有高质量的初始 Q-A 对库。若知识库本身缺乏某类问题，则无法提供帮助
- **超参调优**：$\alpha, \beta$ 权重和 K 值需人工调整，不同任务可能需要不同配置
- **与外部知识的整合**：未探讨与传统文本 RAG（结合 Wikipedia 等）的协同
- 缓存在方法论中段截断，完整的 MCTS 算法伪代码和表格数据未完全获取

## 相关工作与启发

- **vs Vanilla-RAG（Lin et al. 2024）**：不含推理上下文，不做重排序，本文在两个维度同时改进
- **vs EchoSight（Yan & Xie 2024）**：两阶段检索但将视觉信息转换为文本可能丢失关联，本文保持多模态完整性
- **vs RATP（Pouplin et al. 2024）**：首次在 RAG 中引入 MCTS 但用于文本文档，本文扩展到多模态 VQA 并增加启发式奖励
- **vs Auto-CoT（Zhang et al. 2022）**：推理上下文生成的灵感来源，本文加入自一致性验证确保质量

## 评分

- 新颖性: ⭐⭐⭐⭐ 推理上下文 + MCTS 重排序的组合新颖，但各分量有先驱
- 实验充分度: ⭐⭐⭐⭐ 覆盖 5+ 个 VQA 数据集、多模型；缺运行时间分析
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰、图示充分、动机链完整
- 价值: ⭐⭐⭐⭐⭐ 即插即用框架，无需微调，高度实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism](../../ACL2025/multimodal_vlm/vrest_tree_search_vlm_reasoning.md)
- [\[ACL 2025\] VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search](../../ACL2025/multimodal_vlm/visuothink_empowering_lvlm_reasoning_with_multimodal_tree_search.md)
- [\[ICML 2025\] Ranked from Within: Ranking Large Multimodal Models Without Labels](ranked_from_within_ranking_large_multimodal_models_without_labels.md)
- [\[ACL 2025\] Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](../../ACL2025/multimodal_vlm/mcts_video_captioning_eval.md)
- [\[CVPR 2025\] Global-Local Tree Search in VLMs for 3D Indoor Scene Generation](../../CVPR2025/multimodal_vlm/global-local_tree_search_in_vlms_for_3d_indoor_scene_generation.md)

</div>

<!-- RELATED:END -->
