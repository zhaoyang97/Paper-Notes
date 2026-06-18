---
title: >-
  [论文解读] Progressive Multimodal Reasoning via Active Retrieval
description: >-
  [ACL 2025][多模态VLM][多模态推理] 本文提出AR-MCTS框架，将主动检索（Active Retrieval）与蒙特卡洛树搜索（MCTS）结合，在多步多模态推理的每一步动态检索关键知识来替代传统beam search采样，自动生成逐步推理标注以渐进式对齐过程奖励模型（PRM），在MathVista、We-Math和GAOKAO-MM上显著提升了多种MLLM的推理性能。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "多模态推理"
  - "蒙特卡洛树搜索"
  - "主动检索"
  - "过程奖励模型"
  - "检索增强生成"
---

# Progressive Multimodal Reasoning via Active Retrieval

**会议**: ACL 2025  
**arXiv**: [2412.14835](https://arxiv.org/abs/2412.14835)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 多模态推理, 蒙特卡洛树搜索, 主动检索, 过程奖励模型, 检索增强生成

## 一句话总结
本文提出AR-MCTS框架，将主动检索（Active Retrieval）与蒙特卡洛树搜索（MCTS）结合，在多步多模态推理的每一步动态检索关键知识来替代传统beam search采样，自动生成逐步推理标注以渐进式对齐过程奖励模型（PRM），在MathVista、We-Math和GAOKAO-MM上显著提升了多种MLLM的推理性能。

## 研究背景与动机
多步多模态推理是多模态大语言模型（MLLM）面临的核心挑战之一。现有基于MCTS的方法已在纯文本LLM中取得成功，但在多模态场景中存在两个关键局限：

**扩展阶段的知识不足**：传统MCTS在扩展阶段依赖beam search，利用模型内部知识进行采样。对于文本任务，LLM的内部知识足够支撑推理路径扩展；但在多模态推理中，不同模态输入之间的交互经常出现对齐错误，导致内部知识不足以支撑可靠的路径采样。

**误差累积效应**：多步推理中每一步依赖前一步，小错误会随步骤增加而放大。

核心矛盾在于：如何在MCTS的扩展阶段为多模态推理提供可靠的外部知识支持，同时保持采样空间的多样性和准确性？

本文的切入角度是：在MCTS扩展的每一步动态检索不同的问题解决洞察，替代传统beam search的单一采样策略，从而提升采样质量和多样性。

核心 idea：**用主动检索替代beam search作为MCTS扩展策略，实现外部知识驱动的多模态推理路径采样与验证**。

## 方法详解

### 整体框架
AR-MCTS由两大核心组件构成：
1. **统一检索模块**：构建混合模态检索语料库 + 多模态检索模块 + 知识概念过滤
2. **基于MCTS与主动检索的推理标注**：利用MCTS的四步操作（选择、扩展、模拟、回溯），在扩展阶段引入主动检索，自动获取逐步推理标注，进而**渐进式对齐PRM**

### 关键设计

1. **混合模态检索语料库构建**:

    - 数学专用知识：整合GSM8K、MATH（纯文本）+ MathVista、MathVerse、MathVision、We-Math（多模态），共22K文本QA对 + 12.5K多模态样本对
    - 通用推理知识：利用Wikipedia和COIG大规模题库
    - 通过正则表达式去除与测试集重叠部分，防止数据泄露
    - 每个样本包含问题q、解题过程p和答案a，拼接为统一格式

2. **统一多模态检索模块**:

    - **文本检索**：使用Contriever作为密集检索器，计算query与文档的点积相似度
    - **跨模态检索**：使用CLIP双流架构，对图像编码器E_I和文本编码器E_T的输出取平均作为混合表示，利用FAISS索引进行高效检索
    - 对于混合模态语料库中的纯文本样本和多模态样本采用不同的编码策略

3. **知识概念过滤**:

    - 观察到多模态推理对细粒度知识概念的一致性高度敏感（如代数知识无法帮助解决三角形问题）
    - 利用数据集自带的知识概念标签（如"角度与长度"），同时设置检索相似度阈值T_r和知识概念一致性阈值T_kc进行双重过滤
    - 只有同时满足两个阈值的样本才作为问题解决洞察

4. **MCTS中的主动检索策略（核心创新）**:

    - **选择**：从根节点通过UCB公式递归选择子节点
    - **扩展（关键改进）**：在每一步扩展时，将当前状态的查询与之前的推理步骤拼接，动态从洞察库D_ins中检索该步所需的候选洞察r_i，替换上一步的洞察r_{i-1}。不同的检索结果产生不同的推理分支，增强采样多样性
    - **模拟**：使用one-step rollout评估每个节点的价值V(s_i)，通过判断推理路径是否能推导出正确答案来赋值
    - **回溯**：反向更新访问计数和Q值

5. **课程式过程奖励建模（两阶段PRM训练）**:

    - **第一阶段：逐步DPO预对齐**。MCTS的扩展和评估过程自然产生正负样本对（value > 0.8为正，value = 0为负），使用step-level DPO目标训练PRM区分推理步骤的正确性
    - **第二阶段：逐点微调**。在预对齐的PRM上应用逐步交叉熵目标，进一步解锁其推理评分能力，实现从易到难的泛化

### 损失函数 / 训练策略
- 第一阶段：Step-level DPO损失（L_SDPO），最大化正样本y+相对于负样本y-的似然
- 第二阶段：逐点交叉熵损失（L_PFT），对每个状态的golden label（0/1）进行sigmoid评分
- 推理阶段设置early stopping为4轮，提取每轮最高分节点，丢弃低质量路径

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文(AR-MCTS) | 之前SOTA(ORM/SC) | 提升 |
|--------|------|------|----------|------|
| MathVista(ALL) GPT-4o | Accuracy | 62.6% | 61.9%(ORM) | +0.7% |
| We-Math(AVG) GPT-4o | Score | 46.8% | 45.2%(SC) | +1.6% |
| MathVista(ALL) Qwen2-VL-7B | Accuracy | 64.1% | 62.3%(ORM) | +1.8% |
| We-Math(AVG) Qwen2-VL-7B | Score | 28.1% | 26.4%(ORM) | +1.7% |
| MathVista(ALL) InternVL2-8B | Accuracy | 63.1% | 61.8%(SC) | +1.3% |
| We-Math(S3) InternVL2-8B | Score | 43.6% | 35.1%(SC) | +8.5% |
| GAOKAO-MM GPT-4o | Overall | 52.2% | 47.8%(SC) | +4.4% |

### 消融实验

| 配置 | MathVista(ALL) | We-Math(S3) | 说明 |
|------|---------|------|------|
| AR-MCTS (完整) | 64.1% | 40.6% | 基线 |
| w/o PRM | 61.0% (-3.1) | 37.7% (-2.9) | PRM对推理验证至关重要 |
| w/o Active Retrieval | 61.9% (-2.2) | 38.7% (-1.9) | 主动检索显著提升采样质量 |
| w/o Filtering | 62.8% (-1.3) | 39.5% (-1.1) | 知识概念过滤减少噪声 |

### 关键发现
- MLLM难以自我纠正推理错误：Self-Correction策略在多数模型上导致性能下降，Qwen2-VL-7B在MathVista上下降超过8%
- PRM在复杂推理任务中优于ORM：尤其在We-Math的S3指标上（GPT-4o: 56.4% vs 50.3%）
- AR-MCTS对较弱MLLM的提升更为显著：Qwen2-VL-7B在MathVista上提升5.3%，在We-Math上提升8.3%，表明小模型有正确推理的潜力但缺乏有效解码策略
- 采样多样性分析：AR-MCTS相比beam search产生更多聚类中心（38 vs 46）和更分散的语义表示分布
- 跨领域验证：在中文GAOKAO-MM上也取得一致性提升，数学+12.5%，物理+7.7%，历史+20%

## 亮点与洞察
- 理论建模清晰：通过公式(1)将MCTS的扩展和模拟过程统一建模，揭示了传统方法在多模态场景下的核心局限
- 创新性地将RAG思想引入MCTS的扩展阶段，用动态检索替代beam search，开辟了新的采样策略思路
- 两阶段课程式PRM训练（DPO预对齐 → 逐点微调）设计合理，符合从易到难的学习范式
- 框架的通用性好，作为plug-and-play框架适用于各种MLLM骨干
- 采样空间的准确性和多样性分析提供了直观的可视化证据

## 局限与展望
- 检索语料库主要覆盖数学推理，对其他领域（如逻辑推理、常识推理）的适用性有待验证
- 知识概念过滤依赖数据集自带的类别标签，但很多真实场景中缺少这样的标注
- 检索模块的实时性和效率问题：每步都需要动态检索，推理开销较大
- 可探索的方向：将主动检索策略与推测解码（speculative decoding）结合以提升效率；探索自适应的检索触发机制（不是每步都检索）

## 相关工作与启发
- MCTS在LLM推理中的应用（AlphaCode、o1等）：本文将其扩展到多模态场景并解决了关键的扩展策略问题
- RAG在多模态领域的应用（MuRAG等）：本文创新性地将RAG与MCTS结合，实现逐步检索
- PRM的训练方法（Math-Shepherd等）：本文提出了自动化的多模态PRM标注和训练方案

## 评分
- 新颖性: ⭐⭐⭐⭐ 将主动检索引入MCTS扩展阶段是全新的思路，但整体框架是已有技术的组合
- 实验充分度: ⭐⭐⭐⭐ 三个benchmark、多种MLLM骨干、详细的消融和分析，但缺少计算开销对比
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，理论建模规范，但符号较多可能增加阅读难度
- 价值: ⭐⭐⭐⭐ 提供了一个通用的多模态推理增强框架，对推理验证和检索增强社区都有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval](megapairs_massive_data_synthesis_for_universal_multimodal_retrieval.md)
- [\[ICML 2026\] Spectral-Progressive Thought Flow for Lightweight Multimodal Reasoning](../../ICML2026/multimodal_vlm/spectral-progressive_thought_flow_for_lightweight_multimodal_reasoning.md)
- [\[ACL 2025\] VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search](visuothink_empowering_lvlm_reasoning_with_multimodal_tree_search.md)
- [\[ACL 2025\] OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval](omgm_orchestrate_multiple_granularities_and_modalities_for_efficient_multimodal_.md)
- [\[ACL 2025\] VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism](vrest_tree_search_vlm_reasoning.md)

</div>

<!-- RELATED:END -->
