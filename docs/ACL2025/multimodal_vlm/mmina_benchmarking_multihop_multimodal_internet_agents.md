---
title: >-
  [论文解读] MMINA: Benchmarking Multihop Multimodal Internet Agents
description: >-
  [ACL 2025][多模态VLM][Web Agent] 提出MMInA基准，包含1,050个人工编写的多跳多模态网页任务（覆盖14个真实动态网站，平均2.85跳），并设计逐跳评估协议和记忆增强方法，揭示当前最强Agent（GPT-4V仅21.8%任务成功率）在多跳网页导航上与人类（96.3%）的巨大差距。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "Web Agent"
  - "多跳推理"
  - "多模态基准"
  - "网页浏览"
  - "记忆增强"
---

# MMINA: Benchmarking Multihop Multimodal Internet Agents

**会议**: ACL 2025  
**arXiv**: [2404.09992](https://arxiv.org/abs/2404.09992)  
**作者**: Shulin Tian, Ziniu Zhang, Liangyu Chen, Ziwei Liu (NTU S-Lab)  
**代码**: [github.com/shulin16/MMInA](https://github.com/shulin16/MMInA)  
**领域**: 多模态VLM  
**关键词**: Web Agent, 多跳推理, 多模态基准, 网页浏览, 记忆增强  

## 一句话总结

提出MMInA基准，包含1,050个人工编写的多跳多模态网页任务（覆盖14个真实动态网站，平均2.85跳），并设计逐跳评估协议和记忆增强方法，揭示当前最强Agent（GPT-4V仅21.8%任务成功率）在多跳网页导航上与人类（96.3%）的巨大差距。

## 研究背景与动机

### 问题背景
构建能自主导航互联网、完成复杂用户任务的具身智能体是AI领域的长期挑战。现实中的网页任务天然具有组合性——用户经常需要跨多个网站收集信息或执行操作（如"订机票→查旅游攻略→租车→订酒店"），这要求Agent具备长程规划和多模态推理能力。

### 已有工作的不足
- **单跳局限**：现有基准（MiniWoB++、WebShop、Mind2Web、WebArena等）绝大多数任务仅涉及单个网站，平均跳数接近1.0，无法评估跨网站的组合推理
- **文本为主**：WebArena、Mind2Web等基准主要依赖文本信息（accessibility tree），忽视了图像在真实网页任务中的关键作用（如"买一件蓝色棉质衬衫"需要视觉判断颜色）
- **静态环境**：多数基准使用静态快照或本地部署的网站，无法反映真实网页的动态变化特性
- **评估粗粒度**：仅使用任务级成功率评估，在多跳场景下往往接近零，难以提供有价值的分析洞察

### 核心动机
填补多跳+多模态+动态真实网站三重空白，建立更贴近真实场景的Internet Agent评估体系。

## 方法详解

### 关键设计1：基准构建与环境设计

**环境建模**：将网页浏览形式化为部分可观测马尔可夫决策过程 $\langle S, A, P, R \rangle$。Agent在每个时间步接收部分观测 $o_t \in \Omega$（包含accessibility tree、页面图片、历史动作），执行12种标准化动作之一（点击、滚动、键盘输入等）。

**数据集构建**：
- 1,050个人工编写任务，覆盖购物、旅行、搜索、票务预订等多个领域
- 跨14个真实动态网站，包含2,989个子跳
- 跳数范围1-10跳，平均2.85跳，平均每个任务需12.9个动作
- 标注者采用"极简主义"策略：以全知视角用最短路径完成任务，记录关键URL节点

**多模态设计**：所有任务均需同时处理视觉和文本信息。环境自动提取accessibility tree的同时识别并下载当前视图中的图片，图片上标注元素ID供Agent引用。

### 关键设计2：多跳评估协议

**单跳评估**：采用两种方法——
- `must_include`：关键词匹配，Agent回答必须包含所有预定义关键词
- `fuzzy_match`：利用GPT-3.5-Turbo进行语义匹配，处理如"gold"与"yellow"的语义等价

**多跳评估**：维护一个包含各跳完成条件的队列（长度为 $N+1$，末尾为END标记）。Agent必须按顺序完成每一跳——仅当当前跳正确完成后才能进入下一跳。同时计算跳成功率（hop SR）和任务成功率（task SR），提供更细粒度的性能分析。

### 关键设计3：记忆增强方法

提出三层记忆系统增强Agent：
- **语义记忆**（Semantic Memory）：编码在LLM权重中的通用世界知识
- **片段记忆**（Episodic Memory）：临时保存当前任务的逐步动作轨迹，作为自回归模型的上下文
- **过程记忆**（Procedural Memory）：任务完成后编码完整动作序列和结果，为未来相似任务提供经验回放

核心思路是通过过程记忆回放（replaying past action trajectories）让Agent在执行相似任务时参考历史成功轨迹，显著提升单跳和多跳性能。

## 实验关键数据

### 实验1：主要基准结果

| Agent | 输入类型 | 1跳 Hop SR | 2-4跳 Hop SR | 5+跳 Hop SR | 总Hop SR | 1跳 Task SR | 2-4跳 Task SR | 5+跳 Task SR | 总Task SR |
|-------|---------|-----------|-------------|-------------|---------|-----------|-------------|-------------|---------|
| GPT-4 (文本) | Tree | 14.37 | 30.56 | 5.23 | 12.26 | 14.37 | 9.09 | 0 | 9.34 |
| GPT-4 (文本+描述) | Tree+Caption | 38.58 | 20.70 | 3.43 | 13.50 | 38.58 | 3.79 | 0 | 19.85 |
| DeepSeek-R1-32B (文本+描述) | Tree+Caption | 47.68 | 3.84 | 4.68 | 11.11 | 47.68 | 0 | 0 | 23.07 |
| GPT-4V (多模态) | Tree+Image | 42.91 | 21.23 | 3.99 | 13.89 | 42.91 | 3.03 | 0 | 21.77 |
| Gemini-Pro-Vision (多模态+记忆) | Tree+Image+History | 39.17 | 23.93 | 4.78 | 14.27 | 39.17 | 10.61 | 1.13 | 20.13 |
| **人类基线** | 原始网页 | **99.02** | **97.91** | **93.77** | **98.43** | **99.02** | **95.34** | **88.12** | **96.25** |

### 实验2：按跳数分解的失败模式分析（GPT-4V）

| 任务总跳数 | 第1跳SR | 第2跳SR | 第3跳SR | 第4跳SR | 第5跳SR | 第6跳SR |
|-----------|--------|--------|--------|--------|--------|--------|
| 2跳任务 | 56.50 | 11.00 | - | - | - | - |
| 3跳任务 | 22.73 | 4.55 | 0.00 | - | - | - |
| 4跳任务 | 12.50 | 0.00 | 0.00 | 0.00 | - | - |
| 5跳任务 | 12.28 | 1.75 | 0.00 | 0.00 | 0.00 | - |
| 6跳任务 | 16.67 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

关键发现：即使是完全相同语义的第1跳，在总跳数更多的任务中成功率急剧下降（2跳任务56.5% → 6跳任务16.7%）。

### 实验3：记忆增强效果

记忆增强方法在GPT-4V和Gemini-Pro-Vision上均带来显著提升：
- GPT-4V + 记忆增强：总Hop SR从13.89%提升至约16%+，2-4跳Task SR从3.03%提升
- Gemini-Pro-Vision + 记忆增强：2-4跳Task SR从1.51%提升至10.61%（约7倍），5+跳Task SR从0提升至1.13%

## 关键发现

1. **早期失败效应**：Agent在多跳任务中倾向于在早期跳就失败，且总跳数越多，即使是第1跳的成功率也越低——这不是简单的单跳性能叠加
2. **搜索空间爆炸**：多跳任务提示中包含多个网站URL，Agent失败后倾向于切换到其他网站而非重试当前网站，导致过度探索
3. **终止条件失识**：Agent常无法识别单跳的终止条件，在已完成的跳中徘徊而非前进
4. **多模态优势**：多模态模型整体优于纯文本模型，视觉信息对准确执行网页任务至关重要
5. **推理模型悖论**：DeepSeek-R1在单跳任务上表现最优（47.68%），但多跳任务急剧退化（2-4跳仅3.84%），暴露了推理模型在长上下文保持上的弱点

## 亮点

- **真实动态环境**：唯一在持续变化的真实网站上运行的基准，确保高度真实性
- **多跳设计贴近实际**：最多10跳、平均2.85跳的任务设置远超现有基准（多数平均≈1跳），真正考察组合推理能力
- **逐跳评估协议**：突破任务级0/1评估的局限，提供过程性洞察（如发现"早期失败效应"）
- **重要发现**：揭示Agent在多跳场景下的系统性失败模式——不是能力不够而是规划和记忆机制不足
- **记忆增强方法通用性强**：模型无关的轻量方法，可直接应用于任意LMM

## 局限性

- **网站覆盖有限**：仅14个网站，部分网站因反爬机制限制只能使用离线/开源版本，削弱了真实性主张
- **评估基于URL匹配**：以访问正确URL序列作为成功标准，可能遗漏Agent实际完成任务但路径不同的情况
- **动态网站的可复现性**：真实网站内容持续变化，不同时间运行的评估结果难以严格对比
- **记忆方法评估不充分**：记忆增强方法的消融实验和详细分析相对有限，未充分探讨不同记忆长度的影响
- **缺乏开源模型的深度评估**：主要依赖API模型（GPT-4V、Gemini），对开源LMM的评估不够全面
- **任务分布偏向购物和搜索**：可能未充分覆盖其他重要网页交互场景

## 与相关工作的对比

- **WebArena / VisualWebArena**：使用本地部署的静态网站，最多2跳（平均≈1跳）；MMInA使用真实动态网站，最多10跳
- **WebVoyager**：同样使用真实网站但最多4跳（平均2.4跳），且未设计逐跳评估协议
- **Mind2Web**：131个网站但为静态快照、纯文本、单跳，且采用多选题而非开放式动作
- **GAIA / OpenAGI**：多模态通用Agent基准，但不专注于网页浏览环境
- **CogAgent / SeeAct**：Web Agent模型而非基准，在MMInA上表现较弱（CogAgent仅3.35% Task SR）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个同时强调多跳+多模态+动态真实网站的Web Agent基准
- 实验充分度: ⭐⭐⭐⭐ — 覆盖多类模型和人类基线，逐跳分析深入，但记忆方法实验偏少
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分，图表丰富
- 价值: ⭐⭐⭐⭐ — 揭示了多跳Web任务的核心难点，为Web Agent研究指明方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] PunchBench: Benchmarking MLLMs in Multimodal Punchline Comprehension](punchbench_mllm_punchline.md)
- [\[ACL 2025\] Attacking Vision-Language Computer Agents via Pop-ups](attacking_vl_agents_popups.md)
- [\[ACL 2025\] MMSciBench: Benchmarking Language Models on Chinese Multimodal Scientific Problems](mmscibench_benchmarking_language_models_on_chinese_multimodal_scientific_problem.md)
- [\[ACL 2025\] MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](cant_see_the_forest_for_the.md)
- [\[ACL 2025\] Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents](agent_rewardbench.md)

</div>

<!-- RELATED:END -->
