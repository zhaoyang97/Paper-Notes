---
title: >-
  [论文解读] Critic-V: VLM Critics Help Catch VLM Errors in Multimodal Reasoning
description: >-
  [CVPR 2025][多模态VLM][VLM推理纠错] 本文提出Critic-V框架，将VLM推理过程解耦为Reasoner（推理器）和Critic（评价器），通过DPO训练的Critic模型提供自然语言反馈迭代优化推理路径，在8个基准上的5个超越GPT-4V，数学推理任务提升尤为显著（MathVista +11.8%）。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "VLM推理纠错"
  - "Actor-Critic"
  - "DPO偏好优化"
  - "自然语言反馈"
  - "多模态推理"
---

# Critic-V: VLM Critics Help Catch VLM Errors in Multimodal Reasoning

**会议**: CVPR 2025  
**arXiv**: [2411.18203](https://arxiv.org/abs/2411.18203)  
**代码**: [GitHub](https://github.com/kyrieLei/Critic-V)  
**领域**: 多模态VLM  
**关键词**: VLM推理纠错, Actor-Critic, DPO偏好优化, 自然语言反馈, 多模态推理

## 一句话总结
本文提出Critic-V框架，将VLM推理过程解耦为Reasoner（推理器）和Critic（评价器），通过DPO训练的Critic模型提供自然语言反馈迭代优化推理路径，在8个基准上的5个超越GPT-4V，数学推理任务提升尤为显著（MathVista +11.8%）。

## 研究背景与动机
视觉语言模型在多模态推理任务上取得了显著进展，但仍然持续产生不准确或无关的回答。主要问题包括：图像理解的幻觉（hallucination）——模型"看到"了图像中不存在的内容；推理路径的粗糙——中间步骤的小错误会在链式推理中级联放大；以及对内部知识的过度依赖——忽视实际的视觉上下文。

现有的改进方案主要依赖模型自身的内在能力：Self-Refine让模型自我反思修正，Self-Consistency通过多次采样投票，DPO/RLHF在训练阶段对齐人类偏好。但这些方法有一个共同弱点：**没有引入外部独立的质量评估**。Self-Refine等方法完全依赖模型自身判断，而研究表明LLM往往能纠正已知错误但无法自主发现错误。

本文的核心观点是：需要一个独立训练的、专门擅长发现错误的Critic模型，在推理时为Reasoner提供具体的自然语言反馈（而非标量奖励），从而实现更精准的错误定位和推理路径优化。灵感来源于强化学习中Actor-Critic范式的分工思想。

## 方法详解

### 整体框架
Critic-V包含三个核心阶段：
1. Critic模型的离线训练（通过VEST生成训练数据 + DPO优化）
2. 推理时Reasoner-Critic的交互循环
3. 基于自然语言反馈的迭代优化

整个流程：给定图像和问题 → Reasoner生成初始推理回答 → Critic评估回答并指出问题 → Reasoner基于反馈修正回答 → 迭代直到Critic满意或达到最大轮次。

### 关键设计
1. **Reasoner（推理器）**:

    - 直接使用现有的VLM（如Qwen2-VL-7B、DeepSeek-VL-7B等）作为Reasoner，不修改其参数
    - 核心创新：用动态文本prompt替代传统RL中的参数化策略。推理策略通过prompt变化来实现，而非梯度更新
    - Critic的反馈 $\delta P^{reasoner}$ 被直接拼接到下一轮的prompt中，引导推理方向
    - 借鉴TextGrad框架进行prompt更新：将Critic的自然语言反馈视为"文本梯度"，指导推理路径的调整
    - 即"plug-and-play"——无需对Reasoner做任何训练或微调

2. **Critic模型训练**:

    - **VEST（Vision Error inSertion Technique）数据构造**：
        - 从VQA数据集收集问题-图像对，用GPT-4o在正确答案中插入1-5个虚假细节（伪造的错误答案）
        - 让GLM-4V-9B、GPT-4o mini、MiniCPM-V三个VLM分别对错误答案生成批评意见
        - 总计构建29,012个多模态QA对及对应的批评数据
    - **Rule-based Reward（RBR）评分**：
        - 结合Jaccard指数和GPT评分来评估批评质量：$Score(i) = Jaccard(i) + \alpha \times GPT(i)$
        - Jaccard指数：$J(G,C) = |G \cap C| / |G \cup C|$，其中G为插入的错误集合，C为批评检出的错误集合
        - Jaccard指数的引入关键在于防止"长批评倾向"——过长的批评可能包含更多假阳性（nitpicks）
        - 根据RBR得分构建偏好对（preferred vs disfavored critique）
    - **DPO训练**：
        - 基于Qwen2-VL-7B训练Critic模型
        - 使用标准DPO损失函数优化，鼓励模型给高质量批评分配更高概率
        - 偏好数据集 $\mathcal{D}_{cri} = \{(Q, I, C_w, C_l)\}$ ，其中 $C_w$ 为偏好批评，$C_l$ 为非偏好批评

3. **Reasoner-Critic交互框架**:

    - Reasoner生成初始回答 → Critic在问题、图像和回答的完整上下文下评估 → 以自然语言形式给出反馈（指出哪里错了、为什么错、应该如何修正）
    - Reasoner将反馈纳入新的prompt继续推理 → 循环直到Critic认为满意或达到最大迭代次数
    - 自然语言反馈比标量奖励更具信息量：能精确指出错误位置和类型，而不只是给一个"好/坏"的信号
    - 每次Critic评估仅消耗几十个额外token，计算开销可忽略

### 损失函数 / 训练策略
- Critic使用DPO损失训练：$\mathcal{L}_{DPO} = -\mathbb{E}[\log \sigma f(\pi_\theta; \pi_{ref})]$
- temperature设为0或接近0以保证输出稳定性
- 推理时采用两轮对话：第一轮Reasoner回答，第二轮Critic评价后Reasoner修正
- Critic的plug-and-play特性：同一个Critic可以搭配不同的Reasoner VLM

## 实验关键数据

### 主实验

| 模型 | RealWorldQA | MMBench | MathVista | MathVerse | MMT-Bench |
|------|-------------|---------|-----------|-----------|-----------|
| Qwen2-VL-7B基线 | 70.1 | 80.7 | 61.4 | 25.8 | 60.4 |
| **+Critic-V** | **74.9(+4.8)** | **82.8(+2.1)** | **73.2(+11.8)** | **32.9(+7.1)** | **62.0(+1.6)** |
| DeepSeek-VL-7B基线 | 58.1 | 73.5 | 35.3 | 18.4 | 46.5 |
| **+Critic-V** | **62.1(+4.0)** | **79.0(+5.5)** | **53.1(+17.8)** | **28.9(+10.5)** | **53.6(+7.1)** |
| LLaVA-v1.5-7B基线 | 50.7 | 68.4 | 37.8 | 26.0 | 36.0 |
| **+Critic-V** | **63.5(+12.8)** | **73.8(+5.4)** | **53.1(+15.3)** | **30.5(+4.5)** | **47.4(+11.4)** |
| GPT-4V | 61.4 | 74.3 | 49.9 | 54.4 | 55.5 |

### 与其他方法对比（LLaVA-V1.5-7B基础）

| 方法 | RealWorldQA | MMStar | MMBench | SEEDBench | MMT-Bench |
|------|-------------|--------|---------|-----------|-----------|
| +POVID | 51.8 | 33.6 | 71.6 | 65.4 | 33.4 |
| +SCL（最强竞争者） | 53.2 | 35.8 | 70.8 | 68.6 | 39.6 |
| **+Critic-V** | **63.5** | **38.4** | **73.8** | **70.1** | **49.7** |

### 消融实验

| 配置 | MathVista | MMT-Bench | MMBench | 说明 |
|------|-----------|-----------|---------|------|
| Qwen2-VL-7B基线 | 61.4 | 60.4 | 80.7 | - |
| +Self-Refine（无DPO） | 63.4 | 57.8 | 82.1 | 反而降低MMT-Bench |
| +Critic-V（含DPO） | **73.2** | **62.0** | **82.8** | DPO至关重要 |
| +仅特殊prompt（无Critic） | 61.8 | 59.0 | 81.0 | 排除prompt设计的影响 |

### 关键发现
- Critic-V在24组对比实验中的23组取得了提升，普适性极强
- 数学推理任务提升最为显著（MathVista +11.8%~+17.8%），"逻辑密集型任务最受益于外部纠错"
- Qwen2-VL-7B+Critic-V在8个基准中5个超越GPT-4V
- Self-Refine（自我纠错）效果远不如外部Critic——说明模型确实难以自我发现错误
- DPO训练是Critic-V有效性的关键，未经DPO训练的模型做Critic反而可能有害
- 每次Critic评估仅消耗额外几十个token，计算开销极低

## 亮点与洞察
1. 将RL中的Actor-Critic范式巧妙迁移到VLM推理场景，用自然语言反馈替代标量奖励是关键创新
2. VEST数据构造方法非常实用：通过插入已知错误来自动生成Critic训练数据，避免了人工标注
3. Jaccard指数在RBR中的使用解决了"批评越长分越高"的偏差问题
4. Plug-and-play设计意味着一个Critic可以服务多种不同的VLM，实用价值高
5. 验证了一个重要假设：VLM可以在被指出错误后有效修正（Tyen et al.的发现在VLM领域的延伸）
6. LLaVA-v1.5-7B在RealWorldQA上从50.7提升到63.5（+12.8），说明即使是弱模型也能从强Critic中大幅受益

## 局限与展望
- Critic模型本身基于Qwen2-VL-7B训练，其能力上限受限于基座模型——对于超出其能力范围的错误可能无法识别
- VEST中的错误插入依赖GPT-4o，存在成本和可复现性的问题
- 迭代次数的选择缺乏自适应机制——目前固定为2轮，不同难度题目可能需要不同轮次
- MMStar基准上Qwen2-VL-7B+Critic-V出现了-4.5%的下降，说明Critic的反馈有时可能误导而非帮助
- 仅评估了7B量级的模型作为Reasoner，未验证在更大模型（如70B+）上是否仍然有效
- Critic和Reasoner使用不同的VLM时的组合效果未被系统探索

## 相关工作与启发
- 与CriticGPT的关系：CriticGPT专注于文本LLM的代码审查，Critic-V将该思路推广到多模态视觉推理
- 与Self-Refine/Self-Consistency的对比：外部Critic强于自我纠正，说明"自检能力"是独立于推理能力的另一种能力
- DPO在Critic-V中的应用不同于传统用法：传统DPO对齐模型的生成偏好，这里用来训练Critic的评估偏好
- 对具身智能和自动驾驶有直接应用价值——这些场景对推理可靠性有极高要求
- 启发：能否将Critic训练为更细粒度的"bug检测器"，不仅指出推理步骤错误，还定位到具体的视觉区域？

## 评分
- 新颖性: ⭐⭐⭐⭐ Actor-Critic到VLM的迁移有新意，VEST+RBR的数据构造流程设计精良
- 实验充分度: ⭐⭐⭐⭐ 8个基准、3个Reasoner VLM、充分的消融；但缺少大模型实验
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，公式推导完整但部分地方冗余
- 价值: ⭐⭐⭐⭐⭐ 实用性极强，即插即用的设计、显著的性能提升、低计算开销

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] LLaVA-Critic: Learning to Evaluate Multimodal Models](llava-critic_learning_to_evaluate_multimodal_models.md)
- [\[CVPR 2025\] DocVLM: Make Your VLM an Efficient Reader](docvlm_make_your_vlm_an_efficient_reader.md)
- [\[CVPR 2025\] Self-Evolving Visual Concept Library using Vision-Language Critics](self-evolving_visual_concept_library_using_vision-language_critics.md)
- [\[CVPR 2025\] FLAIR: VLM with Fine-grained Language-informed Image Representations](flair_vlm_with_fine-grained_language-informed_image_representations.md)
- [\[CVPR 2025\] VisionArena: 230K Real World User-VLM Conversations with Preference Labels](visionarena_230k_real_world_user-vlm_conversations_with_preference_labels.md)

</div>

<!-- RELATED:END -->
