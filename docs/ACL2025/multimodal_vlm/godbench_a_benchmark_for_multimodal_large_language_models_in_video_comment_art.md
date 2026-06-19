---
title: >-
  [论文解读] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art
description: >-
  [ACL 2025][多模态VLM][视频弹幕艺术] GODBench 构建了首个系统性评估多模态大模型（MLLM）视频弹幕/评论创作能力的基准，定义了5个创意维度和25个子类别，并提出受物理波纹传播启发的 Ripple of Thought（RoT）多步推理框架来增强模型的创意生成能力。 领域现状：视频弹幕/评论艺术（Co…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "视频弹幕艺术"
  - "多模态大模型"
  - "创意评测基准"
  - "思维涟漪"
  - "视频理解"
---

# GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art

**会议**: ACL 2025  
**arXiv**: [2505.11436](https://arxiv.org/abs/2505.11436)  
**代码**: [https://github.com/stan-lei/GODBench-ACL2025](https://github.com/stan-lei/GODBench-ACL2025)  
**领域**: 多模态VLM / 创意生成  
**关键词**: 视频弹幕艺术, 多模态大模型, 创意评测基准, 思维涟漪, 视频理解

## 一句话总结

GODBench 构建了首个系统性评估多模态大模型（MLLM）视频弹幕/评论创作能力的基准，定义了5个创意维度和25个子类别，并提出受物理波纹传播启发的 Ripple of Thought（RoT）多步推理框架来增强模型的创意生成能力。

## 研究背景与动机

**领域现状**：视频弹幕/评论艺术（Comment Art）是互联网文化中极为活跃的一种创意表达形式。高质量的视频评论需要对视频内容深度理解，并具备幽默、讽刺、共情等高级创意能力。随着多模态大语言模型（如 GPT-4o、Gemini）的快速发展，这些模型在 STEM 任务上展现出强推理能力，但在创意表达领域的能力尚未被系统评估。

**现有痛点**：现有的创意评测基准存在两个核心问题：(1) 模态受限——大多只涉及纯文本或图文，缺乏视频+文本的多模态创意基准；(2) 类别不足——已有基准通常只覆盖少数创意类型（如笑话生成），无法系统评估"理解视频上下文并做出有创意回应"这种综合能力。

**核心矛盾**：创意生成需要深层的文化理解、情境感知和"意外性"（出人意料但合理的联想），这与 MLLM 擅长的模式化推理有本质区别。现有的 Chain-of-Thought 方法虽然能提升逻辑推理，但对创意发散帮助有限——过于线性的推理链条反而可能抑制创意跳跃。

**本文目标**：(1) 构建一个覆盖多维度创意能力的视频评论基准；(2) 设计一种适合创意生成的推理框架，替代线性的 CoT。

**切入角度**：作者从物理学中波纹（ripple）的传播模式获得灵感——一颗石子投入水中，涟漪向外逐层扩散，每一层都可能激发新的扩展。类比到创意思维，一个初始触发点可以引发多层次、多方向的联想扩散。

**核心 idea**：提出 Ripple of Thought（RoT）框架，将创意思维抽象为"触发→扩散→共振→叠加→收束"五个阶段，引导 MLLM 像涟漪一样逐层拓展联想，最终生成有创意的视频评论。

## 方法详解

### 整体框架

GODBench 包含两大贡献：(1) 基准数据集——收集视频+评论对，定义5个创意维度（Resonant Thinking, Divergent Association, Witty Twist, Imaginary Completion, Emotional Resonance），每个维度下有5个子类别共25类，覆盖判别式（选择、排名、分类、解释）和生成式（创意写作）两大任务类型；(2) RoT 推理框架——替代标准 CoT，通过五个传播阶段引导 MLLM 进行创意推理。

### 关键设计

1. **五维度创意分类体系**:

    - 功能：系统化定义和覆盖视频评论创作中的不同创意能力
    - 核心思路：通过对大量视频评论的分析和专家标注，将评论创意分为五个维度：
        - **Resonant Thinking (RT, 共鸣思维)**：能引起观众深层情感共鸣的评论
        - **Divergent Association (DA, 发散联想)**：将视频内容与看似不相关的概念巧妙关联
        - **Witty Twist (WT, 机智反转)**：通过意外转折制造幽默效果
        - **Imaginary Completion (IV, 想象补全)**：对视频中未展示的部分进行创造性补充
        - **Emotional Resonance (ER, 情感共振)**：精准捕捉并表达视频的情感基调
    - 设计动机：单一指标无法全面衡量创意能力。五维度分类确保了评估的系统性和全面性，也帮助研究者定位模型在哪类创意上更强/更弱。

2. **多任务评测设计**:

    - 功能：从判别和生成两个角度全面评估 MLLM 的创意能力
    - 核心思路：设计五种任务类型——选择题（SEL，从候选评论中选最佳）、排名题（RNK，对多条评论按创意度排序）、分类题（CLS，判断评论属于哪个创意维度）、解释题（EXP，解释为什么某条评论有创意）、创作题（CRE，根据视频直接创作评论）。前四种为判别式任务，使用 Exact Match Accuracy 评估；最后一种为生成式任务，使用 GPT-4o 评分和人工投票评估。
    - 设计动机：创意能力是多层次的——"识别好评论"和"写出好评论"需要不同层次的能力。多任务设计能更精确地诊断模型在创意理解 vs 创意生成上的差距。

3. **Ripple of Thought (RoT) 推理框架**:

    - 功能：替代标准 CoT，增强 MLLM 的创意推理能力
    - 核心思路：受物理波纹传播启发，RoT 将创意推理分为五个阶段：
        - **Trigger（触发）**：识别视频中的核心信息和情感触发点
        - **Propagation（扩散）**：从触发点向外发散联想，生成多条联想链
        - **Resonance（共振）**：在多条联想链中寻找互相增强的共振点
        - **Superposition（叠加）**：将多个共振点叠加组合，形成多层次含义
        - **Convergence（收束）**：从叠加的丰富联想中收束为一条精炼的创意评论
    - 设计动机：标准 CoT 的线性推理适合逻辑问题但不利于创意发散。RoT 的"先散后收"策略模拟了人类创意思维的实际过程——先大范围联想再聚焦提炼。五个阶段的显式分离也使得模型在每个阶段专注于不同的认知操作。

### 损失函数 / 训练策略

RoT 框架主要通过精心设计的 prompting 策略实现，不需要额外训练。对于部分实验，作者使用 LoRA 在 GODBench 训练集上微调了几个开源 MLLM，以评估数据适应效果。

## 实验关键数据

### 主实验

在判别式任务上使用 Exact Match Accuracy (EMA)，对比多个 MLLM 的表现：

| 模型 | Size | SEL | RNK | CLS | EXP | 平均 |
|------|------|-----|-----|-----|-----|------|
| GPT-4o | - | 最高档 | 最高档 | 最高档 | 最高档 | 最佳 |
| Gemini 1.5 Pro | - | 高 | 高 | 中高 | 高 | 第二档 |
| InternVL2 | 26B | 中 | 中 | 中 | 中 | 中等 |
| VideoLLaMA2 | 7B | 低 | 低 | 低 | 低 | 较差 |
| LLaVA-Video | 7B | 低 | 低 | 低 | 低 | 较差 |

生成式任务上（创作评论），RoT 对比标准 CoT 和直接生成：

| 方法 | GPT-4o评分 | 人工投票偏好 | 说明 |
|------|-----------|-------------|------|
| Direct (直接生成) | 基线 | 基线 | 无推理引导 |
| Standard CoT | 小幅提升 | 微弱偏好 | 线性推理对创意帮助有限 |
| RoT (本文) | 显著提升 | 明显偏好 | 先散后收策略有效促进创意 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| RoT (完整五阶段) | 最佳 | 五个阶段都有贡献 |
| w/o Resonance | 下降 | 共振阶段帮助筛选高质量联想 |
| w/o Propagation | 显著下降 | 发散是创意生成的核心 |
| w/o Convergence | 质量下降 | 缺少收束导致评论松散 |
| 减少 Propagation 分支数 | 下降 | 更多联想链→更高创意多样性 |

### 关键发现

- **MLLM 在创意理解/生成上明显不如 STEM 推理**：即使是 GPT-4o 在某些创意维度上的准确率也不高，特别是在需要深层文化理解的 Witty Twist 和 Divergent Association 上。
- **闭源模型远超开源模型**：GPT-4o 和 Gemini 在创意任务上的表现远好于 7B-26B 的开源模型，说明创意能力与模型规模/训练数据质量高度相关。
- **RoT 对创意生成有实质帮助**：相比标准 CoT，RoT 生成的评论在人工评估中被明显更多地偏好。关键在于 Propagation 阶段提供的多方向联想。
- **LoRA 微调在判别任务上有提升但在生成任务上效果有限**：说明创意生成能力难以通过小规模微调获得，可能需要更本质的模型能力提升。
- **模型在 Imaginary Completion 上表现相对好，在 Witty Twist 上最差**：前者更依赖叙事能力（MLLM 擅长），后者需要真正的幽默感和意外性。

## 亮点与洞察

- **五维度创意分类体系**设计得很有体系性，既有理论支撑又有实际可操作性。这个分类框架可以推广到其他创意评测场景（如广告文案、社交媒体内容创作）。
- **RoT 的"物理波纹"类比**很直觉——把创意思维比作涟漪扩散，先发散再收束。这个框架可以迁移到其他需要发散-收束思维的任务，如头脑风暴、故事创作、产品设计。
- **69 页论文 + 66 张图**，工作量巨大，数据集构建和人工标注非常扎实。

## 局限与展望

- **文化依赖性强**：视频弹幕文化主要来自特定网络社区（如 B站、YouTube），评估标准和创意规范可能不具有文化普适性。
- **评估仍依赖人工或 GPT-4o**：创意质量的评估本身就是主观的，自动化指标的信效度有待验证。
- **RoT 增加了推理步骤和 token 消耗**：五阶段推理需要较长的 prompt 和多轮生成，实际应用的效率需要考虑。
- **数据集以中文视频为主（推测）**：可能限制了非中文模型的公平评估。
- **改进方向**：可以将 RoT 的阶段设计为可训练的模块而非纯 prompt 方案，或探索利用人类创意评论做 RLHF 来直接提升模型的创意能力。

## 相关工作与启发

- **vs CreativeBench / HumorBench**: 之前的创意基准主要是纯文本或单图，GODBench 首次引入视频模态，且创意维度更全面。
- **vs Chain-of-Thought (CoT)**: 标准 CoT 是线性推理链，适合逻辑问题但抑制创意发散；RoT 的"先散后收"明确设计为促进创意而非逻辑。
- **vs Tree-of-Thought (ToT)**: ToT 也有分支探索，但主要用于搜索最优解而非激发创意多样性；RoT 的 Resonance 和 Superposition 阶段是 ToT 不具备的。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个视频弹幕创意评测基准，RoT框架有新意，五维度分类体系有价值
- 实验充分度: ⭐⭐⭐⭐ 多模型对比+判别和生成双评估+消融+人工评估，非常全面
- 写作质量: ⭐⭐⭐⭐ 论文69页极其详尽，分类和案例展示丰富，但篇幅过长可能影响可读性
- 价值: ⭐⭐⭐⭐ 填补了视频创意理解和生成评测的空白，RoT框架对创意任务的prompt设计有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](mcts_video_captioning_eval.md)
- [\[ACL 2025\] COSMMIC: Comment-Sensitive Multimodal Multilingual Indian Corpus](cosmmic_commentsensitive_multimodal_multilingual_indian_corpus.md)
- [\[ACL 2025\] Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues](speaking_beyond_language.md)
- [\[NeurIPS 2025\] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs](../../NeurIPS2025/multimodal_vlm/video-safetybench_a_benchmark_for_safety_evaluation_of_video_lvlms.md)
- [\[ACL 2025\] COLING-UniA at SciVQA 2025: Few-Shot Example Retrieval and Confidence-Informed Ensembling for Multimodal Large Language Models](coling-unia_at_scivqa_2025_few-shot_example_retrieval_and_confidence-informed_en.md)

</div>

<!-- RELATED:END -->
