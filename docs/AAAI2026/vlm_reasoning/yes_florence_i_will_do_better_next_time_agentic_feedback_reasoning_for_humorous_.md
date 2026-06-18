---
title: >-
  [论文解读] Yes FLoReNce, I Will Do Better Next Time! Agentic Feedback Reasoning for Humorous Meme Detection
description: >-
  [AAAI 2026 Oral][多模态VLM][幽默 meme 检测] 提出 FLoReNce 框架，将幽默 meme 理解建模为闭环控制系统，通过 Judge 反馈+PID 控制器+非参数知识库的闭环学习，在推理时通过检索相似经验调制 prompt，使冻结的 VLM 实现自适应推理，无需微调即可显著提升预测和解释质量。
tags:
  - "AAAI 2026 Oral"
  - "多模态VLM"
  - "幽默 meme 检测"
  - "反馈闭环推理"
  - "非参数知识库"
  - "PID 控制器"
  - "视觉语言模型"
---

# Yes FLoReNce, I Will Do Better Next Time! Agentic Feedback Reasoning for Humorous Meme Detection

**会议**: AAAI 2026 Oral  
**arXiv**: [2601.07232](https://arxiv.org/abs/2601.07232)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 幽默 meme 检测, 反馈闭环推理, 非参数知识库, PID 控制器, 视觉语言模型

## 一句话总结

提出 FLoReNce 框架，将幽默 meme 理解建模为闭环控制系统，通过 Judge 反馈+PID 控制器+非参数知识库的闭环学习，在推理时通过检索相似经验调制 prompt，使冻结的 VLM 实现自适应推理，无需微调即可显著提升预测和解释质量。

## 研究背景与动机

幽默 meme 融合视觉和文本线索来传达讽刺、社会评论等含义，对 AI 系统构成独特挑战——必须理解意图而非表面关联。现有方法存在两大局限：

**静态分类方法**（如 MemeCLIP、CLIP+分类器）虽然能融合多模态特征，但只学习像素与词语的相关性，无法捕捉幽默的深层不协调性（incongruity）

**基于推理的方法**（如 Chain-of-Thought、多智能体辩论框架）虽然改善了可解释性，但本质上仍是开环的（open-loop）：一旦模型产生错误或浅显的推理，没有任何纠正、反思或基于反馈的自适应机制

人类的幽默理解本质是动态的——通过批评、社会反馈和接触新的文化语境来不断精炼解读。这种反馈过程就像控制系统调节输出以最小化误差。缺乏这种闭环机制，AI 推理往往在过预测和欠预测幽默之间振荡，缺乏自我纠正能力。

**核心动机**：将幽默推理建模为闭环状态空间系统 → 让 VLM 能从自身的解释历史中学习并逐步稳定对幽默的感知。

## 方法详解

### 整体框架

FLoReNce（Feedback-Loop Reasoner with Non-parametric Experience）由两个阶段组成：

- **闭环学习阶段（训练）**：VLM 推理 → Judge 评判 → PID 控制器生成控制信号 → 存入知识库
- **开环推理阶段（测试）**：从知识库检索相似经验 → 组装控制向量 → 调制 prompt → VLM 自适应推理

### 关键设计

1. **视觉-语言推理智能体（Reasoning Agent）**

   使用冻结的 Qwen2.5-VL-32B-Instruct 作为推理 Agent。给定 meme 的图像 $x^{img}$、OCR 文本 $x^{text}$ 和引导 prompt $p$，输出：
    - 幽默概率 $\hat{y} \in [0,1]$
    - 文本推理依据 $r$
    - 隐藏层 embedding $emb \in \mathbb{R}^d$（最后隐藏层的均值池化）

   Prompt 由 prompt-mapper $\Psi$ 从控制向量 $c$ 映射而来：$p = \Psi(c)$，将连续反馈转化为可解释的提示指令（如"检查讽刺"、"验证铺垫→反转"）。

2. **Judge Agent 与 PID 控制器**

   Judge $\mathcal{J}_\phi$ 有权访问真实标签，输出三个信号：
    - 标量误差 $e_t = y - \hat{y}_t$
    - 文本批评 $fb\_text_t$
    - 低维反馈向量 $f_t \in \mathbb{R}^3$（沿讽刺/叙事结构/布局线索三个可解释轴）

   PID 控制器将误差历史转化为稳定动作：

    $u_t = K_P e_t + K_I \sum_{\tau=1}^{t} e_\tau + K_D (e_t - e_{t-1})$

   最终控制向量 $c_t = [u_t, f_t^\top, k_t^\top]^\top \in \mathbb{R}^7$，其中 $k_t$ 是 KB 紧凑信号。

   **设计动机**：PID 控制器提供了数值上的稳定性——比例项响应当前误差，积分项消除稳态偏差，微分项抑制振荡。这使得推理行为像受控系统一样逐步收敛。

3. **反馈知识库（Feedback-informed KB）**

   与传统 RAG 不同，KB 存储的不是原始训练样本，而是完整的推理经验：
   
    $\mathcal{K} \leftarrow \mathcal{K} \cup \{(id, emb_t, r_t, fb\_text_t)\}$

   即同时存储 embedding、推理文本和 Judge 批评。这使后续检索是"经验感知"的——系统不仅记住"看过什么"，还记住"如何被纠正的"。

   推理时，计算查询 embedding 的余弦相似度检索 top-K 邻居，汇总为紧凑 KB 信号 $k \in \mathbb{R}^3$。

### 损失函数 / 训练策略

- **无参数更新**：整个过程中没有模型权重被更新。闭环学习阶段仅构建 KB，所有自适应通过 prompt 调制实现
- **PID 超参数**：$(K_P, K_I, K_D) = (1.0, 0.5, 0.1)$
- KB 格式为 JSONL，使用 CPU 张量进行检索
- 生成最大 128 tokens（推理文本）
- 硬件：NVIDIA L40S (48GB)

## 实验关键数据

### 主实验

数据集 PrideMM：5,063 张与 LGBTQ+ 运动相关的文本嵌入图像，按 85/5/10 划分 train/val/test。

| 模型 | Backbone | Accuracy | Macro-F1 | MCC | RQ(%) |
|------|----------|----------|----------|-----|-------|
| Visual Only | ResNet50+MLP | 66.08 | 61.67 | 0.33 | - |
| Text Only | T5+MLP | 67.85 | 66.10 | 0.36 | - |
| MemeCLIP | CLIP | 78.30 | 76.99 | 0.57 | - |
| MOMENTA | CLIP | 73.57 | 69.92 | 0.47 | - |
| PromptHate | RoBERTa | 73.77 | 73.46 | 0.49 | - |
| LoReHM | LLaVA-34B | 70.09 | 64.07 | 0.39 | 64.8 |
| COLA | GPT-3.5-Turbo | 53.25 | 59.34 | 0.07 | 58.5 |
| MiND | Qwen2.5-VL-32B | 54.45 | 50.43 | 0.05 | 52.6 |
| **FLoReNce (K=1)** | Qwen2.5-VL-32B | 73.40 | **77.08** | 0.48 | 74.0 |
| **FLoReNce (K=3)** | Qwen2.5-VL-32B | 73.73 | **77.36** | 0.48 | 74.3 |
| **FLoReNce (K=5)** | Qwen2.5-VL-32B | 73.80 | **77.33** | 0.48 | 74.4 |

关键发现：FLoReNce 的 Macro-F1 增益大于 Accuracy 增益 → 反馈知识库对较难类别尤为有效，提升了类别平衡性能。

### 消融实验

| 变体 | Accuracy | Macro-F1 | MCC |
|------|----------|----------|-----|
| Base VLM（无 KB 无控制） | 64.20 | 58.10 | 0.22 |
| + KB only（无控制） | 68.30 | 63.90 | 0.35 |
| + Control only（无 KB） | 72.00 | 69.40 | 0.44 |
| − $f_t$（PID+KB 但去掉反馈向量） | 73.00 | 70.20 | 0.46 |
| − PID（仅 KB 信号） | 72.60 | 70.00 | 0.45 |
| **Full FLoReNce** | **73.73** | **77.36** | **0.48** |

### 关键发现

- KB alone → +4.10% Acc / +5.80% F1，说明检索反馈经验本身就有价值
- Control alone → 比 KB only 略低但显著优于 base，控制在有意义的记忆信号上更有效
- 三者（PID+KB+$f_t$）缺一不可：Judge 批评携带 embedding 无法恢复的信息
- K=1 到 K=10 性能非常稳定 → KB 一旦填充，即使最少检索也足以产生一致的反馈对齐推理
- 使用与 MiND 相同的 Qwen2.5-VL-32B backbone，FLoReNce 的 F1 从 50.43% → 77.36%（+27%），证明闭环反馈机制的巨大价值

## 亮点与洞察

1. **控制论视角的创新**：首次将幽默理解形式化为状态空间闭环控制问题，PID 控制器提供数值稳定性
2. **经验感知的非参数记忆**：KB 存储"推理+纠正"的完整经验而非原始样本，使检索天然带有自我纠正信息
3. **零微调自适应**：冻结 VLM 权重不变，所有自适应通过 prompt 调制实现 → 资源高效且即插即用
4. **实用性强**：即使 K=1 也能达到 F1=77.08%，说明方法对 KB 大小不敏感

## 局限与展望

- 仅在 PrideMM 一个数据集验证，泛化性待考察
- Case Study 暴露了区分"批评权力的讽刺"vs."针对边缘群体的嘲笑"的界限问题
- PID 参数为手动设定（虽然实验中表现稳定），自适应 PID 可能更优
- KB 随训练规模线性增长，大规模场景下的检索效率需优化
- Accuracy 略低于 MemeCLIP（73.73 vs 78.30），但 F1 更高；两者是否可结合？

## 相关工作与启发

- 与 RAG 的区别：RAG 存储原始文档，FLoReNce 存储推理经验+Judge 反馈
- 与 Self-Refine 的区别：Self-Refine 是开环的自我完善，FLoReNce 是闭环的基于 Judge 反馈的有状态进化
- PID 控制在 NLP 中的应用尚少，本文开创了"控制论 + VLM prompting"的新范式
- 可扩展到 hate speech、sarcasm、misinformation 等其他主观性强的多模态任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（控制论视角+反馈知识库的结合非常新颖）
- 实验充分度: ⭐⭐⭐（仅一个数据集，消融完整但缺乏跨域验证）
- 写作质量: ⭐⭐⭐⭐（formulation 清晰，控制论角度阐述到位）
- 价值: ⭐⭐⭐⭐（为主观任务的自适应推理提供了通用思路）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] CAMU: Context Augmentation for Meme Understanding](trace_textual_relevance_augmentation_and_contextual_encoding_for_multimodal_hate.md)
- [\[ICML 2026\] Learning GUI Grounding with Spatial Reasoning from Visual Feedback](../../ICML2026/multimodal_vlm/learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)
- [\[CVPR 2026\] ANTS: Adaptive Negative Textual Space Shaping for OOD Detection via Test-Time MLLM Understanding and Reasoning](../../CVPR2026/multimodal_vlm/ants_adaptive_negative_textual_space_shaping_for_ood_detection_via_test-time_mll.md)
- [\[ACL 2026\] All Changes May Have Invariant Principles: Improving Ever-Shifting Harmful Meme Detection via Design Concept Reproduction](../../ACL2026/multimodal_vlm/all_changes_may_have_invariant_principles_improving_ever-shifting_harmful_meme_d.md)
- [\[CVPR 2026\] Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models](../../CVPR2026/multimodal_vlm/activation_matters_test-time_activated_negative_labels_for_ood_detection_with_vi.md)

</div>

<!-- RELATED:END -->
