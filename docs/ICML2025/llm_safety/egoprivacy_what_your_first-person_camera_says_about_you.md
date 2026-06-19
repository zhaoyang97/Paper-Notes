---
title: >-
  [论文解读] EgoPrivacy: What Your First-Person Camera Says About You?
description: >-
  [ICML2025][LLM安全][egocentric vision] 提出 EgoPrivacy——首个大规模第一人称视频隐私基准，定义三类隐私（人口统计/个体/情境）七大任务，并设计检索增强攻击 (RAA) 将 ego-to-exo 检索与分类联合，证明基础模型零样本即可以 70–80% 准确率推断佩戴者性别、种族等敏感属性。
tags:
  - "ICML2025"
  - "LLM安全"
  - "egocentric vision"
  - "privacy benchmark"
  - "demographic attack"
  - "retrieval-augmented attack"
  - "对比学习"
---

# EgoPrivacy: What Your First-Person Camera Says About You?

**会议**: ICML2025  
**arXiv**: [2506.12258](https://arxiv.org/abs/2506.12258)  
**代码**: [GitHub](https://github.com/williamium3000/ego-privacy)  
**领域**: 隐私 / AI安全  
**关键词**: egocentric vision, privacy benchmark, demographic attack, retrieval-augmented attack, contrastive learning

## 一句话总结

提出 EgoPrivacy——首个大规模第一人称视频隐私基准，定义三类隐私（人口统计/个体/情境）七大任务，并设计检索增强攻击 (RAA) 将 ego-to-exo 检索与分类联合，证明基础模型零样本即可以 70–80% 准确率推断佩戴者性别、种族等敏感属性。

## 研究背景与动机

- **问题来源**：可穿戴相机（AR 眼镜、GoPro）日益普及，第一人称视频被持续采集用于活动识别、行为分析、生活日志等任务。已有隐私研究主要关注画面中出现的第三方面孔，但**对相机佩戴者自身的隐私威胁**几乎未被系统研究。
- **核心问题**：仅从第一人称视频中能推测出佩戴者多少隐私？佩戴者的性别、种族、年龄、身份、所处场景与时间等信息是否可被还原？
- **现有差距**：已有第一人称隐私数据集（FPSI、EVPR、IITMD）规模极小（6–32 人）、仅覆盖身份识别单一维度、无人口统计标注、无 OOD 测试集。
- **动机**：系统定义第一人称视频隐私的攻击面并建立全面基准，量化不同能力攻击者的信息泄露程度，为后续隐私防御奠定基础。

## 方法详解

### 1. 隐私定义与任务体系

将佩戴者隐私分为三大类、七项任务：

| 隐私类别 | 任务 | 形式 | 评估指标 |
|---|---|---|---|
| 人口统计隐私 | 性别 / 种族 / 年龄分类 | 分类 | Accuracy |
| 个体隐私 | ego-to-ego / ego-to-exo 身份检索 | 检索 | HR@k |
| 情境隐私 | 场景检索 / 时刻检索 | 检索 | HR@k |

**人口统计隐私**建模为分类问题：

$$\text{Acc}(\mathcal{D}; f) = \frac{1}{|\mathcal{D}|} \sum_{(\mathbf{x}, a) \in \mathcal{D}} \mathbb{1}[f(\mathbf{x}) = a]$$

**个体 / 情境隐私**建模为检索问题，以 Hit Rate@k 度量风险：

$$\text{HR@}k(\mathcal{D}; g) = \frac{1}{|\mathcal{D}|} \sum_{(\mathbf{x}, I) \in \mathcal{D}} \mathbb{1}[g^k(\mathbf{x}) \cap \mathcal{T}_I \neq \emptyset]$$

### 2. 威胁模型（Attack Capability）

定义四级递增攻击能力：

- **Capability ⓪ (Zero-shot)**：攻击者无训练数据，直接使用基础模型零样本推断。
- **Capability ① (Fine-tuned)**：攻击者可用带标注训练集微调模型。
- **Capability ② (Retrieval-Augmented)**：攻击者拥有 ego-exo 配对训练集 + 外部第三人称视频池。
- **Capability ③ (Identity-level)**：攻击者可判断两段 ego 视频是否属于同一身份。

### 3. Ego-Exo 联合嵌入

采用 **Supervised Contrastive Learning (SupCon)** 学习 ego-exo 联合嵌入空间：

$$L(g, g') = -\sum_{i=1}^{N} \frac{1}{|P(i)|} \sum_{k \in P(i)} \log \frac{\exp(\langle \mathbf{z}_i^E, \mathbf{z}_k^X \rangle / \tau)}{\sum_{j \in N(i)} \exp(\langle \mathbf{z}_i^E, \mathbf{z}_j^X \rangle / \tau)}$$

其中 $P(i)$ 为正对集合（按隐私类型定义），$N(i)$ 为负对集合，$\tau$ 是温度系数。通过改变 $P(i)$ 的定义统一个体隐私（同一佩戴者所有 exo 视频为正对）与情境隐私（同步录制的 exo 片段为正对）。

### 4. 检索增强攻击 (RAA)

核心思路："先检索，再预测"——利用 ego-to-exo 检索弥补第一人称视频对面部/身体遮挡的缺陷。

1. 给定 ego 查询 $\mathbf{x}^E$，用 ego-exo 检索器 $g$ 从外部 exo 池 $\mathcal{D}^X$ 中取 Top-M 最相似片段 $\{\mathbf{x}_{1:M}^X\}$。
2. 分别用 ego 分类器 $f$ 和 exo 分类器 $f'$ 对各输入做隐私属性预测。
3. 通过投票聚合得到最终结果：

$$f^{\text{RAA}}(\mathbf{x}^E, \{\mathbf{x}_{1:M}^X\}) = \mathcal{A}\big(f(\mathbf{x}^E),\; f'(\mathbf{x}_1^X),\;\dots,\; f'(\mathbf{x}_M^X)\big)$$

聚合函数 $\mathcal{A}$ 可为硬投票（majority voting）或软投票（加权 pooling）。

### 5. EgoPrivacy 基准构建

- 基于 **Ego-Exo4D** (5,625 clips, 839 人, 131 场景) + **Charades-Ego** (4,000 clips, 112 人) 构建。
- 人口统计标注通过 Amazon Mechanical Turk 对 exo 视频中可见佩戴者进行标注，标签集：Gender {Female, Male}、Race {Asian, Black, White}、Age {Young, Middle-aged, Senior}。
- 支持 ID (Ego-Exo4D train/test) 和 OOD (train=Ego-Exo4D, test=Charades-Ego) 两种评估。

## 实验关键数据

### 人口统计隐私攻击（OOD, Charades-Ego）

| 方法 | Capability | Gender | Race | Age |
|---|---|---|---|---|
| Random Chance | — | 50.00 | 33.33 | 33.33 |
| Prior (多数类) | — | 60.74 | 54.17 | 79.48 |
| CLIP H/14 zero-shot (ego) | ⓪ | 57.89 | 45.21 | 72.02 |
| CLIP H/14 fine-tuned (ego) | ① | 68.87 | 70.92 | 79.73 |
| CLIP H/14 + RAA | ①+② | 76.98 (+8.11) | 71.92 (+1.00) | 79.73 |
| CLIP H/14 zero-shot + RAA | ⓪+② | 67.35 (+9.46) | 60.98 (+15.77) | 76.23 (+4.21) |

**关键发现**：
- 零样本基础模型对性别/种族/场景的推断准确率达 **70–80%**，远超随机基线。
- RAA 在零样本设置下对种族攻击准确率提升多达 **+15.77%**，性别 **+9.46%**。
- 微调后 ego 攻击已接近 exo 攻击水平，说明 ego 视频的"天然遮挡"保护有限。

### 个体与情境隐私

- Ego-to-ego 身份检索：微调 CLIP 在 ID 评估中 HR@1 显著高于零样本，验证手势/环境线索足以暴露身份。
- Ego-to-exo 身份检索：SupCon 训练后 HR@1 大幅提升，表明 ego-exo 跨视角身份关联可学习。
- 场景 / 时刻检索：基础模型零样本即具有较强场景匹配能力，微调进一步提升。

## 亮点与洞察

1. **首个系统性 ego 隐私基准**：将佩戴者隐私细化为三类七任务，比已有数据集（6–32 人）扩大到 839 人 / 131 场景，填补重要空白。
2. **RAA 攻击策略新颖实用**：模拟真实场景（监控摄像头与 ego 设备同时拍摄同一人），通过 ego-to-exo 检索桥接两个视角，无需直接人脸匹配即可显著提升攻击成功率。
3. **零样本即高威胁**：开源基础模型无任何额外数据即可恢复敏感人口统计属性，对隐私法规和设备设计敲响警钟。
4. **统一公式化**：用 SupCon 损失的正对定义变化统一个体/情境两大类检索任务，优雅简洁。

## 局限与展望

1. **标签体系粗糙**：性别仅 Male/Female，种族仅 Asian/Black/White（由标注者主观判断），年龄仅三档，遗漏大量多样性。
2. **数据集偏差**：Ego-Exo4D 以实验室/特定活动场景为主，Charades-Ego 限于家庭室内，缺少户外/城市/驾驶等高频 ego 场景。
3. **仅评估攻击**：未提出相应的防御方法或隐私保护策略（如差分隐私表征、对抗扰动），对应用落地的指导有限。
4. **RAA 假设较强**：要求攻击者拥有含目标身份的 exo 视频池，实际获取难度需进一步讨论。
5. **静态帧采样**：主要使用帧级特征，未充分利用时序动作/步态等动态隐私线索。

## 相关工作与启发

- 与 VISPR、PIPA 等社交媒体隐私基准互补，将研究扩展到第一人称视角。
- RAA 的 "检索增强" 思路类似于 NLP 中的 RAG，可启发更多视觉隐私攻防场景。
- 为可穿戴设备厂商（Meta Ray-Ban、Apple Vision Pro）的隐私设计提供量化参考。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统化定义 ego 视频佩戴者隐私攻击面，RAA 策略新颖
- 实验充分度: ⭐⭐⭐⭐ — 七任务 × 四级威胁模型 × ID/OOD 全面实验，消融充分
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，公式化统一，图表丰富
- 价值: ⭐⭐⭐⭐ — 填补 ego 隐私研究空白，对可穿戴设备隐私设计和法规制定有重要参考意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Attention Smoothing Is All You Need For Unlearning](../../ICLR2026/llm_safety/attention_smoothing_is_all_you_need_for_unlearning.md)
- [\[ICML 2025\] Improving Your Model Ranking on Chatbot Arena by Vote Rigging](improving_your_model_ranking_on_chatbot_arena_by_vote_rigging.md)
- [\[ICML 2025\] Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs](is_your_model_fairly_certain_uncertainty-aware_fairness_evaluation_for_llms.md)
- [\[ACL 2026\] Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens](../../ACL2026/llm_safety/forget_what_matters_keep_the_rest_selective_unlearning_of_informative_tokens.md)
- [\[ICML 2025\] Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models](watch_out_your_album_on_the_inadvertent_privacy_memorization_in_multi-modal_larg.md)

</div>

<!-- RELATED:END -->
