---
title: >-
  [论文解读] Stop Learning It All to Mitigate Visual Hallucination, Focus on the Hallucination Target
description: >-
  [CVPR 2025][幻觉检测][视觉幻觉] 提出**TL-DPO**（Target-Learning DPO），将传统DPO的全句级偏好学习限制到**幻觉发生的目标chunk**和**对应的图像区域**，通过目标生成损失和目标条件损失排除无关信号，在LLaVA-1.5上将CHAIR_s从66.8降至20.1，同时LLaVA-Bench从63.4提升至71.2。
tags:
  - "CVPR 2025"
  - "幻觉检测"
  - "视觉幻觉"
  - "目标学习"
  - "DPO"
  - "偏好优化"
  - "多模态大模型"
---

# Stop Learning It All to Mitigate Visual Hallucination, Focus on the Hallucination Target

**会议**: CVPR 2025  
**arXiv**: [2506.11417](https://arxiv.org/abs/2506.11417)  
**代码**: 未公开  
**领域**: 幻觉检测  
**关键词**: 视觉幻觉, 目标学习, DPO, 偏好优化, 多模态大模型

## 一句话总结

提出**TL-DPO**（Target-Learning DPO），将传统DPO的全句级偏好学习限制到**幻觉发生的目标chunk**和**对应的图像区域**，通过目标生成损失和目标条件损失排除无关信号，在LLaVA-1.5上将CHAIR_s从66.8降至20.1，同时LLaVA-Bench从63.4提升至71.2。

## 研究背景与动机

多模态大语言模型（MLLM）在视觉-语言任务中表现出色，但存在严重的**幻觉问题**——生成图像中不存在的物体信息或不准确的空间描述。偏好学习方法（如RLHF、DPO）被广泛用于缓解幻觉，但效果有限。

**现有痛点**：
1. **全局学习的局限**——传统偏好学习在完整响应级别进行优化，但幻觉可能只出现在响应的某个局部（如"时钟显示11:20"中只有"11:20"是错误的），全局优化会学到大量与幻觉无关的信号
2. **注意力偏移**——通过分析注意力图发现，传统DPO训练后模型可能将注意力从图像目标物体转移到文本信号上，导致过拟合文本模式而非修正视觉理解
3. **无关信号干扰**——偏好学习中"好/差"响应对的大部分内容是相同的，真正的差异仅在幻觉部分，但模型被迫在整个响应上学习偏好，效率低且可能学到错误信号

**核心矛盾**：人类修正错误时只修改出错的地方（如改掉一个错字），但现有偏好学习方法要求模型"重写全部内容"，导致学习效率低且引入副作用。

**切入角度**：像人类一样只关注出错的地方——将偏好学习限制在幻觉目标（target）上，包括响应中出错的文本chunk和图像中导致幻觉的对象区域。

## 方法详解

### 整体框架

TL-DPO包含两个互补的损失函数：(1) **目标生成损失**——仅在幻觉发生的文本chunk上计算DPO损失，过滤掉无关的文本信号；(2) **目标条件损失**——通过遮罩图像中导致幻觉的目标对象，训练模型学会利用目标区域的视觉信息来给出正确回答。训练数据基于Visual Genome数据集构建，包含幻觉响应、正确响应、以及幻觉目标位置信息（包括文本chunk位置和图像bounding box）。

### 关键设计

1. **目标生成损失（Target Generation Loss）**
    - **功能**：将DPO的偏好比较从全句级别缩小到幻觉chunk级别，排除响应中与幻觉无关的部分
    - **核心思路**：假设响应 $y$ 中只有部分chunk $y^t$ 包含幻觉信息。DPO标准损失比较整个响应 $(y_r, y_h)$ 的奖励差，TL-DPO只比较目标chunk $(y_r^t, y_h^t)$ 的奖励差：
     $$\mathcal{L}_t = -\mathbb{E}_{(x, y_r^t, y_h^t) \sim D} [\log \sigma(u(x, y_r^t, y_h^t))]$$
     例如，对于"时钟显示约11:20（错误）"和"时钟显示约15:26（正确）"，$y_h^t$ = "11:20"，$y_r^t$ = "15:26"，只在这两个片段上计算偏好
    - **设计动机**：理论证明（Theorem 3.3），在假设3.1（幻觉无关信号不影响奖励差异）下，目标级DPO与完整DPO等价，但前者假设空间更小（Proposition 1），需要更少的样本达到相同泛化误差

2. **目标条件损失（Target Condition Loss）**
    - **功能**：训练模型学会利用图像中目标区域的视觉信息，而非依赖文本先验来回答
    - **核心思路**：给定幻觉相关的图像区域 $m_i^t$（由bounding box标记），构造遮罩图像 $\tilde{m}_i^t$（将该区域遮罩），形成偏好对 $(m_i, q, y_r)$（完整图像+正确回答）vs $(\tilde{m}_i^t, q, y_r)$（遮罩图像+正确回答），训练模型偏好使用完整图像信息：
     $$\mathcal{L}_c = -\mathbb{E}_{(m_i, \tilde{m}_i^t, x, y_r) \sim D} [\log \sigma(u^*(m_i, \tilde{m}_i^t, x, y_r))]$$
     其中 $u^* = r(m_i, x, y_r) - r(\tilde{m}_i^t, x, y_r)$
    - **设计动机**：解决偏好学习中模型可能过拟合文本模式而忽略图像信息的问题。通过遮罩-未遮罩图像对，显式引导模型关注与幻觉相关的视觉区域

3. **最终训练目标**
    - **功能**：综合文本目标和视觉目标的偏好学习
    - **核心思路**：$\mathcal{L}_{TL-DPO} = \mathcal{L}_t + \mathcal{L}_c$，两个损失互补——目标生成损失确保文本层面的精准修正，目标条件损失确保视觉层面的正确关注
    - **设计动机**：单独使用任一损失都不够——仅用目标生成损失可能导致模型仍然不看图像，仅用目标条件损失可能不够精确地修正文本幻觉

### 损失函数 / 训练策略

基于LoRA微调LLaVA-v1.5-7B，batch size=32，3个epoch，学习率1e-5，cosine调度，warm-up 0.1。DPO的β=0.1，LoRA α=128，rank=64。训练数据基于VG数据集，用基线模型生成响应，GPT-4判断正确/错误并生成修正，构建包含目标位置信息的偏好数据集。

## 实验关键数据

### 主实验（LLaVA-1.5基线，与其他偏好学习方法对比）

| 方法 | CHAIR_s ↓ | CHAIR_i ↓ | POPE ↑ | MMHal ↑ | MMBench ↑ | LLaVA-Bench ↑ |
|------|-----------|-----------|--------|---------|-----------|---------------|
| LLaVA-1.5 | 66.8 | 12.7 | 85.9 | 2.42 | 63.0 | 63.4 |
| +RLHF-V | 44.6 | 7.9 | 86.2 | 2.59 | 63.6 | 65.4 |
| +HA-DPO | 37.2 | 10.0 | 86.9 | 1.97 | 64.0 | 66.2 |
| +HALVA | 46.6 | 23.1 | 87.0 | 2.25 | 66.1 | 67.2 |
| **+TL-DPO** | **20.1** | **5.2** | **86.95** | **2.72** | **67.8** | **71.2** |

### 跨模型泛化

| 模型 | CHAIR_s (基线→+TL-DPO) | POPE (基线→+TL-DPO) | MMBench (基线→+TL-DPO) |
|------|----------------------|---------------------|----------------------|
| LLaVA-1.5 | 66.8→20.1 | 85.9→87.0 | 63.0→67.8 |
| LLaVA-Next | 29.1→25.1 | 84.8→87.1 | 63.0→63.1 |
| InternVL-2.5(8B) | 18.4→7.6 | 86.5→87.0 | 68.6→80.0 |
| Llama3 | 5.5→7.1 | 82.8→87.1 | 85.8→87.3 |

### 消融实验

| 配置 | CHAIR_s ↓ | CHAIR_i ↓ | POPE ↑ | MMHal ↑ | LLaVA-Bench ↑ |
|------|-----------|-----------|--------|---------|---------------|
| LLaVA-1.5基线 | 66.8 | 12.7 | 85.9 | 2.42 | 63.4 |
| +目标条件仅 | 32.4 | 8.6 | 84.4 | 2.58 | 66.5 |
| +目标生成仅 | 14.6 | 6.1 | **89.6** | 2.70 | 68.7 |
| **TL-DPO（两者结合）** | **20.1** | **5.2** | 87.0 | **2.72** | **71.2** |

### 关键发现

1. TL-DPO在CHAIR_s上相比最强基线HALVA降低了57%（20.1 vs 46.6），同时综合性能全面提升
2. 仅目标生成损失的CHAIR_s更低（14.6），但POPE略降；两者结合实现最佳平衡
3. TL-DPO对多个模型（LLaVA、Qwen、InternVL等）都有效，展现良好泛化性
4. InternVL-2.5上CHAIR_s从18.4大幅降至7.6，MMBench从68.6提升至80.0（+11.4），效果惊人

## 亮点与洞察

1. **"像人类一样改错"的直觉**——类比人类修正绘画或文字时只改出错部分，而非重画重写，这一简洁直觉催生了target learning的核心思想
2. **理论支撑扎实**——从Bradley-Terry模型出发证明了目标级DPO与完整DPO的等价性（Theorem 3.3），并证明目标学习需要更少样本（Proposition 1），理论与实验一致
3. **双重损失互补**——目标生成损失解决"改什么文本"，目标条件损失解决"看什么图像区域"，两者结合实现文本和视觉的双重精准修正
4. **不牺牲综合性能**——多数幻觉缓解方法会降低综合benchmark分数，TL-DPO反而在MMBench、LLaVA-Bench上大幅提升，说明排除无关信号不仅减少幻觉还改善了整体学习

## 局限性

1. 损失函数中目标生成损失和目标条件损失权重固定为1:1，未探索加权的影响
2. 训练数据构建依赖GPT-4判断幻觉和生成修正，数据质量受GPT-4能力制约
3. 部分模型（如Qwen VL Chat）加TL-DPO后某些综合指标反而下降，泛化性不完全一致
4. 假设3.1（幻觉无关信号不影响奖励差异）在实际中可能不完全成立

## 相关工作与启发

- **偏好学习**: 从RLHF到DPO的发展使偏好学习更简洁，本文进一步将DPO从全句级精炼到目标级，是偏好学习粒度的重要进步
- **视觉幻觉**: HA-DPO、POVID、RLHF-V等方法从数据或全局优化角度缓解幻觉，本文首次从"排除无关信号"角度出发
- **启发**: 在偏好学习中，"少学"可能比"多学"更有效——排除无关信号不仅提高效率还提高效果，这一原则可能适用于其他偏好学习场景

## 评分

⭐⭐⭐⭐ — 直觉清晰、理论扎实、效果显著（CHAIR_s降低70%），跨模型泛化验证充分，但假设的实际合理性和数据构建对GPT-4的依赖是潜在限制

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Learning from Fine-Grained Visual Discrepancies: Mitigating Multimodal Hallucinations via In-Context Visual Contrastive Optimization](../../ICML2026/hallucination/learning_from_fine-grained_visual_discrepancies_mitigating_multimodal_hallucinat.md)
- [\[CVPR 2025\] PhD: A ChatGPT-Prompted Visual Hallucination Evaluation Dataset](phd_a_chatgpt-prompted_visual_hallucination_evaluation_dataset.md)
- [\[ACL 2025\] Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models](../../ACL2025/hallucination/retrieval_visual_contrastive_decoding_to_mitigate_object_hallucinations_in_large.md)
- [\[CVPR 2026\] First Logit Boosting: Visual Grounding Method to Mitigate Object Hallucination in Large Vision-Language Models](../../CVPR2026/hallucination/first_logit_boosting_visual_grounding_method_to_mitigate_object_hallucination_in.md)
- [\[ICML 2026\] A Unified Definition of Hallucination: It's The World Model, Stupid!](../../ICML2026/hallucination/a_unified_definition_of_hallucination_its_the_world_model_stupid.md)

</div>

<!-- RELATED:END -->
