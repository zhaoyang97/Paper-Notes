---
title: >-
  [论文解读] Multi-Agent VLMs Guided Self-Training with PNU Loss for Low-Resource Offensive Content Detection
description: >-
  [AAAI 2026][多模态VLM][自训练] 本文提出了一种多智能体视觉语言模型（MA-VLMs）引导的自训练框架，结合新颖的PNU损失函数，在仅有少量标注数据（如50个）的低资源场景下实现高质量攻击性内容检测，性能接近大规模模型。 社交媒体上的攻击性内容（仇恨言论、厌女症、骚扰等）威胁着公共安全和民主讨论…
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "自训练"
  - "多智能体VLM"
  - "伪标签"
  - "PNU损失"
  - "攻击性内容检测"
---

# Multi-Agent VLMs Guided Self-Training with PNU Loss for Low-Resource Offensive Content Detection

**会议**: AAAI 2026  
**arXiv**: [2511.13759](https://arxiv.org/abs/2511.13759)  
**代码**: [github](https://github.com/Social-AI-Studio/MA-VLM.git)  
**领域**: 多模态VLM  
**关键词**: 自训练, 多智能体VLM, 伪标签, PNU损失, 攻击性内容检测

## 一句话总结
本文提出了一种多智能体视觉语言模型（MA-VLMs）引导的自训练框架，结合新颖的PNU损失函数，在仅有少量标注数据（如50个）的低资源场景下实现高质量攻击性内容检测，性能接近大规模模型。

## 研究背景与动机

社交媒体上的攻击性内容（仇恨言论、厌女症、骚扰等）威胁着公共安全和民主讨论。现有内容审核系统在覆盖面、公平性和跨语言/文化适应性上存在不足。

**核心痛点**：构建鲁棒的攻击性内容检测系统需要大量高质量标注数据，但这类数据极度稀缺。原因有二：（1）攻击性样本本身比例低，常被平台删除；（2）人工标注成本高且需要理解上下文、讽刺和隐含伤害。

**现有方法的局限**：
- **LLM提示**：few-shot效果好但推理成本高，无法大规模部署
- **数据增强**：局限于文本，语义多样性不足
- **迁移学习**：仍需要中等规模目标域标注
- **传统自训练**：初始模型弱时伪标签质量差，错误传播严重

**核心矛盾**：低资源场景下，模型初始能力弱→伪标签不可靠→自训练失败；而VLM虽然理解能力强，但推理成本高无法直接部署。

**本文切入角度**：能否让VLM作为"验证者"而非"执行者"，引导轻量级分类器的自训练过程？进一步地，攻击性内容本质上存在标注歧义（审核员倾向保守 vs 用户倾向自由表达），能否通过模拟这种社会张力来提升伪标签质量？

## 方法详解

### 整体框架
本文提出MA-VLMs引导的自训练流水线（图2），核心思路是用轻量级分类器（CLIP-Large + 1层MLP）做预测，用冻结的多智能体VLM（Qwen-2.5-VL-72B）做验证，通过分类器与VLM的共识/分歧来区分样本类型，最后用PNU损失统一训练。

### 关键设计

#### 1. **MA-VLMs引导的自训练流水线**
自训练分5个步骤迭代进行：
- **训练**：用 $n$（如100）个标注样本训练分类器
- **预测与排序**：分类器对未标注样本预测并按置信度排序
- **共识/分歧判定**：取Top-$k$（$k=500$）高置信样本交由MA-VLMs验证。三方一致→**Agreed-Unknown**（伪标签正/负），存在分歧→**Disagreed-Unknown**（保留为无标签）
- **重训练**：用PNU损失整合所有数据类型重新训练分类器
- **验证检查**：若开发集性能提升则继续，否则回退

设计动机：传统自训练仅依赖分类器自身判断，当初始模型弱时容易错误累积。引入VLM作为外部验证器可显著提升伪标签质量。Top-$k$选择策略确保每轮只纳入最可靠的样本。

#### 2. **多智能体VLM（MA-VLMs）提示格式**
创新性地设计了两个VLM角色：
- **审核员（Moderator）**：安全优先偏向，倾向判定为攻击性内容
- **用户（User）**：捍卫表达自由，倾向判定为非攻击性

两个智能体各自给出初始判断和理由，然后审阅对方的判断后给出最终决定。只有当两个智能体都与分类器一致时，才标记为Agreed-Unknown。

设计动机：现实世界的内容审核本质上是"过度审查"与"保护不足"之间的张力。模拟这种社会动态可以更好地捕捉标注歧义，通过多视角协商发现隐含仇恨（如将贬低女性包装为赞美的meme）。

#### 3. **PNU损失函数**
核心公式如下：

$$\mathcal{L}_{\text{pnu}} = \begin{cases} (1-\gamma) \cdot (\mathcal{L}_{\text{pn}} + \mathcal{L}_{\text{soft-pn}}) + \gamma \cdot \mathcal{L}_{\text{pu}}, & \text{if } \gamma \geq 0 \\ (1+\gamma) \cdot (\mathcal{L}_{\text{pn}} + \mathcal{L}_{\text{soft-pn}}) - \gamma \cdot \mathcal{L}_{\text{nu}}, & \text{if } \gamma < 0 \end{cases}$$

三类数据对应三个损失项：
- **$\mathcal{L}_{\text{pn}}$**：标准PN损失，用于有真实标签的数据
- **$\mathcal{L}_{\text{soft-pn}}$**：软PN损失，用于Agreed-Unknown数据（软标签 $\hat{y}_p=0.67, \hat{y}_n=0.33$），缓解过拟合伪标签
- **$\mathcal{L}_{\text{pu}}$ / $\mathcal{L}_{\text{nu}}$**：PU/NU学习损失，用于Disagreed-Unknown数据

参数 $\gamma \in [-1,1]$ 控制PU/NU学习的强度：$\gamma=0$ 退化为PN学习，$\gamma>0$ 为PU学习，$\gamma<0$ 为NU学习。

设计动机：传统方法丢弃分歧样本，浪费了有价值的训练信号。PNU损失通过区分数据可靠性来最大化利用所有可用数据，同时控制伪标签噪声的影响。

### 损失函数 / 训练策略
- 分类损失 $\ell$ 使用交叉熵
- 正类先验 $\pi_p$ 固定为0.5（偏离会引入类别偏差）
- $\gamma$ 通过消融实验确定：FHM用 $\gamma=0.0$，其他数据集用 $\gamma=0.1$
- 每轮选择Top-$k=500$个高置信样本进行验证
- 训练10个epoch，根据开发集选最优epoch

## 实验关键数据

### 主实验

| 模型 | 训练策略 | FHM M-F1 | MAMI M-F1 | HSOL M-F1 | Sent140 M-F1 |
|------|----------|----------|-----------|-----------|--------------|
| Qwen7B | SupOnly | 70.41 | 76.06 | 84.89 | 78.19 |
| CLIP | SupOnly | 59.24 | 62.18 | 85.30 | 64.22 |
| CLIP | SelfTrain(CLIP) | 70.00 | 67.03 | 86.48 | 73.05 |
| CLIP | SelfTrain(Qwen72B) | 65.22 | 67.42 | 81.06 | 75.57 |
| **CLIP** | **SelfTrain(CLIP+Qwen72B)** | **72.68** | **73.49** | **86.69** | **77.11** |

（n=100，即仅100个标注样本）

### 不同标注量实验（FHM数据集）

| 标注量n | Qwen7B SupOnly M-F1 | CLIP SupOnly M-F1 | CLIP SelfTrain M-F1 |
|---------|---------------------|-------------------|---------------------|
| 50 | 39.11 | 48.76 | **71.27** |
| 100 | 70.41 | 59.24 | **72.68** |
| 250 | **75.88** | 69.67 | 72.97 |

### 消融实验

| 配置 | FHM M-F1 | MAMI M-F1 | 说明 |
|------|----------|-----------|------|
| γ = -0.1 | 68.35 | 59.98 | NU学习无效 |
| γ = 0.0 | **72.68** | 68.84 | FHM最优 |
| γ = 0.1 | 71.50 | **73.49** | MAMI最优 |
| γ = 0.2 | 71.79 | 72.42 | 过强PU有害 |

| 提示格式 | FHM M-F1 | MAMI M-F1 | 说明 |
|----------|----------|-----------|------|
| Zero-Shot | 74.46 | 79.17 | 基线 |
| Few-Shot | 71.09 | 75.08 | 示例引入偏差 |
| CoT | 74.43 | 78.28 | 社会任务CoT无效 |
| **MA-VLMs** | **74.62** | **81.64** | 多智能体最优 |

### 关键发现
1. 仅50个标注样本时，自训练CLIP（M-F1=71.27）大幅领先SupOnly Qwen7B（M-F1=39.11），证明框架在极端低资源下的有效性
2. CLIP+Qwen72B联合伪标签在所有数据集上优于单一模型伪标签，验证了互补验证的价值
3. MA-VLMs提示格式在标注歧义更大的MAMI数据集上优势更明显（+2.47 vs Zero-Shot），说明多智能体协商善于处理模糊概念
4. γ的最优值与数据集特性相关：类别平衡时γ=0.1更好，不平衡时γ=0.0更优

## 亮点与洞察
1. **多视角审核机制设计精巧**：模拟审核员与用户的社会张力，不仅提升了伪标签质量，还增强了系统的公平性和可解释性
2. **PNU损失的统一框架**：优雅地整合了三类不同可靠性的数据，避免了传统方法中丢弃分歧样本的浪费
3. **轻量级部署**：最终推理只需CLIP-Large（428M参数），VLM仅用于训练阶段的伪标签生成，兼顾了性能和效率
4. **伪标签分析揭示了ground truth标注错误**：说明多方共识可能比人工标注更可靠

## 局限与展望
1. 依赖Qwen72B进行伪标签验证，训练阶段仍需大模型推理资源
2. 每轮固定选择Top-k=500，未自适应调整选择数量
3. γ需要针对每个数据集手动调优，缺乏自动选择机制
4. 仅在二分类任务上验证，多类别攻击性内容检测（如细粒度分类）未探索
5. 未考虑多语言/多文化场景的适应性

## 相关工作与启发
- PU学习在社会任务中的应用是新颖的视角，可推广到其他标注歧义大的任务（如情感分析、立场检测）
- 多智能体协商机制可以启发其他需要平衡不同立场的NLP任务
- 知识蒸馏思路（大模型→小模型）在低资源场景值得更多探索

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Contamination Detection for VLMs using Multi-Modal Semantic Perturbation](../../ICLR2026/multimodal_vlm/contamination_detection_for_vlms_using_multi-modal_semantic_perturbation.md)
- [\[CVPR 2026\] VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments](../../CVPR2026/multimodal_vlm/vs_bench_evaluating_vlms_for_strategic_abilities_in_multi_agent_environments.md)
- [\[CVPR 2026\] Hierarchical Attacks for Multi-Modal Multi-Agent Reasoning](../../CVPR2026/multimodal_vlm/hierarchical_attacks_for_multi-modal_multi-agent_reasoning.md)
- [\[CVPR 2026\] Multimodal Learning on Low-Quality Data with Conformal Predictive Self-Calibration](../../CVPR2026/multimodal_vlm/multimodal_learning_on_low-quality_data_with_conformal_predictive_self-calibrati.md)
- [\[CVPR 2026\] Rethinking BCE Loss for Multi-Label Image Recognition with Fine-Tuning](../../CVPR2026/multimodal_vlm/rethinking_bce_loss_for_multi-label_image_recognition_with_fine-tuning.md)

</div>

<!-- RELATED:END -->
