---
title: >-
  [论文解读] MacVQA: Adaptive Memory Allocation and Global Noise Filtering for Continual Visual Question Answering
description: >-
  [AAAI 2026][多模态VLM][持续学习] 本文提出MacVQA框架，通过全局噪声过滤（GonF）增强视觉特征的鲁棒性，并通过自适应记忆分配（AMA）基于原型检索和记忆衰减优化知识保留与更新，在VQA v2的10个持续学习任务上实现43.38%平均准确率（+3.57%）和2.32%遗忘率。 领域现状 视觉问答（VQA…
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "持续学习"
  - "视觉问答"
  - "噪声过滤"
  - "记忆分配"
  - "原型学习"
---

# MacVQA: Adaptive Memory Allocation and Global Noise Filtering for Continual Visual Question Answering

**会议**: AAAI 2026  
**arXiv**: [2601.01926](https://arxiv.org/abs/2601.01926)  
**代码**: [https://github.com/HubuKG/MacVQA](https://github.com/HubuKG/MacVQA)  
**领域**: 多模态VLM  
**关键词**: 持续学习, 视觉问答, 噪声过滤, 记忆分配, 原型学习

## 一句话总结

本文提出MacVQA框架，通过全局噪声过滤（GonF）增强视觉特征的鲁棒性，并通过自适应记忆分配（AMA）基于原型检索和记忆衰减优化知识保留与更新，在VQA v2的10个持续学习任务上实现43.38%平均准确率（+3.57%）和2.32%遗忘率。

## 研究背景与动机

### 领域现状
视觉问答（VQA）融合视觉和文本信息来回答关于图像的问题，在自动驾驶、医学诊断、无障碍辅助等领域有广泛应用。持续学习（Continual Learning）在VQA中逐渐受到关注，旨在让模型在学习新任务的同时保留已学知识，解决不断演变的环境和动态任务的需求。

### 现有痛点
现有持续VQA学习方法存在多方面局限：

**正则化方法**（EWC, MAS）：通过保护关键参数来保留知识，但在多模态架构中难以处理模态冲突，且对细粒度交互建模不足

**回放方法**（ER, DER）：通过存储样本的回放来缓解遗忘，但受限于存储开销和隐私约束

**近期VQA专用方法**各有不足：
   - **VQACL**：评估未见技能-概念组合，但对动态任务适应性差
   - **PROOF**：使用可扩展投影，但扩展性受限
   - **QUAD**：通过对齐一致性蒸馏消除视觉存储，但对噪声特征不够鲁棒
   - **ProtoGroup**：多原型分组稳定表示，但计算开销大且对聚类参数敏感

**共性不足**：缺乏对多模态特征质量的关注（噪声视觉区域干扰推理）；记忆管理效率低，未能平衡保留、适应和泛化。

### 核心idea
从两个互补角度解决问题：
- **特征侧**：通过全局噪声过滤消除无关视觉区域，增强多模态特征的鲁棒性
- **记忆侧**：通过自适应记忆分配，基于原型检索和时间衰减策略高效管理跨任务知识

## 方法详解

### 整体框架
MacVQA基于VL-T5 backbone，包含三个核心模块：
1. **全局噪声过滤（GonF）**：打分→加权→去噪自编码器→全局特征融合，提升视觉特征质量
2. **记忆池**：存储视觉和文本原型作为动态参考
3. **自适应记忆分配（AMA）**：原型检索→门控融合→时间插值更新，优化知识的获取与保留

### 关键设计

#### 1. **全局噪声过滤（GonF）**

GonF的目标是消除视觉表示中的噪声和无关区域特征。

**特征提取**：
- 视觉特征：预训练Faster R-CNN提取n个区域特征 $\mathbf{V} \in \mathbb{R}^{n \times d}$（d=2048）
- 文本特征：编码输入查询为嵌入 $\mathbf{Q} \in \mathbb{R}^{L \times d}$

**噪声过滤三步走**：

**Step 1 - 特征打分**：通过线性变换+softmax为每个图像区域分配注意力权重

$$\omega_m = \frac{\exp(\text{score}(\mathbf{V}_m))}{\sum_{l=1}^{n} \exp(\text{score}(\mathbf{V}_l))}$$

**Step 2 - 全局特征生成**：加权求和生成全局上下文向量

$$\mathbf{G} = \sum_{m=1}^{n} \omega_m \cdot \mathbf{V}_m$$

**Step 3 - 去噪+融合**：去噪自编码器（DAE）将 $\mathbf{V}$ 细化为去噪特征 $\mathbf{V}'$，再与全局特征 $\mathbf{G}$ 融合为最终增强特征 $\mathbf{V}''$

**GonF损失**：重建准确性 + 注意力分布平滑性

$$\mathcal{L}_{\text{GonF}} = \frac{1}{n} \sum_{m=1}^{n} \left( \|\mathbf{V}_m - \mathbf{V}'_m\|_2^2 - \theta_1 \omega_m \log \omega_m \right)$$

$\theta_1$ 控制去噪精度与注意力分布平滑性的平衡（通常在[0.1, 0.5]）。

**设计动机**：VQA中图像区域并非都相关（如问"what color is the car"时，背景的天空和树木是噪声）。GonF先全局评分找重点，再用DAE去噪，最后用全局信息补充局部视角，是一个"打分→去噪→融合"的三阶段pipeline。

#### 2. **自适应记忆分配（AMA）**

AMA维护一个包含视觉原型集 $\{\mathbf{V}_m\}$ 和文本原型集 $\{\mathbf{Q}_n\}$ 的记忆池。

**原型检索**：
- 将多模态融合后的隐藏状态 $\mathbf{H}$ 投影到视觉和文本子空间
- 在记忆池中通过余弦相似度检索top-k最相似的视觉和文本原型

$$\text{sim}(\mathbf{H}_v, \mathbf{V}_m) = \frac{\mathbf{H}_v \cdot \mathbf{V}_m}{\|\mathbf{H}_v\| \cdot \|\mathbf{V}_m\|}$$

**门控融合**：动态权重向量 $\mathbf{g}$ 控制当前输入与检索原型的融合比例

$$\mathbf{g} = \sigma(\mathbf{W}_g \cdot [\mathbf{H}; \mathbf{Q}_p; \mathbf{V}_p])$$

最终融合特征：$\mathbf{H}' = \mathbf{H} + \bm{\alpha} \cdot \mathbf{Q}_p + \bm{\beta} \cdot \mathbf{V}_p$

**记忆更新（时间插值策略）**：

$$\mathbf{P}_i = \lambda \cdot \mathbf{P}_{i-1} + (1-\lambda) \cdot \mathbf{H}$$

$\lambda$ 控制旧记忆与新上下文的平衡，实现渐进式知识积累。

**AMA损失**：

$$\mathcal{L}_{\text{AMA}} = -\sum_{m=1}^{k} \text{sim}(\mathbf{H}_v, \mathbf{V}_{p_m}) - \sum_{n=1}^{k} \text{sim}(\mathbf{H}_q, \mathbf{Q}_{p_n}) + \theta_2(g_q + g_v - 1)^2 + \theta_3 \|\mathbf{H}' - \mathbf{H}\|_2^2$$

包含四项：相似度最大化 + 门控平衡正则 + 更新幅度约束。

**设计动机**：传统记忆回放是被动的"重播旧样本"，AMA是主动的"检索相关知识并动态融合"。时间插值策略让记忆池随任务流平滑演化，兼顾保留和适应。

### 损失函数 / 训练策略

总损失：

$$\mathcal{L}_{total} = \phi_1 \mathcal{L}_{\text{GonF}} + \phi_2 \mathcal{L}_{\text{AMA}} + \phi_3 \mathcal{L}_{\text{decoder}}$$

其中 $\phi_1 + \phi_2 + \phi_3 = 1$。解码器损失为标准负对数似然。

训练配置：
- Backbone：VL-T5（所有方法共用）
- 优化器：Adam，学习率3e-5，梯度裁剪5
- 记忆缓冲区大小：5000（可配置）
- 两种评估范式：标准测试（已见技能-概念组合）和新组合测试（未见组合）

## 实验关键数据

### 主实验

**VQA v2 标准测试（10个任务的平均准确率/遗忘率）**：

| 方法 | AP(%) ↑ | AF(%) ↓ | 特点 |
|------|---------|---------|------|
| Vanilla | 14.49 | 30.15 | 灾难性遗忘 |
| EWC | 15.77 | 28.38 | 正则化 |
| ER | 36.99 | 4.80 | 经验回放 |
| VQACL | 38.77 | 2.90 | VQA专用 |
| ProtoGroup | 39.81 | 2.87 | 多原型分组 |
| **MacVQA** | **43.38** | **2.32** | GonF+AMA |

**新组合测试（Novel Composition）**：

| 方法 | AP(%) ↑ | AF(%) ↓ |
|------|---------|---------|
| VQACL | 35.40 | 4.90 |
| QUAD | 40.00 | 3.81 |
| **MacVQA** | **42.53** | **3.60** |

MacVQA在标准测试上比最佳基线提升+3.57%准确率，遗忘率降低-0.55%；多个任务类型上有显著提升：如Color任务56.95%（vs ProtoGroup 44.66%），Action任务58.13%（vs 50.79%）。

### 消融实验

**GonF和AMA模块的贡献**：

| GonF | AMA | AP(%) | AF(%) | 说明 |
|------|-----|-------|-------|------|
| × | × | 38.77 | 2.90 | 基线(VQACL) |
| ✓ | × | 41.75 | 2.14 | 噪声过滤减少遗忘 |
| × | ✓ | 40.97 | 2.34 | 记忆分配提升准确率 |
| ✓ | ✓ | **43.38** | **2.32** | 组件互补协同 |

**原型选择策略**：

| 策略 | 标准AP | 标准AF | 新组合AP |
|------|--------|--------|---------|
| Random | 40.29 | 3.46 | 41.25 |
| **Max-Similarity** | **43.38** | **2.32** | **42.53** |

Max-Similarity策略比随机的效果显著更好，验证了基于相似度的原型检索的重要性。

### 关键发现

1. **GonF和AMA的互补性**：GonF擅长提升感知类任务（Recognition, Location），AMA擅长记忆敏感的推理任务（Color, Type）。GonF的早期噪声过滤阻止错误传播，AMA的原型优先选择支持长期记忆保留
2. **记忆缓冲区大小敏感性**：MacVQA在所有缓冲区规模下都保持最高AP和最低AF，说明高效的记忆利用——不仅仅是存得多就好
3. **超参数敏感性**：视觉主导任务（Recognition, Location）在 $\alpha=1.0, \beta=0.4$ 时最佳；文本主导任务（Commonsense, Judge）在 $\alpha=0.4, \beta=1.0$ 时最佳——证实了模态自适应加权的必要性
4. **定性分析**：MacVQA能在新组合测试中正确回答训练中未见过的物体-推理组合（如在Location任务中处理训练时未见的teddy bear类型）

## 亮点与洞察

1. **问题定义的清晰性**：从"特征质量"和"记忆管理"两个正交维度改进持续VQA，每个维度对应一个模块，设计干净
2. **GonF的三阶段pipeline**（打分→去噪→全局融合）虽然每个组件不算新颖，但整体组合在持续学习场景中首次应用，效果显著
3. **AMA的门控融合+时间插值更新**是一个简洁实用的记忆管理方案：门控决定"用多少"，时间插值决定"怎么更新"
4. **两种测试范式**（标准+新组合）的设计全面评估了模型的保留和泛化能力
5. 超参数分析揭示了任务类型与最优模态权重的关联——为实际应用中的自适应策略提供了依据

## 局限与展望

- Faster R-CNN作为视觉特征提取器相对过时，是否可以用更现代的ViT或CLIP视觉编码器？
- 10个任务的任务序列是固定的，不同任务顺序对结果的影响未分析
- 记忆池的大小（原型数量M, N）似乎是预设的，自适应调整可能更好
- 所有方法共用VL-T5 backbone——在更强的backbone（如BLIP-2, LLaVA）上的效果如何？
- 无法处理任务边界未知的在线持续学习场景

## 相关工作与启发

- VQACL引入的技能-概念组合测试范式是持续VQA的标准评估协议
- ProtoGroup的多原型分组思想与MacVQA的原型记忆池有概念联系，但MacVQA的门控融合+时间衰减机制更加灵活
- GonF的DAE去噪可以看作是一种在特征空间的数据增强，与持续学习中的knowledge distillation有互补性
- 将原型学习与持续学习结合的趋势正在增强——MacVQA是又一个成功案例

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering](../../ICLR2026/multimodal_vlm/meta-adaptive_prompt_distillation_for_few-shot_visual_question_answering.md)
- [\[CVPR 2026\] VQ-VA World: Towards High-Quality Visual Question-Visual Answering](../../CVPR2026/multimodal_vlm/vq-va_world_towards_high-quality_visual_question-visual_answering.md)
- [\[ACL 2026\] WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering](../../ACL2026/multimodal_vlm/wikiseeker_rethinking_the_role_of_vision-language_models_in_knowledge-based_visu.md)
- [\[ACL 2025\] MAGIC-VQA: Multimodal and Grounded Inference with Commonsense Knowledge for Visual Question Answering](../../ACL2025/multimodal_vlm/magic-vqa_multimodal_and_grounded_inference_with_commonsense_knowledge_for_visua.md)
- [\[AAAI 2026\] Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models](branch_or_layer_zeroth-order_optimization_for_continual_lear.md)

</div>

<!-- RELATED:END -->
