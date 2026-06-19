---
title: >-
  [论文解读] Staining and Locking Computer Vision Models without Retraining
description: >-
  [ICCV 2025][AI安全][模型水印] 本文提出了无需重训练或微调即可对预训练视觉模型进行"染色"（水印嵌入）和"锁定"（使用保护）的新算法，通过直接修改少量权重植入高选择性检测神经元，并提供了可计算的误报率理论保证，在图像分类和目标检测模型上验证了有效性。 深度学习模型的设计、训练和验证成本高昂…
tags:
  - "ICCV 2025"
  - "AI安全"
  - "模型水印"
  - "模型锁定"
  - "知识产权保护"
  - "免训练"
  - "计算机视觉"
---

# Staining and Locking Computer Vision Models without Retraining

**会议**: ICCV 2025  
**arXiv**: [2507.22000](https://arxiv.org/abs/2507.22000)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 模型水印, 模型锁定, 知识产权保护, 免训练, 计算机视觉

## 一句话总结
本文提出了无需重训练或微调即可对预训练视觉模型进行"染色"（水印嵌入）和"锁定"（使用保护）的新算法，通过直接修改少量权重植入高选择性检测神经元，并提供了可计算的误报率理论保证，在图像分类和目标检测模型上验证了有效性。

## 研究背景与动机
深度学习模型的设计、训练和验证成本高昂，模型权重一旦泄露或被盗用，对企业的损害巨大。现有的模型知识产权保护方法主要分为两类：**染色（Staining/Watermarking）** 和 **锁定（Locking）**。

**现有痛点**：
- 文献中所有现有的染色和锁定方法都依赖于模型重训练或微调，这意味着：(1) 每个客户需要单独训练一个模型，成本高；(2) 重训练可能以不可预知的方式改变模型行为；(3) 需要访问训练或验证数据
- 后门式水印方法通过操纵训练数据来实现，固有地改变了模型对自然图像的响应，风险不可控
- 现有方法缺乏可证明的误报率保证

**核心矛盾**：如何在不重训练模型的前提下，实现可靠的模型保护，同时提供理论保证？

**切入角度**：受隐蔽攻击（stealth attacks）的启发，利用现代模型特征空间的集中性质，直接通过修改少量权重来植入高选择性的检测神经元。

**核心idea**：通过在模型中植入一个随机采样的检测神经元，利用特征空间的高维集中现象确保该神经元几乎不响应自然输入，但可以通过优化得到一个能强烈激活它的触发输入，从而实现无需重训练的染色和锁定。

## 方法详解

### 整体框架
整个方法围绕一个核心组件——**检测神经元（Detector Neuron）** 构建。染色（staining）是将检测神经元植入模型作为识别指纹；锁定（locking）则在检测神经元基础上增加**干扰器（Disruptor）**，使模型在没有触发输入时无法正常工作。整个过程直接修改模型权重，无需训练数据和梯度反向传播训练。

### 关键设计
1. **检测神经元植入（Detector Neuron Implantation）**:

    - 功能：在模型指定层植入一个高选择性检测神经元，使其仅对特定触发输入产生强响应
    - 核心思路：从单位球面 $\mathcal{U}(\mathbb{S}^{m-1})$ 均匀采样一个检测权重向量 $v$，然后通过梯度下降优化触发输入 $x^* \in \arg\max_{z \in S} v \cdot \phi(z)$。利用偏置参数 $\delta$ 和响应值 $\Delta$ 控制检测器行为：自然输入响应为 $\delta \ll 0$（被ReLU截断为0），触发输入响应为 $\Delta \gg 0$
    - 设计动机：高维空间的集中现象保证随机向量几乎不会与自然数据的特征表示强烈对齐，但总可以优化到一个触发输入使其强烈响应

2. **非加性与加性染色（Non-additive & Additive Staining）**:

    - 功能：两种将检测器嵌入模型的方式
    - 核心思路：非加性染色直接替换目标神经元权重为 $u = \frac{\Delta - \beta}{v \cdot \phi(x^*)} v$；加性染色将检测器权重加到现有神经元上，$u = w + \frac{\Delta - \beta - w \cdot \phi(x^*)}{v \cdot \phi(x^*)} v$
    - 设计动机：非加性染色产生"静默"神经元易于检测并移除；加性染色隐蔽性更好，因为检测器权重被融合到原始权重中

3. **内部锁定（Internal Locking）**:

    - 功能：使模型在没有触发补丁（trigger patch）时无法正常工作
    - 核心思路：(1) 在早期卷积层植入检测器，限定激活位置为图像角落 $(a,b)$，仅优化该感受野内的小补丁作为触发器；(2) 通过"导管"（序列恒等卷积核）将检测信号传播到后层；(3) 在logits层用随机干扰向量 $u$ 替换偏置：锁定时偏置为 $su + t$，解锁时通过检测信号 $\gamma$ 恢复原始偏置
    - 设计动机：触发补丁小（早期层感受野有限），模型外观与原始一致，但缺少补丁就无法正确推理

4. **Squeeze-and-Excite锁定**:

    - 功能：提供架构无关的通用锁定方案
    - 核心思路：利用 Sq-Ex 块 $s(x) = x \odot q(x)$ 的全局平均池化特性横向传播检测信号。干扰器嵌入 Sq-Ex 块的参数 $S_2$ 和 $\tau_2$ 中
    - 设计动机：内部锁定受限于检测器感受野必须覆盖触发补丁，Sq-Ex 块通过全局池化打破了这个限制，可添加到任何预训练模型中且计算开销极低

### 损失函数 / 训练策略
本方法完全无需训练。所有操作均为直接的权重修改：
- 检测器权重从球面均匀分布采样
- 触发输入通过梯度下降优化（针对固定的检测器权重）
- 干扰器权重从球面均匀分布采样
- 无需训练数据、无需损失函数、无需反向传播更新

## 实验关键数据

### 主实验
实验在 ResNet50、VGG16（图像分类，ImageNet）和 SSDLite-MobileNetV3、Faster-RCNN-ResNet50（目标检测，COCO）上进行。

| 模型 | 任务 | 原始性能 | 染色后性能 | 锁定(无补丁) | 解锁(有补丁) |
|------|------|----------|-----------|-------------|-------------|
| ResNet50 | 分类(Acc) | 76.1% | ≈76.1% (无损) | 大幅下降 | ≈原始性能 |
| VGG16 | 分类(Acc) | 71.6% | ≈71.6% (无损) | 大幅下降 | ≈原始性能 |
| SSDLite | 检测(AP) | 21.3 | ≈21.3 (无损) | 大幅下降 | ≈原始性能 |
| Faster-RCNN | 检测(AP) | 36.4 | ≈36.4 (无损) | 大幅下降 | ≈原始性能 |

所有实验中染色均无误报发生（0/50次采样 × 全验证集）。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 检测器层位置（早→晚） | 误报率递减，触发补丁递增 | 早期层维度低，误报风险高但补丁小 |
| 内部锁 vs Sq-Ex锁 | Sq-Ex锁更通用 | Sq-Ex可添加到任何架构 |
| 添加Sq-Ex块对性能影响 | 几乎无影响 | "edited"模型性能≈原始 |
| DC-GAN + 染色/锁定 | 有效 | 扩展到生成模型 |
| ViT-B-16 + 染色/锁定 | 有效 | 扩展到Transformer架构 |

### 关键发现
- 理论上界（Theorem 1几何界 + Theorem 2数据驱动界）与实验观测高度吻合
- 染色对模型性能的影响可忽略不计
- 锁定状态下模型性能显著下降，解锁后恢复接近原始性能
- 方法可扩展到GAN和ViT架构

## 亮点与洞察
- **完全免训练**：这是该方向最大的突破。一个基础模型可以为不同客户生成不同的水印/锁，无需重新训练
- **可证明的保证**：Theorem 1（基于PCA维度的几何界）和 Theorem 2（基于经验数据的界）提供了误报率的上界，这在文献中是独一无二的
- **实用的触发补丁设计**：通过限定检测器位置生成小的触发补丁，可插入图像角落，对正常推理影响极小
- **揭示了伪造攻击风险**：免训练特性同时意味着攻击者也可以轻松植入伪造水印，这对水印系统是个警示

## 局限与展望
- 论文主要聚焦于最简单场景（单检测器+单干扰器），实际部署应使用多个
- 隐蔽性和混淆技术仅简要讨论，未深入实现
- 内部锁定对架构有一定依赖（需要合适的干扰器位置）
- 未深入测试对模型蒸馏、权重剪枝等攻击的鲁棒性
- 触发补丁的大小受检测层位置约束，存在安全性-隐蔽性的权衡

## 相关工作与启发
- 与隐蔽攻击（stealth attacks）的理论联系为双向的：本文理论保证可反过来分析隐蔽攻击
- Sq-Ex锁定思路可能扩展到其他使用全局池化的架构（如注意力机制）
- 免训练特性启示：模型权重空间中存在大量"空闲"方向可被利用
- 对联邦学习中的模型保护有潜在应用价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 完全免训练的染色/锁定方法是全新的贡献，理论保证也是该领域首次
- 实验充分度: ⭐⭐⭐⭐ 覆盖了分类和检测任务，多种架构，但缺乏对抗攻击的鲁棒性测试
- 写作质量: ⭐⭐⭐⭐⭐ 思路清晰，理论和实验紧密结合，算法伪代码详尽
- 价值: ⭐⭐⭐⭐⭐ 解决了模型IP保护的核心痛点（免训练），具有很强的实际部署价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Retraining with Predicted Hard Labels Provably Increases Model Accuracy](../../ICML2025/ai_safety/retraining_with_predicted_hard_labels_provably_increases_model_accuracy.md)
- [\[ICML 2025\] A Certified Unlearning Approach without Access to Source Data](../../ICML2025/ai_safety/a_certified_unlearning_approach_without_access_to_source_data.md)
- [\[CVPR 2026\] Unlearning without Forgetting: Securely Removing Targeted Concepts from Large-Scale Vision-Language Open-Vocabulary Detectors](../../CVPR2026/ai_safety/unlearning_without_forgetting_securely_removing_targeted_concepts_from_large-sca.md)
- [\[CVPR 2026\] A Unified Perspective on Adversarial Membership Manipulation in Vision Models](../../CVPR2026/ai_safety/a_unified_perspective_on_adversarial_membership_manipulation_in_vision_models.md)
- [\[CVPR 2026\] Hierarchically Robust Zero-shot Vision-language Models](../../CVPR2026/ai_safety/hierarchically_robust_zero-shot_vision-language_models.md)

</div>

<!-- RELATED:END -->
