---
title: >-
  [论文解读] MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders
description: >-
  [CVPR 2025][多模态VLM][知识蒸馏] 本文提出MoVE-KD——首个从知识蒸馏角度将多个视觉编码器（CLIP/EVA/ConvNeXt/SAM）的特长融合到单个编码器的框架，通过Mixture-of-LoRA-Experts (MoLE)缓解多教师知识冲突、利用CLIP的[CLS]注意力自适应加权蒸馏token和教师，在LLaVA/LLaVA-NeXT上实现一致提升。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "知识蒸馏"
  - "视觉编码器融合"
  - "混合专家"
  - "LoRA"
  - "注意力引导"
---

# MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders

**会议**: CVPR 2025  
**arXiv**: [2501.01709](https://arxiv.org/abs/2501.01709)  
**代码**: [https://github.com/hey-cjj/MoVE-KD](https://github.com/hey-cjj/MoVE-KD)  
**领域**: 多模态VLM  
**关键词**: 知识蒸馏, 视觉编码器融合, 混合专家, LoRA, 注意力引导

## 一句话总结

本文提出MoVE-KD——首个从知识蒸馏角度将多个视觉编码器（CLIP/EVA/ConvNeXt/SAM）的特长融合到单个编码器的框架，通过Mixture-of-LoRA-Experts (MoLE)缓解多教师知识冲突、利用CLIP的[CLS]注意力自适应加权蒸馏token和教师，在LLaVA/LLaVA-NeXT上实现一致提升。

## 研究背景与动机

不同预训练视觉编码器（CLIP、EVA、ConvNeXt、DINOv2等）各有所长，在不同VL任务中表现各异。→ 当前利用多编码器的方法（Eagle、Mini-Gemini、S²）通过特征拼接/注意力机制并行使用多编码器，但计算成本与编码器数量线性增长。→ AM-RADIO尝试用单模型多head复制多个基础模型的预测，但共享backbone学习多种不同特性会产生冲突。→ 核心矛盾：如何在保持单编码器效率的前提下，无冲突地吸收多个教师编码器的各自优势？→ 本文切入角度：知识蒸馏 + MoE + LoRA的组合——用MoLE让不同token根据输入选择不同LoRA专家，用CLIP [CLS]注意力引导蒸馏聚焦有价值的特征。

## 方法详解

### 整体框架

训练流程：多个教师编码器（CLIP/EVA/ConvNeXt，可选SAM）各自处理图像得到visual tokens → 每个教师的token通过独立的2层MLP编码器适配器投影到统一空间 → 基于CLIP预训练的[CLS]注意力计算token权重 $W^{(tok)}$ 和教师权重 $W^{(tea)}$ → 加权MSE损失蒸馏到学生编码器 → 学生编码器内部集成MoLE结构防止知识冲突 → 总损失 = 文本损失 + KD损失。推理时只用学生编码器。

### 关键设计

1. **Mixture-of-LoRA-Experts (MoLE)**:
    - 功能：在学生编码器中缓解多教师知识冲突
    - 核心思路：在学生编码器每层FFN中嵌入MoE架构，每个专家为一个参数高效的LoRA模块（两个低秩矩阵）；Router线性层根据输入动态选择top-1专家：$F^\star(x) = F(x) + E_i(x)$，$i = \text{argmax}(\text{Softmax}(f(x)))$
    - 设计动机：直接微调学生编码器→过拟合+灾难性遗忘+训练崩溃（loss异常）；用完整FFN做专家→参数暴增；LoRA既参数高效（仅0.3%总参数）又有更好的泛化性；消融实验证实不加MoLE时KD甚至会使性能低于baseline（VQAv2: 76.7→77.4 有MoLE vs 76.7 无MoLE无KD）

2. **注意力引导的KD正则化（Token权重+教师权重）**:
    - 功能：自适应地识别哪些visual token和哪个教师更值得蒸馏
    - 核心思路：利用CLIP预训练的[CLS] token与其他visual token的cross-attention作为重要性度量
    - Token权重：$W^{(tok)} = \text{Softmax}(\frac{V^{(cls)}W^{(Q)} \cdot (V^{(res)}W^{(V)})^T}{\sqrt{d}})$，让学生聚焦于语义丰富区域（如前景物体），忽略背景
    - 教师权重：$W^{(tea)} = \text{Softmax}(\text{mean}(\frac{V^{(cls)} \cdot V_i^{(t)T}}{\sqrt{d}}))$，反映各教师对特定图像的响应强度
    - 设计动机：[CLS]注意力天然聚焦关键区域（可视化显示CLIP的[CLS]集中在有意义的物体上），比可学习token更高效且泛化更好；不同教师在不同图像上的贡献应该不同，统一权重会限制各教师发挥特长

3. **编码器适配器（Encoder Adapter）**:
    - 功能：将不同教师编码器输出对齐到统一表示空间
    - 核心思路：每个教师配一个独立的2层MLP适配器，映射到与学生相同维度
    - 设计动机：不同预训练源的编码器输出空间不一致，简单线性插值（interpolation）无法桥接差异，消融显示用适配器比插值好0.4% (avg: 66.3→66.7)

### 损失函数 / 训练策略

总损失：$\mathcal{L}_{total} = \mathcal{L}_{text} + \lambda_{kd} \cdot \mathcal{L}_{kd}$

- $\mathcal{L}_{text}$：VLM的标准文本生成交叉熵损失
- $\mathcal{L}_{kd}$：带token权重+教师权重的加权MSE蒸馏损失
- $\lambda_{kd} = 0.5$，CLIP教师固定权重0.8（因学生从CLIP初始化），MoLE专家数=3，LoRA rank=32
- 两阶段训练：预训练阶段冻结LLM只训练MoLE/适配器/投影层；微调阶段解冻除教师外的所有参数
- 16×A800 GPU训练

## 实验关键数据

### 主实验（LLaVA-1.5 7B + MoVE-KD）

| 方法 | VQAv2 | GQA | TextVQA | POPE | SQA | MME | MMB | Avg |
|------|-------|-----|---------|------|-----|-----|-----|-----|
| LLaVA-1.5 | 78.5 | 62.0 | 58.2 | 85.9 | 66.8 | 1510.7 | 64.3 | 66.5 |
| +RADIO | 76.3 | 63.0 | 56.3 | 86.2 | — | — | — | — |
| +MoVE-KD v1.0 | 79.5 | 63.2 | 58.3 | 86.9 | 69.3 | 1524.5 | 66.3 | 68.0 |
| **+MoVE-KD v1.1** | **79.9** | **63.9** | **59.6** | 86.3 | **69.8** | 1509.1 | **67.4** | — |

### 消融实验

| 配置（逐步添加） | VQAv2 | GQA | VizWiz | POPE | MMB | Avg |
|-----------------|-------|-----|--------|------|-----|-----|
| LLaVA-1.5 (baseline) | 78.5 | 62.0 | 50.0 | 85.9 | 64.3 | 66.5 |
| + KD (interpolation) | 79.0 | 62.4 | 50.9 | 84.7 | 62.9 | 66.3 |
| + Encoder adapter | 79.3 | 62.4 | 51.2 | 85.2 | 63.8 | 66.7 |
| + MoLE | 79.1 | 62.8 | 51.9 | **86.4** | 65.4 | 67.4 |
| + Token weight | 79.3 | 63.1 | 52.5 | 86.7 | 66.0 | 67.7 |
| **+ Teacher weight (full)** | **79.5** | **63.2** | **52.3** | **86.9** | **66.3** | **68.0** |

### 关键发现

- RADIO在VQAv2/TextVQA上明显遗忘（-2.2/-1.9），MoVE-KD则全面提升——说明MoLE有效缓解了知识冲突
- 单独引入MoLE但不做KD不会提升性能（Table 3），排除了"参数增加带来提升"的假设
- 直接解冻编码器微调会导致性能下降，证明MoVE-KD的提升不是因为encoder unfreezing
- v1.0→v1.1（增加SAM教师）进一步提升，尤其TextVQA (+1.3)，说明方法具有良好的教师扩展性
- CLIP教师权重0.8最优，过低（0.6）会导致原始CLIP知识遗忘过多

## 亮点与洞察

- **首创视角**：首次从KD角度解决VLM多编码器融合，比特征拼接更优雅且推理无额外开销
- **MoLE设计精巧**：LoRA作为MoE专家=参数效率+知识分治，仅0.3%参数增量
- **[CLS]注意力复用**：利用CLIP的[CLS]注意力作为蒸馏指导信号是一个巧妙的"免费午餐"
- **即插即用**：可直接应用于LLaVA/LLaVA-NeXT等主流VLM，无需修改推理架构

## 局限与展望

- 训练时仍需多个教师编码器前向传播，训练成本高（16×A800）
- [CLS]注意力完全依赖CLIP，对CLIP关注不到的区域蒸馏可能不足
- 仅在LLaVA系列验证，未在Qwen-VL/InternVL等架构上测试通用性
- MoLE目前仅top-1路由，未探索top-2或软路由策略
- TextVQA上偶有轻微下降，可能因为增强视觉对文本理解有反作用

## 相关工作与启发

- **AM-RADIO**：多head单backbone蒸馏，用DataComp-1B训练；MoVE-KD无需额外数据且效果更好
- **Eagle/Mini-Gemini**：使用额外编码器做高分辨率refinement，推理成本更高
- **OneS**：多教师LLM蒸馏的先驱，MoVE-KD将多教师KD引入VLM的视觉侧
- 启发：[CLS]注意力是被低估的信号——它编码了"图像中哪里重要"的信息，可广泛用于蒸馏/剪枝/token压缩等场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次在VLM视觉编码器上做多教师KD+MoLE，视角新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖LLaVA/LLaVA-NeXT/多规模(1.7B~13B)、8个benchmark、详细消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，消融设计合理
- 价值: ⭐⭐⭐⭐ 实用性强，即插即用提升VLM；教师扩展性好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Instruction-based Image Manipulation by Watching How Things Move](instruction-based_image_manipulation_by_watching_how_things_move.md)
- [\[CVPR 2026\] Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/uncertainty-aware_knowledge_distillation_for_multimodal_large_language_models.md)
- [\[ICCV 2025\] LLaVA-KD: A Framework of Distilling Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/llava-kd_a_framework_of_distilling_multimodal_large_language_models.md)
- [\[CVPR 2025\] Galaxy Walker: Geometry-aware VLMs For Galaxy-scale Understanding](galaxy_walker_geometry-aware_vlms_for_galaxy-scale_understanding.md)
- [\[CVPR 2025\] Harnessing Frozen Unimodal Encoders for Flexible Multimodal Alignment](harnessing_frozen_unimodal_encoders_for_flexible_multimodal_alignment.md)

</div>

<!-- RELATED:END -->
