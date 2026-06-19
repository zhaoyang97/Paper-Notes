---
title: >-
  [论文解读] Chimera: Improving Generalist Model with Domain-Specific Experts
description: >-
  [ICCV 2025][多模态VLM][多模态推理] 提出 Chimera，一个可扩展的低成本多模态管道，通过轻量路由模块动态选择领域专家模型、渐进式训练策略以及 Generalist-Specialist Collaboration Masking（GSCM）机制，将领域专家知识（表格、图表、数学、文档）集成到通用多模态大模型中，在 MathVista 上达到 64.9%（SOTA），在多个视觉结构提取任务上也达到或超越专家模型水平。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "多模态推理"
  - "专家模型融合"
  - "领域适配"
  - "路由机制"
  - "视觉内容提取"
---

# Chimera: Improving Generalist Model with Domain-Specific Experts

**会议**: ICCV 2025  
**arXiv**: [2412.05983](https://arxiv.org/abs/2412.05983)  
**代码**: 开源（权重、数据、评估）  
**领域**: 多模态大模型 / VLM  
**关键词**: 多模态推理, 专家模型融合, 领域适配, 路由机制, 视觉内容提取

## 一句话总结

提出 Chimera，一个可扩展的低成本多模态管道，通过轻量路由模块动态选择领域专家模型、渐进式训练策略以及 Generalist-Specialist Collaboration Masking（GSCM）机制，将领域专家知识（表格、图表、数学、文档）集成到通用多模态大模型中，在 MathVista 上达到 64.9%（SOTA），在多个视觉结构提取任务上也达到或超越专家模型水平。

## 研究背景与动机

大型多模态模型（LMM）在通用任务上表现出色，但在**专业领域任务**上仍显不足：

- **通用 LMM 的局限**：训练数据以自然图像为主，但专业任务涉及图表、表格、几何图形、函数图等，这些内容文本密度更高、更抽象
- **"One for One" 范式问题**：针对各领域训练的专家模型性能强但泛化差，不同子领域之间分布差距大
- **数据私有性问题**：领域特定数据通常是私有的，无法用于后训练 LMM

直接集成专家模型面临两个核心挑战：

**表征差距**：跨领域编码器之间的分布偏移大

**优化不平衡**：通用视觉编码器已与语言模型对齐良好，导致模型过度依赖通用编码器，忽视专家特征

## 方法详解

### 整体框架

Chimera 由以下组件构成：
- 通用视觉编码器 $E^g$ + 通用投影器 $P^g$ + 语言模型 $f$（初始化自预训练 LMM，如 InternVL2）
- 路由器 $R$（线性层，基于通用编码器的 CLS token 预测）
- 专家模型集 $S^e = \{E^{table}, E^{chart}, E^{math}\}$
- 专家投影器集 $S^p = \{P^{table}, P^{chart}, P^{math}\}$

推理流程：路由器根据视觉输入决定是否调用专家及调用哪个专家 → 专家特征与通用特征拼接 → 送入语言模型。

### 关键设计

1. **轻量路由模块（Router）**：

    - 输入通用编码器的 CLS token $\mathcal{Z}_v^{cls}$
    - 线性层输出 $N_e + 1$ 维预测值（$N_e$ 个专家 + 1 个不调用选项）
    - $i = \arg\max_i(\mathcal{H}_r)_i$，选择最高置信度的专家
    - 在 MathVista 上实现 95.4% 的路由准确率，所有错误均为"通用 vs 专家"混淆，专家之间无混淆
    - 路由损失采用分类交叉熵：$\mathcal{L}_c = -\sum_{i=0}^{N_e+1}\log P(c_i|\mathcal{X}_v, \theta)$

2. **Generalist-Specialist Collaboration Masking（GSCM）**：

    - **核心问题**：通用编码器已与语言模型充分对齐，直接拼接专家特征会导致模型"偷懒"——只用通用特征完成任务，忽视专家特征
    - **解决方案**：训练时按均匀分布随机采样一定比例的通用视觉 token，将其 attention mask 设为 False
    - 效果：强制模型利用专家模型提供的领域信息作为通用信息的补充
    - 均匀分布防止了因集中遮蔽图像中心或特定区域引入的偏置
    - 注意力分析验证：使用 GSCM 后，模型输出对专家视觉 token 的注意力显著增加

3. **渐进式训练策略**：

    - **阶段1 - 领域通用知识对齐**：冻结通用编码器、专家模型和语言模型，仅训练路由器、通用投影器和专家投影器
        - 数据：自然图像描述、表格格式转换、图表结构提取、数学图表描述、段落级OCR
    - **阶段2 - 视觉指令微调**：解冻路由器、投影器和语言模型（专家编码器始终冻结），应用 GSCM
        - 数据：多领域指令跟随数据集

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \mathcal{L}_c + \mathcal{L}_m$

- $\mathcal{L}_m$：自回归建模的 token 级交叉熵损失
- $\mathcal{L}_c$：路由器分类损失

偏好优化（可选阶段3）：使用 60K 公开数据构建偏好对，进行 DPO，将 Chimera 推升至 68.3%。

## 实验关键数据

### 主实验 (表格)

**MathVista testmini 准确率（%）**

| 模型 | 规模 | 整体 | GPS | MWP | TQA | GEO |
|------|------|------|-----|-----|-----|-----|
| GPT-4o | - | 63.8 | - | - | - | - |
| InternVL2-8B | 8B | 61.6 | 64.4 | 61.3 | 64.6 | 61.9 |
| Qwen2-VL | 7B | 58.2 | - | - | - | - |
| Math-LLaVA* | 13B | 46.6 | 57.7 | 56.5 | 51.3 | 56.5 |
| **Chimera-8B** | **8B** | **64.9** | **71.6** | **72.6** | **65.2** | **69.5** |
| **Chimera†-8B** | **8B** | **68.3** | **76.9** | **80.1** | 60.8 | **74.5** |

**MathVerse 准确率（%）**

| 模型 | 规模 | 整体 | Text Dominant | Vision Only |
|------|------|------|---------------|-------------|
| GPT-4V | - | 39.4 | 54.7 | 31.6 |
| MAVIS-7B* | 7B | 27.5 | 41.4 | 14.6 |
| InternVL2-8B | 8B | 31.3 | 38.8 | 17.0 |
| **Chimera-8B** | **8B** | **32.4** | **39.6** | **19.3** |

**视觉结构提取（ChartQA-SE AP@strict / Table-SE TEDS）**

| 方法 | ChartQA-SE AP@strict | Table-SE TEDS | Table-SE Edit Dist↓ |
|------|---------------------|---------------|---------------------|
| GOT* | 74.7 | **0.745** | 0.257 |
| InternVL-2 | 73.7 | 0.676 | 0.229 |
| **Chimera** | **74.1** | 0.740 | **0.165** |

### 消融实验 (表格)

**领域级别分析（MathVista testmini）**

| 模型 | 整体 | 通用 | 图表 | 表格 | 数学 |
|------|------|------|------|------|------|
| InternVL2-2B | 48.3 | 45.3 | 58.9 | 50.0 | 44.2 |
| Chimera-2B | 53.1 | 46.0 | 60.3 | 62.9 | 56.1 |
| InternVL2-4B | 57.0 | 50.1 | 66.2 | 65.7 | 58.3 |
| Chimera-4B | 61.3 | 54.0 | 64.8 | 72.9 | 66.9 |
| InternVL2-8B | 61.6 | 52.7 | 71.2 | 67.1 | 66.5 |
| Chimera-8B | 64.9 | 57.5 | 71.2 | 62.9 | 71.9 |

**路由器错误统计（MathVista testmini）**

| 真实\预测 | 通用 | 表格 | 图表 | 数学 |
|---------|------|------|------|------|
| 通用 | – | 0 | 16 | 6 |
| 表格 | 1 | – | 0 | 0 |
| 图表 | 1 | 0 | – | 0 |
| 数学 | 22 | 0 | 0 | – |

路由准确率：95.4%。所有误分类都是通用↔专家之间，专家间无混淆。

### 关键发现

- Chimera-8B 在 MathVista 上以 64.9% 超越 GPT-4o（63.8%），设立了 70B 以下开源 LMM 的新 SOTA
- **专家知识甚至提升了通用场景表现**：Chimera 在"通用"类别上也优于 InternVL2 基线（57.5% vs 52.7%），说明领域知识提供了多样化视角
- DPO 后训练仅用 60K 数据即可将性能从 64.9% 提升到 68.3%（+3.4%），展示了框架的可扩展性
- GSCM 注意力分析验证：有 masking 时专家 token 获得显著更多注意力，无 masking 时模型几乎完全依赖通用编码器
- 数学专家一致性最强——跨所有模型规模都带来提升；表格专家过于专业化可能在 8B 模型上引入噪声
- Chimera 在文档结构提取（Doc-SE）的英文和中文任务上均大幅领先 InternVL2

## 亮点与洞察

- **"One for All + Domain Experts" 范式**：既不牺牲通用性，又能获得专业能力，比纯专家模型或纯通用模型更有实用价值
- **GSCM 机制的洞察深刻**：识别并解决了"预训练对齐优势导致的优化不平衡"这一非直觉问题
- **极简路由设计**：仅一个线性层即可达到 95.4% 的路由准确率，说明不同领域的视觉内容在特征空间中已有很好的区分性
- **DPO 的高效集成**：展示了 Chimera 作为基础框架可以与偏好优化无缝结合

## 局限与展望

- Chimera-8B 在表格领域反而低于 InternVL2-8B（62.9% vs 67.1%），因为表格专家（StructEqTable）过于专业化，任务差距引入噪声
- 路由标签基于数据集级别而非图像级别标注，"通用"类别可能包含混合领域图像
- 当前仅集成了表格、图表、数学三个领域的专家，扩展到更多领域的效果需验证
- 专家模型始终冻结，限制了领域知识的进一步适配
- 推理时需要运行所有编码器（通用+选中的专家），计算开销大于纯通用模型

## 相关工作与启发

- **InternVL2**：Chimera 的基础模型，渐进式对齐策略借鉴了其预训练-微调范式
- **GOT**：文档结构提取的专家模型，使用百万级私有数据训练
- **ChartVLM**：图表 QA 的路由结构启发了 Chimera 的路由设计
- **MAVIS / Math-LLaVA**：数学推理专家模型，展示了领域专精的重要性但缺乏跨领域泛化
- **MoE 思想**：Chimera 的路由+专家设计与 Mixture of Experts 有相似之处，但更轻量且面向预训练专家模型的集成

## 评分

- **新颖性**: ⭐⭐⭐⭐ — GSCM 机制解决了一个重要但容易被忽视的优化不平衡问题，"融合现有专家"的思路实用且可扩展
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖推理和提取两大场景，2B/4B/8B三个规模，多个基准（MathVista/MathVerse/ChartQA-SE/Table-SE/Doc-SE），领域级分析细致
- **写作质量**: ⭐⭐⭐⭐ — 框架图清晰，渐进式方法描述合理，训练策略有层次
- **价值**: ⭐⭐⭐⭐⭐ — 为 LMM 快速获取领域能力提供了低成本、高效的框架方案，使用公开数据和模型即可复现，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy](dita_scaling_diffusion_transformer_for_generalist_visionlang.md)
- [\[ICCV 2025\] A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition](a_qualityguided_mixture_of_scorefusion_experts_framework_for.md)
- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)
- [\[CVPR 2025\] From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons](../../CVPR2025/multimodal_vlm/from_multimodal_llms_to_generalist_embodied_agents_methods_and_lessons.md)

</div>

<!-- RELATED:END -->
