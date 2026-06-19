---
title: >-
  [论文解读] Mitigating Sexual Content Generation via Embedding Distortion in Text-conditioned Diffusion Models
description: >-
  [NeurIPS 2025][图像生成][不安全内容缓解] 提出Distorting Embedding Space (DES)，一种基于文本编码器的防御框架，通过将不安全嵌入变换到安全区域、保持安全嵌入不变、中和"裸露"语义三管齐下，在FLUX.1和SD v1.5上实现SOTA的性内容缓解效果（ASR分别降至9.47%和0.52%），同时保持良好的良性图像质量。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "不安全内容缓解"
  - "嵌入空间扭曲"
  - "文本编码器"
  - "对抗攻击防御"
  - "NSFW过滤"
---

# Mitigating Sexual Content Generation via Embedding Distortion in Text-conditioned Diffusion Models

**会议**: NeurIPS 2025  
**arXiv**: [2501.18877](https://arxiv.org/abs/2501.18877)  
**代码**: 暂无  
**领域**: 扩散模型 / 安全防护  
**关键词**: 不安全内容缓解, 嵌入空间扭曲, 文本编码器, 对抗攻击防御, NSFW过滤

## 一句话总结

提出Distorting Embedding Space (DES)，一种基于文本编码器的防御框架，通过将不安全嵌入变换到安全区域、保持安全嵌入不变、中和"裸露"语义三管齐下，在FLUX.1和SD v1.5上实现SOTA的性内容缓解效果（ASR分别降至9.47%和0.52%），同时保持良好的良性图像质量。

## 研究背景与动机

扩散模型（SD、DALL-E等）虽然生成能力强大，但可能被滥用生成色情、NSFW内容。现有防御方法各有不足：

**过滤方法**（黑名单文本过滤、安全检查器）：容易被恶意提示词绕过

**概念移除方法**（ESD、SalUn）：修改U-Net导致生成质量下降或对对抗攻击不鲁棒

**性内容缓解方法**（SafeGen、ShieldDiff）：SafeGen产生明显伪影，ShieldDiff未经对抗攻击评估

关键观察：概念相关参数分散在U-Net各层中，难以精准移除。而文本编码器中的属性存储在局部化组件中，更适合作为干预对象。进一步的insight来自持续学习：特征位置的维持能减少灾难性遗忘，反过来，**控制特征远离其原始位置**可能有效移除不安全信息。

## 方法详解

### 整体框架

DES分为两个阶段：(1) 目标向量生成：为每个不安全提示计算最优的安全变换目标；(2) 训练：微调文本编码器以扭曲不安全嵌入空间同时保留安全嵌入。

### 关键设计

1. **目标向量生成 (Target Vector Generation)**

   为每个不安全向量 $u_i$ 找到相似度最低的安全向量：
    $s_i^* = \arg\min_{s_i} \frac{u_i \cdot s_i}{\|u_i\|\|s_i\|}$
   
   然后减去"裸露"方向（$n$ 为"nudity"向量），生成反相关目标向量：
    $t_i = s_i^* - \alpha \frac{n}{\|n\|}$
   
   其中 $\alpha$ 为缩放因子（$\alpha=200$）。设计动机：选择最不相似的安全向量作为基础，再减去裸露方向确保目标向量与裸露概念反相关，增大嵌入空间扭曲程度提高鲁棒性。作者观察到即使被选中的安全向量也与裸露向量存在正相关，因此减法操作是必要的。

2. **不安全嵌入空间扭曲 + 安全嵌入保护**

   **不安全损失**：将当前不安全向量对齐到目标安全向量：
    $\mathcal{L}_u = \frac{1}{B}\sum_{i=1}^B \left(1 - \frac{\tilde{u}_i \cdot t_i}{\|\tilde{u}_i\|\|t_i\|}\right)$
   
   **安全损失**（带自适应调节）：维持安全向量与原始向量的相似度，并通过裸露集成向量 $\tilde{s}'_i = \tilde{s}_i + \alpha\frac{n}{\|n\|}$ 实现自适应权重：
    $\mathcal{L}_s = \frac{1}{B}\sum_{i=1}^B \left[\left(1 - \frac{\tilde{s}_i \cdot s_i}{\|\tilde{s}_i\|\|s_i\|}\right) + \left(1 - \frac{\tilde{s}'_i \cdot s_i}{\|\tilde{s}'_i\|\|s_i\|}\right)\right]$
   
   自适应机制：与裸露向量相关度低的安全向量获得更大的保留损失，相关度高的安全向量则被温和调整（因为它们可能包含隐性不安全语义）。

3. **裸露嵌入中和 (Nudity Neutralization)**

   将"nudity"向量对齐到中性空向量 $e_0$（对应空字符串""的嵌入）：
    $\mathcal{L}_n = 1 - \frac{\tilde{n} \cdot e_0}{\|\tilde{n}\|\|e_0\|}$
   
   动机：防止基于概念提取的攻击（如Ring-A-Bell使用遗传算法找到与裸露概念相似的提示）。中和后攻击者只能提取到语义无意义的嵌入。

### 损失函数 / 训练策略

总损失：$\mathcal{L}_t = \lambda \mathcal{L}_s + (1-\lambda)(\mathcal{L}_u + \mathcal{L}_n)$，$\lambda=0.3$

三个损失互补不冲突：裸露中和操作于当前"nudity"向量，不安全损失使用预计算的裸露向量做目标偏移，安全损失也使用预计算值做相似度计算。

训练极其高效：仅需90秒，推理零开销。训练数据：CoPro数据集的6911组安全-不安全提示对。

## 实验关键数据

### 主实验

**I2P显式提示防御（SD v1.5, NudeNet检测）：**

| 方法 | 裸露Total↓ | FID↓ | CLIP↑ |
|------|-----------|------|-------|
| SD v1.5 (无防御) | 851 | 16.57 | 26.46 |
| SLD-strong | 511 | 31.38 | 24.61 |
| Safe-CLIP | 404 | 17.49 | 25.73 |
| UCE | 216 | 16.99 | 26.16 |
| SalUn | 21 | 21.14 | 24.78 |
| AdvUnlearn | 27 | 18.94 | 23.82 |
| **DES** | **16** | **15.44** | **25.52** |

**对抗提示防御（黑盒攻击，SD v1.5, ASR↓）：**

| 方法 | Sneaky | MMA | Ring-A-Bell | P4D | 平均ASR↓ |
|------|--------|-----|-------------|-----|----------|
| SD v1.5 | 45.16 | 73.93 | 98.13 | 94.93 | 78.04 |
| AdvUnlearn | 1.61 | 2.10 | 0.93 | 1.10 | 1.44 |
| **DES** | **0.00** | **0.40** | **0.93** | **0.74** | **0.52** |

**FLUX.1上：** DES平均ASR 8.86% vs EraseAnything 43.23%，降低约80%。

### 消融实验

| 配置 | 作用 | 效果说明 |
|------|------|---------|
| 仅 $\mathcal{L}_u$ | 扭曲不安全嵌入 | 有效但安全图像质量下降 |
| $\mathcal{L}_u + \mathcal{L}_s$ | 加安全保留 | FID和CLIP score恢复 |
| $\mathcal{L}_u + \mathcal{L}_s + \mathcal{L}_n$ | 加裸露中和 | 对提取式攻击更鲁棒 |
| $\alpha$ 缩放因子 | 控制目标偏移程度 | $\alpha=200$最优 |

**白盒自适应攻击：**

| 方法 | MMA↓ | UDA↓ | Ring-A-Bell↓ | CCE↓ | 平均↓ |
|------|------|------|-------------|------|-------|
| ESD | 8.50 | 60.56 | 26.17 | 18.12 | 28.34 |
| AdvUnlearn | 2.73 | 19.72 | 0.00 | 6.15 | 7.15 |
| **DES** | **1.82** | **18.31** | **0.00** | 5.76 | **6.47** |

### 关键发现

- DES在所有攻击类型上均实现SOTA或接近SOTA的ASR，且跨攻击类型的方差极小（std 0.41）
- 关键优势在于生成质量保持：FID 15.44实际上优于原始SD v1.5的16.57（可能因为移除了影响FID的不安全内容）
- 文本编码器级干预优于U-Net级：AdvUnlearn和DES均优于ESD和UCE
- DES在FLUX.1（多文本编码器架构）上也有效，独立训练每个编码器
- 训练仅需90秒，推理无额外开销，是目前最高效的防御方案

## 亮点与洞察

- 从持续学习的"特征位置影响遗忘"推导出"扭曲不安全特征位置实现遗忘"的insight非常巧妙
- 三重损失设计（扭曲+保留+中和）互补且不冲突，形成完整的嵌入空间控制体系
- 安全损失中的自适应权重机制体现了对嵌入空间结构的深入理解
- 训练90秒 + 推理零开销的极致效率使其具有即时部署价值

## 局限与展望

- 目标向量生成依赖预定义的安全/不安全提示集，覆盖范围可能有限
- "nudity"单向量中和可能过于简化，性相关语义可能分布在多个维度
- 对新型攻击（如嵌入空间插值攻击）的鲁棒性待验证
- I2I任务上ASR仍有20%左右，有改进空间

## 相关工作与启发

- 与AdvUnlearn的对比凸显了嵌入空间控制优于对抗训练的优势（后者损害生成质量）
- 目标向量的减法操作（减去裸露方向）可推广到其他概念移除任务
- 自适应安全损失的设计可启发其他选择性遗忘/保留的任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ 嵌入空间扭曲的思路新颖，自适应安全损失和裸露中和设计精巧
- **实验充分度**: ⭐⭐⭐⭐⭐ 多种攻击场景（显式/黑盒/白盒/自适应）、两个模型、T2I+I2I全面评估
- **写作质量**: ⭐⭐⭐⭐ 方法展示清晰，但安全相关的内容处理需谨慎
- **价值**: ⭐⭐⭐⭐⭐ 90秒训练+零开销推理+SOTA防御性能，对实际部署极具价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models](diffusion_adaptive_text_embedding_for_texttoimage_diffusion.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] Text Embedding Knows How to Quantize Text-Guided Diffusion Models](../../ICCV2025/image_generation/text_embedding_knows_how_to_quantize_text-guided_diffusion_models.md)
- [\[NeurIPS 2025\] DiffEye: Diffusion-Based Continuous Eye-Tracking Data Generation Conditioned on Natural Images](diffeye_diffusion-based_continuous_eye-tracking_data_generation_conditioned_on_n.md)
- [\[NeurIPS 2025\] Mitigating Intra- and Inter-modal Forgetting in Continual Learning of Unified Multimodal Models](mitigating_intra-_and_inter-modal_forgetting_in_continual_learning_of_unified_mu.md)

</div>

<!-- RELATED:END -->
