---
title: >-
  [论文解读] Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains
description: >-
  [ACL 2025][多模态VLM][多图推理] 提出 Focus-Centric Visual Chain 多图推理范式，通过问题分解和逐步聚焦关键视觉信息实现跨图推理，并构建 VISC-150K 数据集，在七个多图基准上实现 2-3% 的一致性提升。 视觉语言模型（VLM）在单图任务上已达到人类水平…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "多图推理"
  - "视觉语言模型"
  - "数据合成"
  - "Chain-of-Thought"
  - "多模态推理"
---

# Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains

**会议**: ACL 2025  
**arXiv**: [2504.20199](https://arxiv.org/abs/2504.20199)  
**代码**: [GitHub - VISC](https://github.com/zhangjuntian/VISC)  
**领域**: 多模态VLM  
**关键词**: 多图推理, 视觉语言模型, 数据合成, Chain-of-Thought, 多模态推理

## 一句话总结

提出 Focus-Centric Visual Chain 多图推理范式，通过问题分解和逐步聚焦关键视觉信息实现跨图推理，并构建 VISC-150K 数据集，在七个多图基准上实现 2-3% 的一致性提升。

## 研究背景与动机

视觉语言模型（VLM）在单图任务上已达到人类水平，但在多图场景中性能显著下降。多图任务的两大挑战：

**跨图关联（Cross-image correlations）**：图像之间存在多样化的关系（时间、空间、语义），需要整体理解其上下文联系

**视觉不连续性（Visual discontinuity）**：信息在图像之间碎片化分布，使得准确捕捉跨图关系变得困难

现有解决方案的不足：
- 直接用多模态模型生成推理链的可靠性不足，即使 GPT-4o 在多图任务上也表现不稳定
- 从更强模型蒸馏数据成本高昂，难以扩展
- 现有多图推理数据极度稀缺

## 方法详解

### 整体框架

方法包含两部分：（1）Focus-Centric Visual Chain 推理范式——通过问题分解和逐步聚焦进行多步推理；（2）Focus-Centric Data Synthesis（FCDS）框架——自底向上合成高质量多图推理数据。

### 关键设计

1. **Focus-Centric Visual Chain 推理范式**：给定图像集合 $\mathcal{G} = \{I_k\}$ 和问题 $Q$，模型 $\mathcal{M}$ 逐步构建推理链 $\mathcal{R}$。在第 $i$ 步，模型生成子问题 $q_i$ 并识别对应的视觉焦点子集 $G_i$（最小化的视觉信息子集），通过联合分析 $q_i$ 和 $G_i$ 得到中间答案 $a_i$。模型动态判断是否继续推理（停止信号 $z_i$），最终从所有 QA 对合成最终答案。核心思想是将复杂多图任务分解为一系列聚焦局部视觉输入的子任务。

2. **Feature Extraction 模块**：为每张图像构建详细的文本画像（profile），包含总体视图、背景描述、物体属性和物体交互四个要素。使用 LLaVA-OneVision-7B 作为 Extractor 基础模型，通过视觉编码器 $f_e$、视觉-语言连接器 $f_c$ 和 LLM $f_\phi$ 三部分生成画像。

3. **Pair Connection 模块**：通过两个标准判断图像节点间的关联：（1）物体导向——图像共享相同物体；（2）事件导向——图像描述相关事件。使用 Qwen2.5-7B-Instruct 作为 Connector，基于画像集合识别有效的成对连接。

4. **Relevance Annotation 模块**：将图像对间的关联分为三类：

    - **时间关系（Temporal）**：图像描述时间序列
    - **空间关系（Spatial）**：视觉元素呈现几何和位置关联
    - **语义关系（Semantic）**：包含主题、逻辑和因果关联的抽象联系
   使用 LLaVA-OneVision-7B 作为 Annotator，对每对连接图像进行关系标注。

5. **Question Generation 模块**：沿推理路径采样连续的节点链，为每对连接图像基于其关系标注和画像生成子问题，最终合成综合性问题。使用 Qwen2.5-7B-Instruct 作为 Questioner。这种自底向上的设计确保数据质量的同时保持计算效率，全程仅使用开源模型。

### 训练策略

- 基于 LLaVA-OneVision-7B 和 Qwen2-VL-7B-Instruct 进行 LoRA 微调
- 在 VISC-150K 上训练 1 个 epoch，batch size 8，学习率 1e-5
- Warmup ratio 0.05，余弦调度器
- 最大上下文长度 32,768

## 实验关键数据

### 主实验

| 模型 | MMIU | MuirBench | MIRB | BLINK | NLVR2 | Mantis-Eval | MVBench |
|------|------|-----------|------|-------|-------|-------------|---------|
| LLaVA-OneVision-7B | 40.32 | 41.77 | 51.18 | 48.20 | 89.40 | 64.20 | 56.70 |
| +VISC-150K | **46.52**(↑6.20) | **49.62**(↑7.85) | **53.02**(↑1.84) | **50.24**(↑2.04) | **89.88**(↑0.48) | **66.36**(↑2.16) | **58.23**(↑1.53) |
| Qwen2-VL-7B | 50.00 | 39.12 | 58.67 | 53.20 | 86.42 | 69.60 | 67.00 |
| +VISC-150K | **52.76**(↑2.76) | **44.50**(↑5.38) | **60.16**(↑1.49) | **55.34**(↑2.14) | **89.82**(↑3.40) | 69.12(↓0.48) | **68.01**(↑1.01) |

### 消融实验

| 实验问题 | 关键结果 | 说明 |
|---------|---------|------|
| 数据规模影响（RQ1） | 0→25K 快速提升，125K→150K 渐趋收敛 | 25K 数据即可激活多图推理能力 |
| 子任务效果（RQ2） | 12个MuirBench子任务中8个显著提升 | 相似性分析和比较推理提升最大 |
| 输入图像数（RQ3） | 3-8张图提升最显著，15+张略有退化 | 中等规模图像集受益最大 |
| 通用能力影响（RQ4） | 4个单图基准保持或略有提升 | 不牺牲通用视觉语言能力 |
| 数据质量（RQ5） | 97.5% 整体准确率（3人评审） | Fleiss' κ=0.637，可靠性较高 |

### 关键发现

- LLaVA-OneVision 在 MMIU 和 MuirBench 上分别提升 6.20% 和 7.85%
- 在 7 个基准中的 4 个上建立新 SOTA（MMIU、MIRB、BLINK、NLVR2）
- 即使已经强大的 Qwen2-VL 也获得平均 2.24% 的提升
- 方法在视频基准 MVBench 上也有提升，证明范式的领域无关性
- 数据合成过程全部使用开源模型，97.5% 准确率验证了方法的可靠性

## 亮点与洞察

- **从推理范式到数据合成的完整闭环**：推理范式（自顶向下分解）和数据合成（自底向上构建）形成对偶设计，逻辑上非常优雅
- **纯开源方案**：数据合成全部使用 7B 级别的开源模型，成本可控且可复现
- **跨架构一致性**：在 LLaVA-OneVision 和 Qwen2-VL 两种不同架构上均有提升，证明数据的通用价值
- **不损害通用能力**：在 HallusionBench、MMStar 等单图基准上保持甚至略有提升
- **25K 数据激活效应**：少量数据即可解锁模型的多图推理潜力，暗示这是一种"能力激活"而非新能力学习

## 局限与展望

- 成对图像关联标注的二次复杂度限制了图像集规模的扩展
- 数据集主要覆盖真实照片和漫画，对图表、代码截图等结构化视觉内容的效果未经验证
- 推理步数受限于基础语言模型的内在能力
- 复杂空间动态理解和领域专业知识性视觉任务仍然是短板
- 输入超过 15 张图像时性能略有退化，长序列图像处理仍需改进

## 相关工作与启发

- 与 Chain-of-Thought 在文本推理中的成功类似，Visual Chain 将分步推理引入多图视觉理解
- 数据合成框架的自底向上设计避免了依赖闭源模型的高成本和不可靠性
- 启发：多图推理的核心在于动态选择性注意（selective attention），Focus-Centric 机制本质上实现了这一能力
- 未来方向：将该范式扩展到视频理解（已有初步验证）和更长序列的图像理解

## 评分

- 新颖性: ⭐⭐⭐⭐ Focus-Centric 推理范式和自底向上数据合成是新颖的组合创新
- 实验充分度: ⭐⭐⭐⭐⭐ 7个基准、2种架构、5个研究问题、人工质量评估，非常充分
- 写作质量: ⭐⭐⭐⭐ 方法描述系统化，公式形式化清晰，但部分符号较冗余
- 价值: ⭐⭐⭐⭐ 高质量的 150K 数据集和有效的推理范式，对多图 VLM 研究有重要贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Vision-Language Models Struggle to Align Entities across Modalities](vision-language_models_struggle_to_align_entities_across_modalities.md)
- [\[ACL 2025\] Jailbreak Large Vision-Language Models Through Multi-Modal Linkage](jailbreak_large_vision-language_models_through_multi-modal_linkage.md)
- [\[ACL 2025\] Symmetrical Visual Contrastive Optimization: Aligning Vision-Language Models with Minimal Contrastive Images](symmetrical_visual_contrastive_optimization_aligning_visionlanguage.md)
- [\[ACL 2025\] Performance Gap in Entity Knowledge Extraction Across Modalities in Vision Language Models](performance_gap_in_entity_knowledge_extraction_across_modalities_in_vision_langu.md)
- [\[ACL 2025\] Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback](improving_medical_large_vision-language_models_with_abnormal-aware_feedback.md)

</div>

<!-- RELATED:END -->
