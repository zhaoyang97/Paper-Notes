---
title: >-
  [论文解读] VSC: Visual Search Compositional Text-to-Image Diffusion Model
description: >-
  [ICCV 2025][图像生成][组合文本到图像生成] 提出 VSC，一种基于视觉搜索的组合文本到图像扩散生成方法，通过为每个属性-对象对单独生成参考图像并融合视觉原型嵌入，结合分割引导的交叉注意力定位训练，显著提升多属性-对象绑定的准确性和扩展性。 文本到图像扩散模型在生成逼真图像方面表现出色，但在处理包含多个属性-对象…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "组合文本到图像生成"
  - "属性-对象绑定"
  - "扩散模型"
  - "视觉嵌入融合"
  - "交叉注意力定位"
---

# VSC: Visual Search Compositional Text-to-Image Diffusion Model

**会议**: ICCV 2025  
**arXiv**: [2505.01104](https://arxiv.org/abs/2505.01104)  
**代码**: 未公开  
**领域**: 图像分割  
**关键词**: 组合文本到图像生成, 属性-对象绑定, 扩散模型, 视觉嵌入融合, 交叉注意力定位

## 一句话总结

提出 VSC，一种基于视觉搜索的组合文本到图像扩散生成方法，通过为每个属性-对象对单独生成参考图像并融合视觉原型嵌入，结合分割引导的交叉注意力定位训练，显著提升多属性-对象绑定的准确性和扩展性。

## 研究背景与动机

文本到图像扩散模型在生成逼真图像方面表现出色，但在处理包含多个属性-对象绑定对的复杂提示时常常失败。例如给定 "a red car and a yellow bicycle"，模型可能将颜色混淆或错误分配。

这一问题的根源在于 CLIP 等文本编码器的局限性：

**语言结构理解不足**：CLIP 倾向于将文本表示为"概念袋"（bag-of-concepts），忽略层次化关系和修饰词

**交叉注意力图错误**：扩散模型 UNet 的交叉注意力图本身就是不准确的，属性 token 的注意力分布与对象不对齐

**复杂度扩展性差**：现有推理时注意力控制方法（如 Attend-and-Excite）随绑定对数量增加性能急剧下降

作者从认知心理学的**连续搜索理论（Serial Search Theory）**获得灵感：人类在面对多特征对象时，大脑会逐个处理并绑定特征。同时观察到 CLIP 和扩散模型在**单绑定对**设置下已表现极好（SD2.1 单对 color/texture/shape 得分分别达 0.95/0.94/0.85），因此关键思路是：**先为每个属性-对象对独立生成参考图像，再将视觉信息融合到完整生成中**。

## 方法详解

### 整体框架

VSC 基于 Stable Diffusion，核心流程：
1. **分解**：将复杂提示分解为多个属性-对象子提示
2. **搜索**：为每个子提示使用 SD 生成 $m$ 张参考图像
3. **融合**：通过图像编码器提取视觉原型，经 MLP 与文本嵌入融合
4. **生成**：用增强后的文本嵌入驱动冻结的 SD 生成最终组合图像

### 关键设计一：视觉嵌入融合（Visual Embedding Fusion）

对每个绑定对 $[a_n, o_n]$ 生成 $m$ 张参考图像，用图像编码器 $\phi$ 提取嵌入并取均值作为视觉原型：

$$e_j = \frac{1}{m}\sum_{k=0}^{m} v_j^k = \frac{1}{m}\sum_{k=0}^{m} \phi(r_j^k), \quad j \in \mathcal{I}$$

然后通过可训练的 MLP 将视觉原型与对应位置的文本嵌入融合：

$$c'_i = \begin{cases} c_i, & i \notin \mathcal{I} \\ \text{MLP}([c_i, e_j]), & i = i_j \in \mathcal{I} \end{cases}$$

只有属于绑定对索引的 token 才会被视觉信息增强，其他 token 保持不变。

### 关键设计二：分割引导的交叉注意力定位训练

SD 的交叉注意力图是错误的，直接使用会导致属性错绑定。因此引入定位损失，让属性和对象的注意力图都趋近于分割掩码：

$$L_{\text{loc}} = \frac{1}{n}\sum_{j=1}^{n}\left[(\text{mean}(A_{i_a} \odot \bar{M}_j) - \text{mean}(A_{i_a} \odot M_j)) + (\text{mean}(A_{i_o} \odot \bar{M}_j) - \text{mean}(A_{i_o} \odot M_j))\right]$$

总训练目标：$L = L_{\text{noise}} + \lambda L_{\text{loc}}$

### 关键设计三：合成数据集构建

利用 SD3.5、Flux 和 SynGen 三个模型为 T2I-CompBench 的每个提示生成 300 张图像，使用 Mask2Former 做实例分割、OpenCLIP 计算对齐分数。选取最高 VQA 分数的 45 张/提示，最终得到 90,000 张图像及精确分割标注。

### 训练细节

- 仅训练 MLP 和图像编码器最后 2-3 层，SD UNet 全程冻结
- 训练 60,000 步，学习率 1e-5，batch size 8
- 支持 SD 1.4、2.1 和 3.5 三个版本

## 实验

### 主实验（T2I CompBench）

| 模型 | Color | Texture | Shape | HM |
|------|-------|---------|-------|----|
| **SD 2.1** | | | | |
| Baseline | 0.50 | 0.49 | 0.42 | 0.467 |
| Attn-Excite | 0.64 | 0.59 | 0.45 | 0.547 |
| SynGen | 0.71 | 0.61 | 0.50 | 0.594 |
| **VSC** | **0.74** | **0.64** | **0.53** | **0.608** |
| **SD 3.5** | | | | |
| Baseline | 0.76 | 0.67 | 0.53 | 0.639 |
| SynGen | 0.82 | 0.74 | 0.59 | 0.703 |
| **VSC** | **0.85** | **0.79** | **0.63** | **0.727** |

VSC 在所有 SD 版本上均取得最佳性能，SD 3.5 + VSC 达到最高 HM 0.727。

### 扩展性实验（不同绑定对数量）

| 方法 | 3对 | 4对 | 5对 |
|------|-----|-----|-----|
| SynGen (SD2.1) | 0.678 | 0.485 | 0.119 |
| **VSC (SD2.1)** | **0.688** | **0.548** | **0.190** |
| SynGen (SD3.5) | 0.808 | 0.566 | 0.182 |
| **VSC (SD3.5)** | **0.813** | **0.618** | **0.246** |

关键发现：
- VSC 的性能优势随绑定对数量增加而扩大（5对时 VSC 比 SynGen 提升 60%+）
- 这证实了分割引导的交叉注意力训练有效修复了注意力图的错误

### 消融分析

- **数据规模**：5K→30K 图像时性能稳步提升，30K→90K 提升趋缓
- **可迁移性**：仅用 Color 类别训练的模型在 Texture 和 Shape 上也优于 baseline，证明组合能力可跨属性迁移
- **图像编码器微调**：冻结图像编码器会导致定位损失无法收敛，微调最后 2-3 层是必要的
- **人类评估**：VSC 获得 30.01% 的多数票，显著优于 SynGen（25.18%）和 SD3.5（12.61%）

### 推理效率

VSC 通过并行生成参考图像实现高效扩展：推理时间约为 baseline 的 2.26 倍且保持恒定，而 SynGen 和 A&E 随绑定对数增加而显著增长。5 对时 VSC 已最快（除 baseline）。

## 亮点与洞察

1. **认知科学启发**：连续搜索理论在生成模型中的巧妙应用——先逐个"搜索"再统一合成
2. **训练极轻量**：仅需训练 MLP + 图像编码器最后几层，UNet 完全冻结
3. **扩展性优势**：与基于推理时注意力控制的方法不同，VSC 的定位损失在训练时已纠正注意力图，推理时无额外开销
4. **全合成数据训练**：不需要任何人工标注数据

## 局限性

1. 每个绑定对需要生成参考图像，每个额外提示需约 1.1GB VRAM
2. 合成数据的质量受参考模型能力限制
3. Shape 属性的提升幅度始终小于 Color 和 Texture
4. 依赖分割模型（Mask2Former）获取训练标注

## 相关工作

- **组合生成**：Attend-and-Excite 推理时控制注意力；SynGen 训练时优化交叉注意力损失
- **主体驱动生成**：DreamBooth、FastComposer 基于参考图像的个性化生成
- **布局引导生成**：GLIGEN、SpaText 利用边界框/分割图控制空间信息

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 视觉搜索+融合的思路新颖，但整体框架基于已有组件
- **技术质量**: ⭐⭐⭐⭐ — 实验全面含人类评估和扩展性测试，分析充分
- **实用性**: ⭐⭐⭐⭐ — 适用于多种 SD 版本，但推理需额外生成参考图像
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，心理学理论引入自然

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](holistic_unlearning_benchmark_a_multi-faceted_evaluation_for_text-to-image_diffu.md)
- [\[ICCV 2025\] Addressing Text Embedding Leakage in Diffusion-Based Image Editing](addressing_text_embedding_leakage_in_diffusion-based_image_editing.md)
- [\[NeurIPS 2025\] Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation](../../NeurIPS2025/image_generation/evaluating_the_evaluators_metrics_for_compositional_text-to-image_generation.md)
- [\[ICCV 2025\] CompSlider: Compositional Slider for Disentangled Multiple-Attribute Image Generation](compslider_compositional_slider_for_disentangled_multiple-attribute_image_genera.md)
- [\[ICCV 2025\] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
