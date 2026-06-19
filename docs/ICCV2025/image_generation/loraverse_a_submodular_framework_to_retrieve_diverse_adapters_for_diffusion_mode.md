---
title: >-
  [论文解读] LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models
description: >-
  [图像生成] 将从100K+ LoRA适配器库中检索相关且多样化的LoRA组合建模为组合优化问题，提出基于子模函数最大化的LoRAverse框架，通过概念提取+子模检索实现兼顾相关性和多样性的LoRA选择。 CivitAI等平台托管着超过100K个LoRA模型，每个针对特定风格/主题/角色微调。用户面临三重挑战： 数量爆炸：…
tags:
  - "图像生成"
---

# LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2510.15022](https://arxiv.org/abs/2510.15022)
- **代码**: 未公开
- **领域**: 图像生成
- **关键词**: LoRA模型检索, 子模优化, 多样性, 概念提取, CivitAI

## 一句话总结
将从100K+ LoRA适配器库中检索相关且多样化的LoRA组合建模为组合优化问题，提出基于子模函数最大化的LoRAverse框架，通过概念提取+子模检索实现兼顾相关性和多样性的LoRA选择。

## 研究背景与动机

CivitAI等平台托管着超过100K个LoRA模型，每个针对特定风格/主题/角色微调。用户面临三重挑战：

**数量爆炸**：无法手动浏览评估海量LoRA

**冗余选择**：简单的余弦相似度检索返回高度相似的模型，缺乏多样性

**概念对齐**：用户提示包含多个概念，需要精确关联不同LoRA到对应概念

现有方法Stylus使用top-K排名+LLM筛选，但top-K固有地倾向相似模型，LLM进一步引入输入偏差，限制了多样性的上限。

**核心思路**：将LoRA选择建模为单调子模函数最大化问题——这保证贪心算法可在$(1-1/e)$近似比下求解。

## 方法详解

### 整体框架：概念提取器 + 子模检索器

### 1. 概念提取器

使用LLM将用户提示分解为不重叠的概念块：

$$\mathcal{C}(s) = \{t_i | t_i \in \mathcal{T}(s), t_i \subseteq s\}$$

例如"一只英短猫在樱花园中玩耍" → ["英短猫", "樱花园"]

### 2. 子模检索器

**相关性目标**（模函数，因此子模）：
$$\mathcal{F}_{\text{relevance}}(\mathcal{P}) = \sum_{a_i \in \mathcal{P}} \mathcal{F}_{\text{sim}}(\phi(a_i), \phi(s))$$

**多样性目标**（利用聚类的子模函数）：
$$\mathcal{F}_{\text{diversity}}(\mathcal{P}) = \sum_{k=1}^{K} \log\left(1 + \sum_{a_i \in \mathcal{C}_k \cap \mathcal{P}} \mathcal{F}_{\text{reward}}(\phi(a_i))\right)$$

$\log(1+\cdot)$的凹性保证了递减收益性质：从已覆盖的聚类中选择额外模型的边际增益递减。

**总目标**：
$$\mathcal{F}(\mathcal{P}) = \lambda_1 \mathcal{F}_{\text{relevance}}(\mathcal{P}) + \lambda_2 \mathcal{F}_{\text{diversity}}(\mathcal{P})$$

论文证明了此函数的子模性，贪心算法保证$(1-1/e) \approx 0.63$的近似比。

### 安全检查

使用GPT-4o作为适配器安全检查器，过滤包含不当内容的LoRA。

## 实验

### 定量对比（CFG=7，Realistic-Vision-v6检查点）

| 方法 | CLIP↑ | TCE↑ | TIE↑ | I2I↓ | 用户偏好↑ |
|------|-------|------|------|------|---------|
| SD v1.5 | 25.88 | 19.43 | 38.12 | 0.846 | 22.55% |
| Stylus | 25.41 | 20.30 | 38.53 | 0.825 | 29.90% |
| **LoRAverse** | 25.07 | **22.63** | **40.06** | **0.784** | **47.55%** |

- TCE（截断CLIP熵）提升16.5%：语义多样性大幅增强
- TIE（截断Inception熵）提升5.1%：视觉多样性提升
- I2I降低7.3%：图像间相异度更高
- CLIP仅降3.1%：多样性提升的同时文本对齐基本保持

### 消融：检索算法对比

| 方法 | CLIP↑ | TCE↑ | TIE↑ | I2I↓ |
|------|-------|------|------|------|
| 余弦相似度 | **24.67** | 23.47 | 41.99 | 0.781 |
| **子模检索** | 24.50 | **23.97** | **42.43** | **0.762** |

子模方法在多样性指标上全面胜出，仅以0.7% CLIP代价换取全方位多样性提升。

### 推理时间分析

LoRAverse额外开销约26s，主要来自聚类（23.2s）。此开销固定不随批量增大，大批量场景下边际成本趋零。

## 亮点与洞察

1. **优雅的数学建模**：将直觉上的"相关且多样"精确转化为子模优化，有理论保证
2. **概念级检索**：不对整个提示做单一检索，而是分解概念后分别检索再组合
3. **VLM-as-Judge**：使用GPT-4o评估多样性/质量/文本对齐的三维指标
4. **鲁棒性**：对聚类数和概念数均不敏感，超参数$\lambda_1=7.0, \lambda_2=1.0$即可

## 局限性

- 聚类质量影响最终结果：若相似LoRA被分到不同聚类，会降低有效多样性
- 多LoRA组合可能导致风格漂移，需要debias提示缓解
- 检索的LoRA可能放大训练数据中的社会偏见

## 相关工作

- **LoRA检索**: Stylus, RAG-based方法
- **子模优化**: 经典的diversity-promoting子集选择
- **扩散模型个性化**: DreamBooth, LoRA

## 评分
- 新颖性：★★★★☆ — 子模框架+概念提取的组合新颖
- 技术深度：★★★★☆ — 有完整的子模性证明
- 实用性：★★★★☆ — 直接面向CivitAI用户的实际痛点

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Trans-Adapter: A Plug-and-Play Framework for Transparent Image Inpainting](trans-adapter_a_plug-and-play_framework_for_transparent_image_inpainting.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [\[CVPR 2025\] EasyCraft: A Robust and Efficient Framework for Automatic Avatar Crafting](../../CVPR2025/image_generation/easycraft_a_robust_and_efficient_framework_for_automatic_avatar_crafting.md)
- [\[ICCV 2025\] Less is More: Improving Motion Diffusion Models with Sparse Keyframes](less_is_more_improving_motion_diffusion_models_with_sparse_keyframes.md)
- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](../../CVPR2026/image_generation/tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)

</div>

<!-- RELATED:END -->
