---
title: >-
  [论文解读] VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning
description: >-
  [NeurIPS 2025][多模态VLM][少样本学习] 提出VT-FSL框架，通过跨模态迭代提示（CIP）联合利用类名和支持图像驱动LLM生成精确文本描述并零样本合成语义一致图像，再通过核化体积对比学习（CGA）实现全局非线性跨模态对齐，在10个少样本学习基准上平均提升4.2%分类准确率。 问题背景 少样本学习（FSL）…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "少样本学习"
  - "大语言模型"
  - "跨模态对齐"
  - "文本提示"
  - "视觉合成"
  - "对比学习"
---

# VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning

**会议**: NeurIPS 2025  
**arXiv**: [2509.25033](https://arxiv.org/abs/2509.25033)  
**作者**: Wenhao Li, Qiangchang Wang, Xianjing Meng, Zhibin Wu, Yilong Yin (山东大学, 深圳Loop Area研究院, 山东财经大学)  
**代码**: [GitHub](https://github.com/peacelwh/VT-FSL)  
**领域**: 多模态VLM  
**关键词**: 少样本学习, 大语言模型, 跨模态对齐, 文本提示, 视觉合成, 对比学习  

## 一句话总结

提出VT-FSL框架，通过跨模态迭代提示（CIP）联合利用类名和支持图像驱动LLM生成精确文本描述并零样本合成语义一致图像，再通过核化体积对比学习（CGA）实现全局非线性跨模态对齐，在10个少样本学习基准上平均提升4.2%分类准确率。

## 研究背景与动机

### 问题背景
少样本学习（FSL）旨在仅用少量标注样本识别新类别。度量学习方法通过构建类原型进行最近邻分类，但有限样本导致原型偏离真实类中心。引入文本模态的语义信息是改进原型表示的重要思路。

### 已有工作的不足
- **类名信息不足**：AM3、SP等方法仅利用类名，提供的上下文信息极其有限
- **语义幻觉问题**：SemFew、ECER等方法虽利用LLM生成更丰富的描述，但仅以类名为条件，忽略了支持图像的视觉模式，导致生成文本与实际视觉证据不一致，需要额外的人工或算法矫正
- **朴素提示策略**：现有方法对LLM使用简单输入提示，未充分发挥LLM的推理和生成能力
- **对齐方式局限**：CLIP风格的成对对比学习仅将每个表示对齐到单一锚点，忽略多模态间的全局结构关系

### 核心动机
同时利用类名和支持图像生成视觉接地的文本描述，消除语义幻觉；生成互补的文本和视觉提示，分别提供高层类语义和低层类内多样性；通过几何感知对齐实现全特征的一致性融合。

## 方法详解

### 整体框架
VT-FSL由两个核心模块组成：跨模态迭代提示（CIP）和跨模态几何对齐（CGA）。CIP负责生成文本和视觉提示，CGA负责将所有模态表示进行全局对齐。

### 跨模态迭代提示（CIP）
CIP联合利用类标签和$K$-shot支持图像，通过结构化推理生成精确且视觉接地的类描述。受Chain-of-Thought启发，将生成过程分解为四个阶段：

1. **策略（Strategy）**：概述问题，明确需要从支持图像中提取的关键视觉属性
2. **感知（Perception）**：解读支持图像中的视觉模式，提取共享的类别特征
3. **精化（Refinement）**：逐步推理，消除不一致信息，优化文本描述质量
4. **结论（Conclusion）**：生成最终的精确类描述

每个阶段使用结构化标签标记，整个过程在**单次推理**中完成，无需多轮交互，降低延迟和人工成本。

生成的描述随后输入文本到图像模型（Janus-Pro），零样本合成语义一致的图像。通过LLM成对比较策略，选择与文本描述最匹配的top-$K$张图像，构建扩充的$N$-way $(K+K)$-shot支持集。

### 跨模态融合
利用CLIP编码的文本特征$Z_t$与支持特征$z_s$进行融合：
- **通道维度**：通过两层MLP生成调制向量$\beta$，沿通道维度增强支持特征
- **空间维度**：将$Z_t$和$z_s$沿空间维度拼接，通过Transformer的多头自注意力捕获token间的语义关联，得到增强的支持嵌入$Z_s$

### 跨模态几何对齐（CGA）——核化体积对比学习
不同于传统成对对比学习，CGA通过测量多向量张成的$k$维平行多面体的体积来度量对齐程度：

$$\mathrm{Vol}_{\mathcal{H}}(\mathbf{v}_1,\dots,\mathbf{v}_k)=\sqrt{\det(\mathbf{K})},\quad \mathbf{K}_{ij}=\kappa(\mathbf{v}_i,\mathbf{v}_j)$$

其中$\kappa$为RBF核函数，将对齐扩展到再生核Hilbert空间（RKHS），捕获非线性关系。以文本模态为锚，定义双向对比损失$\mathcal{L}_{\text{D2A}}$和$\mathcal{L}_{\text{A2D}}$，最终总损失为：

$$\mathcal{L}_{\text{total}}=\sum_{i=1}^{M}\text{CE}(p_i,y_i)+\frac{1}{2}(\mathcal{L}_{\text{D2A}}+\mathcal{L}_{\text{A2D}})$$

### 推理阶段
分别计算文本增强原型$c_t$和视觉扩充原型$c_v$，通过凸组合得到最终分类原型：$C=uc_t+(1-u)c_v$，融合因子$u$在验证集上网格搜索确定。

## 实验关键数据

### 实验1：标准少样本分类（miniImageNet & tieredImageNet）

| 方法 | Backbone | miniImageNet 1-shot | miniImageNet 5-shot | tieredImageNet 1-shot | tieredImageNet 5-shot |
|------|----------|-------------------|-------------------|---------------------|---------------------|
| ProtoNet | ResNet-12 | 62.39 | 80.53 | 68.23 | 84.03 |
| SemFew (CVPR'24) | Swin-T (29M) | 78.94 | 86.49 | 82.37 | 89.89 |
| UAP (NeurIPS'24) | ResNet-12 | 81.63 | 79.05 | 79.68 | 76.78 |
| ECER (AAAI'25) | Visformer-T | 81.14 | - | 81.81 | - |
| **VT-FSL** | **Visformer-T (10M)** | **83.66** | **88.38** | **88.02** | **91.71** |

VT-FSL使用轻量Visformer-T骨干（10M参数），超越所有使用更大骨干（ViT-S 22M、Swin-T 29M、WRN28-10 36.5M）的方法。相比SemFew在1-shot上提升3.7%-5.7%。

### 实验2：细粒度与跨域少样本分类

| 场景 | 数据集 | 次优方法 1-shot | VT-FSL 1-shot | 提升 |
|------|--------|---------------|--------------|------|
| 细粒度 | CUB-200 | 86.02 (SUITED) | **91.08** | +5.06 |
| 细粒度 | Stanford-Dogs | 76.55 (SUITED) | **86.58** | +10.03 |
| 细粒度 | Stanford-Cars | 89.97 (SUITED) | **92.95** | +2.98 |
| 跨域 | mini→CUB | 51.55 (MEFP) | **66.86** | +15.31 |
| 跨域 | mini→Places | 59.07 (SVasP) | **73.68** | +14.61 |
| 跨域 | mini→Plantae | 41.55 (MEFP) | **45.90** | +4.35 |

跨域1-shot场景提升尤为显著（最高15.31%），证明跨模态语义的迁移能力。

### 消融实验核心发现

| 文本提示 | 视觉提示 | 对齐损失 | miniImageNet 1-shot | CIFAR-FS 1-shot |
|---------|---------|---------|-------------------|----------------|
| ✗ | ✗ | ✗ | 68.47 | 76.43 |
| ✓ | ✗ | ✗ | 78.82 | 84.76 |
| ✓ | ✓ | ✗ | 82.08 | 87.72 |
| ✓ | ✓ | ✓ | **83.66** | **88.67** |

三个组件均持续带来增益，文本提示贡献最大（+10.35），视觉提示进一步补充（+3.26），对齐损失锦上添花（+1.58）。

### 效率对比

| 方法 | 提示生成(h) | 训练时间(min) | 推理时间(ms) | 准确率(%) |
|------|-----------|-------------|-------------|----------|
| SP | - | 1.7 | 78 | 72.31 |
| SemFew | 1.5 | 2.3 | 105 | 78.94 |
| ECER | 0.7 | 3.0 | 119 | 81.14 |
| **VT-FSL** | **0.7** | **1.1** | **76** | **83.66** |

VT-FSL训练最快、推理最快、精度最高，因为跨模态提示通过一次性离线生成，下游模型无额外开销。

## 亮点

- **视觉接地的文本生成**：首次联合利用类名+支持图像条件化LLM，消除纯文本驱动方法的语义幻觉问题，四阶段结构化推理在单次推理中完成
- **互补跨模态提示设计**：文本提示提供高层类语义，合成图像提供低层类内多样性，二者形成互补；插件式设计使提示可被任意下游FSL模型直接使用
- **核化体积对比学习**：首次将基于体积的对比学习引入FSL，通过RKHS中的核Gram矩阵行列式捕获全局非线性跨模态关系，超越InfoNCE和线性体积方法
- **全面SOTA**：在10个基准（标准/细粒度/跨域）上全面刷新SOTA，平均提升4.2%，且使用最轻量骨干

## 局限与展望

- **依赖LLM和生成模型质量**：CIP依赖Qwen2.5-VL-32B的视觉理解和推理能力，合成图像依赖Janus-Pro的生成质量，对更弱的模型效果未知
- **离线生成开销**：虽然提示生成是一次性的，但对于大规模数据集仍需0.7小时，且需要GPU资源运行大模型
- **融合因子手动搜索**：推理时的融合因子$u$需要在验证集上网格搜索，不够自动化
- **合成图像数量敏感**：超过$K$张合成图像后性能下降甚至退化，说明生成质量仍有上限
- **仅验证图像分类**：未在检测、分割等更复杂的少样本任务上验证泛化性
- **计算资源要求**：需要NVIDIA RTX 6000 Ada级别GPU，对资源受限场景不友好

## 与相关工作的对比

- **AM3 (NeurIPS'19)**：仅利用类名的语义特征与视觉原型自适应融合，信息量有限；VT-FSL生成丰富的视觉接地描述
- **CaFo (CVPR'22)**：基于类名生成合成图像扩充数据，但文本描述缺乏视觉接地；VT-FSL从支持图像提取视觉线索消除幻觉
- **SemFew (CVPR'24)**：用LLM基于类名生成连贯描述增强原型，但朴素提示限制了语义质量，且需4.3M参数的融合模块；VT-FSL仅需0.7M参数的两层MLP
- **ECER (AAAI'25)**：提取属性级文本信息，仅以类名为条件；VT-FSL联合类名和图像，结构化推理生成更精确描述
- **InfoNCE对比学习**：成对对齐忽略多模态间全局结构关系；VT-FSL的核化体积损失同时考虑三模态的全局几何关系
- **SUITED (AAAI'25)**：细粒度FSL次优方法；VT-FSL在1-shot上超越3.0%-10.3%

## 评分

- 新颖性: ⭐⭐⭐⭐ — 核化体积对比学习和视觉接地CIP设计新颖，但整体框架为已有组件的巧妙组合
- 实验充分度: ⭐⭐⭐⭐⭐ — 10个基准、三种场景、详细消融、效率对比、可视化分析，实验极为充分
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机明确，图表丰富，数学推导完整
- 价值: ⭐⭐⭐⭐⭐ — 全面刷新SOTA且方法实用，插件式设计便于落地，代码已开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](../../ICCV2025/multimodal_vlm/causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [\[CVPR 2026\] Training-Only Heterogeneous Image-Patch-Text Graph Supervision for Advancing Few-Shot Learning Adapters](../../CVPR2026/multimodal_vlm/training-only_heterogeneous_image-patch-text_graph_supervision_for_advancing_few.md)
- [\[CVPR 2025\] UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning](../../CVPR2025/multimodal_vlm/unem_unrolled_generalized_em_for_transductive_few-shot_learning.md)
- [\[NeurIPS 2025\] CyIN: Cyclic Informative Latent Space for Bridging Complete and Incomplete Multimodal Learning](cyin_cyclic_informative_latent_space_for_bridging_complete_and_incomplete_multim.md)
- [\[NeurIPS 2025\] Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning](praxisvlm_visiongrounded_decision_making_via_textdriven_rein.md)

</div>

<!-- RELATED:END -->
