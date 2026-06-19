---
title: >-
  [论文解读] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations
description: >-
  [ICCV 2025][可解释性][视觉电路发现] 提出 Granular Concept Circuit (GCC) 方法，通过迭代评估神经元间的功能依赖性（Neuron Sensitivity Score）和语义一致性（Semantic Flow Score），自动发现深度视觉模型中编码特定概念的细粒度视觉电路——这是首个能在单个query中发现多个概念级电路的方法。
tags:
  - "ICCV 2025"
  - "可解释性"
  - "视觉电路发现"
  - "概念表征"
  - "神经元连接性"
  - "机械可解释性"
---

# Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations

**会议**: ICCV 2025  
**arXiv**: [2508.01728](https://arxiv.org/abs/2508.01728)  
**代码**: [https://github.com/daheekwon/GCC](https://github.com/daheekwon/GCC)  
**领域**: 可解释性  
**关键词**: 可解释性, 视觉电路发现, 概念表征, 神经元连接性, 机械可解释性

## 一句话总结

提出 Granular Concept Circuit (GCC) 方法，通过迭代评估神经元间的功能依赖性（Neuron Sensitivity Score）和语义一致性（Semantic Flow Score），自动发现深度视觉模型中编码特定概念的细粒度视觉电路——这是首个能在单个query中发现多个概念级电路的方法。

## 研究背景与动机

深度视觉模型通过层次化架构形成概念表征——从低层的边缘、纹理到高层的物体、场景。理解这些概念在模型中如何被编码是可解释AI的核心问题。

现有方法的不足：

**单神经元分析**（NetDissect、CLIP-Dissect等）：将概念关联到单个神经元，忽视了表征的分布式本质——概念通过多个神经元跨层协作编码

**VCC**（Visual Concept Connectome）：用概念激活向量（CAV）分析层间连接，但与网络结构不对齐，无法精确定位概念在哪里出现

**ADVC**（Rajaram等人的方法）：使用梯度×激活值的跨层归因迭代发现电路，但仅构建一个与类别标签相关的统一电路，缺乏概念级细粒度分析
4. 所有现有方法都**不支持**将模型响应分解为多个概念级电路——不同概念（如天空、旗帜、时钟）被混在一个电路中

## 方法详解

### 整体框架

GCC 的目标是为给定query发现多个细粒度概念电路，每个电路对应一个与query相关的特定概念。流程：(1) 提取根节点 → (2) 评估跨层连接 → (3) 迭代追踪直到无进一步连接 → (4) 对所有根节点重复，得到完整电路集合。

### 关键设计

1. **Neuron Sensitivity Score ($S_{NS}$)**：基于干预的功能依赖性度量。

   通过将源神经元置零（mute），观察目标神经元激活的变化来量化连接强度：

    $\tilde{S}_{NS,c} = \max(0, f^{l+1}(a_c^l) - f^{l+1}(\hat{a}_c^l))$
    $S_{NS} = \frac{\tilde{S}_{NS}}{\sum \tilde{S}_{NS}}$

   其中 $\hat{a}_c^l$ 是将第c个神经元置零后的层l激活。高 $S_{NS}$ 表示目标神经元强烈依赖源神经元。进行正值裁剪以仅关注正相关关系。

    - 设计动机：因果干预比梯度更准确地捕获真实依赖关系
    - 使用一阶近似（逐个神经元干预）避免 $O(2^{|N|})$ 的全组合搜索

2. **Semantic Flow Score ($S_{SF}$)**：语义一致性约束。

    $S_{SF} = \frac{|\mathcal{S}_{src} \cap \mathcal{S}_{tgt}|}{|\mathcal{S}_{src}|}$

   其中 $\mathcal{S}_{src}$ 和 $\mathcal{S}_{tgt}$ 分别是源和目标神经元的 top-k 高激活样本集合。高重叠表示两个神经元编码了类似的语义信息。

    - 设计动机：仅有高 $S_{NS}$ 不够——非线性可能产生伪连接（功能依赖但语义不相关），$S_{SF}$ 过滤掉这些虚假连接

3. **电路构建算法**：

    - **根节点提取**：选择在所有样本中激活排名前1%的神经元
    - **连接判定**：同时满足 $S_{NS} > \tau_{NS}$ 且 $S_{SF} > \tau_{SF}$；$\tau_{NS}$ 通过极值理论中的 Peak-over-Threshold (POT) 方法自动确定；$\tau_{SF}$ 使用所有节点的平均分数
    - **迭代扩展**：新加入的节点成为新的起点，向下一层继续搜索，直到无满足条件的连接
    - **高效计算**：复用已计算的源节点连接，递归技术避免重复计算

### 损失函数 / 训练策略

GCC 是一种后训练分析方法，不涉及训练过程。它直接在预训练模型（VGG19、ResNet50/101、MobileNetV3、ViT等）上运行，使用 ImageNet1K 验证集作为参考样本集。

## 实验关键数据

### 主实验

**忠实性与完备性评估**（在100个随机ImageNet1K query上测试，消融电路内/外神经元后观察logit变化）：

| 消融条件 | ResNet50 | ResNet101 | VGG19 | MobileNetV3 | 平均下降 |
|---------|---------|----------|-------|------------|---------|
| 原始（无消融） | 17.17 | 17.46 | 20.94 | 17.34 | — |
| 随机消融神经元 | 15.66 | 13.80 | 19.03 | 15.01 | ▼2.35 |
| 消融GCC内神经元 | 6.41 | 6.18 | 12.93 | 12.95 | **▼8.60** |
| 消融GCC外神经元 | 16.12 | 14.58 | 19.93 | 15.88 | ▼1.74 |

消融GCC内神经元导致logit大幅下降（8.60），远超随机消融（2.35）；消融GCC外神经元影响极小（1.74），说明电路既忠实又完备。

### 消融实验（用户研究）

33名参与者对GCC五个方面打分（5分制）：

| 评估维度 | 平均分 |
|---------|-------|
| Query相关性：GCC是否与query相关 | 3.65/5 |
| 多样性：GCC是否捕获多种概念 | 4.00/5 |
| 原型性：GCC是否代表给定多query的共性 | 4.45/5 |
| 连接合理性：节点间和query-节点连接是否合理 | >90%认为合理 |
| 与VCC对比：GCC vs VCC谁捕获更有意义的概念 | 70%选择GCC |

边消融/插入实验：按 $S_{NS}$ 排序移除/添加边，移除高排名边导致性能快速下降，添加高排名边带来显著增益。

### 关键发现

- **首次实现概念级视觉电路分解**：一个"scoreboard"图像可以被分解为17个GCC——分别对应天空背景、旗帜、时钟等不同概念
- **跨模型/数据集通用**：在CNN（VGG19、ResNet50/101、MobileNetV3）和Transformer（ViT）上均有效
- **概念层次化演进**：孔雀图像的"蓝色纹理"概念从第一层的"广泛蓝色调"逐步细化为"蓝色鳞片"→"结构化蓝色图案"→"装饰性蓝色图案"
- **跨类别共享概念发现**：能在不同类别（雏菊和孔雀）中发现共享的"辐射状"概念，在坦克/小巴/货车中发现"轮子"概念
- **错误分类审计**：可定位导致误分类的具体概念电路并通过刺激/抑制验证

## 亮点与洞察

- **双分数设计互补**：$S_{NS}$ 捕获功能依赖、$S_{SF}$ 确保语义一致，缺一不可——单靠 $S_{NS}$ 会产生伪连接
- **POT自动阈值**：避免手动调参，使方法更具实用性
- **从"一个电路"到"多个电路"的范式转变**：之前的方法（VCC、ADVC）为每个query只构建一个统一电路，GCC首次拆分为多个概念特定电路
- **可解释性与实用性并重**：错误分类审计、跨类别概念发现等应用场景有实际价值
- **Sankey图可视化**：用连接粗细表示连接强度、用高激活样本裁剪展示神经元语义，可视化方案直观清晰

## 局限与展望

- 连接评估采用逐神经元干预的一阶近似，可能遗漏高阶交互
- 在严格阈值下，单个概念可能分布在多条电路路径中
- 部分高 $S_{NS}$ 连接仍难以用人类语言解释
- 仅在分类模型上验证，生成模型（GAN/Diffusion）的电路发现值得探索
- 语义覆盖的参考数据集（ImageNet1K验证集）可能引入偏见——某些概念的激活样本不够丰富
- 可扩展到视频模型和更大规模Transformer（如ViT-L/H）

## 相关工作与启发

- 与机械可解释性（NLP领域的Conmy等人）对应，GCC首次在视觉领域实现细粒度电路发现
- CRP求解条件相关性但限于两两层间、CRAFT从分类器层反推但只能提取分类相关概念——GCC正向传播，发现更广泛的概念
- 神经科学中Hebb学习规则和突触可塑性的启发：功能连接+信息流保持
- GCC的正向电路发现能找到跨类别的抽象特征，这是基于类别logit反推的方法（ADVC）做不到的

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次实现视觉模型中概念级细粒度电路发现，双分数设计原创且有理论支撑
- **实验充分度**: ⭐⭐⭐⭐ 忠实性/完备性定量验证+用户研究+边消融+多模型多数据集，较全面
- **写作质量**: ⭐⭐⭐⭐ 概念阐述清晰，可视化丰富；算法伪代码规范
- **价值**: ⭐⭐⭐⭐ 为视觉模型的可解释性提供了新工具，错误审计等应用场景有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Towards Human-Understandable Multi-Dimensional Concept Discovery](../../CVPR2025/interpretability/towards_human-understandable_multi-dimensional_concept_discovery.md)
- [\[ACL 2025\] Position-aware Automatic Circuit Discovery](../../ACL2025/interpretability/position-aware_automatic_circuit_discovery.md)
- [\[ICCV 2025\] CE-FAM: Concept-Based Explanation via Fusion of Activation Maps](ce-fam_concept-based_explanation_via_fusion_of_activation_maps.md)
- [\[ACL 2025\] Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers](../../ACL2025/interpretability/separating_tongue_from_thought_activation_patching_reveals_language-agnostic_con.md)
- [\[ICML 2026\] CB-SLICE: Concept-Based Interpretable Error Slice Discovery](../../ICML2026/interpretability/cb-slice_concept-based_interpretable_error_slice_discovery.md)

</div>

<!-- RELATED:END -->
