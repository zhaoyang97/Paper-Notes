---
title: >-
  [论文解读] SceneForge: Enhancing 3D-text alignment with Structured Scene Compositions
description: >-
  [NeurIPS 2025][3D视觉][3D-文本对比学习] 提出SceneForge框架，通过将单个3D点云对象组合成带显式空间关系的多物体场景，配合LLM精炼的组合描述，增强3D-文本对比学习的数据多样性和复杂度，在多个下游任务上带来一致性能提升。 大规模对比学习已经彻底改变了视觉-语言建模，CLIP/ALIGN在2D…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "3D-文本对比学习"
  - "组合增强"
  - "点云场景组合"
  - "空间关系"
  - "零样本分类"
---

# SceneForge: Enhancing 3D-text alignment with Structured Scene Compositions

**会议**: NeurIPS 2025  
**arXiv**: [2509.15693](https://arxiv.org/abs/2509.15693)  
**代码**: 暂无  
**领域**: 3D视觉  
**关键词**: 3D-文本对比学习, 组合增强, 点云场景组合, 空间关系, 零样本分类

## 一句话总结

提出SceneForge框架，通过将单个3D点云对象组合成带显式空间关系的多物体场景，配合LLM精炼的组合描述，增强3D-文本对比学习的数据多样性和复杂度，在多个下游任务上带来一致性能提升。

## 研究背景与动机

大规模对比学习已经彻底改变了视觉-语言建模，CLIP/ALIGN在2D领域的成功激励了3D方向的研究（如Uni3D、OmniBind、OpenShape）。但将对比学习扩展到3D仍面临核心挑战：

**3D-文本数据稀缺**：与2D图像-文本数据集相比，大规模3D-文本数据集极度匮乏。即使是最大的OpenShape数据集，规模也远不及2D领域

**现有增强方法的局限**：PointCutMix和PointMixup等3D增强方法要么随机混合点导致物体语义破坏，要么插值坐标产生不真实的形状

**缺乏空间关系建模**：现有3D对比学习方法主要处理单物体，模型没有学到物体间空间关系的理解能力

本文的两个核心洞察：

- **3D的天然优势**：与2D图像不同，单个3D点云可以自由组合成结构化场景而不产生视觉伪影（没有背景、光照、视角的缠绕问题）
- **空间可控性**：3D数据允许显式控制物体定位，这在2D中很难实现。组合后的场景可以自然地配上包含空间关系的文本描述（如"A在B上方"）

## 方法详解

### 整体框架

SceneForge作为数据增强模块嵌入任意3D-文本对比学习管道中。在每个训练batch中，以概率 $\alpha$ 将样本标记为组合样本（其余保持单物体）。标记的样本经SceneForge模块组合 $K$ 个物体成场景，并生成对应的组合文本描述。

### 关键设计

1. **3D Scene Forge（3D场景组合）**：按空间关系依次放置物体，确保语义连贯的组合。

    - 定义三种空间关系："over"（在...上方）、"under"（在...下方）、"next to"（在...旁边）
    - 物体放置基于边界框约束：
        - "over"关系：将 $p_i$ 的最小z对齐到 $p_{i-1}$ 的最大z之上：$\mathcal{P}(p_i, p_{i-1}, \text{"over"}) = \max_{\mathbf{z}}(p_{i-1}) - \min_{\mathbf{z}}(p_i)$
        - "next to"关系：在xy平面采样水平单位向量 $\mathbf{d}$，沿该方向对齐：$\mathcal{P}(p_i, p_{i-1}, \text{"next to"}) = (\max_{\mathbf{x} \in p_{i-1}} \langle \mathbf{x}, \mathbf{d} \rangle - \min_{\mathbf{y} \in p_i} \langle \mathbf{y}, \mathbf{d} \rangle) \mathbf{d}$
    - 添加固定偏移 $\delta$ 和高斯噪声 $\epsilon$ 防止完美对齐，引入自然随机性
    - 最终将组合点云下采样到目标点数 $P = 10k$

2. **Scene Caption Forge（场景描述生成）**：镜像3D组合过程，顺序拼接每个物体的描述和空间关系，然后用LLM（Qwen2.5-7B-Instruct）精炼：

    - 修正语法、标点和句子结构
    - 保持原始语义和空间关系
    - 增强描述的多样性和流畅度
    - 同时改进了OpenShape原始BLIP生成的不够精确的描述

3. **训练方案设计**：

    - **损失分区**：组合样本只参与text-3D对比损失（因为无法实时渲染2D视角），单物体样本同时参与text-3D和image-3D损失。对image-3D损失乘以缩放因子 $\frac{1}{1-\alpha}$ 平衡梯度贡献
    $\mathcal{L} = \underbrace{\frac{1}{2}[\mathcal{L}_{3D \to txt} + \mathcal{L}_{txt \to 3D}]}_{\text{所有N样本}} + \frac{1}{1-\alpha} \underbrace{\frac{1}{2}[\mathcal{L}_{3D \to 2D} + \mathcal{L}_{2D \to 3D}]}_{\text{仅单物体}}$
    - **增强约束**：待组合物体禁止平移增强（避免组合后不一致），允许垂直轴完整旋转但限制其他轴的旋转幅度（保证"上方"/"下方"语义正确）
    - **模型无关性**：在三种不同编码器（OpenShape-PointBERT、Uni3D-G、ViT-Lens-G）上均有效，CLIP编码器始终冻结

### 损失函数 / 训练策略

使用标准InfoNCE对比损失：
$$\mathcal{L}_{m \to n}(\mathcal{S}) = -\frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \log \frac{\exp(\langle e_i^m, e_i^n \rangle / \tau)}{\sum_{j \in \mathcal{S}} \exp(\langle e_i^m, e_j^n \rangle / \tau)}$$

训练200 epoch，全局batch size 1152，$\alpha = 0.5$，最大组合物体数 $N = 3$。SceneForge只需额外一个GPU运行轻量LLM。采用生产者-消费者并行策略：训练batch $t$ 时并行准备batch $t+M$。

## 实验关键数据

### 主实验

**零样本分类精度（Top-1%，ensemble训练集含LVIS）：**

| 方法 | LVIS | ModelNet | ScanObjNN | ScanNet | 平均Δ |
|------|------|----------|-----------|---------|-------|
| ULIP-2 | 50.6 | 84.7 | 51.5 | 38.9 | — |
| MixCon3D | 52.5 | 86.8 | 58.6 | 44.1 | — |
| OmniBind-L | 54.0 | 86.6 | 64.7 | 46.3 | — |
| Uni3D | 53.5 | 87.3 | 63.9 | 45.8 | — |
| **SF-Uni3D** | **54.7** | **88.2** | **65.2** | **49.4** | **+1.75** |

**ScanQA 3D视觉问答：**

| 方法 | B-4 | CIDEr | EM | ΔB-4 | ΔCIDEr |
|------|-----|-------|-----|------|--------|
| OmniBind-L + BLIP2 | 8.5 | 62.9 | 17.1 | — | — |
| Uni3D + BLIP2 | 7.5 | 58.3 | 16.4 | — | — |
| **SF-Uni3D + BLIP2** | **10.4** | **66.7** | **20.5** | **+2.9** | **+8.4** |

### 消融实验

| 配置 | LVIS T1 | ModelNet T1 | ScanObjNN T1 | ScanNet T1 | 说明 |
|------|---------|-------------|-------------|-----------|------|
| N=1（基线Uni3D） | 53.5 | 87.3 | 63.9 | 45.8 | 无组合 |
| N=2 | 53.9 | 87.6 | 64.5 | 48.2 | 开始提升 |
| **N=3** | **54.7** | **88.2** | **65.2** | **49.4** | **最佳** |
| N=4 | ≈54 | ≈88 | ≈65 | ≈48 | 开始下降 |
| N=5 | <54 | <88 | <65 | <47 | 点云过度碎片化 |

**3D组合方法对比（N=2，Uni3D骨干）：**

| 方法 | LVIS T1 | ModelNet T1 | ScanObjNN T1 | ScanNet T1 |
|------|---------|-------------|-------------|-----------|
| Uni3D（无组合） | 53.5 | 87.3 | 63.9 | 45.8 |
| PointMixup | 39.2 | 78.7 | 41.4 | 30.2 |
| PointCutMix-K | 44.7 | 83.0 | 45.1 | 34.8 |
| PointCutMix-R | 53.5 | 87.1 | 64.1 | 47.5 |
| **SF-Uni3D (N=2)** | **53.9** | **87.6** | **64.5** | **48.2** |

### 关键发现

- **最优组合数N=3**：从1→3单调提升，4开始平台期或下降，5全面下降。原因是固定10k点预算下5个物体会碎片化显著几何特征
- **α=0.5最优**：组合样本比例过高（>0.5）会牺牲单物体理解能力
- PointMixup和PointCutMix-K大幅破坏性能，因为它们破坏了物体语义完整性；PointCutMix-R随机混合整体物体略好但仍不如结构化组合
- SF-Uni3D在ScanQA空间推理问题上提升最大，说明多物体训练确实增强了空间关系理解
- **N物体跨模态检索**：先前模型在N=2时就急剧下降到50%以下，而SceneForge N=3模型在N=6时仍保持70%+

## 亮点与洞察

- **正交且模型无关的增强策略**：不修改模型架构或损失函数，仅在数据层面增强，可即插即用到任何3D-文本对比学习管道中
- **"整体大于部分之和"的实证**：多物体组合训练不仅提升多物体场景理解，还提升了单物体分类性能，这与CutMix/MixUp在2D中的正则化效应一致
- **3D的独特优势**：利用了3D数据天然无背景、可自由组合的特性，这是2D数据增强无法轻易实现的
- **空间关系理解的泛化**：只用3种简单关系（over/under/next to）训练，却能泛化到ScanQA中更复杂的空间关系（attached to、sitting on等）

## 局限与展望

- 空间关系仅定义了3种（over/under/next to），更丰富的关系（inside、behind等）可能带来进一步提升
- 无法实时渲染组合场景的2D视角，导致组合样本只能用text-3D损失训练，限制了2D-3D对齐的增强
- LLM描述精炼引入额外计算开销（0-50%训练减速），虽然通过并行化缓解但仍是瓶颈
- 当N≥4时组合场景过于拥挤，10k点预算不足以表达所有物体细节

## 相关工作与启发

本文与MixCon3D（混合多视角渲染和点云）形成互补，MixCon3D关注多模态表示的混合，而SceneForge关注通过结构化组合增加数据多样性。与OmniBind（多模型集成）相比，SceneForge用单一模型以更低推理成本达到更高性能。这种结构化数据增强的思路可以推广到其他3D-文本任务，如3D grounding、3D captioning等。

## 评分

- 新颖性: ⭐⭐⭐⭐ 利用3D数据天然可组合的特性做结构化增强，思路简洁优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖分类/分割/VQA/检索/微调多任务，三种骨干网络验证模型无关性，消融极为详尽
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，消融设计合理，但部分表格较密集
- 价值: ⭐⭐⭐⭐ 提供了一种低成本提升3D-文本对比学习的通用策略，具有广泛适用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] VA-GS: Enhancing the Geometric Representation of Gaussian Splatting via View Alignment](va-gs_enhancing_the_geometric_representation_of_gaussian_splatting_via_view_alig.md)
- [\[CVPR 2025\] CrossOver: 3D Scene Cross-Modal Alignment](../../CVPR2025/3d_vision/crossover_3d_scene_cross-modal_alignment.md)
- [\[CVPR 2026\] DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis](../../CVPR2026/3d_vision/dmaligner_enhancing_image_alignment_via_diffusion_model_based_view_synthesis.md)
- [\[NeurIPS 2025\] Enhancing Multilingual LLM Pretraining with Model-Based Data Selection](enhancing_multilingual_llm_pretraining_with_model-based_data_selection.md)
- [\[NeurIPS 2025\] D$^2$USt3R: Enhancing 3D Reconstruction for Dynamic Scenes](d2ust3r_enhancing_3d_reconstruction_for_dynamic_scenes.md)

</div>

<!-- RELATED:END -->
