---
title: >-
  [论文解读] Hypergraph Vision Transformers: Images are More than Nodes, More than Edges
description: >-
  [CVPR 2025][图学习][超图] 提出HgVT，将层次化二部超图结构嵌入ViT中，通过主图像patch顶点和虚拟顶点的分离处理、动态余弦邻接构建和超边通信池三层注意力机制，无需聚类即可捕获patch间高阶语义关系，在ImageNet-1K上HgVT-Ti以7.7M参数达到76.2%准确率（超ViHGNN-Ti 1.9%），并在图像检索中达到73.23% mAP@10。
tags:
  - "CVPR 2025"
  - "图学习"
  - "超图"
  - "Transformer"
  - "高阶关系"
  - "虚拟顶点"
  - "动态邻接"
---

# Hypergraph Vision Transformers: Images are More than Nodes, More than Edges

**会议**: CVPR 2025  
**arXiv**: [2504.08710](https://arxiv.org/abs/2504.08710)  
**代码**: 无  
**领域**: 图学习  
**关键词**: 超图、Vision Transformer、高阶关系、虚拟顶点、动态邻接

## 一句话总结
提出HgVT，将层次化二部超图结构嵌入ViT中，通过主图像patch顶点和虚拟顶点的分离处理、动态余弦邻接构建和超边通信池三层注意力机制，无需聚类即可捕获patch间高阶语义关系，在ImageNet-1K上HgVT-Ti以7.7M参数达到76.2%准确率（超ViHGNN-Ti 1.9%），并在图像检索中达到73.23% mAP@10。

## 研究背景与动机

**领域现状**：ViT通过自注意力建模patch间关系但复杂度为二次方；Vision GNN（ViG）将patch作为图节点建模关系但依赖KNN/FCM等聚类算法构建图结构，计算开销大且固定。

**现有痛点**：（1）ViT的全局注意力缺乏显式的高阶关系建模能力——只能处理成对关系；（2）ViG类方法的图结构依赖聚类，推理时增加延迟；（3）ViHGNN虽引入超图但仍需额外的图构建步骤。

**核心矛盾**：图像中的高阶语义结构（如"这组patch共同构成一只鸟的翅膀"）需要超越成对关系的建模能力，但高效构建这种高阶结构是挑战。

**本文目标** 设计一种无需聚类、计算高效的超图Vision Transformer，能够自适应地捕获patch间的高阶语义关系。

**切入角度**：引入虚拟顶点和超边作为"通信池"，通过动态余弦相似度构建稀疏邻接矩阵，让超边自动发现和组织语义相关的patch群组。

**核心 idea**：用可学习的虚拟顶点和超边替代聚类构建超图，通过三层注意力（顶点自注意力→顶点聚合到超边→超边分发回顶点）实现高阶关系的高效建模。

## 方法详解

### 整体框架
输入图像分为patch后，每个patch为主顶点，额外引入可学习的虚拟顶点。通过动态余弦邻接构建超图结构（哪些顶点属于哪条超边），然后通过三层注意力在超图上传播信息。最终通过Expert Edge Pooling融合虚拟超边的信息进行分类。

### 关键设计

1. **层次化二部超图结构**:

    - 功能：将图像patch组织为高阶语义群组
    - 核心思路：定义两类顶点——主顶点$i\mathcal{V}$（图像patch）和虚拟顶点$v\mathcal{V}$（可学习参数）；两类超边——主超边$p\mathcal{E}$（连接所有顶点）和虚拟超边$v\mathcal{E}$（仅连接虚拟顶点）。动态邻接矩阵$A = \sigma(\alpha \cdot \tilde{X}^{adj(v)} [\tilde{X}^{adj(e)}]^T)$通过顶点和超边特征的余弦相似度计算，$\alpha=4$为尖锐化因子。硬邻接$\hat{A} = [A > 0.5]$用于稀疏注意力掩码
    - 设计动机：无需KNN/FCM等聚类算法，邻接矩阵端到端学习，复杂度$O(|V| \cdot E)$其中$E < |V|$

2. **超边通信池三层注意力**:

    - 功能：在超图上实现高效的高阶特征传播
    - 核心思路：三步顺序注意力——（a）顶点自注意力($\mathcal{V} \to \mathcal{V}$)：同一超边内的顶点交互；（b）边聚合注意力($\mathcal{V} \to \mathcal{E}$)：顶点特征聚合到超边；（c）边分发注意力($\mathcal{E} \to \mathcal{V}$)：超边信息分发回顶点。通过稀疏掩码避免全局注意力的二次复杂度
    - 设计动机：超边作为"通信池"中介，近似实现了全局特征传递但复杂度更低；三步分解比直接多对多注意力更稳定

3. **Expert Edge Pooling + 正则化**:

    - 功能：利用超图结构信息进行分类
    - 核心思路：虚拟超边作为"专家"生成置信度分数，通过top-k选择加权平均（类似MoE）。两种正则化——多样性正则化惩罚虚拟顶点嵌入间的余弦相似度（防止坍缩），种群正则化控制每条超边连接的顶点数在上下界$[\beta, \gamma]$内（防止过稀疏/过密集）
    - 设计动机：没有正则化时虚拟顶点会坍缩到相同表示，超图退化为普通图

### 损失函数 / 训练策略
分类交叉熵 + 多样性正则化 + 种群正则化。三种模型变体：HgVT-Lt (6.8M/0.92B)、HgVT-Ti (7.7M/1.8B)、HgVT-S (23M/5.5B)。

## 实验关键数据

### 主实验（ImageNet-1K）

| 模型 | Params | FLOPs | Top-1 | vs同级对比 |
|------|--------|-------|-------|----------|
| HgVT-Ti | 7.7M | 1.8B | 76.2% | ViHGNN-Ti 74.3% (+1.9%) |
| HgVT-S | 22.9M | 5.5B | 81.2% | ViG-S 80.4% (+0.8%) |
| DeiT-Ti（参考） | 5.7M | 1.3B | 72.2% | HgVT-Ti高4.0% |

HgVT-Ti与DeiT-B (86.4M) 的ReaL精度相当（83.2% vs 86.7%），但参数仅1/11。

### 消融实验（ImageNet-100, HgVT-Lt）

| 配置 | Top-1 | 说明 |
|------|-------|------|
| Full model | 84.36% | 基准 |
| w/o 多样性正则化 | 80.79% | 掉3.57%，表示坍缩 |
| w/o 种群正则化 | 81.79% | 掉2.57%，超图退化 |
| w/o 顶点自注意力 | -4%~-6% | 对特征分离至关重要 |
| Expert+Image pooling | 84.36% | 最佳组合 |
| 仅Expert pooling | 82.52% | 单独不够 |

### 关键发现
- 多样性和种群正则化都不可缺少，去掉任一个性能下降3-4%，超图退化
- 虚拟顶点作为"噪声摘要元素"而非纯图摘要器工作，Image pooling与DINO特征对齐更好
- 图像检索上73.23% mAP@10超越MRL基线65.04%，Expert pooling展现宏观类别聚类（狗类、鸟类自动分组）
- 超边熵和轮廓系数指标与分类精度正相关，验证了超图结构质量与性能的关联

## 亮点与洞察
- **无聚类的超图构建**：通过可学习的余弦邻接完全避免了KNN等聚类操作，使超图结构端到端可微且推理高效
- **虚拟顶点的"通信池"抽象**：虚拟顶点不直接对应图像内容，而是学习语义分组的抽象概念，这种设计类似Slot Attention但在超图框架下更自然
- **图像检索的意外发现**：Expert pooling自动产生了宏观语义聚类，说明超图结构确实捕获了高阶语义信息

## 局限与展望
- 仅在分类和检索上验证，密集预测（检测/分割）的超图结构如何适配未探索
- 虚拟顶点和超边数量是手动设定的超参数
- HgVT-S在Top-1上略低于ViHGNN-S (81.2% vs 81.5%)，大模型上优势减弱
- 三层注意力的顺序执行限制了并行度

## 相关工作与启发
- **vs ViG**: ViG用KNN构建固定图结构；HgVT用动态超图无需聚类且捕获高阶关系
- **vs ViHGNN**: 同为超图ViT，但HgVT用虚拟顶点+动态邻接避免了聚类瓶颈，Ti级模型精度高1.9%
- **vs DeiT**: HgVT-Ti用7.7M参数超越DeiT-Ti (72.2%)达76.2%，证明超图归纳偏置的价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 二部超图+虚拟顶点+三层注意力的组合独特，无聚类设计优雅
- 实验充分度: ⭐⭐⭐⭐ ImageNet分类+检索+详细消融，但缺密集预测任务
- 写作质量: ⭐⭐⭐⭐ 超图形式化严谨，但篇幅偏长
- 价值: ⭐⭐⭐⭐ 为视觉Transformer引入高阶关系建模提供了新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DVHGNN: Multi-Scale Dilated Vision HGNN for Efficient Vision Recognition](dvhgnn_multi-scale_dilated_vision_hgnn_for_efficient_vision_recognition.md)
- [\[NeurIPS 2025\] Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking](../../NeurIPS2025/graph_learning/diagnosing_and_addressing_pitfalls_in_kg-rag_datasets_toward_more_reliable_bench.md)
- [\[NeurIPS 2025\] S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning](../../NeurIPS2025/graph_learning/smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)
- [\[ICLR 2026\] Actions Speak Louder than Prompts: A Large-Scale Study of LLMs for Graph Inference](../../ICLR2026/graph_learning/actions_speak_louder_than_prompts_a_large-scale_study_of_llms_for_graph_inferenc.md)
- [\[ACL 2025\] Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](../../ACL2025/graph_learning/kg_llm_trustworthy_qa.md)

</div>

<!-- RELATED:END -->
