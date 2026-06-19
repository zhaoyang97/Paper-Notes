---
title: >-
  [论文解读] FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation
description: >-
  [CVPR 2026][AI安全][联邦学习] 提出 FedAFD 框架，通过双层对抗对齐、粒度感知特征融合和相似度引导的集成蒸馏三阶段设计，在多模态联邦学习中同时提升异构客户端和服务器的模型性能。 多模态联邦学习（MFL）允许不同模态的客户端在不共享原始数据的前提下协作训练模型，但面临三大挑战： 模态/任务异构性：不同客户…
tags:
  - "CVPR 2026"
  - "AI安全"
  - "联邦学习"
  - "Adversarial Alignment"
  - "Feature Fusion"
  - "知识蒸馏"
  - "Model Heterogeneity"
---

# FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation

**会议**: CVPR 2026  
**arXiv**: [2603.04890](https://arxiv.org/abs/2603.04890)  
**代码**: [Chao2433/FedAFD](https://github.com/Chao2433/FedAFD)  
**领域**: AI安全 / 联邦学习  
**关键词**: Multimodal Federated Learning, Adversarial Alignment, Feature Fusion, knowledge distillation, Model Heterogeneity

## 一句话总结

提出 FedAFD 框架，通过双层对抗对齐、粒度感知特征融合和相似度引导的集成蒸馏三阶段设计，在多模态联邦学习中同时提升异构客户端和服务器的模型性能。

## 研究背景与动机

多模态联邦学习（MFL）允许不同模态的客户端在不共享原始数据的前提下协作训练模型，但面临三大挑战：

**模态/任务异构性**：不同客户端可能处理不同模态（图像、文本）和不同任务（分类、检索），导致特征空间不一致，产生模型漂移

**个性化不足**：现有方法为提升全局模型性能往往牺牲了本地模型性能

**模型异构性**：不同客户端使用不同架构的编码器，无法直接进行参数级聚合

现有方法如 CreamFL 只关注全局模型性能，忽视了本地个性化，且在处理模态/任务差异时缺乏有效机制。FedAFD 的核心思路是通过"边缘-云"协作框架，同时增强全局和本地模型性能。

## 方法详解

### 整体框架

FedAFD 包含三个阶段的迭代训练：
- **阶段①**：服务器在公共数据集上训练并提取全局公共特征
- **阶段②**：客户端接收全局表示和编码器，在私有数据上通过双层对抗对齐 + 粒度感知融合训练本地模型
- **阶段③**：客户端在公共数据上提取本地特征并上传至服务器，服务器执行相似度引导的集成蒸馏更新全局模型

系统包含三类客户端：$N_I$ 个单模态图像客户端（图像分类）、$N_T$ 个单模态文本客户端（文本分类）、$N_M$ 个多模态客户端（图文检索），以及一个公共多模态数据集 $\mathcal{P}$。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    S1["服务器：公共数据集 P 上训练<br/>提取全局公共特征与编码器"]
    S1 -->|下发全局表示与编码器| C0["客户端：私有数据 + 接收的全局表示"]
    C0 --> BAA["双层对抗对齐 BAA<br/>模态内 + 跨模态判别器拉近客户端-服务器分布"]
    BAA --> GFF["粒度感知特征融合 GFF<br/>注意力两级自适应融合本地与全局特征"]
    GFF --> CT["更新本地模型<br/>任务损失 + β·对抗损失"]
    CT -->|公共数据上提取本地特征并上传| SED["相似度引导的集成蒸馏 SED<br/>按表示相似度加权聚合教师表示"]
    SED -->|蒸馏更新全局模型，进入下一轮通信| S1
```

### 关键设计

**1. 双层对抗对齐 BAA：把客户端-服务器的表示错位当成域适应来消**

不同模态/任务的客户端和服务器各自学出来的表示对不上，会造成模型漂移。FedAFD 把这种不一致直接建模成联邦域适应问题，给每个客户端配两个判别器：**模态内判别器** $\mathcal{D}_c^{in}$ 区分同模态下的本地/全局表示（如 $i_p^{c,k}$ vs $i_p^{g,k}$），**跨模态判别器** $\mathcal{D}_c^{cr}$ 区分不同模态的本地/全局表示（如 $i_p^{c,k}$ vs $t_p^{g,k}$）。对抗损失为 $\mathcal{L}_{adv} = \frac{1}{|\mathcal{P}|}\sum_{k=1}^{|\mathcal{P}|}(\mathcal{L}_{in}^k + \mathcal{L}_{cr}^k)$，其中 $\mathcal{L}_{in}^k = \log \mathcal{D}_c^{in}(i_p^{g,k}) + \log(1-\mathcal{D}_c^{in}(i_p^{c,k}))$、跨模态同理。判别器最大化、编码器最小化这个损失，就把客户端-服务器之间的表示分布差异拉近。

**2. 粒度感知特征融合 GFF：对齐之后别让全局知识淹掉本地个性化**

BAA 把分布对齐了，但塞进太多全局知识会反过来压低本地性能。GFF 用注意力在样本级自适应地融合本地与全局特征，分两级细化：第一级 $h_c^k = M(i_c^k + i_g^k) \otimes i_c^k + (1-M(i_c^k + i_g^k)) \otimes i_g^k$，第二级 $\widetilde{i}_c^k = M(h_c^k) \otimes i_c^k + (1-M(h_c^k)) \otimes i_g^k$。注意力权重 $M(x) = \sigma(T_1(x) + T_2(x))$ 由两个并行非线性变换 $T_1, T_2$ 捕获多尺度上下文。融合后的特征拿去算任务损失 $\mathcal{L}_{task}$，这样每个样本自己决定要多少全局、留多少本地。消融里去掉 GFF 客户端性能暴跌，印证它是个性化的命门。

**3. 相似度引导的集成蒸馏 SED：异构架构没法参数聚合，就在表示级按相似度蒸馏**

各客户端编码器架构不同，服务器没法做参数级聚合。SED 改在表示层做，并按特征相似度动态分配聚合权重：相似度分数 $s^{c,k} = \log \frac{\exp(sim(i_p^{c,k}, i_p^{g,k}))}{\sum_{j=1}^{|\mathcal{P}|}\exp(sim(i_p^{c,k}, i_p^{g,j}))}$，softmax 归一化得 $w^{c,k} = \frac{\exp(s^{c,k})}{\sum_{c'\in\pi_{img}}\exp(s^{c',k})}$，再加权得聚合教师表示 $i_{agg}^k = \sum_{c\in\pi_{img}} w^{c,k} \cdot i_p^{c,k}$。和服务器越像的客户端贡献越大，从而无需参数一致就能跨异构模型传知识。

### 损失函数 / 训练策略

- **客户端损失**：$\mathcal{L}_{task} + \beta \cdot \mathcal{L}_{adv}$，$\beta=0.5$
- **服务器蒸馏损失**：$\mathcal{L}_{kd} = \frac{1}{|\mathcal{P}|}\sum_{k}(\|i_{agg}^k - i_p^{g,k}\|_2 + \|t_{agg}^k - t_p^{g,k}\|_2)$，$\gamma=0.4$
- 训练策略：40 轮通信，每轮 5 个本地 epoch，共 200 次本地更新
- 客户端判别器和编码器交替对抗训练

## 实验关键数据

### 主实验

设定：3 个图像客户端（CIFAR-100）、3 个文本客户端（AGNEWS）、4 个多模态客户端（Flickr30k），服务器任务 MS-COCO 检索。

| 方法 | CIFAR-100 acc@1 | AGNEWS acc@1 | Flickr30k i2t R@1 | MS-COCO rsum R@1 | 收敛轮数 |
|------|----------------|-------------|-------------------|-----------------|---------|
| LOCAL | 28.07 | 48.35 | 22.33 | 57.54 | 29 |
| FedMD | 22.54 | 48.18 | 19.13 | 58.47 | 25 |
| FedGEMS | 22.84 | 48.30 | 18.93 | 58.62 | 27 |
| CreamFL | 22.14 | 42.16 | 18.38 | 59.61 | 21 |
| FedET | 31.86 | 49.38 | 22.63 | 58.92 | 27 |
| FedMKD | 24.99 | 47.99 | 22.33 | 59.18 | 21 |
| FedDFA | 23.09 | 43.79 | 19.68 | 59.10 | 26 |
| **FedAFD** | **33.18** | **51.98** | **32.48** | **60.16** | **20** |

Non-IID 设置。IID 设置下优势更大：CIFAR-100 上 FedAFD 61.04% vs FedET 46.44%，AGNEWS 89.34% vs 86.07%。FedAFD 在客户端和服务器端均显著优于所有基线，尤其 Flickr30k i2t 检索提升 **+10 个点**。注意许多 baseline 的 client 性能甚至低于 LOCAL，说明全局优化损害了个性化——FedAFD 是唯一能同时提升两端的方法。

### 消融实验

| 配置 | CIFAR-100 | AGNEWS | MS-COCO rsum | 说明 |
|------|----------|--------|-------------|------|
| FedAFD (Full) | 33.18 | 51.98 | 60.16 | 完整框架 |
| w/o BAA | 33.56 | 49.03 | 59.29 | 去掉对抗对齐，服务器性能下降 |
| w/o GFF | 24.94 | 44.46 | 59.72 | 去掉特征融合，客户端性能暴跌 |
| w/o SED | 32.21 | 50.20 | 59.56 | 去掉集成蒸馏，全局性能下降 |

### 关键发现

- **GFF 对客户端性能至关重要**：去掉 GFF 后 CIFAR-100 准确率从 33.18% 降到 24.94%（-8.24%）
- **BAA 主要提升服务器性能**：去掉后 rsum 从 60.16 降到 59.29，同时文本客户端也受影响
- **公共数据量影响**：公共数据从 10k 增至 30k，服务器 rsum 从 60.16 提升到 78.09，但客户端性能略有下降
- **收敛效率**：FedAFD 仅需 20 轮达到基线目标（57.50），其他方法需 21-29 轮

## 亮点与洞察

1. **三阶段统一设计**：首次在一个框架中同时解决跨模态/任务对齐、任务感知个性化和架构无关聚合
2. **将MFL建模为域适应问题**：通过对抗学习最小化客户端-服务器表示分布差异，理论基础清晰
3. **双向优化**：不同于仅关注全局或本地的方法，FedAFD 同时提升两端性能
4. **表示级蒸馏**：无需参数级一致性即可跨异构模型传递知识

## 局限与展望

1. **公共数据依赖**：框架依赖一个公共多模态数据集 $\mathcal{P}$，在数据敏感场景中获取公共数据可能受限
2. **判别器开销**：每个客户端需额外维护两个判别器，增加了计算和通信负担
3. **扩展性验证不足**：实验仅包含 10 个客户端，大规模场景（100+ 客户端）下的表现未知
4. **模态类型有限**：仅验证了图像和文本两种模态，音频/视频等多模态场景待探索

## 相关工作与启发

- **CreamFL**：使用模态内/跨模态对比正则化，但忽视本地性能，FedAFD 的 GFF 模块解决了这一问题
- **FedDFA**：边界感知蒸馏权重，FedAFD 的 SED 进一步引入样本级动态权重
- **域适应理论**：将联邦学习中的模态/任务差异建模为域适应问题，为 MFL 提供了新视角

## 评分

- **新颖性**: ⭐⭐⭐⭐ 三模块协同设计系统性强，域适应视角新颖
- **实验充分度**: ⭐⭐⭐⭐ 消融实验完整，IID/Non-IID双设置 + T-SNE可视化 + 公共数据量分析，附录含超参和通信开销分析
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，公式推导详实
- **价值**: ⭐⭐⭐⭐ 是 MFL 领域较为完整的解决方案，对异构联邦学习有参考意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] HealSplit: Towards Self-Healing through Adversarial Distillation in Split Federated Learning](../../AAAI2026/ai_safety/healsplit_towards_self-healing_through_adversarial_distillation_in_split_federat.md)
- [\[AAAI 2026\] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation](../../AAAI2026/ai_safety/core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)
- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)
- [\[CVPR 2026\] Domain-Skewed Federated Learning with Feature Decoupling and Calibration](domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)
- [\[CVPR 2026\] ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning](proxyfl_a_proxy-guided_framework_for_federated_semi-supervised_learning.md)

</div>

<!-- RELATED:END -->
