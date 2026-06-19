---
title: >-
  [论文解读] Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning
description: >-
  [CVPR 2025][医学图像][第三磨牙] 本文比较了局部学习(LL)、联邦学习(FL)和集中式学习(CL)三种范式在全景X光片上自动分类第三磨牙与下颌管重叠关系的性能，使用预训练ResNet-34作为骨干网络，发现集中式训练性能最优(AUC 0.831)，而FL在隐私保护前提下显著优于纯局部训练。
tags:
  - "CVPR 2025"
  - "医学图像"
  - "第三磨牙"
  - "下颌管"
  - "联邦学习"
  - "全景X光片"
  - "深度学习分类"
---

# Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning

**会议**: CVPR 2025  
**arXiv**: [2603.11850](https://arxiv.org/abs/2603.11850)  
**代码**: 无  
**领域**: 医学图像  
**关键词**: 第三磨牙, 下颌管, 联邦学习, 全景X光片, 深度学习分类

## 一句话总结

本文比较了局部学习(LL)、联邦学习(FL)和集中式学习(CL)三种范式在全景X光片上自动分类第三磨牙与下颌管重叠关系的性能，使用预训练ResNet-34作为骨干网络，发现集中式训练性能最优(AUC 0.831)，而FL在隐私保护前提下显著优于纯局部训练。

## 研究背景与动机

**领域现状**：下颌第三磨牙（智齿）嵌入且靠近下颌管时，拔牙手术可能导致下齿槽神经损伤。临床上常规使用全景X光片评估磨牙与下颌管的位置关系，但这一评估过程依赖医生的主观判断，存在标注者间一致性较低的问题。CBCT可以提供三维信息，但并非所有病例都需要做CBCT，自动化二分类（重叠/不重叠）有助于临床分诊。

**现有痛点**：传统的集中式深度学习需要将多个医疗机构的数据汇集到一处进行训练，这在隐私法规（如GDPR和HIPAA）日趋严格的背景下变得不可行。各医疗中心独立训练的局部模型由于数据量有限、标注偏差等问题，泛化能力往往较差。

**核心矛盾**：医学影像AI需要大量多样化数据来获得良好泛化性，但数据隐私法规禁止跨机构数据共享，这构成了性能与隐私之间的根本矛盾。

**本文目标**：系统性比较三种学习范式——局部学习(LL)、联邦学习(FL)和集中式学习(CL)——在多标注者全景X光片数据集上的分类性能，定量评估FL作为隐私保护替代方案的可行性。

**切入角度**：将来自8位独立标注者的数据视为8个独立客户端/中心，模拟多中心场景下的三种训练策略，通过per-client指标和全局阈值指标全面评估各范式的性能差异。

**核心 idea**：通过严格对比实验证明，联邦学习可以在不共享原始数据的前提下，显著超越各中心独立训练的局部模型，虽然略逊于集中式训练，但提供了有效的隐私-性能折中方案。

## 方法详解

### 整体框架

输入为裁剪后的全景X光片（以第三磨牙区域为中心），经过预训练ResNet-34提取特征，最终二分类输出磨牙与下颌管的重叠/不重叠关系。三种训练范式使用相同的网络架构和超参数，仅在数据组织和参数聚合策略上有所不同。

### 关键设计

1. **数据划分与多客户端模拟**:

    - 功能：将全景X光数据集按标注者划分为8个独立客户端，模拟真实的多中心场景
    - 核心思路：8位独立标注者分别标注了一批全景X光片上第三磨牙与下颌管的重叠关系（二分类：overlap vs no-overlap），每位标注者的数据作为一个"客户端"的本地数据集。每个客户端内部进一步划分训练集和测试集
    - 设计动机：这种划分方式不仅模拟了现实中多院区各自拥有不同标注数据的情况，还自然引入了标注者间的偏差，使实验更贴近实际应用场景

2. **三种学习范式的统一比较**:

    - 功能：在相同网络架构(ResNet-34)和超参数条件下，公平比较LL、FL、CL三种方案
    - 核心思路：**局部学习(LL)** — 每个客户端独立用本地数据微调ResNet-34；**联邦学习(FL)** — 采用FedAvg算法，各客户端本地训练后上传模型参数到服务器端聚合，再将聚合参数下发到各客户端进行下一轮训练；**集中式学习(CL)** — 将所有8个客户端数据合并成一个大数据集统一训练。三种范式均使用ImageNet预训练的ResNet-34
    - 设计动机：使用同一骨干网络可以排除架构差异的影响，纯粹评估数据组织和聚合策略对性能的影响

3. **双重评估策略**:

    - 功能：分别使用per-client局部最优阈值和全局统一阈值评估模型
    - 核心思路：对每个客户端的测试数据，首先通过ROC曲线找到该客户端特定的最优阈值进行per-client评估；然后在所有客户端的pooled测试集上使用统一阈值进行全局评估。评估指标包括AUC、准确率、灵敏度、特异度等
    - 设计动机：per-client评估反映模型在各个独立中心的适应性，全局阈值评估反映模型的跨中心普适性

### 损失函数 / 训练策略

使用标准的二分类交叉熵损失函数训练ResNet-34。FL训练采用FedAvg策略，每轮各客户端本地训练若干epoch后上传参数。训练过程中通过Grad-CAM可视化监控模型注意力区域是否聚焦于解剖学相关结构。

## 实验关键数据

### 主实验

| 学习范式 | AUC (pooled) | 准确率 (pooled) | 备注 |
|---------|-------------|----------------|------|
| 集中式学习 (CL) | **0.831** | **0.782** | 性能最优 |
| 联邦学习 (FL) | 0.757 | 0.703 | 隐私保护下次优 |
| 局部学习 (LL) | 0.619–0.734 (均值0.672) | — | 各客户端差异大 |

### Per-Client AUC对比

| 客户端 | LL (AUC) | FL (AUC) | CL (AUC) |
|--------|---------|---------|---------|
| 最佳客户端 | 0.734 | — | — |
| 最差客户端 | 0.619 | — | — |
| 均值 | 0.672 | 0.757 | 0.831 |

### 关键发现

- **CL全面领先**：集中式训练的AUC比FL高约0.074，比LL均值高约0.159，表明数据量和多样性对模型性能的正面影响显著
- **FL有效弥补数据孤岛**：FL在不共享数据的前提下，AUC比最好的LL客户端还高0.023，显著优于大多数LL模型，证明了联邦聚合的有效性
- **LL存在明显过拟合**：训练曲线显示LL模型过拟合最为严重，特别是在数据量较小的客户端上，泛化能力极差
- **Grad-CAM分析**：CL和FL模型的注意力更聚焦于磨牙-下颌管交界的解剖区域，而LL模型的注意力往往更分散，说明更多数据帮助模型学到更有意义的特征

## 亮点与洞察

- **系统性的三范式比较框架**：在医学影像这一隐私敏感领域，本文提供了LL/FL/CL的完整对比基线，为后续研究者选择训练策略提供了清晰参考。这种实验设计思路可以迁移到其他医学影像分类任务中
- **标注者作为客户端的巧妙设计**：利用多标注者的自然分组作为FL中的客户端划分，既模拟了现实中的多中心场景，又避免了人工划分数据带来的偏差
- **Grad-CAM的可解释性分析**：不仅报告数字指标，还通过注意力可视化解释了为什么CL/FL优于LL——因为它们学到了更有解剖学意义的特征表示

## 局限与展望

- **数据集规模有限**：仅来自8位标注者，真实的多中心场景可能涉及数十乃至上百个机构，数据异质性更强
- **任务相对简单**：二分类（重叠/不重叠）是一个较为粗粒度的评估。临床上可能更需要回归或多分类来评估重叠程度和风险等级
- **缺乏跨数据集验证**：所有数据来自相同来源，未验证模型在完全不同设备或机构拍摄的X光片上的泛化性
- **FL隐私保证的量化分析不足**：未讨论差分隐私等更强隐私保证机制的引入，也未分析隐私泄露风险
- 改进方向：可以结合差分隐私(DP)的FedAvg，探索更强隐私保证下的性能折中；也可以将方法推广到多分类或检测任务

## 相关工作与启发

- **vs 传统集中式深度学习**: 传统方法需要数据集中化，本文的CL结果验证了其上限，但FL提供了不需要数据共享的可行替代
- **vs 纯联邦学习方法**: 本文使用最基础的FedAvg，未涉及FedProx、SCAFFOLD等更先进的FL算法，留下了改进空间
- **vs 局部独立训练**: 实验清楚表明局部模型存在严重过拟合和泛化不足，强调了跨中心协作的必要性

## 评分

- 新颖性: ⭐⭐⭐ 方法上无新颖贡献，主要是已有技术在特定医学场景的应用性比较
- 实验充分度: ⭐⭐⭐⭐ 包含多种评估指标和Grad-CAM分析，但数据规模较小
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，实验分析较为详尽
- 价值: ⭐⭐⭐⭐ 为牙科影像AI领域提供了FL的实证参考，有一定临床指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DFLMoE: Decentralized Federated Learning via Mixture of Experts for Medical Data](dflmoe_decentralized_federated_learning_via_mixture_of_experts_for_medical_data_.md)
- [\[CVPR 2025\] CycleULM: A Unified Label-Free Deep Learning Framework for Ultrasound Localisation Microscopy](cycleulm_a_unified_label-free_deep_learning_framework_for_ultrasound_localisatio.md)
- [\[CVPR 2025\] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ovary_using_deep_learning_models.md)
- [\[CVPR 2025\] Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)
- [\[CVPR 2026\] OralGPT-Plus: Learning to Use Visual Tools via Reinforcement Learning for Panoramic X-ray Analysis](../../CVPR2026/medical_imaging/oralgpt-plus_learning_to_use_visual_tools_via_reinforcement_learning_for_panoram.md)

</div>

<!-- RELATED:END -->
