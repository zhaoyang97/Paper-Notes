---
title: >-
  [论文解读] Experience with Single Domain Generalization in Real World Medical Imaging Deployments
description: >-
  [AAAI2026][医学图像][单域泛化] 提出DL+EKE框架，将领域不变的专家知识与深度学习集成，解决医学影像中稀有类（rare class）的单域泛化（SDG）问题，在糖尿病视网膜病变分级、rs-fMRI癫痫灶定位和应激心电图CAD检测三个真实部署场景中显著优于SOTA SDG方法。 域泛化在医学影像中的重要性 域泛…
tags:
  - "AAAI2026"
  - "医学图像"
  - "单域泛化"
  - "稀有类检测"
  - "专家知识集成"
  - "医学影像部署"
  - "糖尿病视网膜病变"
  - "癫痫灶定位"
  - "冠状动脉疾病"
---

# Experience with Single Domain Generalization in Real World Medical Imaging Deployments

**会议**: AAAI2026  
**arXiv**: [2601.16359](https://arxiv.org/abs/2601.16359)  
**作者**: Ayan Banerjee (ASU), Komandoor Srivathsan, Sandeep K.S. Gupta (ASU)  
**代码**: 未公开  
**领域**: 医学图像  
**关键词**: 单域泛化, 稀有类检测, 专家知识集成, 医学影像部署, 糖尿病视网膜病变, 癫痫灶定位, 冠状动脉疾病  

## 一句话总结

提出DL+EKE框架，将领域不变的专家知识与深度学习集成，解决医学影像中稀有类（rare class）的单域泛化（SDG）问题，在糖尿病视网膜病变分级、rs-fMRI癫痫灶定位和应激心电图CAD检测三个真实部署场景中显著优于SOTA SDG方法。

## 研究背景与动机

### 域泛化在医学影像中的重要性
域泛化（Domain Generalization）是部署就绪AI的核心能力。在医学影像领域，扫描仪硬件差异、采集协议变化、患者年龄性别等因素导致不同临床中心之间存在严重的域偏移（domain shift）。多源域泛化（MSDG）和域适应（DA）需要来自多中心的数据，存在隐私和数据获取难题。单域泛化（SDG）仅需单个源域数据，是更实用的方案，但在稀有类场景下挑战更为严峻。

### 稀有类检测的特殊困难
医学领域的关键诊断任务往往对应稀有类，如5期糖尿病视网膜病变、rs-fMRI中的癫痫发作起始区（SOZ，仅占5–10%的独立成分）、应激心电图中的冠状动脉疾病阳性。稀有类具有四个属性：判别性（Discrimination）、稀缺性（Scarcity）、重要性（Significance）和重叠性（Overlap）。由于观测样本极少，纯数据驱动的DL方法无法充分估计稀有类分布，根据Cramer-Rao下界理论，稀有类的类熵$\theta_r$越高，参数估计误差越大。

### 部署失败的真实案例驱动
作者在Mayo Clinic部署ViT模型进行应激心电图CAD检测时，模型在2010年训练/测试数据上PPV达79%、NPV达81.8%，但在2025年的盲测数据上PPV骤降至46%、NPV降至49%。原因是2012年分诊策略变化：此前仅S-T压低患者被转到冠脉造影（ICA），之后即使无S-T压低也可能被转诊。新数据中CAD阳性的心电图包含导联间关系等专家知识因素，这些在旧数据中不存在，导致严重的域偏移。

## 核心问题

如何在仅有单一源域数据的条件下，实现对医学影像中稀有类的可靠泛化检测？SOTA的SDG方法在稀有类场景下性能不足，需要一种新范式来弥补纯数据驱动方法在稀有类分布估计上的固有缺陷。

## 方法详解

### 稀有类定义与量化
利用基于类熵（class-wise entropy）的度量来量化稀有性。给定观测$y_i$的表示$x_i$，定义密度函数：

$$\lambda(x_i) = \frac{1}{|Q(x_i)|} \sum_{j=1}^{|Q(x_i)|} \frac{1}{\text{dist}(x_i, x_j)}$$

类平均密度$\gamma(x_i) = \lambda(x_i) / \sum_{j=1}^{|c_r|} \lambda(x_j)$，类熵定义为：

$$\theta_r = \sum_{i=1}^{|c_r|} (-\gamma(x_i) \log_2 \gamma(x_i))$$

若$\theta_r^z > \theta_M^z + \sigma_\theta^z$或$\theta_r^z < \theta_M^z - \sigma_\theta^z$，则类$c_r$为稀有类。

### RareSaGe算法框架
RareSaGe（RARE class classification for Single domain Generalization）核心流程：

1. **稀有类识别**：用CLIP提取类无关特征嵌入，计算各类的类熵，识别满足Definition 2的稀有类
2. **重叠类定位**：计算稀有类与各非稀有类的CLIP特征余弦相似度，找到最相似的"重叠类"$c_o$
3. **DL机器编排**：用DL技术（ViT/CNN/LVM）区分重叠类vs非重叠类
4. **知识机器编排**：用基于专家知识的分类器（SVM+逻辑规则）区分稀有类vs非稀有类
5. **标签预测器**：集成DL和EKE——DL先分类重叠/非重叠，对非重叠样本直接采用EKE标签；对DL判为重叠但EKE以高置信度（>$t_c$=0.9）判为稀有类的样本，覆盖为稀有类标签

### SOZ检测中的专家知识
利用两类专家知识：(1) 解剖学知识——脑区位置通过图像处理提取；(2) SOZ特异性知识——基于文献的逻辑连接规则：

$$\kappa_{SOZ} = p_1 \wedge \neg p_s \wedge p_a \wedge [p_g \wedge (\neg p_w \vee (p_w \wedge p_v))]$$

其中$p_1$=单一激活簇、$p_s$=正弦域稀疏性、$p_a$=activelet域稀疏性、$p_g$=灰质激活、$p_w$=白质重叠、$p_v$=血管区域重叠。

### CAD检测中的专家知识集成
K1配置整合了最高等级的专家知识：选择5个专家指定导联（而非全部12导联），仅使用最大MET水平数据。对比K2–K4的消融配置验证了专家知识的有效性。

## 实验关键数据

### SOZ跨试验验证（跨中心SDG）

| 方法 | 准确率 | 精确度 | 灵敏度 | F1 | 平均F1 | 消融 |
|------|--------|--------|--------|-----|--------|------|
| Pre-trained ViT (A→B) | 64.5% | 86.9% | 71.4% | 78.4% | 77.2% | DL only |
| Pre-trained ViT (B→A) | 61.5% | 91.4% | 65.3% | 76.1% | — | DL only |
| Knowledge (A→B) | 83.8% | 89.6% | 92.8% | 91.2% | 78.9% | Knowledge only |
| Knowledge (B→A) | 50.0% | 89.6% | 53.0% | 66.6% | — | Knowledge only |
| **DL+EKE (A→B)** | **90.3%** | **90.3%** | **100%** | **94.9%** | **90.2%** | DL+EKE |
| **DL+EKE (B→A)** | **75.0%** | **92.8%** | **79.5%** | **85.6%** | — | DL+EKE |

### CAD检测结果（应激心电图）

| 指标 | 方法 | 验证集 | 测试(2010) | 盲测(2025) |
|------|------|--------|-----------|-----------|
| PPV | ViT | 80.4% | 79.0% | 46.0% |
| PPV | K1 (DL+EKE) | 91.2% | 91.2% | **75.0%** |
| NPV | ViT | 83.0% | 81.8% | 49.0% |
| NPV | K1 (DL+EKE) | 93.0% | 93.0% | **76.0%** |

K1在盲测上PPV和NPV分别比纯ViT高29%和27%，5折交叉验证ROC AUC达92.2(±1.1)。

## 亮点

- **真实部署验证**：不同于纯benchmark的学术工作，本文在Mayo Clinic（CAD）和UNC（SOZ）进行了真实跨中心部署验证，具有极高的实际参考价值
- **理论分析扎实**：通过Cramer-Rao下界和類熵的关系，从信息论角度解释了为何纯DL方法在稀有类上必然失败
- **通用框架设计**：RareSaGe框架是泛化的——任何具备稀有类属性的应用都可实例化，已在DR、SOZ、CAD三个截然不同的领域验证
- **临床交互反馈**：文中记录了Mayo Clinic心内科专家在盲测后提出的实际问题（LIME注意力图解释、数据质量评估、性别差异分析），展现了部署AI的真实挑战

## 局限与展望

- **专家知识编码成本高**：每个应用场景都需要大量人工提取和编码领域专家知识，可扩展性有限
- **知识覆盖阈值$t_c$敏感**：ROC分析显示知识覆盖阈值的选择对性能影响显著，缺乏自动化选择策略
- **仅验证分类任务**：未涉及分割、检测等其他医学影像任务类型
- **跨中心差异预判困难**：文中坦承许多跨中心变异难以预先发现，需要临床与工程团队的持续协作

## 与相关工作的对比

- **SOTA SDG方法**（EPVT, DiMix, ERM++等）：在DR基准上表现一般，DL+EKE在across-trial和aggregate-trial验证中均显著优于这些方法
- **知识增强DL**（Daniele & Serafini 2019）：对输出施加符号约束，但无法建模类内变异性；DL+EKE通过编排策略解决了知识模糊和冲突问题
- **CLIP-DR**（Yu et al. 2024, Li et al. 2022）：language-vision预训练方法在DR分级上各有侧重，DL+EKE在aggregate trial中F1提升4.5%

## 对我的启发

专家知识与DL的集成是一个被低估的方向。在稀有类场景下，增加数据量的边际收益递减，而领域不变的专家知识（如解剖学规则、临床诊断准则）可以提供数据驱动方法无法学到的先验。这种思路可推广到任何存在严重类不平衡的实际部署场景。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 专家知识+DL集成用于SDG稀有类是新视角，但各组件相对标准
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个真实部署场景+DR基准，跨中心验证+盲测，极为充分
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，部署经验描述透彻，附录详尽
- 价值: ⭐⭐⭐⭐ — 对医学AI实际部署有重要指导意义，填补SDG稀有类的空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] PulseMind: A Multi-Modal Medical Model for Real-World Clinical Diagnosis](pulsemind_a_multi-modal_medical_model_for_real-world_clinical_diagnosis.md)
- [\[CVPR 2025\] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](../../CVPR2025/medical_imaging/human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)
- [\[ICML 2026\] Evidential Reasoning Advances Interpretable Real-World Disease Screening](../../ICML2026/medical_imaging/evidential_reasoning_advances_interpretable_real-world_disease_screening.md)
- [\[CVPR 2026\] DK-DDIL: Adaptive Knowledge Retention for Dynamic Domain-Incremental Learning in Medical Imaging](../../CVPR2026/medical_imaging/dk-ddil_adaptive_knowledge_retention_for_dynamic_domain-incremental_learning_in_.md)
- [\[ICML 2025\] LangDAug: Langevin Data Augmentation for Multi-Source Domain Generalization in Medical Image Segmentation](../../ICML2025/medical_imaging/langdaug_langevin_data_augmentation_for_multi-source_domain_generalization_in_me.md)

</div>

<!-- RELATED:END -->
