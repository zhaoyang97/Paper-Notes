---
title: >-
  [论文解读] NeurIPT: Foundation Model for Neural Interfaces
description: >-
  [NeurIPS 2025][医学图像][EEG基础模型] NeurIPT是一个面向多样化脑机接口(BCI)应用的EEG基础模型，通过振幅感知掩码预训练(AAMP)、渐进式专家混合(PMoE)架构、3D电极空间编码和脑叶内/跨脑叶池化(IILP)四大创新设计，在八个下游BCI任务上实现了SOTA性能。
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "EEG基础模型"
  - "自监督预训练"
  - "Mixture-of-Experts"
  - "脑电信号"
  - "脑机接口"
---

# NeurIPT: Foundation Model for Neural Interfaces

**会议**: NeurIPS 2025  
**arXiv**: [2510.16548](https://arxiv.org/abs/2510.16548)  
**代码**: [https://github.com/](https://github.com/) (有，项目页面已提供)  
**领域**: 医学影像 / 脑机接口  
**关键词**: EEG基础模型, 自监督预训练, Mixture-of-Experts, 脑电信号, 脑机接口

## 一句话总结
NeurIPT是一个面向多样化脑机接口(BCI)应用的EEG基础模型，通过振幅感知掩码预训练(AAMP)、渐进式专家混合(PMoE)架构、3D电极空间编码和脑叶内/跨脑叶池化(IILP)四大创新设计，在八个下游BCI任务上实现了SOTA性能。

## 研究背景与动机

脑电图(EEG)因其非侵入性、便携性和高时间分辨率，被广泛应用于临床诊断和脑机接口(BCI)。随着大规模EEG数据集的不断涌现，研究者希望像NLP和CV领域一样，建立EEG的基础模型(FM)来实现跨数据集、跨任务的泛化。

然而，现有EEG基础模型方法面临几个关键挑战：

**空间编码的缺陷**：现有位置编码将电极通道视为可互换的，忽略了电极在三维物理空间中的真实排列关系，严重损害了跨数据集的迁移能力。

**掩码预训练的局限**：类似BERT的随机连续段掩码策略，实际上让模型倾向于"局部插值"而非学习有意义的全局表示——模型只需根据掩码区域的相邻点进行简单插值即可重建。

**下游微调的不足**：传统全连接层或全局池化机制无法显式利用不同脑区的区域特征。

**信号异质性**：EEG数据模式极其多样，从睡眠中的慢波振荡到癫痫发作时的快速尖峰，单一前馈网络难以自适应地捕获这些异质的时域动态。

核心切入点：设计一个同时处理时空异质性的EEG基础模型，分别从时域和空域两个维度提出针对性解决方案。

## 方法详解

### 整体框架

NeurIPT采用编码器-解码器架构，基于Crossformer的分层注意力模块，分为预训练和微调两个阶段。预训练阶段通过自监督学习（不依赖任何标注数据），在超过2000小时的EEG数据上学习鲁棒表示；微调阶段在八个下游BCI数据集上进行分类任务。

### 关键设计

1. **3D电极空间编码(3D-Aligned Spatial Encoding)**：

    - 利用EEG电极在国际10-20系统中的三维物理坐标 $(x_d, y_d, z_d)$，分别对三个空间坐标进行正弦编码，然后拼接：$PE^{(s)}_d = \text{Concat}(PE_x(x_d), PE_y(y_d), PE_z(z_d))$
    - 采用单点嵌入（像素级嵌入）保留时域细节，每个数据点同时编码时域和空间信息：$\mathbf{s}_{t,d} = \mathbf{E}\mathbf{x} + PE^{(t)} + PE^{(s)}$
    - 设计动机：原生支持不同时间长度T和空间维度D的变化，无需额外卷积或填充操作，可无缝适配10-05、10-20等不同电极放置标准

2. **振幅感知掩码预训练(AAMP)**：

    - 核心思想：不是随机掩码连续时间段，而是基于信号振幅进行掩码——对每个通道采样一个随机百分位 $\xi_d \sim \mathcal{U}(0,1)$，选取振幅排序后以该百分位为中心、覆盖 $T \cdot \mathcal{P}$ 个点的区间进行掩码
    - 掩码公式：$\mathcal{M} = \{\mathbb{1}\{x_{t,d} \in [\mathcal{L}_d, \mathcal{U}_d]\}\}$
    - 设计动机：振幅作为信号能量的代理，迫使模型学习底层EEG模式而非简单的局部插值。由于被掩码的点在时间轴上不连续（同一振幅范围的采样点散布在整个时间序列中），模型必须理解全局信号结构才能重建
    - 重建损失：$\mathcal{L}_{AAMP} = \frac{1}{n}(\sum_{i=1}^{n}\|\mathbf{x}^{(i)} - \hat{\mathbf{x}}^{(i)}\|^p)^{1/p}$

3. **渐进式专家混合(PMoE)**：

    - EEG信号包含复杂异质信息（不同频段、瞬态事件、伪迹），单一FFN难以处理。PMoE在更深层逐渐引入更多专家子网络：浅层使用较少专家（如[0,0,2,4,4,6]），深层使用更多专家
    - 每层的输出：$\text{PMoE}^{(l)}(\hat{\mathbf{Z}}^l) = \sum_{e=1}^{E_l} g_e^l \odot Y_e^l + \text{FFN}_{shared}^{(l)}(\hat{\mathbf{Z}}^l)$
    - 共享专家捕获通用模式，逐层引入的专业专家处理越来越特化的信号特征
    - 使用TopKSoftmax稀疏激活来节省计算；辅助损失 $\mathcal{L}_{aux}$ 确保专家利用均衡

4. **脑叶内/跨脑叶池化(IILP)**：

    - 微调阶段使用的两步池化策略。首先沿时间轴平均池化：$\widetilde{\mathbf{V}}_d^l = \frac{1}{T}\sum_{t=1}^{T}\mathbf{Z}_{t,d}^{enc,l}$
    - **脑叶内池化**：将EEG通道按功能脑叶分区（如额叶、枕叶），每个脑叶内取平均：$V_k^l = \frac{1}{|P_k|}\sum_{d \in P_k}\widetilde{\mathbf{V}}_d^l$
    - **跨脑叶拼接**：将各脑叶嵌入拼接，再跨所有编码器层堆叠形成最终表示
    - 设计动机：显式利用不同脑区的功能差异，如癫痫检测和抑郁分类依赖不同脑区的区域信号变化

### 损失函数 / 训练策略
- 预训练：$\ell_p$范数重建损失 + MoE辅助均衡损失
- 预训练使用AdamW优化器 + OneCycle学习率策略，约400K步，8卡RTX 4090，bfloat16混合精度
- 微调：在八个下游数据集上分别微调，采用交叉熵分类损失

## 实验关键数据

### 主实验

| 数据集 | 指标(Balanced Acc) | NeurIPT | 之前SOTA(CBraMod) | 提升 |
|--------|------|------|----------|------|
| MentalArithmetic | Balanced Acc | 86.46 | 72.56 | +13.90 |
| Mumtaz2016 | Balanced Acc | 98.03 | 95.60 | +2.43 |
| PhysioP300 | Balanced Acc | 67.31 | 65.02(EEGPT) | +2.29 |
| Sleep-EDFx | Balanced Acc | 70.47 | 69.17(EEGPT) | +1.30 |
| BCIC-IV-2A | Balanced Acc | 55.04 | 51.38 | +3.66 |
| TUEV | Balanced Acc | 67.61 | 66.71 | +0.90 |

### 消融实验

| 配置 | TUEV | MentalArith | Mumtaz | 说明 |
|------|---------|------|------|------|
| 无任何组件 | 51.80 | 73.36 | 91.83 | 基线 |
| 仅3D PE | 59.64 | 73.61 | 86.07 | 空间编码帮助TUEV+7.8 |
| 仅PMoE | 52.79 | 74.65 | 85.58 | 仅MoE提升有限 |
| 仅IILP | 59.10 | 73.96 | 91.55 | 池化帮助TUEV+7.3 |
| 全部组合 | 68.94 | 75.69 | 97.07 | 组件协同效果最优 |

**池化策略对比**（BCIC-IV-2A数据集）:

| 策略 | Balanced Acc | 说明 |
|------|---------|------|
| 无池化 | 45.14 | — |
| 均值池化 | 37.24 | 丢失区域信息 |
| IILP | 55.04 | 显式利用脑区特征，+17.8 vs均值池化 |

### 关键发现
- PMoE的渐进式策略优于均匀分配和递减分配，且对具体分配方案鲁棒
- IILP在需要跨脑区差异分析的任务（癫痫/抑郁）上提升尤其显著
- BCIC-IV-2A等运动想象任务对空间信息高度敏感（去掉3D PE后性能显著下降）
- 不同任务类别激活不同数量的MoE专家，体现了PMoE的自适应性

## 亮点与洞察
- 振幅感知掩码是一个非常巧妙的设计：利用EEG信号振幅作为掩码依据，在时间轴上产生不连续的掩码模式，迫使模型学习全局而非局部模式
- 3D电极编码使模型天然兼容不同电极系统，解决了跨数据集迁移的核心瓶颈
- 预训练2000+小时数据、8卡4090训练，展示了EEG基础模型的可扩展性
- 注意力分数可视化显示模型确实学到了有意义的脑区交互模式（如手部任务的对侧激活）

## 局限与展望
- 在TUAB数据集的Cohen's Kappa和AUROC上未超过CBraMod，可能与该数据集的特性有关
- 预训练数据规模（2000小时）相比NLP/CV仍然有限，更大规模数据可能带来更多提升
- IILP的脑叶划分目前是固定的，数据驱动的自适应脑区划分可能更优
- 目前仅评估了分类任务，回归任务（如情绪评分）的表现未知

## 相关工作与启发
- BENDR、LaBraM、CBraMod等先前EEG基础模型的迭代启发了本文的设计
- MAE的掩码重建思路被AAMP创新性地改造为振幅感知版本
- MoE在LLM中的成功应用启发了PMoE的设计
- 脑叶池化的思路可推广到其他需要区域特征聚合的多通道生理信号

## 评分
- 新颖性: ⭐⭐⭐⭐ 四个组件均有创新，AAMP尤其有创意，但整体是已有技术的巧妙组合
- 实验充分度: ⭐⭐⭐⭐⭐ 八个数据集全面评估，消融实验详尽
- 写作质量: ⭐⭐⭐⭐ 结构清晰，可视化丰富，但部分细节在附录中
- 价值: ⭐⭐⭐⭐ 推进EEG基础模型的SOTA，为BCI领域提供了实用方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals](brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)
- [\[NeurIPS 2025\] Brain Harmony: A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens](brain_harmony_a_multimodal_foundation_model_unifying_morphology_and_function_int.md)
- [\[NeurIPS 2025\] Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation](toward_a_vision-language_foundation_model_for_medical_data_multimodal_dataset_an.md)
- [\[NeurIPS 2025\] Generalizable, Real-Time Neural Decoding with Hybrid State-Space Models](generalizable_real-time_neural_decoding_with_hybrid_state-space_models.md)
- [\[CVPR 2025\] UNIStainNet: Foundation-Model-Guided Virtual Staining of H&E to IHC](../../CVPR2025/medical_imaging/unistainnet_foundation-model-guided_virtual_staining_of_he_to_ihc.md)

</div>

<!-- RELATED:END -->
