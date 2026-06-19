---
title: >-
  [论文解读] Do ImageNet-trained Models Learn Shortcuts? The Impact of Frequency Shortcuts on Generalization
description: >-
  [CVPR 2025][频率捷径] 提出层次化频率捷径搜索方法（HFSS），首次在ImageNet-1K规模上高效发现CNN和Transformer学到的频率捷径（仅5%频率即可正确分类），揭示频率捷径在保留纹理的OOD测试中反而有益但在风格化测试（IN-R/IN-S）上有害，指出现有OOD评估框架忽视了频率捷径的影响。
tags:
  - "CVPR 2025"
  - "频率捷径"
  - "OOD泛化"
  - "傅里叶分析"
  - "纹理偏置"
  - "模型鲁棒性"
---

# Do ImageNet-trained Models Learn Shortcuts? The Impact of Frequency Shortcuts on Generalization

**会议**: CVPR 2025  
**arXiv**: [2503.03519](https://arxiv.org/abs/2503.03519)  
**代码**: [https://github.com/nis-research/hfss](https://github.com/nis-research/hfss)  
**领域**: LLM评测  
**关键词**: 频率捷径、OOD泛化、傅里叶分析、纹理偏置、模型鲁棒性

## 一句话总结
提出层次化频率捷径搜索方法（HFSS），首次在ImageNet-1K规模上高效发现CNN和Transformer学到的频率捷径（仅5%频率即可正确分类），揭示频率捷径在保留纹理的OOD测试中反而有益但在风格化测试（IN-R/IN-S）上有害，指出现有OOD评估框架忽视了频率捷径的影响。

## 研究背景与动机

**领域现状**：深度模型容易学到训练数据中的捷径特征（spurious correlations）而非真正的语义特征。频率域中的捷径——模型依赖的少量频率子集——更隐蔽且无法通过视觉检查发现。已知ImageNet模型存在纹理偏置，但缺乏大规模的频率捷径分析。

**现有痛点**：现有频率捷径识别方法（Wang et al.）逐个移除单个频率评估影响，计算量与类别数和图像分辨率成正比——在ImageNet-1K上需要约354天（8500小时），完全不可行。且逐频率评估忽略了频率间的联合贡献。

**核心矛盾**：大规模频率捷径分析在计算上不可行，导致我们不知道ImageNet训练的模型是否真的学了频率捷径，以及这些捷径如何影响不同OOD场景下的泛化性。

**本文目标** 开发高效的频率捷径搜索方法，揭示ImageNet模型的频率捷径学习行为，并分析其对不同OOD场景泛化性的影响。

**切入角度**：用层次化搜索（coarse-to-fine）替代穷举搜索，在频率域上分多阶段从大patch到小patch逐步缩小搜索空间，结合随机采样评估频率组合的联合贡献。

**核心 idea**：通过层次化傅里叶频率搜索在ImageNet规模上揭示：模型确实学了频率捷径，但这些捷径在纹理保持的OOD测试中是"好的捷径"，仅在风格化测试中才有害。

## 方法详解

### 整体框架
HFSS分多个搜索阶段，每阶段：（1）将频率谱分成patches，随机采样p%的patch组合成B个候选频率子集；（2）用模型在仅保留这些频率的图像上测试分类损失，评估每个子集的捷径信息量；（3）取top-N子集进入下一阶段更细粒度的搜索。最终输出每个类的Dominant Frequency Map（DFM）。然后将类分为shortcut类和non-shortcut类，分析两组在不同OOD数据上的表现差异。

### 关键设计

1. **层次化频率搜索（HFSS）**:

    - 功能：高效找到模型依赖的频率子集
    - 核心思路：频率谱从大patch（56×56）到小patch（2×2）分6个阶段搜索。每阶段随机采样60%的patch组成候选子集，计算模型在频率过滤图像上的类别损失作为评估指标。后续阶段仅在前一阶段的top-N子集内搜索，搜索空间指数缩小。ImageNet-1K上从354天降到9.2天（38倍加速）
    - 设计动机：随机采样+层次缩小 = 考虑频率联合贡献的同时大幅降低计算量；不同于穷举逐频率评估

2. **捷径度量与分类**:

    - 功能：量化模型的频率捷径学习程度
    - 核心思路：对每个类计算原始图像上的TPR和DFM过滤图像上的$\text{TPR}^{DFM}$。如果$\text{TPR}^{DFM} > t$（阈值），该类被判为shortcut类。计算shortcut类和non-shortcut类分别在ID和OOD数据上的平均TPR，通过两组的表现差异来揭示捷径的影响
    - 设计动机：避免对所有类取均值（non-shortcut类的$\text{TPR}^{DFM}$接近0会淹没分析结果），分组分析更有洞察力

3. **OOD场景差异化分析**:

    - 功能：揭示频率捷径在不同OOD场景下的正面/负面影响
    - 核心思路：在保留纹理信息的OOD测试（IN-v2、IN-C、FGSM对抗）中，shortcut类表现优于non-shortcut类，说明捷径"有用"；在风格化/素描OOD测试（IN-R、IN-S）中，shortcut类表现反差或更差，因为频率捷径对应的纹理信息在这些数据中不存在
    - 设计动机：打破"捷径一定有害"的简单认知，指出影响取决于OOD数据特性

### 损失函数 / 训练策略
HFSS不需要训练，只需对已训练模型做推理评估。用标准交叉熵损失衡量频率子集的分类相关性。实验覆盖ResNet-18/50、ViT-b和CCT四种架构。

## 实验关键数据

### 主实验

| 发现 | 关键数据 |
|------|---------|
| ImageNet模型确实学捷径 | 仅5%频率的DFM图像，shortcut类（t=0.5）TPR>60%（ResNet18） |
| IN-v2/IN-C上捷径有益 | shortcut类TPR > non-shortcut类TPR（所有模型一致） |
| IN-R上捷径有害 | shortcut类TPR < non-shortcut类TPR（rendition改变破坏纹理） |
| IN-S上捷径中性 | 两组TPR接近（sketch保留部分结构信息） |
| FGSM对抗上捷径有益 | shortcut类更鲁棒（对抗噪声难以改变纹理信息） |

### 消融实验

| 配置 | 搜索时间 | 找到的shortcut类数 | 说明 |
|------|---------|------------------|------|
| CF-1 (最完整) | ~7.5h (CIFAR) | 基准 | 完整搜索 |
| CF-2.10 (最高效) | ~0.04h | ~90%覆盖(低阈值) | 200倍加速 |
| Wang et al. [41] | 7.5h (CIFAR) | 仅2类强捷径 | 逐频率评估 |
| HFSS (ours) | 0.5h (CIFAR) | 6类强捷径 | 联合频率贡献 |

### 关键发现
- **CCT的有趣行为**：CCT的shortcut学习程度最高，但同时也学到了其他语义特征。这种"学捷径但不阻碍真正学习"的模式反而使CCT在IN-C（57.73%）上优于类似精度的ResNet50（48.85%）
- HFSS比现有方法更有效找到捷径（6类 vs 2类强捷径），因为考虑了频率的联合贡献
- 频率捷径与纹理偏置直接关联——捷径对应的主要是纹理而非形状信息

## 亮点与洞察
- **颠覆性结论：捷径不一定有害**：在保留纹理的OOD场景中（包括对抗攻击），频率捷径反而增强了模型鲁棒性。这对"消除所有捷径"的研究范式提出了重要挑战
- **层次化搜索的高效性**：将354天降到9.2天，首次使大规模频率分析可行。coarse-to-fine + 随机采样的思路可以迁移到其他大规模搜索问题
- **对OOD评估基准的洞察**：现有OOD评估框架没有区分"捷径是否存在于测试数据中"，可能给出误导性的泛化评估。未来基准应明确考虑频率捷径的影响

## 局限与展望
- HFSS的搜索结果依赖于采样策略和超参数（patch大小、采样比例），可能遗漏某些频率组合
- 分析仅限于分类任务，检测/分割等任务的频率捷径尚未研究
- 仅分析了训练完成后的模型，未探讨训练过程中频率捷径的形成动态
- 缺少捷径缓解方法的提出——发现了问题但没给出解决方案

## 相关工作与启发
- **vs Wang et al. [41]**: 逐频率评估 vs 联合频率搜索，HFSS发现更多且更强的捷径，计算效率提升38倍
- **vs Geirhos et al. [8]（纹理偏置）**: 从频率域角度解释了纹理偏置的形成机制——频率捷径就是纹理偏置的根本原因
- **vs OOD评估基准**: 揭示了IN-R/IN-S等基准测试的有效性部分来自于去除了训练数据中的频率捷径

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次大规模频率捷径分析，"捷径可以有益"的发现有启发性
- 实验充分度: ⭐⭐⭐⭐ 多模型、多OOD场景、与现有方法对比、配置消融
- 写作质量: ⭐⭐⭐⭐⭐ 清晰的实验设计和分析逻辑，图示直观
- 价值: ⭐⭐⭐⭐ 对理解模型泛化和设计OOD基准有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Implicit Reasoning in Transformers is Reasoning through Shortcuts](../../ACL2025/others/implicit_reasoning_in_transformers_is_reasoning_through_shortcuts.md)
- [\[CVPR 2025\] On the Generalization of Handwritten Text Recognition Models](on_the_generalization_of_handwritten_text_recognition_models.md)
- [\[ICML 2025\] How Do Transformers Learn Variable Binding in Symbolic Programs?](../../ICML2025/others/how_do_transformers_learn_variable_binding_in_symbolic_programs.md)
- [\[NeurIPS 2025\] Impact of Layer Norm on Memorization and Generalization in Transformers](../../NeurIPS2025/others/impact_of_layer_norm_on_memorization_and_generalization_in_transformers.md)
- [\[CVPR 2025\] EBS-EKF: Accurate and High Frequency Event-based Star Tracking](ebs-ekf_accurate_and_high_frequency_event-based_star_tracking.md)

</div>

<!-- RELATED:END -->
