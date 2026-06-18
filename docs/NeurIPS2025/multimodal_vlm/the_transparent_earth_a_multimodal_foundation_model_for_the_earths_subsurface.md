---
title: >-
  [论文解读] The Transparent Earth: A Multimodal Foundation Model for the Earth's Subsurface
description: >-
  [NeurIPS 2025][信息检索/RAG][地球科学基础模型] 提出Transparent Earth，一种基于Transformer的多模态基础模型，通过位置编码和文本衍生的模态嵌入融合8种异质地球物理观测数据，实现地球地下属性的零样本推断和上下文学习预测。 地球科学领域的机器学习模型长期面临"学科割裂"问题： 数据…
tags:
  - "NeurIPS 2025"
  - "信息检索/RAG"
  - "地球科学基础模型"
  - "多模态融合"
  - "Transformer"
  - "地下属性重建"
  - "上下文学习"
---

# The Transparent Earth: A Multimodal Foundation Model for the Earth's Subsurface

**会议**: NeurIPS 2025  
**arXiv**: [2509.02783](https://arxiv.org/abs/2509.02783)  
**代码**: 暂无  
**领域**: 信息检索  
**关键词**: 地球科学基础模型, 多模态融合, Transformer, 地下属性重建, 上下文学习

## 一句话总结

提出Transparent Earth，一种基于Transformer的多模态基础模型，通过位置编码和文本衍生的模态嵌入融合8种异质地球物理观测数据，实现地球地下属性的零样本推断和上下文学习预测。

## 研究背景与动机

地球科学领域的机器学习模型长期面临"学科割裂"问题：

**数据异质性**：地球物理观测数据跨越不同模态（应力角、应变角、板块类型、断层类型、盆地类型/年龄、沉积厚度、地幔温度），各模态在空间分辨率、稀疏程度和数据类型上差异巨大

**学科专业化**：现有基础模型局限于单一子领域（地震学、气候预测、海洋学），无法跨学科利用信息

**数据稀疏性**：许多地区的直接观测数据稀缺且昂贵，需要模型从有限观测中泛化

关键洞察是：**这些模态之间存在内在物理关联**——沉积盆地的类型、年龄和厚度与断层类型和位置相关，后者又与地震事件和当前应力场相关。一个统一模型如果能同时学习这些关联，就能在数据稀疏区域做出更准确的预测。

## 方法详解

### 整体框架

Transparent Earth采用编码器-解码器Transformer架构，核心流程为：1) 对每种模态的观测进行输入处理（特征+位置编码+模态嵌入），2) 沿序列维度融合所有模态，3) 通过交叉注意力和自注意力编码到共享隐空间，4) 解码器基于查询位置和任务嵌入生成预测。

### 关键设计

1. **输入处理与模态融合**

   每个观测被变换为一个token嵌入，编码内容和位置信息。对每个模态 $\mathcal{M}_i$，随机采样 $k_i \sim \mathcal{U}(1, k_{\max})$ 个观测点，提取特征向量 $f_i$ 和空间坐标 $c_i$，然后拼接三个组件：

    $x_i = [f_i \| p_i \| m_i]$

   其中 $p_i = \text{PosEnc}(c_i)$ 是正弦位置编码（带深度维度扩展），$m_i = \text{ModEmbed}(\mathcal{M}_i)$ 是使用多语言E5文本嵌入模型从模态名称描述生成的模态嵌入。

   位置编码的频率选择遵循Nyquist-Shannon采样定理：对0.5°×0.5°目标分辨率，$f_{\max}^{lat} = 36$，$f_{\max}^{lon} = 72$：

    $\mathbf{e} = [\sin(\pi\phi \cdot \mathbf{f}_\phi), \cos(\pi\phi \cdot \mathbf{f}_\phi), \sin(\pi\lambda \cdot \mathbf{f}_\lambda), \cos(\pi\lambda \cdot \mathbf{f}_\lambda), z] \in \mathbb{R}^{4F+1}$

   地表模态设 $z=0$，地幔温度等深度依赖模态使用归一化深度值。文本嵌入的**可扩展性优势**：新增模态只需提供文字描述即可生成嵌入，无需重新训练。

2. **编码器设计**

   融合后的序列首先通过交叉注意力层，其中可学习的隐查询向量作为query、输入token作为key/value，将信息压缩到固定大小的隐空间。随后经过3个自注意力+MLP层，捕获高阶跨模态交互。编码器的核心作用是学习模态间的隐式物理关联。

3. **解码器设计**

   查询驱动的解码器支持在任意位置预测任意模态。每个查询点由位置编码 $p_i$ 和任务嵌入 $e_i$ 拼接而成：$Q_i = [p_i \| e_i]$。解码器通过多头交叉注意力从编码器的隐表征中提取信息，后接4层MLP生成预测。

### 损失函数 / 训练策略

针对不同模态使用特定损失函数：
- **角度量（应力角/应变角）**：周期性角度损失 $\mathcal{L}_{angular} = \frac{1}{N}\sum_{i=1}^N\left(\frac{(\hat\theta_i - \theta_i + R/2) \bmod R - R/2}{R/2}\right)^2$
- **分类任务（板块类型等）**：交叉熵损失
- **回归任务（沉积厚度等）**：均方误差

总损失为各模态损失的均匀平均。训练时随机采样观测数量和可用模态，作为模态级dropout促进泛化。

## 实验关键数据

### 主实验——模型缩放性能

| 模型规模 | 应力角MAE↓ | 应变角MAE↓ | 板块分类Acc↑ | 断层分类Acc↑ | 盆地类型Acc↑ |
|---------|-----------|-----------|-------------|-------------|-------------|
| 3M (基线) | ~33° | ~35° | >95% | ~75% | ~88% |
| 30M | 改善 | 改善 | >95% | ~82% | ~92% |
| 243M | 最佳 | 最佳 | >97% | ~88% | >95% |

### 上下文学习——全局推断

| 观测配置 | 应力角MAE | 应变角MAE |
|---------|----------|----------|
| 无输入（纯先验） | ~33° | ~35° |
| 2个同模态观测 | ~25° | ~28° |
| 8个同模态观测 | ~20° | ~22° |
| 2个×8模态（16总） | **~9°** | **~13°** |

多模态融合将应力角误差降低超过3倍。

### 关键发现

1. **缩放定律成立**：从3M到243M参数，所有回归任务的MAE持续降低，分类准确率持续提升
2. **多模态融合优于单模态**：仅用应变模态训练的模型即使有很多观测也难以降低MAE，而全模态训练的基线模型仅需少量观测即可大幅改善
3. **上下文学习能力**：模型能根据输入观测的数量和模态自适应调整预测精度，从"无输入退化到全局先验"到"多观测精准预测"平滑过渡
4. 板块分类任务在所有模型规模下都保持>95%准确率，显示该任务相对简单

## 亮点与洞察

1. **统一多学科**：首个将8种不同分辨率/类型的地球物理模态融入单一模型的尝试，打破学科壁垒
2. **文本嵌入实现可扩展性**：使用预训练文本模型生成模态嵌入，新模态只需文字描述即可加入，无需架构修改
3. **随机采样训练策略**：随机化观测数量和可用模态，天然实现缺失数据鲁棒性
4. **跨模态信息增益显著**：多模态融合带来的误差减少远超同模态增加观测数

## 局限与展望

- 当前仅8种模态，离全面覆盖地下属性还有距离
- 某些模态的空间分辨率较低（如地幔温度5°×5°），可能限制精细预测
- 稀疏模态（如应力角）的数据质量直接影响模型性能
- 未与传统地统计方法（如克里金插值）做系统对比
- 深度依赖模态仅使用归一化深度值，可能不够精细

## 相关工作与启发

- **Aurora/Pangu**: 气候/天气基础模型，但局限于大气层
- **地震基础模型**: 仅处理地震波数据的单模态模型
- **Perceiver IO**: 编码器的交叉注意力设计借鉴了这一架构

## 评分

- 新颖性: ⭐⭐⭐⭐☆ — 多模态地下建模是新颖的问题定义
- 实验充分度: ⭐⭐⭐☆☆ — 缩放实验和上下文学习有说服力，但缺少与传统方法的对比
- 写作质量: ⭐⭐⭐⭐☆ — Workshop论文篇幅有限但结构完整
- 价值: ⭐⭐⭐⭐☆ — 为地球科学基础模型开辟了多模态方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant](../../CVPR2025/information_retrieval/lamra_large_multimodal_model_as_your_advanced_retrieval_assistant.md)
- [\[CVPR 2026\] ProM3E: Probabilistic Masked MultiModal Embedding Model for Ecology](../../CVPR2026/information_retrieval/prom3e_probabilistic_masked_multimodal_embedding_model_for_ecology.md)
- [\[CVPR 2026\] MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model](../../CVPR2026/information_retrieval/muco_multi-turn_contrastive_learning_for_multimodal_embedding_model.md)
- [\[NeurIPS 2025\] Generalized Contrastive Learning for Universal Multimodal Retrieval](generalized_contrastive_learning_for_universal_multimodal_re.md)
- [\[NeurIPS 2025\] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation](windsock_is_dancing_adaptive_multimodal_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
