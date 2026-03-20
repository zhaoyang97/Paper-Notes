# Beyond Brain Decoding: Visual-Semantic Reconstructions to Mental Creation Extension Based on fMRI

**会议**: ICCV 2025  
**arXiv**: 无预印本（仅CVF Open Access）  
**CVF**: [论文页面](https://openaccess.thecvf.com/content/ICCV2025/html/Jing_Beyond_Brain_Decoding_Visual-Semantic_Reconstructions_to_Mental_Creation_Extension_Based_ICCV_2025_paper.html) / [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Jing_Beyond_Brain_Decoding_Visual-Semantic_Reconstructions_to_Mental_Creation_Extension_Based_ICCV_2025_paper.pdf)  
**代码**: 无公开代码  
**作者**: Haodong Jing, Dongyao Jiang, Yongqiang Ma, Haibo Hua, Bo Huang, Nanning Zheng（西安交通大学）  
**领域**: 医学图像/脑科学  
**关键词**: fMRI脑解码, 视觉-语义重建, 心理创造, 大语言模型, 多模态脑架构  

## 一句话总结
提出NeuroCreat——一种结合LLM视觉与文本能力的脑多模态架构，将fMRI解码从单一的视觉刺激重建扩展到**图像重建 + 文本描述（captioning）+ 心理创造（creation）**三个层次，通过Prompt Variant Alignment模块有效弥合fMRI低分辨率信号与高级语义表征之间的鸿沟。

## 背景与动机
从fMRI信号中解码视觉信息是理解大脑如何表征世界的重要途径，也是通用人工智能（AGI）的前沿领域。当前研究的核心局限在于：

1. **目标单一**：主流方法（MindEye、MindEye2、Brain-Diffuser、Unibrain等）几乎全部聚焦于「从fMRI重建所见图像」这一单一任务，很少探索从脑信号生成文字描述或创造全新内容
2. **语义利用不足**：大多方法将fMRI映射到CLIP或SDXL潜空间进行图像重建，但缺乏对高层语义的细粒度挖掘，尤其在文本生成方面能力有限
3. **离"思维创造"尚远**：人类大脑不仅能重现所见，还能从记忆和经验中创造全新的心理图像——这种"创造性"能力在现有脑解码方法中几乎未被探索

同组前序工作"See Through Their Minds"（arXiv:2403.06361）探索了跨被试fMRI解码的可迁移神经表征学习，奠定了本文进一步挖掘fMRI丰富语义内容的基础。

## 核心问题
**如何将fMRI脑解码从单一的视觉刺激重建，扩展到多层次、多模态的脑信号理解——包括视觉重建、语义描述和心理创造？**

具体而言需解决：
- fMRI信号分辨率低、噪声大，如何提取细粒度语义信息？
- 不同输出模态（图像 vs 文本 vs 创造性内容）之间的差异如何处理？
- 如何利用LLM的强大多模态能力来增强脑解码？

## 方法详解

### 整体框架
NeuroCreat是一个**复用式神经解码模型（multiplexed neural decoding model）**，核心思路是将LLM与脑解码有效集成，实现三个层次的输出：

1. **视觉重建（Reconstruction）**：从fMRI重建对应的视觉刺激图像
2. **语义描述（Captioning）**：从fMRI生成对应图像的文本描述
3. **心理创造（Creation）**：从fMRI生成未见过的新内容（embodied implementation）

整体流程大致为：
- **fMRI编码器**：将原始fMRI体素信号编码为紧凑的神经特征表示
- **Prompt Variant Alignment (PVA)**：对齐并区分不同模态的差异
- **LLM解码器**：利用LLM的视觉和文本能力，将对齐后的神经表征转化为不同模态的输出

### 关键设计

#### 1. Prompt Variant Alignment (PVA) 模块
这是NeuroCreat的核心技术创新：
- **动机**：fMRI信号的低分辨率与生成目标（精细图像/准确文本）之间存在巨大鸿沟；不同输出模态对神经表征的需求不同
- **机制**：设计可变prompt来分离不同模态的差异，为每种输出任务构建适配的对齐方式
- **效果**：有效减轻fMRI低分辨率和模态间过度耦合的问题

#### 2. LLM集成策略
- 利用LLM（推测为Vicuna/LLaMA系列）的视觉和文本能力
- 将fMRI编码的神经特征作为LLM的输入条件
- LLM同时服务于captioning和creation任务，实现参数共享和知识迁移

#### 3. 多任务复用设计
- 三个任务共享fMRI编码器和LLM骨架
- 通过PVA模块的不同prompt variant实现任务特异性
- "创造"任务作为全新扩展——在NSD数据集上首次建立此类基准

### 损失函数 / 训练策略
具体损失函数组合需参考原文，推测包含：
- **重建损失**：像素级和/或感知损失（用于图像重建路径）
- **语义对齐损失**：CLIP空间对齐或对比学习损失
- **文本生成损失**：交叉熵/自回归语言建模损失（用于captioning路径）
- 可能采用分阶段训练策略

## 实验关键数据

### 数据集
- **NSD（Natural Scenes Dataset）**：最主流的高质量fMRI视觉解码数据集，7T fMRI，8个被试观看自然图像
- **GOD（Generic Object Decoding）**：另一常用fMRI解码数据集

### 图像重建结果

论文在NSD和GOD数据集上与先前重建方法进行了比较。从snippet中可知"NeuroCreat not only achieves the optimal image..."——在多项重建指标上达到最优。

| 任务 | 数据集 | 对比方法 | 结论 |
|------|--------|----------|------|
| 图像重建 | NSD, GOD | 先前SOTA方法 | 达到最优图像重建效果 |
| Captioning | NSD | 多种方法 | 与多种方法进行了captioning对比 |
| Creation | NSD | 无先前工作（首次） | 首次在NSD上建立creation基准 |

> 注：原文完整的定量数据（PixCorr, SSIM, FID, CLIP-Score等具体数值）需参阅PDF表格。

### 消融实验要点
- PVA模块的有效性验证（移除PVA后各任务性能下降）
- 不同模态输出的prompt variant设计对比
- LLM集成方式的消融

## 亮点
1. **问题定位新颖**：首次明确将fMRI脑解码从"重建所见"扩展到"描述所见 + 创造未见"，对齐了人类认知的层级——感知 → 理解 → 想象
2. **PVA模块设计巧妙**：通过可变prompt对齐不同模态差异，优雅地解决了"一个编码器服务多种输出"的挑战
3. **首创"心理创造"任务**：在NSD上首次建立creation基准，为脑机接口的创造性应用开辟了新方向
4. **LLM + 脑解码的有效集成**：展示了大语言模型在脑信号理解中的潜力，不仅限于图像重建
5. **统一框架**：单一模型同时完成重建、描述、创造三个任务

## 局限性 / 可改进方向
1. **无公开代码**：严重限制可复现性和社区跟进
2. **无arXiv预印本**：仅CVF Open Access，传播和引用受限
3. **fMRI固有限制**：时间/空间分辨率低、被试特异性强、数据采集昂贵，限制了方法的实用性
4. **"创造"的定义和评估**：如何定义和量化"心理创造"的质量缺乏标准化度量
5. **跨被试泛化**：论文是否解决了跨被试适应问题存疑（同组前序工作专门探索了此问题）
6. **LLM选择**：具体使用哪个LLM（规模/版本）及其对结果的影响未知
7. **计算开销**：整合LLM的脑解码框架可能需要大量计算资源

## 与相关工作的对比
| 方法 | 重建 | Captioning | Creation | LLM | 特点 |
|------|:----:|:----------:|:--------:|:---:|------|
| **NeuroCreat (本文)** | ✅ | ✅ | ✅ | ✅ | 首个三合一脑解码框架 |
| MindEye2 | ✅ | ❌ | ❌ | ❌ | 1小时数据达到SOTA重建 |
| Unibrain | ✅ | ✅ | ❌ | ❌ | 统一扩散模型做重建+captioning |
| Brain-Streams | ✅ | ❌ | ❌ | ❌ | 多模态引导的fMRI-to-Image |
| BrainSCUBA | ❌ | ✅ | ❌ | ✅ | 体素级语义 captioning |
| See Through Their Minds | ✅ | ✅ | ❌ | ❌ | 跨被试迁移学习 |

NeuroCreat的核心区分点在于：(1) "creation"是全新任务维度；(2) 通过PVA模块统一处理多模态输出；(3) 深度集成LLM而非仅使用CLIP。

## 启发与关联
1. **Prompt作为模态适配器**：PVA的设计思想（用prompt variant区分不同输出模态）可迁移到其他多任务多模态场景
2. **LLM + 信号处理**：将LLM应用于非文本信号（fMRI）的方式对理解LLM作为通用多模态推理引擎有启发
3. **评估范式思考**："创造"的评估是开放问题——语义合理性？新颖性？与个体经验的一致性？
4. **脑机接口前景**：如果能从脑信号创造新内容，对辅助通信、创意增强等应用有深远影响
5. **与视觉基础模型的衔接**：fMRI解码的进步可能反过来帮助理解视觉基础模型的内部表征

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — "重建→描述→创造"的三层级扩展非常有远见，creation任务是首创
- 实验充分度: ⭐⭐⭐⭐ — 在NSD和GOD上对比多种方法，含captioning和首创的creation评估；但缺代码影响验证
- 写作质量: ⭐⭐⭐⭐ — 问题动机清晰，框架设计逻辑性强
- 对我的价值: ⭐⭐⭐ — 脑解码领域有启发，但距离核心研究方向较远
