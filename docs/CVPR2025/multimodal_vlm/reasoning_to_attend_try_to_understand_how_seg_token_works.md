---
title: >-
  [论文解读] Reasoning to Attend: Try to Understand How \<SEG\> Token Works
description: >-
  [CVPR 2025][多模态VLM][推理分割] 深入分析了 \<SEG\> token 在推理分割任务中的工作机制——发现其学到了与文本直接提及相似的语义特征并用于图像-文本语义对齐，在此基础上提出 READ 方法，将 \<SEG\> token 与图像 token 的相似度图转换为点提示，以即插即用方式指导 SAM 解码器生成更精确的分割掩码。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "推理分割"
  - "SEG token分析"
  - "相似度引导点提示"
  - "语义对齐"
  - "大多模态模型"
---

# Reasoning to Attend: Try to Understand How \<SEG\> Token Works

**会议**: CVPR 2025  
**arXiv**: [2412.17741](https://arxiv.org/abs/2412.17741)  
**代码**: [https://github.com/rui-qian/READ](https://github.com/rui-qian/READ)  
**领域**: 多模态VLM  
**关键词**: 推理分割, SEG token分析, 相似度引导点提示, 语义对齐, 大多模态模型

## 一句话总结

深入分析了 \<SEG\> token 在推理分割任务中的工作机制——发现其学到了与文本直接提及相似的语义特征并用于图像-文本语义对齐，在此基础上提出 READ 方法，将 \<SEG\> token 与图像 token 的相似度图转换为点提示，以即插即用方式指导 SAM 解码器生成更精确的分割掩码。

## 研究背景与动机

在推理分割（reasoning segmentation）任务中，LISA 等先驱工作使用 \<SEG\> token 作为 LLaVA 编码器与 SAM 解码器之间的桥梁：\<SEG\> token 是文本词表中新增的占位符，经LLM微调后其隐藏层嵌入被投射到SAM中生成分割掩码。然而，很少有研究探究 \<SEG\> token 到底"学到了什么"。

作者通过可视化发现了一个惊人的一致性：\<SEG\> token 与图像 token 的相似度图（similarity map）在LLaVA编码器和SAM解码器中都呈现高度一致的激活模式，且这些激活区域与 CLIP 中文本直接提及（如"antler"）的相似度图也高度吻合。这意味着 \<SEG\> token 本质上学到了"语义相似度"能力，充当了隐式文本到视觉的语义桥梁。基于此发现，一个自然的想法是：直接利用相似度图中的高激活点来告诉模型"该注意哪里"。

## 方法详解

### 整体框架

READ 由三个核心模块组成：(1) **LLaVA编码器**：接收图像-文本对，输出文本响应和 \<SEG\> token 的隐藏层嵌入 $\boldsymbol{h}_{seg}$；(2) **SasP模块（Similarity as Points）**：计算 \<SEG\> token 嵌入与图像 token 嵌入的相似度图，将高激活区域转换为连续可微的点坐标；(3) **SAM解码器**：接收 $\boldsymbol{h}_{seg}$、点提示 $\mathcal{P}$ 和图像特征，生成分割掩码 $\hat{\mathbf{M}} = \mathcal{G}_{\mathcal{V}}^{dec}(\mathbf{f}, \boldsymbol{h}_{seg}, \mathcal{P})$。

### 关键设计

1. **相似度即点提示（Similarity as Points, SasP）**:
    - 功能：从 \<SEG\> token 与图像 token 的语义相似度中提取空间位置提示
    - 核心思路：计算参数无关的相似度得分 $\mathcal{S} = \boldsymbol{h}_{img}^{(l_k)} \cdot (\boldsymbol{h}_{seg}^{(l_k)})^T$，其中 $\mathcal{S} \in \mathbb{R}^{N_t}$。根据均值 $\mu$ 和标准差 $\sigma$ 设定阈值划分三类点：正点（$\mathcal{S}_j \geq \mu + 0.5\sigma$，前景）、负点（$\mathcal{S}_j \leq \mu - 0.5\sigma$，背景）、中性点（其余）。恢复每个被选点的绝对坐标后送入 SAM 作为点提示
    - 设计动机：实验证明相似度图中的高激活点已经隐含了目标对象的位置信息（单独用these points提示SAM就能获得27.0% cIoU），将其显式利用可以为SAM提供更强的空间定位线索

2. **离散到连续采样（Discrete to Continuous, DtoC）**:
    - 功能：将离散的、不可微的点坐标转化为连续可微坐标，使梯度可以回传到LMM
    - 核心思路：使用基于距离的高斯加权平均插值。对选定点 $(x_j, y_j)$，计算其与每个网格点的距离权重 $w_i^j = \exp(-d_i^j)$，与 softmax 概率 $p_i$ 结合得到归一化权重 $\hat{w}_i^j$，最终连续坐标为加权平均 $\hat{x}_j = \sum_{i=1}^{h \times w} \mathbf{g}_{x,i} \cdot \hat{w}_i^j$
    - 设计动机：点选择涉及排序和索引操作是不可微的，若不做连续化处理，损失函数的梯度无法传回LLaVA编码器。通过DtoC，模型可以在前向"推理关注位置"的同时在反向"学习更好地关注"

3. **即插即用架构设计**:
    - 功能：SasP 可无缝集成到任何基于 \<SEG\> token 的管线中（如 LISA、SESAME、GSVA 等）
    - 核心思路：SasP 的相似度计算是无参数的（parameter-free），仅需利用已有的 \<SEG\> token 嵌入和图像 token 嵌入，不引入额外参数
    - 设计动机：降低集成成本，保持方法的通用性和极低开销

### 损失函数 / 训练策略

- **总损失**：$\mathcal{L} = \lambda_{txt} \mathcal{L}_{txt} + \lambda_{mask} \mathcal{L}_{mask}$
- **文本生成损失** $\mathcal{L}_{txt}$：交叉熵损失
- **掩码损失** $\mathcal{L}_{mask} = \lambda_{bce} \mathcal{L}_{bce}(\hat{\mathbf{M}}, \mathbf{M}) + \lambda_{dice} \mathcal{L}_{dice}(\hat{\mathbf{M}}, \mathbf{M})$，$\lambda_{bce}=2.0$, $\lambda_{dice}=0.5$
- 使用 LoRA 高效微调 LLaVA，SAM 的图像编码器冻结，仅训练 mask decoder
- 4×3090 GPU，20 epochs，约24小时；AdamW，lr=0.0003

## 实验关键数据

### 主实验

| 数据集 | 指标 | READ-7B | LISA-7B-v1.5(ft) | SESAME | 提升 |
|--------|------|---------|-------------------|--------|------|
| ReasonSeg val | cIoU | **67.6** | 62.9 | 39.1 | +4.7 |
| ReasonSeg test overall | gIoU | **58.5** | 55.6 | 30.5 | +2.9 |
| RefCOCO val | cIoU | **78.1** | 74.9 | 74.7 | +3.2 |
| RefCOCO+ val | cIoU | **68.4** | 65.1 | 64.9 | +3.3 |
| RefCOCOg val(U) | cIoU | **70.1** | 67.9 | 66.1 | +2.2 |
| FP-RefCOCO See | Acc | **82.87** | 51.36 | 79.84 | +3.03 |
| FP-RefCOCO Seg | cIoU | **61.50** | 44.00 | 57.93 | +3.57 |

### 消融实验

| 配置 | gIoU | cIoU | 说明 |
|------|------|------|------|
| \<SEG\>prompt only | 51.2 | 57.6 | 基线LISA方式 |
| + $\mathcal{P}$prompt（离散点） | 56.4 | 64.6 | 点提示贡献+7% cIoU |
| + $\mathcal{P}$DtoC（连续化） | **59.8** | **67.6** | DtoC再贡献+3% cIoU |
| SAM-ViT-Base | 55.6 | 61.9 | - |
| SAM-ViT-Large | 60.1 | 65.2 | - |
| SAM-ViT-Huge | **59.8** | **67.6** | 更大backbone更优 |

### 关键发现

- \<SEG\> token 的定量分析：仅用相似度图中的高/低激活点提示原始SAM，就能达到27.0% cIoU（vs SESAME的30.4%），说明 \<SEG\> token 确实学到了有效的空间定位语义
- 相似度图与ground-truth掩码的 IoU 一致性（$\mathcal{S}$IoU=36.4%）甚至超过了 \<SEG\> token 直接提示 SAM 的结果（30.4%），验证了空间位置信息的存在
- READ 在假前提（false premise）场景中表现出色：FP-RefCOCO 的 See 准确率82.87%（vs LISA的51.36%），说明READ不会盲目生成掩码

## 亮点与洞察

- **首次系统分析 \<SEG\> token 的工作机制**：通过可视化和定量实验揭示了 \<SEG\> token 学到的是"语义相似度"——本质上是将隐式文本推理结果映射到了视觉空间
- **"推理以关注 & 关注以推理"的双向学习**：DtoC 使得梯度可以从分割损失回传到 LLaVA，建立了"注意力引导分割"和"分割反馈优化注意力"的闭环
- **极简且即插即用**：SasP 无额外参数，可直接叠加到任何 \<SEG\>-like 系统上
- **发现 \<SEG\> token 语义等价性**：\<SEG\> token 在隐式推理后获得的嵌入，与直接文本提及（如"antler"）在CLIP空间中的相似度模式高度一致

## 局限与展望

- 当前相似度计算是简单点积，未引入可学习参数（如交叉注意力），有进一步提升空间
- 仅在7B和13B的LLaVA上验证，未在更大或更新的LMM上测试
- 阈值 $\varepsilon=0.5$ 是固定的，自适应阈值可能更优
- 单 \<SEG\> token 的多目标场景尚未充分探索

## 相关工作与启发

- 与 LISA（\<SEG\> → SAM）的关系：READ 在 LISA 的基础上增加了相似度点提示通道，将 \<SEG\> token 的隐式语义显式地转化为空间引导信号
- 与 SESAME（处理假前提）的关系：READ 继承了假前提训练数据，并在 See 和 Segment 两个指标上均大幅超越
- \<SEG\> token 语义等价性的发现，对理解 LLM 中占位 token 的学习行为有更广泛的启示（如 \<IMG\> token 在图像生成中可能有类似行为）

## 评分

- 新颖性: ⭐⭐⭐⭐ 对 \<SEG\> token 机制的分析是全新视角，SasP 设计虽简单但动机清晰
- 实验充分度: ⭐⭐⭐⭐⭐ ReasonSeg + RefCOCO(+/g) + FP-RefCOCO(+/g) 全面覆盖，消融详尽
- 写作质量: ⭐⭐⭐⭐ 分析→发现→方法的逻辑链条清晰，可视化有说服力
- 价值: ⭐⭐⭐⭐ 作为即插即用模块有实用价值，机理分析对后续工作有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation](../../CVPR2026/multimodal_vlm/where_mllms_attend_and_what_they_rely_on_explaining_autoregressive_token_generat.md)
- [\[CVPR 2025\] Vision-Language Models Do Not Understand Negation](vision-language_models_do_not_understand_negation.md)
- [\[CVPR 2025\] Instruction-based Image Manipulation by Watching How Things Move](instruction-based_image_manipulation_by_watching_how_things_move.md)
- [\[CVPR 2025\] Towards Understanding How Knowledge Evolves in Large Vision-Language Models](towards_understanding_how_knowledge_evolves_in_large_vision-language_models.md)
- [\[ACL 2026\] How Do LLMs and VLMs Understand Viewpoint Rotation Without Vision? An Interpretability Study](../../ACL2026/multimodal_vlm/how_do_llms_and_vlms_understand_viewpoint_rotation_without_vision_an_interpretab.md)

</div>

<!-- RELATED:END -->
