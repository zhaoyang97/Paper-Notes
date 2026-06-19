---
title: >-
  [论文解读] MIRe: Enhancing Multimodal Queries Representation via Fusion-Free Modality Interaction
description: >-
  [ACL 2025][多模态VLM][多模态检索] 提出MIRe框架，通过"无融合模态交互"（fusion-free modality interaction）在视觉-文本对齐阶段避免直接融合文本特征，利用查询引导注意力池化模块让文本嵌入引导视觉信息提取但不将文本信号反馈回视觉表示，有效缓解多模态检索中的文本主导问题，在四个基准上取得零样本SOTA。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "多模态检索"
  - "模态交互"
  - "late interaction"
  - "视觉-文本对齐"
  - "文本主导问题"
  - "知识检索"
---

# MIRe: Enhancing Multimodal Queries Representation via Fusion-Free Modality Interaction

**会议**: ACL 2025  
**arXiv**: [2411.08334](https://arxiv.org/abs/2411.08334)  
**作者**: Yeong-Joon Ju, Ho-Joong Kim, Seong-Whan Lee (Korea University)
**代码**: [GitHub](https://github.com/yeongjoonJu/MIRe)  
**领域**: 多模态VLM  
**关键词**: 多模态检索, 模态交互, late interaction, 视觉-文本对齐, 文本主导问题, 知识检索

## 一句话总结

提出MIRe框架，通过"无融合模态交互"（fusion-free modality interaction）在视觉-文本对齐阶段避免直接融合文本特征，利用查询引导注意力池化模块让文本嵌入引导视觉信息提取但不将文本信号反馈回视觉表示，有效缓解多模态检索中的文本主导问题，在四个基准上取得零样本SOTA。

## 研究背景与动机

### 问题背景
多模态查询检索（multimodal query retrieval）旨在根据包含图像和文本的复合查询，从知识库中检索相关段落。现实场景中用户常在查询中附加视觉参考（如复杂物体或命名实体的图片），仅靠文本难以完整表达查询意图。

### 已有工作的不足
现有多模态检索方法（如ReViz、VISTA、PreFLMR）通常在视觉-文本对齐阶段**直接融合两种模态**进行交叉引用：
- **早期token融合**（ReViz/VISTA）：将视觉表示拼接在文本前，通过自注意力层交互
- **交叉注意力融合**（PreFLMR）：用文本嵌入作为key/value进行跨模态注意力

这些方法导致**文本主导问题**（text-dominant issue）：模型过度依赖文本驱动信号，忽略关键视觉信息。当文本查询含糊时（如将"beverage"替换为"object"），模型无法利用视觉线索补偿，导致检索失败。

### 核心动机
设计一种在对齐阶段不融合文本特征的模态交互机制，让文本查询能"读取"视觉信息但不将文本信号"写回"视觉表示，从根本上缓解文本主导问题。

## 方法详解

### 整体架构
MIRe采用双编码器架构：文本编码器 $\mathcal{R}_T$（ColBERTv2）生成token级文本嵌入 $E_t$，视觉编码器 $\mathcal{R}_V$（CLIP ViT）生成全局嵌入 $V_g$（CLS token）和patch级嵌入 $V_m$（倒数第二层）。最终通过late interaction机制（MaxSim）计算查询与段落的相关性分数。

### 关键设计1：查询引导注意力池化（Query-guided Attentive Pooling）

这是MIRe的核心创新模块。与标准交叉注意力不同，该模块让文本嵌入 $E_t$ 作为query，视觉patch嵌入 $V_m$ 作为key/value，计算注意力但**不通过残差连接将文本信号反馈回视觉表示**：

$$\mathcal{A} = \text{Softmax}\left(\frac{E_t \cdot \mathcal{K}_m^\top}{\sqrt{d_t}}\right)$$

其中 $\mathcal{K}_m \in \mathbb{R}^{h \times l_v \times d_t}$ 是视觉patch经线性投影后的key向量，$h$ 为注意力头数。然后通过mean pooling沿序列维度聚合，生成 $h$ 个视觉嵌入：

$$E_m = \text{Linear}\left(\frac{1}{l_t}\sum_{i}^{l_t}(\mathcal{A} \cdot \mathcal{V}_m)\right)$$

关键差异：
- **无残差连接**：标准交叉注意力会通过残差将query信息混入输出，而MIRe的池化操作仅利用 $E_t$ 计算注意力权重 $\mathcal{A}$，不将文本特征直接融入视觉表示
- **mean pooling**：沿序列维度取均值后输出 $h$ 个token，而非保留全部 $l_t$ 个token

### 关键设计2：两阶段训练策略

**对齐阶段**：冻结 $\mathcal{R}_T$ 和 $\mathcal{R}_V$，仅训练投影层和注意力池化模块。关键地，在此阶段**排除文本嵌入 $E_t$**，查询嵌入仅由视觉特征组成：$E_Q = [E_g; E_m]$。这强迫模型学习有效的视觉表示与段落对齐。

**推理/下游微调阶段**：将文本嵌入加回查询：$E_Q = [E_g; E_m; E_t]$，通过late interaction（MaxSim操作）计算最终相关性分数：

$$r_{Q,D} = \sum_{i=1}^{l_Q} \max_{j=1}^{l_D} (E_Q \cdot E_D^T)$$

使用对比损失训练：

$$\mathcal{L}_{CL} = -\sum_{\mathcal{D}} \log \frac{\exp(r_{Q,D}/\tau)}{\exp(r_{Q,D}/\tau) + \sum_{\bar{D} \in \bar{K}} \exp(r_{Q,\bar{D}}/\tau)}$$

### 关键设计3：Response-to-Passage数据构建

现有VQA数据集的回答过于简洁，不适合训练检索器。MIRe提出response-to-passage转换流程：

1. 从视觉对话数据集提取多模态QA对 $S = \{(I, T), R\}$
2. 将简单回答用查询中的名词补充，过滤yes/no类回答
3. 用回答 $R$ 作为查询，通过ColBERTv2从Wikipedia检索top-k段落
4. 将回答插入检索到的段落之间，构造更长、更真实的"伪段落"：$R' = [D_1; R; D_2; \ldots; D_k]$

最终从3个视觉指令数据集和2个VQA数据集构建了135万QA对。

## 实验关键数据

### 实验1：零样本检索性能（MRR@5）

| 方法 | OKVQA-GS | OKVQA-WK11M | ReMuQ | E-VQA |
|------|----------|-------------|-------|-------|
| CLIP | 19.08 | 16.45 | 0.34 | - |
| FLMR | 38.15 | 32.56 | 66.67 | 29.97 |
| ReViz | 45.77 | 44.03 | 23.61 | - |
| UniIR | 53.27 | - | 79.15 | 31.59 |
| VISTA | 55.33 | - | 78.32 | 33.90 |
| PreFLMR† | 59.38 | 45.68 | 52.27 | 30.92 |
| **MIRe** | **63.03** | **51.15** | **83.06** | **41.88** |
| MIRe (ViT-L) | 63.17 | 50.64 | 82.56 | 44.92 |

MIRe在所有四个基准上均大幅超越现有方法。与使用相同数据和设置的PreFLMR†相比，OKVQA-GS上MRR@5提升3.65，E-VQA上提升10.96，验证了架构设计的有效性。

### 实验2：消融实验（MRR@5，零样本）

| 变体 | OK-GS | OK-WK | ReMuQ | E-VQA | 平均 |
|------|-------|-------|-------|-------|------|
| 完整MIRe | 63.03 | 51.15 | 83.06 | 41.88 | 59.78 |
| 去除R2P数据构建 | 60.43 | 42.93 | 81.87 | 38.13 | 55.84 |
| 对齐阶段加残差连接 | 61.65 | 47.95 | 80.47 | 43.06 | 58.28 |
| 对齐阶段加入 $E_t$ | 51.38 | 42.13 | 71.69 | 32.80 | 49.50 |
| 推理时去除 $E_t$ | 36.99 | 36.68 | 2.73 | 11.39 | 21.95 |
| 推理时去除 $E_g$ 和 $E_m$ | 52.46 | 36.00 | 71.69 | 42.48 | 50.66 |

关键发现：对齐阶段加入 $E_t$ 导致平均MRR@5从59.78暴跌至49.50，直接验证了文本主导问题的存在。

### 实验3：微调性能（PR@5 / R@5）

| 方法 | OKVQA-GS (PR@5) | ReMuQ (R@5) |
|------|-----------------|-------------|
| FLMR | 70.63 | 62.76 |
| VISTA | 82.06 | 96.30 |
| MIRe (无预训练) | 74.26 | 92.44 |
| **MIRe** | **83.59** | **94.40** |
| MIRe (ViT-L) | 84.66 | 94.38 |

## 关键发现

1. **文本主导问题的定量验证**：在对齐阶段直接融合 $E_t$ 使平均性能下降17%，仅加残差连接也会下降2.5%。收敛速度更快反而性能更差，说明模型走了"捷径"依赖文本相似度
2. **视觉嵌入的分工**：$E_g$（全局）和 $E_m$（query-guided）捕获互补信息，同时去除二者的损失远大于单独去除
3. **R2P数据构建的关键性**：去除R2P使OK-WK上MRR@5从51.15降至42.93（-16%），是最具影响力的数据设计
4. **多QA对的硬负例效应**：同一图像仅取一个QA对（而非多个）导致显著性能下降，暗示多QA对提供了超越简单视觉对齐的硬负例信号

## 亮点

- **优雅的问题定义与解决**：将文本主导问题归因于对齐阶段的直接融合，通过"单向注意力"（文本→视觉）设计从根本上解决，思路简洁有力
- **两阶段嵌入策略**：对齐阶段仅用视觉嵌入强迫视觉对齐，推理阶段再加回文本嵌入，巧妙平衡了多模态互补与文本主导的矛盾
- **R2P数据构建**：仅用文本检索器将简短QA回答转化为长段落形式的训练数据，无需额外标注或生成模型，成本低且效果显著
- **全面的消融与可视化分析**：UMAP聚类和注意力图直观展示了query-guided pooling如何根据不同文本查询关注不同视觉区域

## 局限与展望

- **仅验证通用领域**：未在医疗、法律等专业领域测试，这些领域的多模态内容可能有不同的模态交互模式
- **未与RAG流水线集成**：检索改进是否能传导到下游生成任务（如VQA答案生成）未被验证
- **知识库依赖性**：R2P数据构建依赖Wikipedia，面对动态更新或领域特定知识库可能需要额外适应
- **模型规模有限**：基于BERT-base（211M参数），未探索更大规模语言模型（如LLaMA）作为文本编码器的效果
- **WiT数据的角色不明确**：消融显示去除WiT仅轻微影响通用基准，但对Infoseek等知识密集型任务至关重要，说明模型的世界知识获取机制还需优化

## 与相关工作的对比

- **FLMR (Lin et al., 2023)**：同样使用late interaction机制，但通过生成caption和RoI来增强视觉查询表示，未处理文本主导问题。MIRe在零样本OKVQA-GS上MRR@5从38.15提升至63.03
- **PreFLMR (Lin et al., 2024)**：FLMR的扩展版本，在相同数据和设置下训练仍大幅落后于MIRe（OKVQA-GS: 59.38 vs 63.03），证明架构差异是关键
- **ReViz (Luo et al., 2023)**：端到端系统，使用VL-ICT从段落构建伪查询进行预训练，但这种方式加剧了文本主导问题（伪查询本身就能匹配段落，无需视觉）
- **VISTA (Zhou et al., 2024)**：将视觉token前置输入文本检索器的早期融合策略，虽有强表现但仍受文本主导限制
- **UniIR (Wei et al., 2024)**：基于指令引导的多模态检索器，需要显式指令输入，架构更复杂但性能不及MIRe
- **ColBERT/ColBERTv2**：MIRe的文本编码器骨干，提供late interaction的token级匹配能力，MIRe在此基础上扩展了多模态支持

## 评分

- 新颖性: ⭐⭐⭐⭐ — "无融合模态交互"思路简洁有效，但核心仍是注意力机制的变体
- 实验充分度: ⭐⭐⭐⭐⭐ — 四个基准、零样本+微调、完整消融、可视化分析、收敛曲线、嵌入分布，覆盖全面
- 写作质量: ⭐⭐⭐⭐ — 问题动机清晰，图示直观，但部分符号定义略显冗余
- 价值: ⭐⭐⭐⭐ — 文本主导问题的揭示和解决方案对多模态检索社区有重要参考意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](../../ICCV2025/multimodal_vlm/hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)
- [\[NeurIPS 2025\] Structure-Aware Fusion with Progressive Injection for Multimodal Molecular Representation Learning](../../NeurIPS2025/multimodal_vlm/structure-aware_fusion_with_progressive_injection_for_multimodal_molecular_repre.md)
- [\[ACL 2025\] Enhancing Multimodal Continual Instruction Tuning with BranchLoRA](branchlora_continual_instruction.md)
- [\[ACL 2025\] DALR: Dual-level Alignment Learning for Multimodal Sentence Representation Learning](dalr_dual-level_alignment_learning_for_multimodal_sentence_representation_learni.md)
- [\[CVPR 2026\] MOON2.0: Dynamic Modality-balanced Multimodal Representation Learning for E-commerce Product Understanding](../../CVPR2026/multimodal_vlm/moon20_dynamic_modality-balanced_multimodal_representation_learning_for_e-commer.md)

</div>

<!-- RELATED:END -->
