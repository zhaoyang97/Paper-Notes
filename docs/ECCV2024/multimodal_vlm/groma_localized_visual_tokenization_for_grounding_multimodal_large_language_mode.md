# Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models

**会议**: ECCV 2024
**arXiv**: [2404.13013](https://arxiv.org/abs/2404.13013)
**代码**: [https://groma-mllm.github.io/](https://groma-mllm.github.io/)
**领域**: 多模态VLM
**关键词**: 视觉定位, Region Tokenization, Grounding, Referring, 多模态大模型

## 一句话总结

Groma提出了将定位能力嵌入视觉tokenization过程的新范式——通过region proposer发现感兴趣区域并编码为region token，使MLLM无需依赖LLM输出坐标或外部模块即可实现高精度的referring和grounding，同时利用GPT-4V+visual prompting构建了首个视觉-文本双prompt的grounded chat数据集Groma Instruct。

## 研究背景与动机

1. **领域现状**：MLLM在图像级理解（captioning、VQA）上表现出色，但普遍缺乏定位（localization）能力，无法将理解关联到视觉上下文中的具体位置。
2. **现有痛点**：
   - **方案A**（LLM输出坐标，如Kosmos-2、Shikra）：LLM计算量大，难以处理高分辨率输入；序列输出不适合密集预测任务（如分割）
   - **方案B**（外部模块，如LISA用SAM）：需要图像被处理两次（MLLM+定位模块），推理延迟高
3. **核心矛盾**：定位本身需要的是感知能力而非高级语义理解，但现有方案要么让沉重的LLM做定位，要么引入额外模块
4. **本文解决的问题**：如何在不增加外部模块、不增加LLM负担的前提下，让MLLM具备精确的定位能力
5. **切入角度**：借鉴open-vocabulary detection的思路，将grounding分解为定位（localization）和识别（recognition）两个子问题，将定位放到visual tokenizer中完成
6. **核心idea**：感知先于理解（perceive-then-understand）——利用visual tokenizer的空间理解能力做定位，LLM只负责语义理解和推理

## 方法详解

### 整体框架

Pipeline: 输入图像(448×448) → DINOv2 Image Encoder → 分两路：(1) 全局image tokens（4合1下采样）→ MLP → LLM, (2) Region Proposer(DDETR) → 发现ROI → Region Encoder(Multi-scale ROIAlign) → region tokens → MLP → LLM → 输出文本（可包含grounded引用）

### 关键设计

1. **Image Encoder (DINOv2)**:
   - 做什么：将输入图像编码为patch-level特征，同时为region proposer和region encoder提供特征金字塔
   - 核心思路：选择DINOv2而非CLIP作为视觉编码器，因为DINOv2在高分辨率输入和细粒度定位特征上表现更优。在COCO检测任务上，448×448的DINOv2达到43.6 AP，而336×336的CLIP仅32.4 AP
   - 设计动机：定位任务需要精确的空间特征，DINOv2的自监督训练使其保留了更好的空间结构信息
   - Token合并：将每4个相邻2D patch token合并为1个，减少LLM输入长度（最终不超过356 visual tokens）

2. **Region Proposer (Deformable DETR)**:
   - 做什么：作为类别无关的检测头，从图像中发现潜在的感兴趣区域（ROI）
   - 核心思路：使用DINOv2最后4层特征构建特征金字塔 → Deformable DETR transformer → 二分类器（基于定位质量评分，而非类别）→ 生成300个proposals → NMS（阈值0.6）+ 置信度过滤（>0.15）→ 取top-100
   - 设计动机：定位是一个低级感知任务，不需要LLM参与。通过专门的detector head完成定位，可以利用大规模检测数据（COCO、Objects365、OpenImages、V3Det、SA1B）预训练。这些数据量级对LLM来说计算上不可行。
   - 训练数据：5.7M检测标注（含2M过滤后的SA1B数据，用于扩展到部件和stuff级别的proposal）

3. **Region Encoder (Multi-scale ROIAlign)**:
   - 做什么：将region proposals（来自proposer或用户输入）编码为region tokens
   - 核心思路：从DINOv2最后3层提取特征金字塔 → multi-scale ROIAlign裁剪并融合区域特征为统一的region token → MLP投影到LLM特征空间
   - 设计动机：与数值坐标表示（Shikra）或离散位置token（Kosmos-2）相比，region token直接携带底层区域的语义信息，对LLM更直观可理解

4. **统一的Referring & Grounding格式**:
   - 做什么：通过proxy tokens `<r1>, <r2>, ..., <rn>` 将region tokens注册到LLM词表中，统一输入（referring）和输出（grounding）
   - 核心思路：grounding输出时，LLM通过引用proxy token来关联文本与图像区域；referring输入时，用户指定的区域同样编码为region token并插入instruction。`[grounding]` special token告知模型生成grounded回复。
   - 设计动机：统一格式避免了为referring和grounding设计不同的编码方式，简化了架构

5. **Groma Instruct数据集（GPT-4V辅助构建）**:
   - 做什么：构建30K条视觉grounded对话数据用于instruction finetuning
   - 核心思路：选择Visual Genome中有密集region标注的图像 → 用SoM技术在图像上标注数字标记 → 将标记后的图像+丰富的文本上下文（region描述、图像描述、QA对）输入GPT-4V → 利用in-context learning生成grounded对话
   - 设计动机：现有对话数据缺乏grounded信息，导致grounded MLLM难以在长文对话中保持grounding能力。这是首个同时使用视觉和文本prompt构建的grounded chat数据集。

### 损失函数 / 训练策略

三阶段训练：
- **Stage 1 - Detection预训练**（12 epochs）：仅训练region proposer（image encoder冻结），使用5.7M检测数据，lr=2e-4, batch=64
- **Stage 2 - Alignment预训练**（2 epochs）：训练region encoder和MLP projector（其他冻结），使用3.2M视觉-语言数据，lr=1e-4, batch=128
- **Stage 3 - Instruction Finetuning**（1 epoch）：解冻LLM，使用857K高质量数据（含Groma Instruct），lr=1e-5, batch=128

训练总耗时：8× A100约8天

## 实验关键数据

### 主实验

| 数据集 | 指标 | Groma | Ferret | MiniGPT-v2 | Shikra | 提升 |
|--------|------|-------|--------|------------|--------|------|
| RefCOCO val | Acc@0.5 | **89.53** | 87.49 | 88.69 | 87.01 | +1.04 |
| RefCOCO testA | Acc@0.5 | **92.09** | 91.35 | 91.65 | 90.61 | +0.44 |
| RefCOCO testB | Acc@0.5 | **86.26** | 82.45 | 85.33 | 80.24 | +0.93 |
| RefCOCO+ val | Acc@0.5 | **83.90** | 80.78 | 79.97 | 81.60 | +2.30 |
| RefCOCOg val | Acc@0.5 | **86.37** | 83.93 | 84.44 | 82.27 | +1.93 |
| REC平均 | Acc@0.5 | **86.52** | 83.91 | 84.29 | 82.93 | +2.23 |
| LVIS-Ground | AR | **28.8** | 16.8 | 11.4 | 4.9 | +12.0 |
| LVIS-Ground | AR@0.75 | **30.3** | 16.3 | 11.2 | 2.0 | +14.0 |

超越同规模所有grounded MLLM，在LVIS-Ground上领先第二名Ferret超过10 AR。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| CLIP 336×336 | 32.4 AP (COCO det) | CLIP定位能力弱 |
| DINOv2 336×336 | 38.9 AP (+6.5) | DINOv2显著优于CLIP |
| DINOv2 448×448 | 43.6 AP (+4.7) | 更高分辨率进一步提升 |
| Frozen LLM (finetune) | Grounding 84.02%, Referring CIDEr 148.0 | 不解冻LLM也能达到强定位能力 |
| Unfrozen LLM (finetune) | Grounding 86.52%, Referring CIDEr 158.4 | 解冻后理解能力更强 |
| 4× Token Merge | REC平均 86.47% | 4合1下采样几乎无损 |
| 不合并 | REC平均 86.55% | 仅+0.08%差异 |

### 关键发现

- **定位能力可与理解解耦**：冻结LLM时，Groma仍保持与Ferret comparable的grounding能力（84.02% vs 83.91%），证明定位确实不需要LLM
- **Token合并几乎无损**：4× downsampling对grounding仅有0.08%影响，验证了解耦设计在效率和精度上的最优性
- **LVIS-Ground暴露现有方法短板**：所有方法在小物体定位（AR@s）上都很差，且多数方法倾向于每个query只预测一个box——因为训练数据（RefCOCO系列）每个query只有一个目标
- Region captioning上与GLaMM（需要分离的refer和ground设计）表现相当或更优

## 亮点与洞察

- **"感知先于理解"的设计哲学**：将定位放到视觉tokenizer中完成而非LLM中，符合人类视觉的处理流程，是一个值得广泛借鉴的范式
- **解耦架构带来的训练效率**：可以在不涉及LLM的情况下用百万级检测数据预训练定位能力，这对传统MLLM是不可能的
- **LVIS-Ground benchmark**：提出的AS-MANY-Protocol评估方式和1203类的grounding benchmark填补了评估空白
- **Groma Instruct的构建方法**：SoM visual prompting + GPT-4V + 文本上下文的组合是构建grounded对话数据的有效方案
- **统一的region token**：同一种token表示同时处理referring输入和grounding输出，简洁优雅

## 局限性 / 可改进方向

- 不支持自由形式的region输入（如点击、涂鸦），仅限bounding box
- 不支持像素级grounding（如分割掩码），作者建议用Mask2Former替换box proposer
- DINOv2特征未与文本天然对齐，导致在conversation和reasoning任务上略弱于CLIP-based方法
- Region proposer的recall是整个系统的性能上限——如果目标区域未被提出，就无法被理解
- 训练数据中小物体标注不足，导致小物体定位能力有限

## 相关工作与启发

- **vs Shikra/Kosmos-2（LLM输出坐标）**：Groma避免了LLM处理高分辨率输入的计算瓶颈，且定位精度更高（LVIS-Ground AR: 28.8 vs 4.9）
- **vs LISA/GLaMM（外部定位模块）**：Groma不需要额外的SAM模块，推理更高效；且统一了refer和ground的表示
- **vs Ferret（空间感知视觉采样）**：Groma通过region proposer主动发现ROI而非被动接收输入，在LVIS-Ground上领先12+ AR
- **vs GPT4RoI（简单pooling提取region特征）**：Groma的multi-scale ROIAlign提取更精细的分层特征

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将定位嵌入视觉tokenization是全新的范式，"感知先于理解"的设计哲学很有启发
- 实验充分度: ⭐⭐⭐⭐☆ 涵盖grounding、referring、VQA三类任务，提出LVIS-Ground新benchmark，消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑非常清晰，三种范式的对比图（Figure 2）直观有力，motivation推导自然
- 价值: ⭐⭐⭐⭐☆ 提出了有影响力的新范式，但受限于不支持mask和自由形式输入，实际应用场景有一定约束
