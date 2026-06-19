---
title: >-
  [论文解读] M3Grounder: Mask-Based Multi-Span and Multi-Granular Grounding for Document QA
description: >-
  [CVPR 2026][多模态VLM][文档问答] M3Grounder 把文档问答的"答案定位"从粗糙的边界框改造成像素级分割：VLM 一边生成答案、一边吐出 `[GROUND]` token，每个 token 经短语 / 行 / 块三个 MLP 头驱动一个可提示分割模块，产出嵌套的多粒度证据掩码，并在四个基准上刷到 SOTA。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "文档问答"
  - "像素级 grounding"
  - "分割"
  - "多粒度"
  - "数据引擎"
---

# M3Grounder: Mask-Based Multi-Span and Multi-Granular Grounding for Document QA

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Venna_M3Grounder_Mask-Based_Multi-Span_and_Multi-Granular_Grounding_for_Document_QA_CVPR_2026_paper.html)  
**代码**: 待确认（论文称将开源代码 / 数据 / 模型）  
**领域**: 多模态VLM  
**关键词**: 文档问答, 像素级 grounding, 分割, 多粒度, 数据引擎

## 一句话总结
M3Grounder 把文档问答的"答案定位"从粗糙的边界框改造成像素级分割：VLM 一边生成答案、一边吐出 `[GROUND]` token，每个 token 经短语 / 行 / 块三个 MLP 头驱动一个可提示分割模块，产出嵌套的多粒度证据掩码，并在四个基准上刷到 SOTA。

## 研究背景与动机

**领域现状**：文档问答（DocVQA）需要视觉-语言模型（VLM）同时理解文字和版面，主流做法把它当成纯文本生成任务，只输出答案文字。少数支持"答案定位"（grounding）的方法（如 DOGR、Qwen3-VL、InternVL3.5）会额外吐出边界框，把框的坐标直接穿插进文本答案里。

**现有痛点**：纯文本生成完全不告诉你答案是从页面哪里读出来的，在医疗、法律、金融这类对可追溯性要求高的场景里很致命。而吐边界框的方法只能给出粗糙的矩形定位——矩形套不住弯曲文字（曲线排版、倾斜表头），还会把大量无关背景圈进去，定位歧义大。

**核心矛盾**：把"语言建模"和"空间定位"耦合在同一条自回归序列里（坐标和文字混在一起生成），会让模型既要管语义正确又要管几何精度，两头都做不好；而矩形这种表示本身就无法刻画文档里真实存在的不规则文字几何。

**本文目标**：(1) 把 grounding 从框预测换成像素级分割，吻合文字真实形状；(2) 支持一个答案对应多个分散证据区域（multi-span）；(3) 沿文档天然的"短语 ⊂ 行 ⊂ 块"层级做多粒度定位；(4) 配套造出带像素掩码标注的大规模数据。

**切入角度**：作者观察到文档有天然的空间层级——词组成行、行组成块，每个粒度对应不同的推理范围（抽取式问题"姓名："只需短语级，概括式问题需要块级）。于是把 grounding 解耦出去交给专门的分割头，让 VLM 专注产出语义准确的答案和 span 边界。

**核心 idea**：VLM 自回归生成答案时在每个证据 span 后紧跟一个 `[GROUND]` token，用该 token 的隐状态去提示一个分割模块，输出短语 / 行 / 块三级嵌套掩码——用"分割代替框、用专用 token 解耦语言与定位"来解决精细 grounding。

## 方法详解

### 整体框架
给定文档图像 $x$ 和问题 $q$，M3Grounder 自回归生成形如 `... <e> yₖ </e>[GROUND] ...` 的答案：用 `<e>...</e>` 标出答案 span，紧随其后的 `[GROUND]` token 触发该 span 对应的多粒度掩码生成。整个模型是"VLM 主干 + 可提示分割模块"的混合体——VLM 负责读图、写答案、决定哪段文字需要定位；分割模块负责把定位落实到像素。

关键在于**解耦**：以往方法把框坐标塞进文本序列，M3Grounder 让语言建模和空间预测各管各的。`[GROUND]` token 的最终隐状态 $\tilde h_k$ 被三个粒度专属 MLP 投影头分别映射成短语 / 行 / 块级提示 $h_k^{(p)}, h_k^{(l)}, h_k^{(b)}$；与此同时分割模块（基于 SAM）抽一次稠密图像特征 $z=F_{enc}(x)$，被所有 span、所有粒度复用；最后掩码解码器 $F_{dec}$ 用提示去解出每级掩码 $\hat M_k^{(i)}$。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["文档图像 x + 问题 q"] --> B["VLM 自回归生成<br/>答案 + 解耦的 [GROUND] token"]
    B -->|每个 [GROUND] 隐状态| C["多粒度分层 grounding<br/>短语/行/块 三 MLP 头 + 嵌套包含约束"]
    A -->|稠密图像特征复用| D["可提示分割模块<br/>SAM 掩码解码器"]
    C --> D
    D --> E["渗漏抑制约束<br/>掩码只落在文字像素内"]
    E --> F["每个 span 的<br/>短语/行/块三级证据掩码"]
    G["GroundingDocQA 数据引擎<br/>200K 文档 / 2M QA"] -.训练监督.-> B
```

### 关键设计

**1. 解耦式 VLM-分割 grounding：用 `[GROUND]` token 把"写答案"和"画掩码"分开**

针对"框坐标塞进文本序列导致语义和几何互相拖累"这个痛点，M3Grounder 在词表里加一个特殊 token `[GROUND]`，VLM 在生成完一个答案 span（`<e>yₖ</e>`）后立刻吐出它，建立 span 与证据区域的一一对应。语言模型这条路只负责答案文字和 span 边界，几何定位完全交给分割头：取 `[GROUND]` token 的最终隐状态 $\tilde h_k$，经 MLP 投影成分割提示，再交给可提示分割模块（VLM 不需要在序列里生成任何坐标数字）。这种分工让 VLM 专注语义、分割头专注几何感知定位，论文实验里它带来的最大好处是在弯曲 / 倾斜文字（CS 子集）上的优势远超框方法，因为掩码能贴合文字真实轮廓而框不能。

**2. 多粒度分层 grounding：三个 MLP 头 + 嵌套包含约束**

针对"不同问题需要不同定位范围、且粗细粒度之间应当空间一致"的需求，每个 `[GROUND]` token 的隐状态被三个独立 MLP 头投影成短语、行、块三级提示，分别解出三张掩码。光各自预测还不够——作者引入分层包含损失 $L_{hier}$ 强制"细掩码必须被粗掩码包住"（$p \subset l \subset b$），其形式是对违反包含关系的像素做惩罚：

$$L_{hier}=\sum_{k=1}^{K}\sum_{(i,j)\in\{(p,l),(l,b)\}}\frac{\sum_{m\in\Omega}\hat M_k^{(i)}(m)\,[1-\hat M_k^{(j)}(m)]}{\sum_{m\in\Omega}\hat M_k^{(i)}(m)+\epsilon}$$

即统计细级掩码 $\hat M_k^{(i)}$ 落在粗级掩码 $\hat M_k^{(j)}$ 之外的像素占比。⚠️ 公式据 OCR 缓存整理，以原文为准。这套约束不仅保证粒度间空间自洽，消融里还能反过来提升整体 grounding 精度和答案准确率（块级 F1g 最高，82.5 / 87.5）。把 SAM 默认的单个共享 MLP 换成三个粒度专属头，是各基准都掉点的关键差异（见消融）。

**3. 渗漏抑制损失 $L_{bleed}$：把掩码摁在文字像素里**

针对"分割掩码容易溢出到非文字背景区域"的问题，作者加一项渗漏抑制损失。记 $M_{ref}$ 为所有文字区域的并集、$\Omega$ 为全部像素坐标，则

$$L_{bleed}=\sum_{i\in\mathcal G}\sum_{k=1}^{K}\frac{\sum_{j\in\Omega}\hat M_k^{(i)}(j)\,[1-M_{ref}(j)]}{\sum_{j\in\Omega}\hat M_k^{(i)}(j)+\epsilon}$$

分子是预测掩码落在文字区域之外（背景）的部分，分母是预测掩码总面积——本质是惩罚"画到背景上的比例"。总损失为 $L_{total}=\lambda_{lm}L_{lm}+\lambda_{seg}L_{seg}+\lambda_{bleed}L_{bleed}+\lambda_{hier}L_{hier}$，其中 $L_{seg}$ 是 Dice + BCE 的组合且对短语 / 行 / 块用递减权重。它带来的增益比 $L_{hier}$ 小但稳定，主要修复证据溢入相邻文字的问题。

**4. GroundingDocQA 数据引擎：造出带像素掩码的多 span / 多粒度监督**

针对"现有文档 grounding 数据只有框、只覆盖抽取式答案、缺层级线索"的空白，作者用三条互补流水线造了 20 万文档、200 万 QA 对的 GroundingDocQA。① **版面感知文档**：用 REPLICA 引擎把文档转成保留空间结构和阅读顺序的高保真 HTML（Fid-HTML），给每个 HTML 元素打唯一 ID，喂给 LLM 生成 QA 对及对应的行级 ID 和关键短语，再把 ID 映射回边界框得到行级框、用关键短语后处理得短语级框、用 HTML 包含层级得块级框，最后统一转成像素掩码（避免坐标幻觉）。② **弯曲文字文档**：先用曲线文字分割得到像素级曲线掩码，把掩码当 highlight 提示喂 VLM 生成 QA，保证 QA 与掩码精确对齐。③ **图表**：执行绘图脚本渲染图表，在渲染时直接拦截绘图库函数记录各元素边界框，避免检测误差。配套还有两阶段验证（句向量相似度过滤语义不一致 + LLM-as-Judge 查事实一致性），以及人工核验的 GroundingDocQA-Bench（2.5K 文档 / 5K QA，含 70% 直排 / 30% 曲排、2820 单 span / 2180 多 span）。

### 损失函数 / 训练策略
端到端联合优化文本生成与掩码预测。$L_{lm}$ 为标准交叉熵；$L_{seg}$ 用 Dice + BCE，且短语 / 行 / 块用递减系数（BCE: 2.0 / 1.0 / 0.5，Dice: 1.0 / 0.5 / 0.25 ⚠️ 以原文为准）。固定权重 $\lambda_{lm}=1$、$\lambda_{seg}=2$、$\lambda_{bleed}=\lambda_{hier}=0.5$。优化器 AdamW，学习率 $2\times10^{-6}$、3% warmup + 余弦衰减、梯度裁剪 1.0；VLM 用混合精度、掩码解码器保持 FP32。两个主干变体：M3Grounder-I（InternVL3.5-8B）与 M3Grounder-Q（Qwen3-VL-8B），均配 SAM 做分割。训练用 64 张 H100 + DeepSpeed ZeRO-3，有效 batch size 128，并混入部分通用 DocVQA 数据以保留文档问答能力。

## 实验关键数据

### 主实验
在四个 grounding 基准上对比，grounding 用 F1g（IoU>0.5），答案质量 AQ 用 G-Eval。SS / MS 分别是单 / 多 span 的 grounding F1，CS 是弯曲 / 倾斜子集 F1。

| 模型 | 规模 | BD-Test F1g | DOGR-Bench F1g | MMDoc IoU | GroundingDocQA F1g | GroundingDocQA CS |
|------|------|------|------|------|------|------|
| Gemini-2.5-Pro（商用） | – | 70.0 | 59.3 | 49.4 | 43.4 | 32.1 |
| InternVL3.5（零样本） | 8B | 41.5 | 12.5 | 15.5 | 7.6 | 6.3 |
| Qwen3-VL（零样本） | 8B | 44.5 | 27.6 | 28.7 | 12.8 | 11.6 |
| DOGR（grounding 专用） | 8B | – | 66.4 | – | – | – |
| Qwen3-VL 微调 | 8B | 62.3 | 35.8 | 43.4 | 60.6 | 38.3 |
| **M3Grounder-I** | 8B | 77.2 | 69.6 | 65.5 | 71.3 | 81.7 |
| **M3Grounder-Q** | 8B | **81.4** | **73.3** | **68.2** | **79.0** | **85.3** |

两个变体在全部基准上都拿到开源 SOTA。最突出的是弯曲 / 倾斜文字（CS）：M3Grounder-Q 的 85.3 远超商用 Gemini-2.5-Pro 的 32.1 和微调 Qwen3-VL 的 38.3，印证分割比框更能贴合不规则文字几何。在 DOGR-Bench 上 M3Grounder-Q 的 73.3 比同规模 DOGR 高 6.9 分。

多粒度结果（GroundingDocQA-Bench，F1g）：

| 模型 | 短语 | 行 | 块 |
|------|------|------|------|
| M3Grounder-I | 71.3 | 74.9 | 82.5 |
| M3Grounder-Q | 79.0 | 81.37 | 87.5 |

粒度越粗 F1g 越高，说明模型能有效利用更大的空间上下文。

### 消融实验
在 BD-T / DR-B / MD-B / GR-B 上分别拆解层级监督、损失项、微调策略（GR-B 给短语 / 行 / 块）。

| 配置 | GR-B 短语 | GR-B 块 | 说明 |
|------|------|------|------|
| 完整 M3Grounder-Q | 79.0 | 87.5 | 完整模型 |
| w/o Hierarchy（只短语级） | 71.3 | – | 去掉多粒度监督，GR-B 从 79.0 掉到 71.3 |
| 单共享 MLP + SAM 默认 | 63.6 | 77.6 | 三粒度专属头换成 SAM 单头，全面掉点 |
| $L_{lm}+L_{seg}$（基础） | 74.0 | 84.4 | 不含两个辅助损失 |
| $+\,L_{hier}$ | 78.2 | 84.7 | 加分层包含损失，稳定提升 |
| $+\,L_{bleed}$ | 77.3 | 86.5 | 加渗漏抑制，增益较小但稳定 |
| LoRA（PEFT） | 61.5 | 75.7 | 只微调注意力投影，远逊全量微调 |

### 关键发现
- **多粒度监督贡献最大**：去掉层级、只训短语级，GroundingDocQA-Bench 从 79.0 掉到 71.3，是最大的单项跌幅。
- **专属 MLP 头不可省**：用 SAM 默认的单共享 MLP（产一个统一提示）替换三个粒度头，各基准全面掉点，验证粒度专属投影的必要性。
- **两个辅助损失方向一致**：$L_{hier}$ 提升更明显（强制嵌套包含稳定了多粒度 grounding），$L_{bleed}$ 增益较小但稳定（抑制证据溢入相邻文字）。
- **精细 grounding 需要全量微调**：LoRA 只动注意力投影时大幅落后，且在最精细的短语级掉得最多——说明像素级文档定位需要更新全部参数。

## 亮点与洞察
- **用 `[GROUND]` token 把语言与定位解耦**：不在文本里塞坐标，而是用一个专用 token 当"指针"去提示分割头，让 VLM 不必学几何、分割头不必懂语义，分工干净且天然支持多 span。这个"特殊 token 当跨模态指针"的范式可迁移到任意"生成文字 + 定位区域"的任务。
- **分层包含损失一举两得**：$L_{hier}$ 本意只是保证短语 ⊂ 行 ⊂ 块的空间自洽，却被发现能反过来提升 grounding 精度和答案准确率——结构先验当正则化用。
- **渲染时拦截绘图函数拿图表 grounding**：造图表数据时不靠检测器、直接在渲染过程中截绘图库函数记录元素框，零检测误差，是个很实用的数据工程 trick。

## 局限与展望
- 作者承认：掩码式 grounding 虽然空间精度更高，但比框方法计算开销更大。
- 当前只在单页粒度上工作，多页 / 长上下文文档的层级 grounding 仍是开放方向。
- 自己看：方法重度依赖数据引擎里 REPLICA、曲线文字分割、绘图库拦截等一串外部组件，标注质量上限受这些工具链制约；评测虽用 G-Eval 但答案质量本质上仍是 LLM 打分，可能引入偏差。
- 改进思路：把短语 / 行 / 块三级掩码做成共享解码、增量预测，或许能缓解多掩码带来的算力膨胀。

## 相关工作与启发
- **vs DOGR**：DOGR 联合预测答案和边界框，比通用 VLM 强但局限于单粒度粗矩形定位；M3Grounder 改用掩码、做短语 / 行 / 块三级，在 DOGR-Bench 上同规模高 6.9 分，且能处理弯曲文字。
- **vs 通用 VLM（Qwen3-VL / InternVL3.5）**：它们能吐框做定位但没专门为 grounding 优化，零样本 grounding 很弱；即便用 GroundingDocQA 微调后，M3Grounder 凭专用分割头仍超过这些微调基线。
- **vs 现有文档 grounding 数据（BoundingDocs / DOGR）**：旧数据只有框、偏抽取式、缺层级；GroundingDocQA 提供多粒度像素掩码标注，覆盖图表 / 曲线 / 表格等异构版面。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把文档 grounding 从框换成像素掩码 + `[GROUND]` token 解耦 + 三级层级监督，是个完整且自洽的新范式。
- 实验充分度: ⭐⭐⭐⭐⭐ 四基准 + 多粒度 + 通用 DocVQA 保持 + 层级 / 损失 / PEFT 三类消融，覆盖很全。
- 写作质量: ⭐⭐⭐⭐ 方法和数据引擎讲得清楚，但损失系数等细节散落、部分公式在缓存里 OCR 受损。
- 价值: ⭐⭐⭐⭐⭐ 同时给出方法、20 万文档数据集和人工核验基准，对可追溯文档问答是实打实的推进。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] M3DocDep: Multi-modal, Multi-page, Multi-document Dependency Chunking with Large Vision-Language Models](m3docdep_multi-modal_multi-page_multi-document_dependency_chunking_with_large_vi.md)
- [\[CVPR 2026\] β-CLIP: Text-Conditioned Contrastive Learning for Multi-Granular Vision-Language Alignment](b-clip_text-conditioned_contrastive_learning_for_multi-granular_vision-language_.md)
- [\[CVPR 2026\] Multi-SpatialMLLM: Multi-Frame Spatial Understanding with Multi-Modal Large Language Models](multi-spatialmllm_multi-frame_spatial_understanding_with_multi-modal_large_langu.md)
- [\[CVPR 2025\] MARTEN: Visual Question Answering with Mask Generation for Multi-Modal Document Understanding](../../CVPR2025/multimodal_vlm/marten_visual_question_answering_with_mask_generation_for_multi-modal_document_u.md)
- [\[CVPR 2026\] GroundingME: Exposing the Visual Grounding Gap in MLLMs through Multi-Dimensional Evaluation](groundingme_exposing_the_visual_grounding_gap_in_mllms_through_multi-dimensional.md)

</div>

<!-- RELATED:END -->
