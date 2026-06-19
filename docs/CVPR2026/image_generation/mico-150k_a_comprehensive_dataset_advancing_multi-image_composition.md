---
title: >-
  [论文解读] MICo-150K: A Comprehensive Dataset Advancing Multi-Image Composition
description: >-
  [CVPR 2026][图像生成][多图合成] 针对"把多张参考图里的人/物/服饰/场景合成进一张连贯图像"（Multi-Image Composition, MICo）缺高质量训练数据的问题，本文用专有模型 Nano-Banana 配合 Compose-by-Retrieval 检索式提示、人在回路过滤与"分解-重组"流程，构建了 15 万级、含身份一致性的 MICo-150K 数据集与 MICo-Bench 评测集，并提出 Weighted-Ref-VIEScore 指标，多个开源 T2I 模型微调后 MICo 能力显著提升、甚至逼近闭源模型。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "多图合成"
  - "身份一致性"
  - "可控生成"
  - "数据集"
  - "评测指标"
---

# MICo-150K: A Comprehensive Dataset Advancing Multi-Image Composition

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Wei_MICo-150K_A_Comprehensive_Dataset_Advancing_Multi-Image_Composition_CVPR_2026_paper.html)  
**代码**: 待确认（论文称数据集与 benchmark 将开源）  
**领域**: 图像生成 / 多图合成数据集  
**关键词**: 多图合成, 身份一致性, 可控生成, 数据集, 评测指标

## 一句话总结
针对"把多张参考图里的人/物/服饰/场景合成进一张连贯图像"（Multi-Image Composition, MICo）缺高质量训练数据的问题，本文用专有模型 Nano-Banana 配合 Compose-by-Retrieval 检索式提示、人在回路过滤与"分解-重组"流程，构建了 15 万级、含身份一致性的 MICo-150K 数据集与 MICo-Bench 评测集，并提出 Weighted-Ref-VIEScore 指标，多个开源 T2I 模型微调后 MICo 能力显著提升、甚至逼近闭源模型。

## 研究背景与动机

**领域现状**：文生图（T2I）和图生图（I2I）已能产出逼真结果，个性化/上下文生成（保持参考图身份一致）是其中最有价值的能力之一。FLUX.Kontext、Qwen-Image 等近期工作在单参考图输入上进展显著。

**现有痛点**：这些系统大多只支持**单张**参考图输入，无法把多个实体（多个人、物、衣服、场景）整合进一张连贯的合成图。而真正的"多图合成"（MICo）开源社区与 GPT-Image-1、Nano-Banana、Seedream 4.0 这类闭源模型差距明显，根因之一是**缺少为该任务量身定制的高质量数据集**。

**核心矛盾**：现有 MICo 数据集有两大硬伤——(1) 很多源图/目标图由少数几个固定 T2I 模型生成，内容同质化、与闭源模型存在明显质量差距；(2) 基于真实照片或视频帧的数据多样性受限，缺乏想象性场景，且偏向"以人为中心"，对"以物为中心"和多主体场景覆盖不足。早期"用 GroundingDINO+SAM 从整图分割出实例当源图、原图当目标"的范式还常产出不完整、语义模糊的样本。

**本文目标**：构建一个覆盖面广、质量高、身份一致的 MICo 数据集；同时补上一个专门的评测基准与可靠指标，把 MICo 这个"挑战大却少有人碰"的任务推动起来。

**切入角度**：与其用弱生成骨干批量造同质数据，不如**直接用最强的闭源模型（Nano-Banana）来合成目标**，并在前端用检索保证源图组合"语义兼容"、在后端用 VLM+人工双重过滤把控质量。

**核心 idea**：用"高质量源图收集 → Compose-by-Retrieval 选语义兼容组合 → 强闭源模型合成 → 自动+人工核验"的流水线造数据，外加一条"分解真实图再重组"的 De&Re 轨道，让数据兼具真实与合成两种构图。

## 方法详解

### 整体框架
MICo-150K 是一个**数据集 + 评测基准**工作，核心是一整套数据构造与评测流水线，而非一个新模型。任务先被系统化为 **3 大类、7 个子任务、27 个细粒度类型**（以物为中心：Object+Object、Object+Scene；以人为中心：Person+Person、Person+Scene；人物交互 HOI：Person+Object、Person+Clothes、Person+Object+Clothes），再加一条独立的"分解-重组"（De&Re）轨道。

整条管线分四步：① **源图收集与清洗**——从 Subject200k、VITON-HD、Headshot、SUN397 等公开数据集收集物体、人物、服饰、场景四类源图，用 Qwen2.5-VL-72B 过滤低质/歧义图、DINO-v3+SigLIP2 特征 + DBSCAN 聚类去冗余，每张配详细 caption；② **Compose-by-Retrieval 组合提示**——不随机搭配源图（否则会出现"男运动员配高跟鞋"这种不兼容组合），而是让 GPT-4o 在候选里挑语义最兼容的组合并生成自然连贯的合成提示；③ **合成与核验**——把提示喂给闭源模型 Nano-Banana 合成目标图，再用 Qwen2.5-VL-72B 核验所有源实体是否正确出现、用 ArcFace 核验人脸身份一致；④ **De&Re 轨道**——把真实单人照片用 Nano-Banana 分解成"人/衣/物/场景"组件、人工核验后再重组，使每组组件同时产出"真实构图"和"重组合成构图"两个版本。评测侧则单独构造 MICo-Bench（1000 例）并提出 Weighted-Ref-VIEScore 指标。

> 说明：本文为纯数据集/基准工作，构造步骤虽多但属于线性数据流水线，按笔记规范不另配框架图，关键贡献以下列设计点讲清。

### 关键设计

**1. 任务体系 + 高质量源图收集去冗：先把"多图合成"切成可控的细粒度子任务，再保证每张源图都干净不重复**

MICo 之所以难做数据，第一步卡在"源图本身质量参差、且不同源图风格同质"。本文先定义 3 大类 / 7 子任务 / 27 细类的 taxonomy（如 Object+Scene 下分 1O1S、2O1S；Person+Person 下分 2M、2W、3M、3W、1M1W 等性别组合），让每个组合类型都有明确的源图采样规则。源图收集后用 Qwen2.5-VL-72B 逐条过滤（去掉含人脸的物体图、模糊/损坏图、多脸/背面/重度遮挡的人像等），再在每个类别内用 DINO-v3 与 SigLIP2 的拼接特征做 DBSCAN 聚类、每个视觉-语义簇只保留一张代表图来消冗余。最终得到 31.5K 物体图、44.6K 人像、约 26.8K 服饰图、11K 场景图，且尽量最大化人脸身份多样性（覆盖 5,403 位名人共 14.4 万张照片清洗后的子集）。

**2. Compose-by-Retrieval：让 GPT-4o 选"语义兼容"的源图组合并写自然提示，而不是随机拼+套模板**

如果直接从各源图池随机抽样组合，很容易得到语义不兼容的搭配，导致合成质量崩坏。Compose-by-Retrieval 的做法是：先确定一张主体（subject）图，再从服饰/场景/物体池里采样若干候选，把主体图、候选图及其详细 caption 一起作为上下文交给 GPT-4o，让它**挑出语义最兼容的组合**用于合成。此外，过去 MICo 方法常把源图 caption 直接拼成提示（如 "Combine 2 images according to ⟨Caption A⟩ and ⟨Caption B⟩"），本文改用 GPT-4o 把 caption 当上下文、生成更连贯自然的合成提示，并额外标注"token→源图"的显式映射，为后续潜空间对齐研究留接口。质量核验上，物体/场景用 Qwen2.5-VL-72B 验、人脸用 ArcFace 提取身份嵌入并用匈牙利算法做源图与生成图的人脸最优匹配，仅当所有匹配对都超过任务相关阈值才算合格。

**3. Decompose-and-Recompose（De&Re）：把真实复杂照片拆成组件再重组，让数据同时拥有真实构图与合成构图两个版本**

纯合成数据再丰富也缺真实世界的复杂性。De&Re 是 MICo-150K 里最复杂的部分：先从 CC12M 收集高质量单人真实照片，用 Nano-Banana 把每张分解为人、衣、物、场景等独立组件；人工标注员做细粒度核验，找出"丢失物体身份""直接复制粘贴""缺乏足够变化"等失败案例，并重写组件级提示以正确抽取目标元素；通过这轮人在回路精修后，再用 Nano-Banana 把组件重组成完整图像。于是每组组件天然得到**一对**目标：一张真实世界构图（原照片）+ 一张重组合成构图（共 11,677 例）。实验进一步发现：用合成目标训练与用真实目标训练效果几乎一致，说明精心策划的合成数据可作为 MICo 训练的有效替代。

**4. Weighted-Ref-VIEScore：用"逐源加权 + 参考图对比"修掉 VLM 评测里跨图注意力过载的硬伤**

传统 VIEScore 把总质量拆成语义一致性 SC 与感知质量 PQ、按 $\text{SC}\times\text{PQ}$ 打分；OmniContext 等沿用并要求把**所有源图同时**喂给 GPT-4o 评。但 VLM 的跨图注意力有限，图一多就会"看不清每张图、判不准每个源是否出现"，导致打分错误（文中例子里人类一致认为 B 远优于 A，VIEScore 却判反）。Weighted-Ref-VIEScore 拆成两件事：**加权**——每张非人源图先与生成图配对喂 Qwen-VL2.5-72B 判断该源是否成功出现、人脸源图改用 ArcFace 核验，得到每个源的贡献权重 $W$；为防止模型靠"把所有源直接复制粘贴"刷权重，再引入**参考机制**——每例先用 Nano-Banana 生成一张经人工核验、忠实包含所有源元素的参考图，评测时 GPT-4o 只把生成图与参考图**逐对**比较（而非一次塞进所有源图），从而拿到更接近人类判断的 SC。最终总分定义为

$$\text{Score} = W \times \sqrt{\text{SR} \times \text{PF} \times \text{PQ}}$$

其中 SR 为主体相似度、PF 为提示遵循度、PQ 仅基于生成图算。⚠️ 公式与权重细节以原文为准。用户研究显示该指标与人类偏好的一致性明显优于现有替代方案。

## 实验关键数据

### 数据集规模

| 任务大类 | 子任务 | 代表细类 | 数量（约） |
|----------|--------|----------|-----------|
| 以物为中心 | Object+Scene | 1O1S / 2O1S | 5,014 / 4,999 |
| 以物为中心 | Object+Object | 2O / 3O / 4O / 5O | 10,007 / 10,012 / 5,001 / 4,998 |
| 以人为中心 | Person+Person | 多种性别组合 | 共 ~2.4 万 |
| 以人为中心 | Person+Scene | 1P1S / 2P1S | 4,986 / 4,994 |
| 人物交互 HOI | Person+Object / +Clothes / +Object+Clothes | 多变体 | 各约 2–2.8 万 |
| 分解-重组 | De&Re | 自适应 | 11,677 |

源图池：物体 31.5K、人像 44.6K、服饰约 26.8K（上装 17.3K、裤装 1.3K、鞋 428、配饰 7.8K）、场景 11K。MICo-Bench 含 1,000 例（7 子任务各 100 + De&Re 300），均经 3 位独立评审一致通过。

### 主实验（MICo-Bench，Overall 分，节选自 Table 2）

| 模型 | base | w/o De&Re | real | synth |
|------|------|-----------|------|-------|
| BLIP3-o | 2.2 | 42.2 | 43.2 | 43.0 |
| Lumina-DiMOO | 4.3 | 32.3 | 34.2 | 33.9 |
| BAGEL | 33.3 | 42.6 | 44.3 | 44.1 |
| Qwen-Image-Edit（→Qwen-MICo） | 38.5 | 56.4 | 58.2 | 58.1 |
| OmniGen2 | 41.0 | 50.6 | 51.2 | 50.7 |
| GPT-4o（闭源） | 59.6 | – | – | – |
| Nano-Banana（闭源） | 60.3 | – | – | – |

注：base = 原模型；w/o = 不含 De&Re 任务微调；real/synth = 用 De&Re 真实/合成目标微调。原本完全没有 MICo 能力的 BLIP3-o（2.2→43.2）和 DiMOO 微调后从零获得能力；BAGEL、Qwen-Image-Edit 本就有"涌现"的 MICo 能力、微调后进一步强化。Qwen-MICo 在 3 图合成上逼近用数百倍大数据训练的 Qwen-Image-2509，且支持**任意张**多图输入（后者仅限 3 图）。

### 消融：真实目标 vs 合成目标（De&Re）

| 配置 | 现象 |
|------|------|
| w/o De&Re | 即便训练数据里没有"人+物+衣+场景"全组合样本，各模型在 MICo-Bench De&Re 子集上仍涌现出一定性能 |
| real 目标 | 用真实构图作目标，Overall 普遍最高（如 BAGEL 50.9） |
| synth 目标 | 用合成构图作目标，性能与 real 几乎持平（BAGEL 50.6），说明合成数据可有效替代 |

### 关键发现
- **强预训练 I2I 模型存在涌现 MICo 能力**：BAGEL、Qwen-Image-Edit 从未在多图合成数据上训练过，仅把多张源图 token 简单拼接喂进去就表现出 MICo 能力，简单 SFT 后大幅增强。
- **合成数据可替代真实数据**：De&Re 真实目标与合成目标训练效果几乎一致，降低了高质量 MICo 数据的获取成本。
- **闭源模型各有所长**：Nano-Banana 定量分更高，但 GPT-4o 鲁棒性更好，更少出现肢体不全、身份完全丢失或"复制粘贴"伪影。

## 亮点与洞察
- **"用最强闭源模型当数据工厂 + 检索控兼容性 + 人在回路把质量"** 是这篇造数据的核心配方：它把"源图同质化"和"组合不兼容"两个老问题分别交给 Compose-by-Retrieval 和强生成骨干解决，思路可迁移到其他需要高质量合成监督的任务。
- **De&Re 一组件双目标** 的设计很巧妙：同一套组件既得真实构图又得合成构图，天然支持"真实 vs 合成"对照实验，并直接证明了合成数据的可用性。
- **Weighted-Ref-VIEScore 把"跨图注意力过载"这一 VLM 评测顽疾具体化并给了可操作解法**（逐源加权 + 只比生成图与参考图），对任何"多输入图→单输出图"的评测都有借鉴价值。
- "强 I2I 模型 token 简单拼接即涌现 MICo 能力"是一个有趣的实证观察，提示多图合成或许更多是数据问题而非架构问题。

## 局限与展望
- **重度依赖闭源模型**：数据合成（Nano-Banana）、提示生成与核验（GPT-4o、Qwen2.5-VL-72B）全靠强闭源/大模型，其偏差与可得性会传导到数据质量，复现成本高。⚠️ 代码/数据开源情况以官方为准。
- **评测仍以 VLM 为裁判**：Weighted-Ref-VIEScore 虽缓解了跨图过载，但 SC 打分仍由 GPT-4o 给出，参考图本身也由 Nano-Banana 生成，存在"用闭源评闭源"的潜在循环偏差。
- **任务边界**：taxonomy 虽细，但仍主要覆盖人/物/衣/场景的组合，对更抽象或强交互（如复杂物理关系、文字排版）场景覆盖有限。
- 改进方向：引入更多开源生成骨干降低对单一闭源模型的依赖；探索论文留出的"token→源图映射"做潜空间对齐的可控生成。

## 相关工作与启发
- **vs 分割式数据范式（Subject Diffusion / MS-Diffusion）**：它们用 GroundingDINO+SAM 从整图抠实例当源图，常得不完整、语义模糊的样本；本文用收集+检索+强生成合成，源图更干净、组合更兼容。
- **vs UNO / OmniGen2 / DreamO**：这些方法用 S2I 或视频多帧造源图，但弱生成骨干导致风格内容同质；本文直接用最强闭源模型合成、并用人在回路把控多样性与质量。
- **vs DreamOmni2**：后者提出三阶段数据构造流水线，但合成数据仍有风格与上下文同质问题；本文额外用 De&Re 真实分解-重组引入真实世界构图，缓解同质化。
- **vs Echo-4o**：据本文统计，非分割式、非 Flux 系列生成的高质量 MICo 数据极稀缺，公开可得的几乎只有 Echo-4o，而 MICo-150K 在源图多样性与提示语义多样性上显著超越它。

## 评分
- 新颖性: ⭐⭐⭐⭐ 任务体系化 + Compose-by-Retrieval + De&Re + 新评测指标组合扎实，但单点方法多为工程化组合而非全新机制。
- 实验充分度: ⭐⭐⭐⭐⭐ 在 5 个异构开源模型上验证，含真实/合成消融与涌现能力分析，覆盖面广。
- 写作质量: ⭐⭐⭐⭐ 流水线与指标讲解清晰，但部分公式/阈值细节需查附录。
- 价值: ⭐⭐⭐⭐⭐ 填补了开源 MICo 训练数据与评测基准的空白，对推动多图合成研究价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] PhotoFramer: Multi-modal Image Composition Instruction](photoframer_multi-modal_image_composition_instruction.md)
- [\[CVPR 2025\] ORIDa: Object-Centric Real-World Image Composition Dataset](../../CVPR2025/image_generation/orida_object-centric_real-world_image_composition_dataset.md)
- [\[CVPR 2026\] Advancing Image Classification with Discrete Diffusion Classification Modeling](advancing_image_classification_with_discrete_diffusion_classification_modeling.md)
- [\[CVPR 2026\] ConsistCompose: Unified Multimodal Layout Control for Image Composition](consistcompose_multimodal_layout_control.md)
- [\[CVPR 2026\] ViStoryBench: Comprehensive Benchmark Suite for Story Visualization](vistorybench_comprehensive_benchmark_suite_for_story_visualization.md)

</div>

<!-- RELATED:END -->
