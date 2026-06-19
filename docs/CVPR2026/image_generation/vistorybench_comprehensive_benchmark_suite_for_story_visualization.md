---
title: >-
  [论文解读] ViStoryBench: Comprehensive Benchmark Suite for Story Visualization
description: >-
  [CVPR 2026][图像生成][故事可视化] ViStoryBench 构建了一个包含 80 个多风格故事、344 个角色、1317 个镜头的综合基准，提出 12 项自动化评估指标（涵盖角色一致性、风格相似度、提示对齐、copy-paste 检测等），系统评估了超过 25 种开源/商业故事可视化方法，填补了该领域缺乏统一评估标准的空白。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "故事可视化"
  - "基准测试"
  - "角色一致性"
  - "多维度评估"
  - "叙事生成"
---

# ViStoryBench: Comprehensive Benchmark Suite for Story Visualization

**会议**: CVPR 2026  
**arXiv**: [2505.24862](https://arxiv.org/abs/2505.24862)  
**代码**: [https://github.com/ViStoryBench/ViStoryBench](https://github.com/ViStoryBench/ViStoryBench)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 故事可视化、基准测试、角色一致性、多维度评估、叙事生成

## 一句话总结
ViStoryBench 构建了一个包含 80 个多风格故事、344 个角色、1317 个镜头的综合基准，提出 12 项自动化评估指标（涵盖角色一致性、风格相似度、提示对齐、copy-paste 检测等），系统评估了超过 25 种开源/商业故事可视化方法，填补了该领域缺乏统一评估标准的空白。

## 研究背景与动机

**领域现状**：故事可视化旨在根据叙事文本和角色参考图生成一组视觉一致的图像序列。近年来，扩散模型和自回归模型的进步推动了这一领域的快速发展，出现了 StoryDiffusion、UNO、USO 等训练无关方法和基于 LLM 的多阶段 pipeline（如 MMStoryAgent、MovieAgent）。

**现有痛点**：现有基准测试存在三大局限：(1) 测试场景单一，多局限于短文本提示或单图生成，无法反映真实叙事的复杂性；(2) 缺少角色参考图，无法测试角色一致性；(3) 评估指标不全面，通常只用 FID/CLIP-Score 等通用指标，忽略了故事可视化特有的维度如角色匹配精度、风格一致性、copy-paste 行为等。

**核心矛盾**：故事可视化本质上是一个多维度问题——需要同时保证角色身份一致、风格统一、叙事对齐、画面美观，但现有评估框架无法系统化地度量这些维度，导致不同方法间的对比缺乏可信度。

**本文目标** (1) 构建多样化的故事脚本+角色参考数据集；(2) 设计覆盖多个关键维度的自动化指标体系；(3) 在统一框架下对比评测大量方法。

**切入角度**：作者从"真实叙事场景"出发，收集文学、电影、民间故事等 80 个故事片段，涵盖 10 种视觉风格，用 LLM 辅助生成结构化剧本（含场景描述、角色动作、镜头设计），并经人工审核。

**核心 idea**：构建首个涵盖多风格、多角色、多指标的故事可视化综合基准，系统性揭示现有方法的优劣。

## 方法详解

### 整体框架
ViStoryBench 要回答一个被现有基准回避的问题：当一堆方法都号称能"根据故事和角色参考图生成连贯插画序列"时，到底谁更好、好在哪一维度。它把这件事拆成三步落地：先从文学、电影、民间故事等多源素材里收集 80 个跨 10 种风格的故事，用 LLM 把每个故事改写成带角色参考图的结构化剧本；再围绕角色、风格、提示对齐、美学、copy-paste 这几条故事可视化特有的维度，设计 12 项自动化指标；最后在同一套数据和指标下，把 25+ 种开源/商业方法（图像方法和视频方法都包含）跑一遍横向评测。难点不在生成本身，而在如何把"角色像不像、风格统不统一、画面对不对得上叙事"这些模糊的人类判断，翻译成可复现的数值。

### 关键设计

**1. 结构化剧本生成：把自然语言故事改写成可逐维评估的剧本**

要做细粒度评估，第一步得有细粒度的 ground truth。如果只给模型一句"小红帽在森林里遇到狼"，评估时根本无从判断它画错的是场景、角色还是动作。ViStoryBench 用 5 种提示工程策略引导 LLM 把故事摘要并拆成镜头（Shot），每个镜头强制包含 5 个标准化组件：场景描述、剧情对应、在场角色列表、静态镜头描述、以及镜头视角设计（细到景别、拍摄类型、机位角度）。所有 LLM 产出再经人工审核，保证叙事连贯、逻辑自洽。这样拆完之后，后面的"角色动作对齐""镜头设计对齐"才有明确的参照可对，而不是笼统地比一张图像不像。

**2. 角色身份相似度 CIDS：先裁出角色再比对，而不是比整张图**

故事可视化最核心的诉求是"同一个角色在不同镜头里得是同一个人"，但直接拿整图算 CLIP 相似度会被背景、构图带偏，量不准角色本身。CIDS 走一条四阶段流水线来对准这个目标：先用 Grounding DINO 从参考图和生成图里把角色区域裁出来；再按风格选特征提取器——非写实风格用 CLIP，写实人脸用 ArcFace / AdaFace / FaceNet——抽成 512 维特征向量；接着对参考角色和生成角色构相似度矩阵、做二分图匹配找最优对应；最后取匹配对的平均余弦相似度。它进一步分成 Cross-CIDS（生成图对参考图，衡量像不像目标角色）和 Self-CIDS（生成图之间互比，衡量同一角色跨镜头稳不稳）。先检测再比对的好处是把"角色一致性"从背景噪声里隔离出来单独度量。

**3. 多粒度提示对齐：把"对不对得上文本"拆成四个层次分开打分**

一个粗粒度的 CLIP-Score 给出 0.3，你无法判断模型究竟是场景错了、镜头机位错了，还是角色交互关系错了。多粒度提示对齐把对齐度拆成 4 个子维度：Scene Score（画面与叙事整体是否对应）、Shot Score（镜头视角是否一致）、Character Interaction（群体交互是否对齐）、Individual Actions（个体动作是否准确）。每个维度交给 VLM 评判——主评估用 Gemini-3-Pro，可复现评估用 Qwen3-VL——按 Likert 5 级量表（0–4）打分，再映射到百分制。拆开之后，"场景对了但动作错了"这类此前被均值掩盖的失败模式就能被单独读出来。

> ⚠️ Gemini-3-Pro / Qwen3-VL 的具体版本以原文为准。

**4. Copy-Paste 检测：识别"高相似度其实是直接粘贴参考图"的作弊**

CIDS 高未必是好事——有些方法（如 Story-Adapter）角色相似度很高，是因为它干脆把参考图原样贴了上去，这种"虚假一致性"会污染排名。Copy-Paste 检测引入同一角色的第二张参考图作为 proxy target 来戳穿它：记生成角色特征为 $g$、被当作输入的那张参考特征为 $r$、同一角色的另一张参考特征为 $t$，若 $g$ 离 $r$ 明显比离 $t$ 更近，就说明它复制的是具体那张输入图而非真正学到了角色身份。

$$\text{Copy-Paste} \;\propto\; \mathbb{1}\big[\,d(g, r) < d(g, t)\,\big]$$

> ⚠️ 上式为示意，原文用几何归一化计算 Copy-Paste Rate，具体公式以原文为准。

把这些判定在数据集上聚合就得到 Copy-Paste Rate。第二张参考图之所以关键，是因为它代表"同一角色但不同呈现"——真正学会角色的模型对 $r$ 和 $t$ 应当大致等距，只有照搬输入的模型才会偏向 $r$。这个用 proxy target 区分"生成的一致"和"粘贴的一致"的思路，可以迁移到任何需要识别模型作弊的评测场景。

### 损失函数 / 训练策略
ViStoryBench 是评测基准，不涉及模型训练，核心贡献在评估协议设计和指标与人工评分的相关性验证上。

## 实验关键数据

### 主实验

| 方法 | CSD-Cross↑ | CIDS-Cross↑ | PA-Avg↑ | OCCM↑ | Inc↑ | Aes↑ |
|------|-----------|------------|---------|-------|------|------|
| OmniGen2 | 0.454 | 0.548 | 2.49 | 70.2 | 11.05 | 5.25 |
| UNO (FLUX1) | 0.391 | 0.485 | 2.30 | 74.2 | 12.40 | 5.23 |
| QwenImageEdit | 0.381 | 0.475 | 2.51 | 59.8 | 13.42 | 5.50 |
| AnimDirector (SD3) | 0.288 | 0.401 | 2.55 | 67.4 | 12.02 | 5.59 |
| Story-Adapter (scale=0) | 0.456 | 0.460 | 1.90 | 69.0 | 12.98 | 4.99 |
| StoryDiffusion (SDXL) | 0.269 | 0.397 | 1.85 | 62.9 | 15.72 | 5.76 |

### 消融实验（指标验证）

| 指标维度 | 人工评估相关性 | 说明 |
|---------|-------------|------|
| CIDS (Cross) | 高 | 与人工角色一致性评分显著正相关 |
| PA (Scene) | 中高 | VLM 评估稳定性分析方差低 |
| Copy-Paste Rate | - | Copy-Paste Baseline 得分 0.474，正常方法 <0.28 |
| Inception Score | 高 | 多样性指标区分度好，StoryDiffusion(15.72) vs SEED-Story(6.30) |

### 关键发现
- **OmniGen2 在角色一致性上表现最佳**（CIDS-Cross=0.548），但 copy-paste 率也最高（0.275），暗示其可能过度依赖参考图复制
- **提示对齐与角色一致性存在 trade-off**：AnimDirector 在 PA-Avg（2.55）上领先，但 CIDS 仅 0.401；Story-Adapter 角色相似度高但提示对齐弱
- **视频方法在场景一致性上更好**：MovieAgent-SD3 的 PA-Avg 达到 2.54，与最佳图像方法持平，但角色一致性和美学得分偏低
- **OCCM 指标揭示角色数量幻觉严重**：最高的 Vlogger 也仅 76.6%，说明多角色场景中角色数量控制是普遍难题

## 亮点与洞察
- **Copy-Paste 检测指标是本文最巧妙的设计**：通过引入同一角色的"第二参考图"作为 proxy target，用几何归一化区分"生成的一致"和"粘贴的一致"，这个思路可以迁移到任何需要检测模型"作弊行为"的评测场景
- **多粒度提示对齐的拆分策略**值得借鉴：将 prompt alignment 拆为 scene/shot/CI/IA 四个子维度，比单一 CLIP-Score 信息量大得多，可推广到视频生成等其他条件生成任务的评估
- **VLM 作为评估器**的方案经过严格的稳定性验证（低方差），为未来大规模自动化评估提供了可靠范式

## 局限与展望
- **角色检测依赖 Grounding DINO**：当检测失败时（特别是非写实风格），CIDS 和 OCCM 指标会受到影响，检测器本身引入的误差未被充分量化
- **数据集规模有限**：80 个故事、344 个角色在统计意义上仍然偏少，特别是某些稀有风格（如 3D 渲染）样本不足
- **缺少时序评估**：对于视频方法只取关键帧评估，丢失了帧间连贯性和动画流畅度信息
- **VLM 评估偏差**：Gemini-3-Pro 的评估标准可能与人类偏好存在系统性偏移，且闭源模型难以复现

## 相关工作与启发
- **vs VinaBench**: VinaBench 虽也面向故事可视化，但缺少角色参考图、仅支持 6 种风格，ViStoryBench 在角色一致性评估和风格覆盖面上更全面
- **vs DreamBench++**: DreamBench++ 聚焦单图生成，本文扩展到多图序列场景，增加了叙事对齐和角色匹配等故事级指标
- **启发**：该基准的指标体系可以直接用于评估长视频生成中的角色一致性和叙事对齐，也可为交互式故事创作系统提供评测标准

## 评分
- 新颖性: ⭐⭐⭐⭐ 指标设计体系全面且有创新（如 copy-paste 检测），但作为 benchmark 论文核心贡献在于"系统性"而非"理论突破"
- 实验充分度: ⭐⭐⭐⭐⭐ 评测了 25+ 种方法，指标与人工评估的相关性验证充分，统计分析严谨
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，不过表格因方法数量过多略显拥挤
- 价值: ⭐⭐⭐⭐⭐ 填补了故事可视化领域缺乏统一评估标准的空白，对后续研究有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DreamingComics: A Story Visualization Pipeline via Subject and Layout Customized Generation using Video Models](dreamingcomics_a_story_visualization_pipeline_via_subject_and_layout_customized_.md)
- [\[CVPR 2026\] EMMA: Concept Erasure Benchmark with Comprehensive Semantic Metrics and Diverse Categories](emma_concept_erasure_benchmark_with_comprehensive_semantic_metrics_and_diverse_c.md)
- [\[CVPR 2026\] RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark](realunify_do_unified_models_truly_benefit_from_unification_a_comprehensive_bench.md)
- [\[CVPR 2026\] ShowTable: Unlocking Creative Table Visualization with Collaborative Reflection and Refinement](showtable_unlocking_creative_table_visualization_with_collaborative_reflection_a.md)
- [\[CVPR 2026\] MICo-150K: A Comprehensive Dataset Advancing Multi-Image Composition](mico-150k_a_comprehensive_dataset_advancing_multi-image_composition.md)

</div>

<!-- RELATED:END -->
