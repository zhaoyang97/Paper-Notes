---
title: >-
  [论文解读] RealBirdID: Benchmarking Bird Species Identification in the Era of MLLMs
description: >-
  [CVPR 2026][多模态VLM][细粒度识别] RealBirdID 是一个面向「答得出就给物种、答不出就给理由」的细粒度鸟类识别基准：它从 iNaturalist 真实争议样本里挖出 3.4k 张「不可答」图片（标注上「需要叫声 / 角度遮挡 / 画质太差」三类拒答理由）配上同属的「可答」样本，并配套三套指标，结果发现 GPT-5、Gemini-2.5 Pro 等顶尖 MLLM 在物种级准确率不足 13%、几乎无法区分可答与不可答、即使拒答理由也大多给错。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "细粒度识别"
  - "拒答(abstention)"
  - "鸟类基准"
  - "分类法层级"
  - "MLLM 校准"
---

# RealBirdID: Benchmarking Bird Species Identification in the Era of MLLMs

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Lawrence_RealBirdID_Benchmarking_Bird_Species_Identification_in_the_Era_of_MLLMs_CVPR_2026_paper.html)  
**代码**: https://github.com/cvl-umass/RealBirdID  
**领域**: 多模态VLM  
**关键词**: 细粒度识别, 拒答(abstention), 鸟类基准, 分类法层级, MLLM 校准

## 一句话总结
RealBirdID 是一个面向「答得出就给物种、答不出就给理由」的细粒度鸟类识别基准：它从 iNaturalist 真实争议样本里挖出 3.4k 张「不可答」图片（标注上「需要叫声 / 角度遮挡 / 画质太差」三类拒答理由）配上同属的「可答」样本，并配套三套指标，结果发现 GPT-5、Gemini-2.5 Pro 等顶尖 MLLM 在物种级准确率不足 13%、几乎无法区分可答与不可答、即使拒答理由也大多给错。

## 研究背景与动机

**领域现状**：鸟类一直是细粒度视觉识别（FGVR）的标尺，CUB-200、NABirds、iNaturalist 等数据集推动了 part/attribute/分类法层级建模；近两年 MLLM + 开放词表 prompting 又把零样本分类拉了上来，似乎细粒度识别在大模型时代已经被「解决」得差不多。

**现有痛点**：但这些 benchmark 几乎都只含「可答」（answerable、in-schema）样本——每张图都有一个标准物种答案。现实部署里大量图片根本无法从单张图判物种：关键线索可能是非视觉的（需要叫声）、或被遮挡 / 角度 / 低分辨率掩盖。在「必须选一个答案」的逼问下，模型会自信地瞎猜乃至幻觉，这在医疗、法律等场景是危险的。

**核心矛盾**：现有评测只奖励「自信作答」，却从不考察「该闭嘴时能不能闭嘴、而且说得出为什么闭嘴」。文本任务里拒答（abstention）已有 SQuAD2.0、AbstentionBench 等，但它们判的是「问题本身不可答」；而本文要的是**问题（prompt）固定不变、模型必须仅凭视觉证据判断该不该拒答**——这块在视觉多模态领域几乎空白。

**本文目标**：把问题拆成两个子能力同时考——(1) 给定一个属（genus）下的物种列表，能不能穷举式地认对物种；(2) 面对该属下真实的「不可答」样本，能不能拒答、并给出正确理由（叫声 / 角度遮挡 / 画质）。

**切入角度**：作者不去造合成的难样本，而是直接挖 iNaturalist 上**社区专家真的吵起来、最终只能定到属一级**的观测——这些自带「为什么定不到种」的人类讨论记录，是天然的「带理由的不可答」标签来源。

**核心 idea**：用「答物种 or 带证据拒答」重新定义细粒度识别任务，造一个分类法**部分标注**（叶子节点可能缺失真值）的基准，并设计能在层级树上评估「准确率 vs 拒答」权衡的指标，逼出当前 MLLM 在「知之为知之」上的真实短板。

## 方法详解

本文是 benchmark 论文，核心不是某个模型 pipeline，而是「任务定义 + 数据集怎么造 + 用什么指标量」。下面按这三块讲清楚；由于是纯数据集/评测协议、没有可串的网络模块，不画框架图。

### 整体框架
RealBirdID 的评测闭环是：对每个属，准备一对子集——**可答集 A**（该属下穷举采样的物种图，有物种真值）与**不可答集 UA**（社区只能定到属、并标了拒答理由的图）。被测模型对一张图输出「物种预测 + 不确定度」或「自由文本回答」；评测端通过扫描不确定度阈值，把预测在「叶子物种 → 中间属节点（=拒答）」之间滑动，画出三类权衡曲线并取曲线下面积作为汇总指标。最终在 248 个属、3442 个物种、35138 张图（A 31885 + UA 3253）的规模上，对 CLIP 系编码器与 MLLM 一起打分。

整个数据集只作 validation 资源、不切 train/val/test，定位是「衡量进展的标尺」而非训练集。

### 关键设计

**1. 任务重定义：答物种或带证据拒答，且 prompt 固定**

本文把任务从「图→物种」改成「图→物种 或 拒答+理由」。和文本拒答基准（AbstentionBench、RGQA）最大的区别是：那些工作靠改写问题文本来制造「不可答」，而 RealBirdID 的 prompt 始终是语义等价的一句「What is the species of this bird?」，模型**只能从视觉内容本身**推断出「这图我答不了」。拒答理由被收敛成三类——需要叫声（Vocalization）、角度/遮挡（Angle/Occlusion）、画质太差（Quality）。这个设定的巧妙在于：它把「拒答能力」从「读懂问题措辞」中剥离出来，单独逼问视觉系统——你看着这张糊掉的、或必须听叫声才能分的鸟，知不知道该停手。

**2. 用 iNaturalist 真实争议挖「带理由的不可答」样本**

不可答样本不是人工合成，而是从 iNaturalist 的真实社区争议里挖出来。流程是：从 140 万张 verifiable 观测出发（**故意不要求 Research Grade**，因为要的就是物种级别上真有分歧的样本），先用 YOLOv3 过滤「有没有鸟」、MANIQA 过滤画质，得到约 41 万张；再保留「至少有属级预测、≥2 位贡献者」的观测，剔除死鸟/蛋/羽毛；然后**用手写正则 + 轻量启发式解析评论与鉴定历史里的歧义信号**（分布/范围歧义、性别/生命阶段二态、换羽磨损、视角不足、杂交/逃逸、分类学不确定、画质低），把每个匹配映射到一个临时拒答理由 schema，得到 5300 个「不可答」候选；最后由懂鸟的专家用 Birds of the World 工具核验「叫声 / 角度遮挡 / 画质」三类失败、剔除太难判或讨论本身有误的，迭代精修文本解析，最终落到 3.4k 张带理由的不可答样本。这一设计让「不可答」既真实又自带人类可对照的理由标签——这是后面能评「拒答理由对不对」的前提。

可答集则反过来：对每个不可答观测所属的属，用 iNaturalist Taxon API **穷举该属下全部后代物种**并采 Research Grade 图（每种最多 200 张），自然得到长尾分布；同时用 SINR 地理-物种模型，把经纬度映射成该地点的物种出现概率向量，给每个不可答样本生成「最可能的候选物种清单」，供后续「用 range map 缩小候选」实验使用。

**3. 三套层级感知指标 + 编码器的概率聚合（TreeGT）**

数据集的「部分不可答」（很多图压根没有物种级真值）让普通准确率失效，作者配了三套扫阈值取面积的指标：

- **Metric 1 — 拒答权衡 UA/A**：在某阈值下分别统计模型在 A 集和 UA 集上的拒答比例，扫阈值得到「UA 被拒答率 vs A 被拒答率」曲线，理想模型应「UA 全拒、A 全不拒」（曲线贴左上角），取曲线下面积为 UA/A。
- **Metric 2 — 分类性能 IG**：借 DARTS 的「准确率 vs 信息增益」曲线——把物种概率向上聚合得到属级、纲级等各层预测，每个预测同时记其「对错」与「信息增益（=分类法深度，物种比属信息量大）」；扫阈值得到「越往深预测、准确率越低」的权衡曲线，面积即 IG。它避免奖励那种「啥都拒答」的退化模型。
- **Metric 3 — 校准 AUC**：固定一个拒答比例，看 A 集的物种/属准确率、UA 集的属准确率，扫阈值取 AUC，衡量「丢掉高熵样本后准确率是否随之上升」的校准性。

另外，CLIP 这类编码器本身**没有拒答类**，作者指出一个关键坑：直接把「属名」当一个文本 prompt 拼进物种列表（Flat List）来当拒答类，效果极差（HM≈0）；改用层级方法 **TreeGT 的概率聚合**——把某属下所有子物种的 softmax 概率求和当作该属概率（如「Crows and Ravens」属概率 = 其下 53 个物种概率之和），才能把编码器纳入同一套层级指标公平比较。对 MLLM 则用 nlg2choice 先生成自由文本、再约束解码抽出最终答案与拒答理由，并检索式地构造物种概率向量再聚合到属级。

## 实验关键数据

评测覆盖 CLIP / MetaCLIP / WildCLIP / SigLIP / BioCLIP 等编码器，与 InternVL3-8B、Qwen2.5-VL-7B、Gemma-3-12B、Llama-3.2-11B-Vision、Gemini-2.5 Pro、GPT-5 等 MLLM。

### 主实验：分类性能与拒答权衡（Metric 2 IG / Metric 1 UA/A）

| 模型 | IG（分类性能↑） | UA/A（拒答权衡↑） |
|------|------|------|
| BioCLIP | **68.9** | 49.6 |
| MetaCLIP-L/14 | 66.0 | 42.5 |
| SigLIP-so400m | 53.7 | **53.2** |
| CLIP-L/14 | 62.0 | 48.1 |
| Gemini-2.5 Pro | 57.7 | 46.2 |
| GPT-5 | 56.4 | 44.1 |
| Qwen2.5-VL-7B | 54.2 | 41.7 |
| Gemma-3-12B | 46.3 | 39.2 |

关键观察：分类最强的 BioCLIP（IG 68.9）在拒答权衡上反而输给 SigLIP（53.2）；编码器内部分类准确率与拒答能力**无显著正相关**（皮尔逊 r=0.60），且同训练家族里增大模型/数据只涨 IG、不涨 UA/A——说明拒答行为由与标准识别**不同的因素**主导。

### 物种/属级准确率与校准（Metric 3，节选）

| 模型 | 可答-物种 Acc | 可答-属 Acc | 不可答-属 Acc |
|------|------|------|------|
| BioCLIP | **17.0** | **57.0** | 57.6 |
| MetaCLIP-L/14 | 11.8 | 56.1 | **63.6** |
| GPT-5 | 10.4 | 45.6 | 58.6 |
| Gemini-2.5 Pro | 12.7 | 52.8 | 60.1 |
| Qwen2.5-VL-7B | 6.7 | 40.6 | 52.6 |
| InternVL3-8B | 1.5 | 16.7 | 34.7 |

物种级 3442 类准确率普遍只有 3.7–17%，连最强的 GPT-5 / Gemini 也 ≤13%；MLLM 整体落后于专用编码器（最强 MLLM Gemini IG 57.7 vs BioCLIP 68.9）。

### 拒答理由是否给对（Tab.4 / Fig.8）

| 真值理由 | Qwen2.5-VL | Llama-3.2V | InternVL3 | Gemma-3 |
|------|------|------|------|------|
| 画质 quality | 0.158 | 0.086 | 0.279 | 0.041 |
| 角度/遮挡 | 0.144 | 0.080 | 0.291 | 0.052 |
| 需要叫声 | 0.098 | 0.077 | 0.278 | 0.044 |

表中为「无论给什么理由，模型在该真值问题下选择拒答的概率」，整体都很低。混淆矩阵显示一个系统性偏差：模型几乎都把失败甩锅给「画质」，Qwen2.5-VL 对真·画质问题 100% 标对，但 42.4% 的「角度/遮挡」被错标成画质；而「需要叫声」几乎**无任何模型会预测**——视觉训练偏置让 MLLM 根本不把「缺音频线索」当作拒答信号。

### 关键发现
- **会拒答 ≠ 拒答得好**：拒答能力与分类准确率解耦，单纯把 FGVR 准确率刷上去无助于可靠拒答，需要显式的拒答感知目标。
- **拒答率极度脆弱**：对 15 种语义等价的指令改写，InternVL3 的拒答率标准差高达 σ=10.10（可在 3%–23% 间乱跳）；Gemma-3 则几乎从不拒答（1–2%）。不明说「可以拒答」时，MLLM 默认几乎不拒答。
- **range map 帮分类不帮拒答**：加入地理范围信息把平均 IG 从 57.2→88.1（属级 AUC 近乎完美），但对拒答权衡只有轻微影响（45.8→47.7），且对 MLLM 的拒答反而**变差**。

## 亮点与洞察
- **拒答标签来自人类真实争议**：不造合成难样本，而是挖 iNaturalist 上社区吵到只能定到属、并自带「为什么定不到种」讨论的观测——这是「带理由的不可答」最干净的来源，可迁移到任何有专家协作标注平台的领域（植物、菌类、医学影像会诊）。
- **prompt 固定、只能视觉拒答**：把拒答能力从「读懂措辞」里剥出来单独考，揭示了一个少有人测的盲区——模型仅凭视觉证据知不知道该闭嘴。
- **层级部分标注 + 信息增益指标**：用「准确率 vs 分类法深度」的权衡，优雅解决了「很多图根本没有叶子真值」的评测困境，对任何带 taxonomy 的识别任务都可复用。
- **「缺叫声」盲点**：MLLM 系统性地不把「需要听声音」当拒答理由，直指视觉训练数据缺乏音频依赖型失败的暴露——这是一个具体且可改的训练缺口。

## 局限与展望
- 数据集只作 validation、无训练集，无法直接驱动「训练出会拒答的模型」；如何把这套理由标签变成可训练监督是开放问题。
- 多图观测里更清晰的鸟常出现在后续帧，本文只取首图、留下多帧处理为 future work，可能低估了部分「角度/遮挡」样本的可答性。
- 不可答理由被压成三类（叫声/角度遮挡/画质），而解析阶段其实涉及更多歧义类型（换羽、杂交、性别二态等），三类之外的细粒度拒答原因被合并/丢弃。
- 拒答判定依赖 nlg2choice 的约束解码与正则解析，文本解析本身可能引入噪声；评测对自由文本回答的「等价判定」也存在主观性。

## 相关工作与启发
- **vs AbstentionBench / SelfAware（文本拒答）**：它们考的是「问题不可答」、靠问题文本触发拒答；RealBirdID 把证据搬到**图像里**、prompt 固定，逼模型纯视觉判断该不该拒答。
- **vs RGQA（视觉拒答）**：RGQA 配对人工标注的不可答问题、仍偏「检测不可答的文本」；本文强调「需要视觉信息才能拒答」，且语义等价 prompt 下考拒答的鲁棒性。
- **vs 层级分类方法（Snæbjarnarson、Tan 等）**：它们假设分类树每个节点都有真值标签；RealBirdID 的 taxonomy 是**部分标注**的（叶子可能缺失），现有大多数层级指标不适用，因而引入 DARTS/TreeGT 的概率聚合与信息增益评测。
- **vs CUB-200 / NABirds / iNat19-Birds**：这些基准全是可答样本（UA=0），RealBirdID 是首个把 3253 张带理由不可答样本与可答集成对、并把物种数推到 3442 的细粒度鸟类拒答基准。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个「prompt 固定、纯视觉拒答 + 带人类理由标签」的细粒度识别基准，定义了一个被忽视的能力维度
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 9+ 编码器与 6 个 MLLM，三套指标 + range map / 指令鲁棒性 / 理由混淆矩阵多角度剖析
- 写作质量: ⭐⭐⭐⭐ 任务动机与发现讲得清楚，但指标定义（IG/UA/A/AUC）密集，初读需反复对照图表
- 价值: ⭐⭐⭐⭐⭐ 直指 MLLM「该闭嘴时不闭嘴、闭嘴也说错理由」的安全短板，为拒答感知识别立了可量化标靶

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Revisiting Model Stitching in the Foundation Model Era](revisiting_model_stitching_in_the_foundation_model.md)
- [\[CVPR 2026\] IF-Bench: Benchmarking and Enhancing MLLMs for Infrared Images with Generative Visual Prompting](if-bench_benchmarking_and_enhancing_mllms_for_infrared_images_with_generative_vi.md)
- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](../../ICCV2025/multimodal_vlm/mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)
- [\[CVPR 2026\] EgoSound: Benchmarking Sound Understanding in Egocentric Videos](egosound_benchmarking_sound_understanding_in_egocentric_videos.md)
- [\[CVPR 2026\] Benchmarking Single-Factor Physical Video-to-Audio Generation](benchmarking_single-factor_physical_video-to-audio_generation.md)

</div>

<!-- RELATED:END -->
