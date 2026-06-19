---
title: >-
  [论文解读] TUNA: Taming Unified Visual Representations for Native Unified Multimodal Models
description: >-
  [CVPR 2026][多模态VLM][统一多模态模型] TUNA 把一个 VAE 编码器和一个语义表示编码器**级联**起来，得到一套同时适配「理解」和「生成」的连续统一视觉表示，再配上自回归文本头 + 流匹配生成头，让 1.5B/7B 规模的单一原生模型在图像/视频理解、图像/视频生成、图像编辑上全面拿到 SOTA（MMStar 61.2、GenEval 0.90）。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "统一多模态模型"
  - "统一视觉表示"
  - "流匹配"
  - "VAE隐空间"
  - "图文生成"
---

# TUNA: Taming Unified Visual Representations for Native Unified Multimodal Models

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_TUNA_Taming_Unified_Visual_Representations_for_Native_Unified_Multimodal_Models_CVPR_2026_paper.html)  
**领域**: 多模态VLM  
**关键词**: 统一多模态模型, 统一视觉表示, 流匹配, VAE隐空间, 图文生成

## 一句话总结
TUNA 把一个 VAE 编码器和一个语义表示编码器**级联**起来，得到一套同时适配「理解」和「生成」的连续统一视觉表示，再配上自回归文本头 + 流匹配生成头，让 1.5B/7B 规模的单一原生模型在图像/视频理解、图像/视频生成、图像编辑上全面拿到 SOTA（MMStar 61.2、GenEval 0.90）。

## 研究背景与动机

**领域现状**：统一多模态模型（UMM）想用一个模型同时做多模态理解和生成。当下主流分两派：一派用**解耦表示**（understanding 用语义编码器如 SigLIP，generation 用 VAE 隐空间），代表是 BAGEL、Mogao；另一派追求**统一表示**，全任务共享一套视觉编码，代表是 Chameleon、Transfusion、Harmon。

**现有痛点**：解耦派要在一个模型里塞进两个格式完全不兼容的视觉编码器——对同一张图，SigLIP 特征和 Wan VAE 隐变量在空间压缩率（16× vs 8×）、时间压缩率（无 vs 4×）、通道维度（1152 vs 16）三处都对不上，于是只能上 MoE 式架构分别处理，徒增参数与训练/推理成本，还会引入表示冲突。统一派本应更省、更优雅，**但实际效果往往打不过解耦派**：单一编码器（VQ-VAE 或 MAR）天生偏科——偏向理解就生成弱、偏向生成就理解弱。Show-o2 想用「双路 late-fusion」把 SigLIP 和 VAE 特征晚期融合来缓解，但作者在 §3.4 的分析发现它融出来的表示**严重偏向语义特征**，导致生成质量受限。

**核心矛盾**：理解任务想要高层语义特征，生成任务想要高保真的可重建隐空间，单一编码器二者不可兼得；而把两类编码器硬拼又会引入格式冲突。问题根本在于——**缺一套"既语义又可重建"的统一视觉表示**。

**切入角度**：作者观察到几个被忽视的事实：① 连续表示（KL-VAE 隐空间）在生成上优于离散表示，理解模型也偏好连续语义特征，所以统一表示应建在**连续 VAE 隐空间**上；② 语义特征反过来能帮生成（REPA、RAE 已证明）；③ VAE 隐变量本就能支撑语义理解（UniTok、TokLIP 等工作）。既然 VAE 隐空间双向都吃得开，那就不该丢掉它，而是**在它之上再叠一层语义编码器**提炼高层特征。

**核心 idea**：直接把一个表示编码器（SigLIP2）接到 VAE 编码器后面级联使用——VAE 负责保留可重建的连续隐空间，语义编码器负责在其上提炼高层语义，得到一套对理解和生成都"足够表达"的统一视觉表示；理解走自回归文本生成，生成走流匹配，全程端到端联合训练。

## 方法详解

### 整体框架

TUNA 是一个**原生** UMM（从头就在理解+生成两个目标上联合预训练，而非把现成理解模型和生成模型用连接器拼起来）。一次前向的数据流是：输入图像/视频 → 3D 因果 VAE 编码器压成连续隐变量 → 按流匹配方式加噪得到带噪隐变量 → 改造过的 SigLIP2 编码器提炼语义 → MLP 连接器投影成统一视觉表示 $z$ → 与文本 token 拼接送入 LLM 解码器 → 根据任务走两个出口：理解走语言建模头做下一 token 预测，生成走流匹配头预测速度场去噪出图。

理解与生成的唯一开关是**加噪时间步 $t$**：理解时固定 $t=1$，让带噪隐变量退化为干净隐变量；生成时在 $[0,1]$ 随机采样 $t$，让模型学会从噪声中恢复图像。同一套表示、同一个解码器主干，靠 $t$ 和注意力掩码切换两类任务。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入图像 / 视频 X"] --> B["3D 因果 VAE 编码器<br/>空间16× 时间4× 压缩"]
    B --> C["级联式统一视觉表示<br/>VAE隐空间加噪 → SigLIP2 → MLP"]
    C -->|视频| D["视频窗口化注意力<br/>4帧窗口独立编码"]
    C --> E["拼接文本 token + timestep token"]
    D --> E
    E --> F["双路输出头的 LLM 解码器<br/>文本因果掩码 / 视觉双向掩码"]
    F -->|理解 t=1| G["语言建模头<br/>自回归文本"]
    F -->|生成 t∈[0,1]| H["流匹配头<br/>预测速度场去噪"]
```

### 关键设计

**1. 级联式统一视觉表示：VAE 编码器后直接叠语义编码器，一套表示双任务通吃**

这是 TUNA 的核心，针对"单编码器偏科、双编码器格式冲突"这个痛点。做法朴素得反直觉：给定输入 $X$，先用 Wan 2.2 的 3D 因果 VAE 编码器压成连续隐变量 $x_1$（空间 16×、时间 4× 下采样）；再用 SigLIP2 视觉编码器 $\Phi$ 在这个隐变量上提语义特征；最后过一个两层 MLP 连接器得到统一表示 $z=\mathrm{MLP}(\Phi'(x_t))$。这里有个关键工程改动：SigLIP2 原本的 patch embedding 是 16×16，但 VAE 已经把图压了 16×，再切 16×16 patch 序列会太短、与原图 $\Phi(X)$ 的 token 长度对不上，所以作者把它换成**随机初始化的 1×1 patch embedding** 层（记为 $\Phi'$），保证序列长度一致。

为什么有效？它和 Show-o2 的 late-fusion 形成鲜明对比：Show-o2 是 VAE 和语义编码器各跑各的、最后一层才融合，作者在 §3.4 用 CKNNA 对齐分析发现这种晚融合让最终表示几乎只继承语义分支（与理解分支 CKNNA=0.45，与生成分支极弱），所以偏科。而 TUNA 是让语义编码器**直接在 VAE 隐变量上从底到顶逐层做深度融合**，得到的表示同时对齐 SigLIP2（语义参考，CKNNA>0.5）和 SD3-Medium（生成参考），更"平衡"，理解生成两头都不丢。

**2. 共享隐空间加噪 + timestep 条件：用流匹配把生成嵌进同一套表示，理解时退化为干净隐变量**

针对"理解要干净特征、生成要带噪去噪"的张力。TUNA 不为生成单独造一套输入，而是在同一份 VAE 隐变量上按流匹配规则插值加噪：

$$x_t = t\,x_1 + (1-t)\,x_0,\quad t\in[0,1],\ x_0\sim\mathcal{N}(0,1)$$

训练时生成任务随机采 $t$、理解任务固定 $t=1$（此时 $x_t=x_1$ 即干净隐变量）。语义编码器 $\Phi'$ 直接吃这个 $x_t$，所以"理解"和"生成"共用完全相同的表示通路，只是噪声水平不同。生成侧再额外把一个表示 $t$ 的 timestep token 前置到 $z$ 上送进解码器，让模型知道当前噪声强度。这样设计的妙处在于：生成的梯度能顺着流匹配头一路回传到表示编码器（Stage 1 就靠这个把表示同时对齐到理解和生成），真正做到"一套表示、两个目标共同塑形"，而不是两套表示各练各的。

**3. 视频窗口化注意力：把帧维折进 batch 维，避免长序列爆炸**

针对视频输入序列过长的效率痛点。视频隐变量是 $x_t\in\mathbb{R}^{b\times c\times f\times h\times w}$，若把所有帧拍平成一条序列喂给 $\Phi'$，序列长度随帧数线性膨胀、显存和算力都吃不消。作者用 einops 的 rearrange 把帧维 $f$ 折进 batch 维：

$$\bar{x}_t=\mathrm{rearrange}(x_t,\ b\,c\,f\,h\,w\to (b f)\,c\,h\,w)$$

$$\bar{z}_v=\mathrm{MLP}(\Phi'(\bar{x}_t)),\quad z_v=\mathrm{rearrange}(\bar{z}_v,\ (b f)\,d\to b\,(f d))$$

这等于让语义编码器**对每个 4 帧窗口独立做注意力**（4 帧来自 VAE 的时间压缩），各窗口互不干扰，编码完再 reshape 回去。代价是窗口间无直接交互，但效率提升显著，让 1.5B 解码器也能扛起视频生成。

**4. 双路输出头的 LLM 解码器：文本因果掩码 + 视觉双向掩码，理解走 LM 头、生成走流匹配头**

针对"同一个解码器要同时干自回归和扩散两件事"的痛点。统一表示 $z$ 与文本 token 拼接后送进 Qwen2.5 解码器，注意力掩码按模态切换：文本 token 用因果掩码（自回归要求），视觉 token 用双向掩码（图像无先后序）。出口分两路——理解任务把解码器输出过语言建模头预测文本 token；生成/编辑任务把完整 token 序列喂给一个随机初始化的流匹配头预测去噪速度场。这个流匹配头复用解码器架构、用 AdaLN-Zero 注入 timestep 条件（沿用 Show-o2 和 DiT 的做法），生成/编辑时还在文本-视觉拼接序列上用多模态 3D-RoPE 来处理交错的指令与视觉内容。一个主干、两个轻量头，避免了解耦派那种 MoE 式重复参数。

### 损失函数 / 训练策略

三阶段渐进训练，逐步把各组件适配到双任务：

- **Stage 1（统一表示 + 流匹配头预训练）**：冻结 LLM 解码器，只训练表示编码器和流匹配头，用图像描述（image captioning）+ 文生图（T2I）两个目标。captioning 对齐 SigLIP2 这类强语义编码器的预训练目标、给表示注入理解能力；T2I 给流匹配头打底，并让生成梯度回传去塑造统一表示。
- **Stage 2（全模型继续预训练）**：解冻 LLM 解码器，2K 步线性 warm-up 后端到端训练，后期加入图像指令跟随、图像编辑、视频描述数据，扩展能力。
- **Stage 3（监督微调 SFT）**：用图像编辑 + 图像/视频指令跟随 + 高质量图像/视频生成数据，以更小学习率（$2\times10^{-5}$）精修。

数据规模上，图像侧用 1.77 亿自有图文对预训练 + FineVision 1300 万对话样本 + OmniEdit 约 200 万编辑样本 + 约 1000 万高质量 SFT 图文对；视频侧约 1000 万自有视频描述对 + LLaVA-Video-178K 约 160 万指令样本。7B 变体因视频训练成本太高，未用视频数据。

## 实验关键数据

### 主实验

理解侧覆盖 9 个 benchmark，TUNA 在 1.5B/7B 两个规模都基本全面 SOTA，且能压过不少体量更大的复合 UMM：

| 规模 | 模型 | MMStar | RealWorldQA | ChartQA | OCRBench |
|------|------|--------|-------------|---------|----------|
| 1.5B | Show-o2 | 43.4 | 56.5 | 40.0 | 24.5 |
| 1.5B | **TUNA** | **54.6** | **62.5** | **82.1** | **71.9** |
| 7B | Show-o2 | 56.6 | 64.7 | 52.3 | 32.4 |
| 7B | **TUNA** | **61.2** | **66.1** | **85.8** | **74.3** |

生成侧在 GenEval / DPG-Bench / VBench / ImgEdit 全面领先同期方法（含解耦派 BAGEL、Mogao）：

| Benchmark | 任务 | TUNA-1.5B | TUNA-7B | 对照 SOTA |
|-----------|------|-----------|---------|-----------|
| GenEval | 图像生成 | 0.88 | **0.90** | Mogao-7B 0.89 |
| DPG-Bench | 图像生成 | 86.03 | **86.76** | Show-o2-7B 86.14 |
| VBench | 视频生成（1.5B 解码器） | **84.06** | — | Show-o2-1.5B 81.34 |
| ImgEdit | 图像编辑 | — | **4.31** | BAGEL-14B 3.20 |

值得注意：7B 的视频生成实际由 1.5B 解码器完成却拿到 VBench 总分 84.06，超过所有有视频生成能力的 UMM；图像编辑 4.31 甚至逼近 FLUX.1 Kontext（4.00）、Qwen-Image（4.27）这类纯生成模型。

### 消融实验

核心消融（Table 6，1.5B 轻量版，两阶段训练）对比三种视觉表示设计：

| ID | 表示设计 | 训练数据 | MMMU | SEED | GenEval | DPG |
|----|---------|---------|------|------|---------|-----|
| 6 | Decoupled（SigLIP2 + VAE 分开） | Und & Gen | 37.2 | 61.4 | 78.3 | 83.50 |
| 7 | TUNA（SigLIP） | Und & Gen | 36.3 | 64.6 | 76.9 | 83.10 |
| 8 | TUNA（SigLIP2） | Und & Gen | 38.1 | 66.5 | **79.4** | **84.20** |
| 9 | TUNA（DINOv3） | Und & Gen | 37.3 | 65.6 | 78.9 | 84.08 |
| 2 | TUNA（SigLIP2） | Und Only | 37.6 | 62.9 | — | — |
| 4 | TUNA（SigLIP2） | Gen Only | — | — | 77.8 | 83.33 |

### 关键发现

- **统一 > 解耦**：Model 8（TUNA 统一）在所有理解+生成指标上稳超 Model 6（解耦），证明把多套视觉表示塞进一个模型确实有冲突，统一表示才是 UMM 的正解。
- **表示编码器越强越好**：Model 7/8/9 对比，SigLIP2（400M）和 DINOv3（800M）全面压过 SigLIP（400M）；最终选 SigLIP2 是因为它理解相当、生成更优、且模型更小，性价比最高。
- **理解-生成互促（synergy）**：Model 8（双任务训练）在理解上超过 Model 2（只训理解）、在生成上超过 Model 4（只训生成）——联合训练让两个任务互相增益而非互相干扰，这是统一表示才有的红利。
- **为何打败 Show-o2**：CKNNA 对齐分析显示，Show-o2 的 late-fusion 表示与 SD3-Medium（生成参考）对齐度明显低于 TUNA，说明它偏向语义、生成受限；根因是其最终融合特征与理解分支强相关（0.45）、与生成分支几乎不相关，而 TUNA 的级联深度融合两头都对齐得更均衡。

## 亮点与洞察
- **"VAE 之上叠语义编码器"这个级联很巧**：它没有发明新模块，只是把两个现成编码器串起来 + 把 patch embedding 改成 1×1，就绕开了双编码器格式冲突，是典型的"简单但有效"。可迁移性强——任何想统一表示的多模态模型都能借这个思路。
- **用 timestep $t$ 当理解/生成的开关**最让人"啊哈"：$t=1$ 是理解、$t\in[0,1]$ 是生成，同一套表示通路两用，把"扩散"无缝嵌进了原本只做自回归的框架，省去了为生成单建一路的工程负担。
- **CKNNA 对齐分析**是说清"为什么统一比 late-fusion 好"的好工具：不靠玄学解释，而是定量证明 late-fusion 会偏科、深度融合更均衡，这个分析方法可复用到任何表示融合的对比研究。
- **视频窗口注意力**把帧维折进 batch 维这个 trick，让小解码器也能扛视频，对算力受限场景很实用。

## 局限与展望
- **视频规模未对齐**：7B 变体因成本未训练视频数据，视频能力只在 1.5B 上验证，统一表示在大规模视频上的可扩展性还是空白。
- **窗口间无交互**：视频窗口化注意力让每个 4 帧窗口独立编码，跨窗口的长时序依赖可能被削弱，长视频理解/生成上或有隐患。
- **依赖大规模自有数据**：1.77 亿图文 + 千万级视频对多为 in-house，复现门槛高，统一表示的收益有多少来自数据规模 vs 架构本身，文中未充分拆解。
- **流匹配头随机初始化**：生成头从零学，Stage 1 的预训练负担不小；是否能用预训练扩散权重初始化以提速，值得探索。
- **CKNNA 分析只在输入层**：表示对齐分析取的是 LLM 解码器输入层特征，深层解码器如何进一步重塑这套统一表示，没有展开。

## 相关工作与启发
- **vs Show-o2（最直接对手）**：Show-o2 用双路 late-fusion 晚期合并 VAE 与语义特征，TUNA 直接在 VAE 隐变量上跑语义编码器做全层深度融合。区别在融合深度，TUNA 因此表示更均衡、不偏科，所有 benchmark 全面领先。
- **vs BAGEL / Mogao（解耦派）**：它们用 MoE 式架构分别处理两类编码器，参数与成本更高且有格式冲突；TUNA 单一表示空间，更省更优，且 7B 在多数指标上压过它们 14B 的体量。
- **vs Chameleon / Transfusion / Harmon（单编码器统一派）**：它们用单一 VQ-VAE 或 MAR 编码器，天生偏科；TUNA 用 VAE+语义编码器级联，兼顾可重建与语义。
- **vs REPA / RAE（表示对齐启发来源）**：REPA 证明扩散对齐语义编码器有益、RAE 证明冻结语义编码器能重建图像，TUNA 把"语义帮生成"的洞察从生成专用模型推广到了统一多模态框架。

## 评分
- 新颖性: ⭐⭐⭐⭐ 级联式统一表示+timestep 统一双任务的设计简洁而有效，虽未发明全新模块，但思路清晰、解决了 UMM 长期的偏科/冲突难题。
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖理解/图生成/视频生成/编辑四类任务、两种规模、统一vs解耦vs late-fusion 三向消融，外加 CKNNA 定量分析，论证扎实。
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰、消融命名规范，但部分实现细节（如各阶段超参、数据配比）压进附录，正文略需对照。
- 价值: ⭐⭐⭐⭐⭐ 1.5B/7B 全任务 SOTA + 互促结论，为"统一视觉表示该怎么建"给出了可复用的强答案，对 UMM 方向有明确指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Unified Multimodal Models as Auto-Encoders](unified_multimodal_models_as_auto-encoders.md)
- [\[CVPR 2026\] DuetSVG: Unified Multimodal SVG Generation with Internal Visual Guidance](duetsvg_unified_multimodal_svg_generation_with_internal_visual_guidance.md)
- [\[ICCV 2025\] Harmonizing Visual Representations for Unified Multimodal Understanding and Generation](../../ICCV2025/multimodal_vlm/harmonizing_visual_representations_for_unified_multimodal_un.md)
- [\[CVPR 2026\] Unified Personalized Understanding, Generating and Editing](unified_personalized_understanding_generating_and_editing.md)
- [\[CVPR 2026\] Rosetta Stone for Unified MLLMs: A Unified Tokenizer to Decipher Understanding and Generation](rosetta_stone_for_unified_mllms_a_unified_tokenizer_to_decipher_understanding_an.md)

</div>

<!-- RELATED:END -->
