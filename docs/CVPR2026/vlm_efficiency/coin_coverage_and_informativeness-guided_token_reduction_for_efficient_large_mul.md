---
title: >-
  [论文解读] CoIn: Coverage and Informativeness-Guided Token Reduction for Efficient Large Multimodal Models
description: >-
  [CVPR 2026][VLM Efficiency][视觉 token 削减] 把多模态大模型的视觉 token 削减重新建模成"最优子集选择"问题，用**信息量**（视觉显著性 + 跨模态对齐）打每个 token 的分、用**覆盖度**（log-det 体积）保证选出的子集张满特征空间，再用一次贪心子模优化端到端选出紧凑子集——无需训练、不依赖注意力、兼容 FlashAttention/KV cache，在 LLaVA-NeXT-7B 上削掉 94.4% 视觉 token 仍保留 86.7% 性能，prefill 提速 6.5×。
tags:
  - "CVPR 2026"
  - "VLM Efficiency"
  - "视觉 token 削减"
  - "训练免费推理加速"
  - "子集选择"
  - "子模优化"
  - "跨模态对齐"
---

# CoIn: Coverage and Informativeness-Guided Token Reduction for Efficient Large Multimodal Models

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Du_CoIn_Coverage_and_Informativeness-Guided_Token_Reduction_for_Efficient_Large_Multimodal_CVPR_2026_paper.html)  
**代码**: 暂未公开  
**领域**: 模型压缩 / 多模态VLM  
**关键词**: 视觉 token 削减, 训练免费推理加速, 子集选择, 子模优化, 跨模态对齐

## 一句话总结
把多模态大模型的视觉 token 削减重新建模成"最优子集选择"问题，用**信息量**（视觉显著性 + 跨模态对齐）打每个 token 的分、用**覆盖度**（log-det 体积）保证选出的子集张满特征空间，再用一次贪心子模优化端到端选出紧凑子集——无需训练、不依赖注意力、兼容 FlashAttention/KV cache，在 LLaVA-NeXT-7B 上削掉 94.4% 视觉 token 仍保留 86.7% 性能，prefill 提速 6.5×。

## 研究背景与动机
**领域现状**：多模态大模型（LMM）把图像编码成一长串视觉 token，和文本 token 拼起来送进 LLM。但视觉 token 数量随分辨率暴涨——LLaVA-1.5 一张图 576 个 token，LLaVA-NeXT 接近 3K，视频模型更夸张。由于 LLM 推理时延和显存随序列长度平方增长，这串视觉 token 成了部署到手机、交互助手等场景的主要瓶颈。免训练地丢掉冗余视觉 token 是最实用的提速手段。

**现有痛点**：现有 token 削减方法各有硬伤。**重要性派**（PDrop、SparseVLM 等）用注意力权重或 `[CLS]` 相似度打分，保留分数最高的 token——但高注意力区域彼此高度相关，选出来一堆冗余、信息密度低；而且依赖注意力会引入 bias，还和 FlashAttention 这类不暴露注意力矩阵的高效实现冲突，`[CLS]` 打分又把方法绑死在特定视觉编码器上，换个 backbone 就失效；它们还只看单模态信息，完全没用上文本 query。**多样性派**（DivPrune、DART 等）通过聚类或惩罚 token 两两相似度来去冗余，但只关注**局部成对**关系、把所有 token 一视同仁，丢掉了 token 本身的显著性和跨模态相关性，导致关键信息缺失。

**核心矛盾**：重要性和多样性是两个**互补但被分开处理**的目标。近期的混合方法（VisionZip、PruMerge、CDPruner）只是把两者**顺序拼接**——先按重要性打分、再做合并/聚类——并没有解决底层指标各自的固有缺陷，仍然不是全局最优的选择。

**本文目标**：把 token 削减重新表述成一个统一的**最优子集选择**问题，让选出的子集既"每个 token 都重要"、又"整体张满场景"。

**核心 idea**：用两个互补准则联合驱动选择——**informativeness（信息量）**= 视觉显著性 + 跨模态对齐，避开注意力和 `[CLS]`；**coverage（覆盖度）**= 体积准则（log-det），用选中 token 张成的子空间体积衡量全局代表性。两者耦合成一个单目标、用贪心一次性求解。

## 方法详解

### 整体框架
CoIn 是一个**单阶段、免训练**的 token 选择器，插在视觉编码器/projector 输出之后、LLM 之前。给定一张图，视觉编码器 + projector 产出 $N$ 个视觉 token $F_V \in \mathbb{R}^{N\times d}$；文本 query 经 tokenizer 编码后取均值得到一个文本表征 $\bar F_T \in \mathbb{R}^{d}$。CoIn 的任务是从 $N$ 个 token 里选出大小为 $K \ll N$ 的子集 $S$，使得 LLM 在 $[S;T]$ 上的输出尽量逼近在全集 $[V;T]$ 上的输出。

它分两路打分、再合并求解：(1) **信息量估计**给每个 token 算一个标量分 $s_{\text{info}}$，融合视觉显著性（特征范数）和跨模态对齐（与文本的余弦相似度）；(2) **覆盖度选择**用 log-det 体积衡量一个子集在特征空间张成多大子空间。最后把"信息量之和"与"覆盖度"耦合进一个统一目标，用**贪心子模优化**一次性选出 $K$ 个 token，送进 LLM 正常推理。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["视觉 token F_V (N×d)<br/>+ 文本均值 F̄_T"] --> B["信息量估计<br/>显著性 ‖F_V‖_p + 跨模态对齐 cos"]
    A --> C["覆盖度·体积子集选择<br/>Vol(S)=log det(FₛᵀFₛ+λI)"]
    B --> D["耦合信息量与覆盖度<br/>(1−α)Σs_info + α·log det"]
    C --> D
    D -->|贪心子模优化 O(NK)| E["紧凑子集 S (K 个 token)"]
    E --> F["送入 LLM 正常推理"]
```

### 关键设计

**1. 信息量估计：不用注意力，靠"视觉显著性 + 跨模态对齐"给每个 token 打分**

针对重要性派依赖注意力/`[CLS]` 带来的 bias、不兼容 FlashAttention、绑死编码器、且只看单模态这一串痛点，CoIn 用两个正交线索算每个 token 的信息量。第一个是**视觉显著性** $s_{\text{vis}}$：直接用 token 嵌入的 $p$-范数（默认 $p=2$）$s_{\text{vis}}=\lVert F_V\rVert_p$ 近似——激活强度越大，说明这个 token 在视觉场景里越突出，且这个量和文本无关、纯靠特征本身。第二个是**跨模态对齐** $s_{\text{align}}$：算每个视觉 token 和文本均值表征的余弦相似度

$$s_{\text{align}}=\frac{F_V \bar F_T^{\top}}{\lVert F_V\rVert\,\lVert \bar F_T^{\top}\rVert},$$

它把 query 语义（如"鞋子"）注入打分，让和问题相关的 token 得高分。由于两者量纲不同，先各自 min–max 归一化到 $[0,1]$，再凸组合：$s_{\text{info}}=\beta\, s_{\text{vis}} + (1-\beta)\, s_{\text{align}}$，$\beta$ 平衡感知显著性与语义相关性。整套打分不碰注意力矩阵、不依赖 `[CLS]`，因此 attention-efficient、bias-free、model-agnostic。论文可视化显示：显著性高亮视觉上突出的区域（不管问什么），对齐则只点亮 query 相关 token，两者互补，组合后一致优于任一单独使用。

**2. 覆盖度·体积子集选择：用 log-det 体积保证选出的 token 张满整个场景**

只按信息量选会让 top-token 全挤在少数高显著区域、漏掉理解全局所需的线索（空间冗余）。传统多样性方法靠惩罚两两相似度（Max-Min Diversity）来去冗余，但只是局部成对关系，大规模 token 选择时次优。CoIn 改用**体积准则（Volume-based Subset Selection, VSS）**：用选中 token 在特征空间张成的几何体积衡量整体代表性——体积越大，子集越能覆盖原集合的多样方向。对大小为 $K$ 的子集 $S$，体积定义为

$$\text{Vol}(S)=\log\det\!\big(F_S^{\top}F_S + \lambda I\big),$$

其中 $F_S$ 是选中 token 的特征矩阵，$\lambda I$ 是保证数值稳定的小 ridge 项。最大化这个 log-det 会偏好在特征空间张成大体积的子集，从而保住原 token 集的全局结构、覆盖到各个方向，天然鼓励非冗余选择。这正是它优于成对相似度的地方：它衡量的是子集**集体**张成多大空间，而非任意两点像不像。

**3. CoIn 耦合目标 + 贪心子模优化：把信息量与覆盖度合成一个目标一次性求解**

针对混合方法"先重要性、再多样性"两阶段顺序处理、治标不治本的问题，CoIn 把信息量和覆盖度**耦合进同一个目标联合优化**。先把信息量分和 log-det 覆盖项都 min–max 归一化到 $[0,1]$，统一目标为

$$S^{*}=\arg\max_{S\subseteq V,\,|S|=K}\Big[(1-\alpha)\sum_{i\in S}s_{\text{info},i} + \alpha\,\log\det\!\big(F_S^{\top}F_S\big)\Big],$$

$\alpha$ 调节"保信息量"与"促多样"的权重。直接求解是组合爆炸的，但这个目标是**单调子模（submodular & monotone）**的——这保证了一个简单的贪心过程就能拿到近最优解。实现上，行列式项用 QR 分解近似、配合**增量 Gram–Schmidt** 避免每步重算整个目标，把总复杂度降到 $O(NK)$，在 $K \ll N$ 时几乎可忽略。正是"耦合成单目标 + 子模性 + 贪心"这套组合，让 CoIn 既是全局一致的选择、又快到能当推理加速器用。⚠️ 完整推导与算法在补充材料，正文未给出贪心每步的具体增量公式。

## 实验关键数据

### 主实验
在 LLaVA-1.5-13B、LLaVA-NeXT-7B、Qwen2.5-VL-7B、LLaVA-OneVision-7B（视频）上评测，覆盖 GQA/MMBench/MME/POPE/TextVQA/VizWiz/OCRBench/SQA/RealWorldQA 等 9 个图像基准 + 3 个视频基准。"Avg." 为相对原模型的平均性能。

LLaVA-NeXT-7B（高分辨率，最多 2880 token）下不同保留预算的对比：

| 保留 token | 削减率 | 本文 CoIn | 次优基线 | 提升 |
|------------|--------|-----------|----------|------|
| 640 | ↓77.8% | **94.0%** | VisionZip 92.7% | +1.3% |
| 320 | ↓88.9% | **90.0%** | VisionZip 88.6% | +1.4% |
| 160 | ↓94.4% | **86.7%** | VisionZip 83.5% | +3.2% |

LLaVA-1.5-13B 在极端 94.4% 削减（32 token）下保留 91.0%（次优 DivPrune 88.6%）；Qwen2.5-VL-7B 保留 256 token（↓80.2%）拿到 96.2%、128 token（↓90.1%）拿到 92.0%，均为最佳。视频任务 LLaVA-OneVision-7B 在 75% 削减下达 99.6%（几乎无损）。

### 效率分析
LLaVA-NeXT-7B + POPE，单张 A100 80GB，把 2880 token 削到 320：

| 方法 | 显存(GB) | Prefill | 加速 | Decoding | 加速 | F1 |
|------|---------|---------|------|----------|------|-----|
| Original | 16.8 | 233ms | 1.0× | 27ms | 1.0× | 86.5 |
| PDrop | 15.5 | 54ms | 4.3× | 23ms | 1.2× | 60.2 |
| VisionZip | 14.9 | 36ms | 6.5× | 21ms | 1.3× | 80.1 |
| DivPrune | 13.9 | 36ms | 6.5× | 21ms | 1.3× | 83.4 |
| **CoIn** | 14.1 | 36ms | **6.5×** | 21ms | **1.3×** | **85.4** |

同样 6.5× prefill 加速下，CoIn 的 F1（85.4）远高于同档加速的 VisionZip（80.1）和 DivPrune（83.4），说明加速不是靠牺牲质量换来的。

### 消融实验
信息量 vs 覆盖度（LLaVA-1.5-7B，保留 64 token）：

| 配置 | POPE | GQA | MME | 说明 |
|------|------|-----|-----|------|
| Original | 85.9 | 62.0 | 1508 | 全 token |
| Info-only | 73.7 | 51.2 | 1288 | 只信息量，token 冗余掉点最多 |
| Cov-only | 74.4 | 53.0 | 1307 | 只覆盖度，保多样但漏信息 |
| **Combination** | **86.2** | **57.8** | 1378 | 完整 CoIn，POPE 甚至超过原模型 |

信息量内部再拆（保留 32 token）：

| 配置 | VizWiz | RealWorldQA | 说明 |
|------|--------|-------------|------|
| IS-only（显著性） | 51.4 | 39.5 | RealWorldQA 上明显弱 |
| CA-only（跨模态对齐） | 51.2 | 41.1 | 语义 grounding 需求强的任务更好 |
| **Combination** | **51.8** | **42.6** | 两者互补，组合最佳 |

### 关键发现
- **两个准则缺一不可且互补**：Info-only 和 Cov-only 都比 Combination 差一大截，Info-only 因冗余掉点最严重；完整 CoIn 在 POPE 上 86.2 反超原模型 85.9——说明紧凑、选得好的子集甚至能改善 grounding 同时省成本。
- **显著性 vs 跨模态对齐分工明确**：感知类任务（VizWiz）两者相近；需要强语言 grounding 的任务（RealWorldQA）跨模态对齐明显更重要，但组合始终最优。
- **削减越激进，CoIn 优势越大**：160 token（94.4% 削减）下对 VisionZip 领先 3.2%，远大于 640 token 时的 +1.3%，说明全局覆盖建模在极端压缩下更关键。
- **超参不敏感**：$\alpha$（信息量↔覆盖度）、$\beta$（显著性↔对齐）在 Qwen 上扫描显示性能不太敏感，简单取 $\alpha\geq 0.5$ 已有强结果。

## 亮点与洞察
- **用 log-det 体积做覆盖度，是从"成对相似度"到"集体子空间"的视角升级**：传统多样性只管两两不像，CoIn 直接最大化子集张成的体积，全局代表性一步到位；且这个目标天然子模，贪心 $O(NK)$ 就近最优——理论漂亮、工程便宜。
- **彻底绕开注意力是它最实用的卖点**：显著性用范数、相关性用余弦，全程不碰注意力矩阵，因此和 FlashAttention/KV cache 完全兼容、换 backbone 不失效——这正是重要性派落地时最头疼的两个坑。
- **"紧凑子集反而超过全 token"很反直觉**：POPE 上 86.2 > 85.9，提示丢掉冗余/干扰 token 能净化 grounding，token 削减不只是省钱、也可能去噪。
- **可迁移性**：信息量 = 显著性 × 跨模态对齐这套打分，几乎可以套到任何"从一堆候选里选代表"的多模态场景（如检索、记忆压缩）；log-det 覆盖项也能即插即用地替换各种基于相似度的去冗余模块。

## 局限与展望
- 作者把贪心每步的增量公式、子模性证明和完整算法都放进了补充材料，正文只给了 $O(NK)$ 复杂度结论，⚠️ 复现细节（如 QR/Gram–Schmidt 的具体更新）需查补充。
- 不同任务对 $\alpha,\beta$ 有轻微偏好，要榨干性能仍需网格搜索（步长 0.1），虽然作者说不太敏感，但跨任务/跨模型的"一组通用超参"并未给出。
- 评测的保留预算（token budget）是预先指定的固定值，论文没有探讨如何**自适应**地按图像内容/query 难度决定该留多少 token——简单图和复杂图用同一预算可能并非最优。
- 体积准则需要算 $F_S^\top F_S$ 的行列式，当特征维度 $d$ 很高、子集较大时，QR 近似的数值稳定性和 $\lambda$ 的选取对结果的影响正文未充分分析。

## 相关工作与启发
- **vs 重要性派（PDrop / SparseVLM）**：它们用注意力/`[CLS]` 打分、只看单模态，选出冗余且绑死编码器、不兼容 FlashAttention；CoIn 用范数+余弦替代注意力、加入文本对齐，model-agnostic 且兼容高效推理。同档加速下 F1 大幅领先（85.4 vs PDrop 60.2）。
- **vs 多样性派（DivPrune / DART）**：它们最大化两两不相似度、忽略 token 显著性，丢关键信息；CoIn 用 log-det 体积做全局覆盖、且联合信息量，既多样又不丢重点。
- **vs 混合派（VisionZip / PruMerge / CDPruner）**：它们"先重要性再多样性"两阶段顺序拼接，治标不治本；CoIn 把两准则耦合成单个子模目标一次性联合优化，是全局一致的选择而非启发式拼接。
- **vs 视频专用（DyCoke / FrameFusion）**：DyCoke 有固定的合并后 token 下界、FrameFusion 逐层渐进削减；CoIn 作为通用 token 选择器在视频基准上同样领先（OneVision 75% 削减下 99.6%）。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把 token 削减建模成 informativeness+coverage 的子模子集选择、用 log-det 体积做覆盖度，视角清晰且自洽
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个 backbone × 多档削减率 × 图像/视频 12 个基准 + 效率/消融/超参全覆盖
- 写作质量: ⭐⭐⭐⭐ 动机递进清楚、图示直观，但贪心算法细节外放补充略影响自洽
- 价值: ⭐⭐⭐⭐⭐ 免训练、兼容 FlashAttention/KV cache、6.5× prefill 加速且几乎无损，落地价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SCoRe: Salience-Coverage Reduction for Vision Token Pruning in Vision-Language Models](score_salience-coverage_reduction_for_vision_token_pruning_in_vision-language_mo.md)
- [\[CVPR 2026\] Rethinking Token Reduction for Large Vision-Language Models](rethinking_token_reduction_for_large_vision-language_models.md)
- [\[CVPR 2026\] EvoComp: Learning Visual Token Compression for Multimodal Large Language Models via Semantic-Guided Evolutionary Labeling](evocomp_learning_visual_token_compression_for_multimodal_large_language_models_v.md)
- [\[CVPR 2026\] OmniZip: Audio-Guided Dynamic Token Compression for Fast Omnimodal Large Language Models](omnizip_audio-guided_dynamic_token_compression_for_fast_omnimodal_large_language.md)
- [\[CVPR 2026\] IF-Prune: Information-Flow Guided Token Pruning for Efficient Vision-Language Models](if-prune_information-flow_guided_token_pruning_for_efficient_vision-language_mod.md)

</div>

<!-- RELATED:END -->
