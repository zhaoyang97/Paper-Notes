---
title: >-
  [论文解读] LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning
description: >-
  [CVPR 2026][多模态VLM][扩散语言模型] 针对"当前多模态大模型几乎全是自回归范式、扩散路线尚未被验证"的空白，本文把视觉指令微调嫁接到掩码扩散语言模型 LLaDA 上，做出纯扩散的多模态大模型 LLaDA-V——靠双向注意力更好地捕捉视觉空间关系，在 18 个基准上不仅刷新纯扩散 MLLM 的 SOTA，还在相同训练数据下于 11 个任务上超过自回归基线 LLaMA3-V。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "扩散语言模型"
  - "多模态大模型"
  - "视觉指令微调"
  - "掩码扩散"
  - "双向注意力"
---

# LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/You_LLaDA-V_Large_Language_Diffusion_Models_with_Visual_Instruction_Tuning_CVPR_2026_paper.html)  
**代码**: https://github.com/ML-GSAI/LLaDA-V  
**领域**: 多模态VLM / 扩散模型  
**关键词**: 扩散语言模型, 多模态大模型, 视觉指令微调, 掩码扩散, 双向注意力

## 一句话总结
针对"当前多模态大模型几乎全是自回归范式、扩散路线尚未被验证"的空白，本文把视觉指令微调嫁接到掩码扩散语言模型 LLaDA 上，做出纯扩散的多模态大模型 LLaDA-V——靠双向注意力更好地捕捉视觉空间关系，在 18 个基准上不仅刷新纯扩散 MLLM 的 SOTA，还在相同训练数据下于 11 个任务上超过自回归基线 LLaMA3-V。

## 研究背景与动机
**领域现状**：多模态大模型（MLLM）能把图像/音频/视频和文本一起处理并生成自然语言回答，但现有方法压倒性地依赖**自回归（AR）**语言模型（LLaVA、Qwen2-VL 等都是 next-token 预测），对"换一种概率建模方式"探索极少。近期把扩散塞进 MLLM 的尝试要么仍靠 AR 提供语言能力（混合路线），要么用语言建模能力很弱的离散扩散，效果都不理想。

**现有痛点**：纯扩散路线一直没被证明能打。直到 LLaDA 把掩码扩散语言模型 scale 到 8B、在大量下游任务上能与 LLaMA3-8B-Instruct 掰手腕，扩散语言模型才第一次有了"能当 LLM 主干"的底气；但 LLaDA 在多模态理解上的潜力**完全没被探索过**。于是核心研究问题是：一个训练和采样都纯扩散的 MLLM，能不能做到和自回归模型competitive？

**核心矛盾**：作者押注的一个判断是——**双向注意力天然更适合多模态理解**。AR 的因果（从左到右）注意力契合顺序文本生成，但视觉输入是有空间关系和上下文依赖的，更需要"同时看所有位置"。扩散语言模型本就用双向注意力，理应能更统一地建模图文输入、带来更强的多模态理解。问题是怎么把 LLaDA 的单模态掩码扩散目标扩展到"多轮、含图像"的多模态对话上。

**本文目标**：(1) 给掩码扩散语言模型设计一个能处理多轮多模态对话的训练目标；(2) 想清楚多模态场景下用因果还是双向注意力；(3) 适配多模态对话的推理流程；(4) 设计多阶段训练策略把语言-视觉对齐、指令跟随、多模态推理逐步建起来。

**切入角度**：不发明新架构，而是把经过验证、在各种 AR-MLLM 上都很有效的**视觉指令微调框架（视觉塔 + MLP 连接器 + 语言塔）**原样搬过来，只把语言塔从 AR 换成扩散的 LLaDA，由此干净地隔离出"扩散 vs 自回归"这一个变量。

**核心 idea**：用掩码扩散替代 next-token 预测做多模态指令微调——图像特征和 prompt 保持不掩码、只随机掩码 response，模型学着在干净图文条件下还原被掩 token；靠 LLaDA 的全局双向注意力更好地捕捉视觉空间依赖。

## 方法详解

### 整体框架
LLaDA-V 沿用经典视觉指令微调的三件套：**视觉塔**（SigLIP 2，siglip2-so400m-patch14-384）+ **MLP 连接器**（两层 MLP，把视觉特征投到 LLaDA 词嵌入空间）+ **语言塔**（LLaDA-8B-Instruct，一个掩码扩散语言模型）。和 AR-MLLM 的根本区别在生成方式：AR 给定图文条件用 next-token 逐个预测 response；LLaDA-V 则把 response 当成被掩码的序列、用掩码扩散的反向过程迭代还原（见论文 Fig. 2）。整条管线分三步理解——训练时只掩 response、图文不掩，学一个预测被掩 token 的目标；推理时从全掩码 response 出发、随时间步从 1 降到 0 迭代揭开 token；训练流程则按"语言-图像对齐 → 视觉指令微调 → 多模态推理增强"三阶段递进。注意力机制上作者实测**双向（no mask）优于因果**，故全程采用双向注意力。

### 关键设计

**1. 多轮多模态对话的掩码扩散训练目标：只掩 response、用扩散损失替代 next-token**

这是把 LLaDA 从纯文本搬到多模态的核心改动，针对"扩散语言模型没有现成的多模态指令微调目标"这一空白。以单图两轮对话 $(v,p^1_0,r^1_0,p^2_0,r^2_0)$ 为例（$v$ 是视觉塔+投影器给出的图像表征，$p$ 是 prompt、$r$ 是 ground-truth response），训练时图像特征 $v$ 和所有 prompt 保持干净不掩，只对 response 按扩散调度随机掩成 $[M]$ token，目标是最大化对被掩 token 的预测对数似然：

$$L(\omega) = -\mathbb{E}\Big[\tfrac{1}{t}\sum_{i}\sum_{j}\mathbf{1}[r^{1,i}_t=[M]\ \text{或}\ r^{2,j}_t=[M]]\cdot\log p_\omega(r^{1,i}_0,r^{2,j}_0\mid v,p^1_0,r^1_t,p^2_0,r^2_t)\Big]$$

其中 $r^1_t,r^2_t$ 是按时间 $t$ 掩码后的 response。该目标被证明是被掩 token 负对数似然的上界，直觉上就是"给定干净图文，去补 response 里被挖掉的 token"——这样就把掩码扩散与视觉指令微调框架对接，让扩散语言模型获得多模态理解能力。

**2. 双向（no-mask）注意力：让模型看全整段对话上下文，更契合视觉空间依赖**

针对"多模态理解到底用因果还是双向注意力"这个关键选择。从系统角度看，因果（对话内早轮看不到晚轮）很诱人——多轮推理时前面轮次的 KV 状态可复用、解码更省。但作者认为双向注意力能在掩码预测时综合理解整段对话上下文（这也正是近期视频扩散模型用双向注意力提升时序一致性的原因）。消融（Tab. 3，12 个基准）显示 no-mask 在 7/12 个基准上更好、平均分（除 MME）49.73 > 因果的 49.03，于是 LLaDA-V 采用双向注意力。进一步的注意力模式分析印证：LLaDA-V 表现出更全局、双向的注意力行为，而 LLaMA3-V 是局部、严格因果的——这种结构让 LLaDA-V 更能捕捉复杂空间依赖，可能正是它在视觉-语言任务上反超的原因。配合 Fast-dLLM 的近似缓存复用，双向带来的效率劣势也被很大程度抹平。

**3. 三阶段训练策略：对齐 → 指令微调 → 推理增强，逐级建能力**

针对"如何把一个纯语言扩散模型一步步养成强多模态模型"。前两阶段沿用 LLaVA-NeXT 等成熟做法，第三阶段是本文额外加的推理增强：

- **Stage 1 语言-图像对齐**：冻结语言塔和视觉塔，只训 MLP 投影器，用 LLaVA-Pretrain 把视觉表征对齐到 LLaDA 词嵌入空间。
- **Stage 2 视觉指令微调**：解冻全模型在大规模指令数据 MAmmoTH-VL 上训练，分两阶段——先用 SI-10M（1000 万单图样本）建单图理解，再用 OV-2M（约 200 万单图/多图/视频混合样本）扩展到多图与视频。
- **Stage 3 多模态推理增强**：先用 VisualWebInstruct（90 万带详细推理链的 QA）做推理训练；但发现模型变得"凡答必先推理"，于是再做**平衡推理训练**——借鉴 Qwen3 的混合思考机制，把 VisualWebInstruct 和 OV-2M 混训，给 OV-2M 的 prompt 加 `/no think`、给 50% 推理数据加 `/think`，让模型既能直接答也能展开推理。

**4. 掩码扩散反向采样推理 + Fast-dLLM 加速：迭代揭开 response，并提供精度-吞吐可调旋钮**

针对"扩散 MLLM 怎么生成、效率能否追上 AR"。推理时给定新 prompt，模型把整段 response 初始化为全 $[M]$ token，沿掩码扩散反向过程从状态 $r_t$ 迭代到 $r_s$（$s<t$、掩码水平递减）：每步先在 $v,p_0,r_t$ 条件下预测所有 $[M]$ token，再把其中比例 $s/t$ 的预测重新掩回 $[M]$、其余 $(1-s/t)$ 保留。remask 策略采用 LLaDA 的**低置信度策略**——优先重掩低置信预测、保住高置信的。由于扩散不能用 KV cache、且一步多解码常掉点，作者把 Fast-dLLM 的近似 KV cache 适配进来：它带一个可配置的缓存刷新间隔 $r$，间隔越大吞吐越高而精度损失很小（MathVista 上 $r$ 从 2 增到 48 提速 3.3× 仅掉 2.3% 精度），等于给部署提供了一个"精度优先调小、时延优先调大"的旋钮。在 MathVerse 上 LLaDA-V（用 Fast-dLLM）达到 32.4 tokens/s、精度 28.5%，与 AR 基线 LLaMA3-V（30.5 tokens/s、29.0%）相当。

## 实验关键数据

语言塔为 LLaDA-8B-Instruct，视觉塔 SigLIP 2，投影器随机初始化两层 MLP。为公平对比，AR 基线 **LLaMA3-V** 把语言塔换成 LLaMA3-8B-Instruct、其余完全一致、训练协议相同（唯一变量是语言塔）。共评测 18 个基准。

### 主实验：与自回归基线及各类 MLLM 对比（部分）

| 模型 | 类型 | 语言塔 | MMMU(val) | MMMU-Pro(std) | MMStar | MMB(en-dev) | MuirBench | MLVU | VideoMME |
|------|------|------|------|------|------|------|------|------|------|
| Qwen2-VL | AR | Qwen2-7B | 54.1 | 43.5 | **60.7** | - | - | - | - |
| LLaMA3-V（AR 基线） | AR | LLaMA3-8B | 45.4 | 28.3 | 56.5 | 79.8 | 47.4 | 57.5 | 55.8 |
| LaViDa-L | Diff. | LLaDA-8B | 43.3 | - | - | 70.5 | - | - | - |
| Dimple | Diff. | Dream-7B | 45.2 | - | - | 74.6 | - | - | - |
| MMaDA | Diff. | LLaDA-8B | 30.2 | - | - | 68.5 | - | - | - |
| **LLaDA-V（本文）** | Diff. | LLaDA-8B | **48.6** | **35.2** | 60.1 | **82.9** | **48.3** | **59.5** | **56.1** |

LLaDA-V 在所有纯扩散/混合扩散 MLLM 中刷新 SOTA；尽管语言塔比 LLaMA3-8B 略弱，仍在 9 个知识/数学基准中的 6 个上反超 LLaMA3-V，且在多图（MuirBench）、视频（MLVU、VideoMME）上全面领先；MMStar 上 60.1 已逼近强 AR 模型 Qwen2-VL 的 60.7。短板在图表/文档（AI2D、DocVQA）和真实场景（RealworldQA）——这些更吃语言塔本身的强度。

### 消融实验：注意力掩码策略（12 基准节选）

| 配置 | MMMU(val) | MMStar | MMB(en-dev) | MuirBench | 平均(除 MME) |
|------|------|------|------|------|------|
| 对话因果掩码 | 42.89 | 49.60 | 75.42 | 28.69 | 49.03 |
| **No Mask（双向，本文）** | **44.67** | **49.79** | **76.71** | **33.88** | **49.73** |

### 关键发现
- **双向注意力是反超的关键机制**：no-mask 在 7/12 基准更好、平均 +0.70，尤其 MuirBench（多图）从 28.69 跳到 33.88——多图/空间任务最吃"同时看所有位置"的能力。
- **语言塔弱却能赢，说明架构本身有红利**：LLaDA-V 语言塔（LLaDA-8B）弱于 LLaMA3-8B，却在 11 个任务上超过同训练配置的 LLaMA3-V，注意力分析显示其更全局/双向的注意力更善捕捉空间依赖。
- **数据可扩展性好**：随指令数据增加 LLaDA-V 持续涨点，在 MMMU/MMMU-Pro 等知识基准上 scaling 甚至优于 LLaMA3-V（MMMU-Pro 上仅 1M 样本就追平 LLaMA3-V 用更多数据的结果）。
- **效率不拖后腿**：靠 Fast-dLLM，MathVerse 上 32.4 vs 30.5 tokens/s、精度 28.5% vs 29.0%，与 AR 基线相当，并能用刷新间隔 $r$ 灵活换吞吐。
- **短板诚实**：图表/文档、真实场景理解仍逊于 LLaMA3-V，且整体不及 Qwen2-VL，主因是 LLaDA-8B 语言主干偏弱、缺偏好对齐。

## 亮点与洞察
- **"只换语言塔、其余全锁死"是干净的对照实验**：LLaMA3-V 与 LLaDA-V 共享视觉塔、投影器、数据、训练协议，唯一差别是 AR vs 扩散语言塔——这让"扩散在多模态上是否有红利"的结论格外可信，是方法学上的亮点。
- **把"双向注意力适合视觉"从直觉做成了可验证的结论**：消融 + 注意力模式分析双重印证，且类比到视频扩散用双向注意力提升时序一致性，给"为什么扩散 MLLM 在空间/多图任务上更强"提供了机制解释。
- **可迁移的范式判断**：在 response 上做掩码扩散、图文条件保持干净，这套目标可直接套用到任何"想把扩散语言模型扩到新模态指令微调"的场景；Fast-dLLM 的可调刷新间隔也是部署侧实用 trick。
- 最"啊哈"的点：一个**更弱的语言塔**配上**更合适的注意力结构**，能在多模态任务上反超更强语言塔的 AR 模型——提示 MLLM 性能不只由语言塔强度决定，建模范式与注意力结构同样关键。

## 局限与展望
- **受限于语言塔**：LLaDA-8B 缺偏好对齐、整体弱于 Qwen2-7B，导致 LLaDA-V 在多数基准上仍不及 Qwen2-VL，图表/文档/真实场景尤其吃亏；天花板被语言塔锁住，需等更强的扩散语言塔出现。
- **效率结构性短板**：扩散语言模型无法原生用 KV cache、一步多解码会掉点，虽靠 Fast-dLLM 的近似缓存缓解，但本质上是用近似换速度，长序列下的精度-吞吐权衡仍需逐场景调。
- **非新架构、偏探索性**：作者明确表示贡献是"验证可行性 + 洞察"，而非提出新架构；很多结论（如双向更优）是经验性的、平均仅 +0.70，跨基准并不一致。
- **真实场景/文档理解偏弱**：这类任务对细粒度 OCR 和语言推理要求高，当前掩码扩散 + 弱语言塔组合还补不上。

## 相关工作与启发
- **vs LLaDA（纯文本扩散语言模型）**: 本文把 LLaDA 从单模态扩到多模态——加视觉塔 + MLP 投影器、改训练目标支持多轮多模态对话；有趣的是 LLaDA 语言能力略逊 LLaMA3-8B，LLaDA-V 却在更多任务上超 LLaMA3 基线，暗示扩散框架对多模态有额外红利。
- **vs LLaMA3-V（自回归基线）**: 同一套视觉塔/投影器/数据/训练协议，只把语言塔换成 AR——LLaDA-V 靠双向注意力在 11 个任务上反超，尤其多图/视频；但图表/真实场景逊于它。
- **vs 其他纯扩散 MLLM（D-DiT / LaViDa / Dimple / MMaDA）**: 这些同期工作也探索离散扩散做多模态理解，但 LLaDA-V 在它们之上刷新 SOTA，且强调数据 scaling 行为、与 AR 的受控对比、注意力模式分析这三点系统性证据。
- **vs 混合 AR-扩散（MetaMorph / Show-o / JanusFlow / Orthus）**: 混合路线仍靠 AR 提供语言能力；本文坚持**训练和采样全扩散**，证明纯扩散路线也能 competitive，是范式层面的差异。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个训练+采样全扩散的 MLLM，但作者自陈非新架构、偏验证可行性
- 实验充分度: ⭐⭐⭐⭐⭐ 18 基准 + 受控 AR 对照 + 数据 scaling + 注意力消融与模式分析，扎实全面
- 写作质量: ⭐⭐⭐⭐⭐ 研究问题与对照设计清晰、对短板诚实交代
- 价值: ⭐⭐⭐⭐ 验证扩散是 AR-MLLM 的可扩展替代路线并开源，但当前性能受弱语言塔与效率短板限制

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)
- [\[ICML 2025\] Parrot: Multilingual Visual Instruction Tuning](../../ICML2025/multimodal_vlm/parrot_multilingual_visual_instruction_tuning.md)
- [\[CVPR 2026\] DEVA: Fine-tuning Multimodal Large Language Models for Visual Perception Tasks](deva_fine-tuning_multimodal_large_language_models_for_visual_perception_tasks.md)
- [\[CVPR 2026\] Streaming Video Instruction Tuning (Streamo)](streaming_video_instruction_tuning.md)
- [\[CVPR 2026\] Harmonious Parameter Adaptation in Continual Visual Instruction Tuning for Safety-Aligned MLLMs](harmonious_parameter_adaptation_in_continual_visual_instruction_tuning_for_safet.md)

</div>

<!-- RELATED:END -->
