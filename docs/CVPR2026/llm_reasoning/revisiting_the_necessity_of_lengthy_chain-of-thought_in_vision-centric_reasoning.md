---
title: >-
  [论文解读] Revisiting the Necessity of Lengthy Chain-of-Thought in Vision-centric Reasoning Generalization
description: >-
  [CVPR 2026][Reasoning][视觉链式推理] 作者用可控的迷宫导航任务系统对比了语言 CoT、grounding CoT、视觉 CoT 三类「think with image」式监督格式，发现更长/更花哨的视觉 CoT 只能加快收敛、抬不高最终天花板，而**只保留最少 grounding 信息的极简 CoT（一条坐标路径）反而泛化最好**，提出「short is long」效应并给出构造可泛化视觉推理 SFT 数据的实操指南。
tags:
  - "CVPR 2026"
  - "Reasoning"
  - "视觉链式推理"
  - "视觉 CoT"
  - "泛化"
  - "SFT-then-RL"
  - "迷宫导航"
---

# Revisiting the Necessity of Lengthy Chain-of-Thought in Vision-centric Reasoning Generalization

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Du_Revisiting_the_Necessity_of_Lengthy_Chain-of-Thought_in_Vision-centric_Reasoning_Generalization_CVPR_2026_paper.html)  
**代码**: https://github.com/RUCAIBox/Revisiting-Visual-CoT  
**领域**: LLM推理 / 多模态VLM  
**关键词**: 视觉链式推理, 视觉 CoT, 泛化, SFT-then-RL, 迷宫导航

## 一句话总结
作者用可控的迷宫导航任务系统对比了语言 CoT、grounding CoT、视觉 CoT 三类「think with image」式监督格式，发现更长/更花哨的视觉 CoT 只能加快收敛、抬不高最终天花板，而**只保留最少 grounding 信息的极简 CoT（一条坐标路径）反而泛化最好**，提出「short is long」效应并给出构造可泛化视觉推理 SFT 数据的实操指南。

## 研究背景与动机
**领域现状**：视觉推理正成为视觉语言模型（VLM）的关键能力，业界普遍用 CoT 数据做监督微调来教模型「想清楚再答」。主流认知是「越长越好」——更长的 CoT 带来多步推演和自我反思，而 o3 式的视觉 CoT（在图上裁剪、画线、标注，再把改过的图喂回去）被认为更贴近人类视觉认知，能进一步提升各类视觉推理 benchmark。

**现有痛点**：这些结论大多是在容易被预训练先验和数据污染干扰的真实 benchmark 上得到的，**到底是哪种 CoT 设计在起作用、为什么起作用、哪种才真正支撑「可泛化」的推理，从来没人讲清楚**。语言、空间坐标、图像操作这三种把中间推理「外化」出来的方式机制完全不同，但被笼统当成「加 CoT 就涨点」。

**核心矛盾**：「让监督信号更丰富（更长、带图像操作）」与「让模型学到可迁移的抽象规则」之间未必正相关——丰富的轨迹可能只是帮模型更快拟合到某个特定布局，而非内化出尺度无关的导航规律。

**本文目标**：在一个干净、可控、难度可调的环境里，把语言 / grounding / 视觉三类 CoT 拆开，分别回答：① 它们各自带来什么收益？② 视觉中心任务里 CoT 到底靠什么能力工作？③ 谁的泛化最好？

**切入角度**：选迷宫（maze）做测试床。理由是它的推理规则完全由视觉输入表达、难度可由网格大小平滑调节（4×4 到 10×10）、当前 VLM 在上面表现极差（Qwen2.5-VL-7B 在 4×4 上成功率 <10%，不会被预训练能力饱和掩盖）、且解和中间步骤都能用规则函数自动合成与过滤，天然避开数据污染。

**核心 idea**：在统一的 SFT-then-RL 流程下公平比较四种 CoT 格式，用「能不能跨迷宫尺寸泛化」而非「训练集成功率」来判优，最终发现**剥到最少的 grounding 信息反而最利于泛化**。

## 方法详解

### 整体框架
这其实是一篇**机制分析 / 数据构造研究**，而非提出一个新模型——它的「方法」是一套严格控制变量的实验协议：以 Qwen2.5-VL-7B 为统一底座，对四种 CoT 格式各自合成 8K 条冷启动轨迹做 SFT、再在迷宫数据上用 RL（GRPO）训到收敛（最多 1000 步），然后在**未见过的更大迷宫**上测泛化。整条链路是「迷宫规则化合成 → 四类 CoT 轨迹格式化 → 各格式独立 SFT 得策略模型 → RLVR 强化 → 跨尺寸泛化评测」。四种格式只在「中间推理怎么外化」这一处变量上不同，其余完全对齐，从而把「格式」对学习与泛化的影响干净地隔离出来。

输入是一张 N×N 迷宫图 $I$ 加指令 $Q$（要求输出从起点 S 到终点 E、不穿墙的坐标路径，最终路径放进 `\boxed{}`）；输出是模型生成的推理过程 `<think>…</think>` 加路径。墙定义在相邻格之间而非占据格子，路径需满足相邻两格之间无墙 $w_{(i_k,j_k)\to(i_{k+1},j_{k+1})}=0$。

### 关键设计

**1. 四种 CoT 格式：把「推理外化方式」做成唯一变量**

这是全文的实验骨架，针对的痛点是「以往把不同 CoT 混为一谈、说不清谁在起作用」。四种格式从「啰嗦」到「极简」排开：

- **语言 CoT（L-CoT）**：纯文本，用「north/south/west/east」描述每一步，轨迹 $R^{lang}_T=r^{(l)}_1,\dots,r^{(l)}_T$，$r^{(l)}_t\in V_{text}$。先用规则函数把路径转成方向序列，再让 Gemini-2.5-Pro 合成自然语言推理。
- **grounding CoT（G-CoT）**：每步把语言引用显式绑定到图上的空间坐标，元素表示为 $g_k=(G_k,C_k)$，$G_k\in\{point,line,region\}$；合成时还**注入反思模式**（故意造撞墙/死胡同的错误路径 + 纠错推理）来加深推理。
- **视觉 CoT（V-CoT）**：在 grounding 基础上允许「动手改图」——用画线操作 $I_{t+1}=\phi_t(I_t,g_t)$ 把当前部分路径画到图上，再把更新后的图喂回模型，形成图文交错推理。
- **G-CoT-least（极简 grounding）**：直接把最终路径坐标序列当作答案，不写额外文字解释也不写绝对坐标——因为迷宫任务的目标输出本身就是一串访问过的格点，**推理已隐式嵌在路径里**。这是「grounding 信息最少」的极端。

把四者放进同一 SFT-then-RL 管线，就能问出「外化越多是否越好」。

**2. SFT-then-RL 训练协议：先冷启动塑形、再用可验证奖励强化**

针对的痛点是「当前 VLM 连像样的迷宫思路都生成不出来，直接 RL 会崩」。流程分两段：SFT 阶段把合成推理包进 `<think></think>`、答案包进 `\boxed{}`，每种格式各 8K 条；视觉 CoT 是图文交错数据，交叉熵只在文本 token 上算。RL 阶段额外合成 20K 迷宫样本，用 GRPO 优化，奖励为

$$r=\alpha\cdot r_{acc}+(1-\alpha)\cdot r_{format},\quad \alpha=0.9$$

其中 $r_{acc}$ 由规则函数判定预测路径是否连通起终点且不穿墙，$r_{format}$ 约束输出格式。关键的方法学贡献在于**训到真收敛**：以往视觉 RL 工作常只训几十到几百步，模型欠训练、天花板看不清；本文一律训到 1000 步、性能收敛，才能公平比较各格式的「最终上限」而非「早期速度」。SFT 时冻结视觉编码器，RL 时解冻。

**3. 用「跨尺寸泛化」而非「训练成功率」判优：揭示 short is long**

这是结论得以成立的判据设计。痛点是「训练集都能刷到 100%，看不出谁真学到规律」。作者改看两类泛化：**单尺度泛化**（只在 6×6 上 SFT+RL，测未见的 7×7）和**跨尺度泛化**（4×4–6×6 上 SFT、7×7–9×9 上 RL，测未见的 10×10）。结果是 G-CoT-least 在两种设置下都稳健保持高成功率，而 V-CoT 约 800 步后饱和、始终落后。机制解释：极简 grounding 迫使模型内化**尺度无关的局部导航规则**（沿走廊走、遇死胡同回溯），而视觉 CoT 容易过拟合到具体视觉布局和操作模式。由此得出「short is long」——简洁但 grounding 良好的监督，比啰嗦重监督更能学到可复用的推理模式。

### 损失函数 / 训练策略
SFT 用标准交叉熵（V-CoT 仅对文本 token 计损）；RL 用 GRPO，奖励见上式（$\alpha=0.9$）。SFT 三个 epoch、学习率 $1\times10^{-5}$、warm-up 比例 0.1、batch 64；RL rollout batch 128、mini-batch 32、每样本 8 次 rollout，训到收敛（≤1000 步）。

## 实验关键数据

### 主实验
核心结论来自迷宫上的训练动态（图 2–5，文中以曲线呈现）与跨任务验证（表 1）。迷宫上的三条关键观察：

| 观察维度 | L-CoT | G-CoT | V-CoT | G-CoT-least |
|---------|-------|-------|-------|-------------|
| RL 收敛速度 | 最慢 | 中等 | 快（≈语言 CoT 的一半步数） | **最快**，超过 V-CoT |
| 训练集最终成功率 | →100% | →100% | →100% | →100%（从未见显式坐标） |
| 7×7 未见迷宫泛化 | 一般 | 较好 | 800 步后饱和、偏低 | **最好且稳定** |

要点：视觉 / 更长 CoT 只**加速收敛、不抬天花板**；剥到最少 grounding 的 G-CoT-least 起点更高、收敛更快，且泛化最强。

把结论外推到其他视觉中心任务（表 1，准确率 %）：

| 模型 | V*Bench Overall | HR-Bench 4K Overall | FrozenLake | Jigsaw |
|------|------|------|------|------|
| Qwen2.5-VL-7B | 72.25 | 72.50 | 20.00 | 0.00 |
| + V-CoT RL | 83.25 | 72.00 | - | - |
| + G-CoT-least RL | **85.86** | **74.12** | **90.33** | **75.60** |

Jigsaw 从 0% 拉到 70%+，FrozenLake 从 20% 拉到 90%+；在 V*Bench / HR-Bench 真实高分辨 VQA 上，不裁图、不画图的 G-CoT-least 反而全面胜过显式裁图的 V-CoT，说明模型能**隐式**完成视觉推理。

### 消融实验
本文形态特殊——它本身就是一组「消融式」对照，核心对照即把 CoT 格式当作被消去/替换的变量：

| 配置 | 关键现象 | 说明 |
|------|---------|------|
| Zero RL（无 SFT 冷启动） | 训练崩溃 | 证明 SFT 冷启动是稳定 RL 的必要条件 |
| L-CoT / G-CoT / V-CoT | 训练集均→100%，但天花板相近 | 视觉/更长 CoT 只快不强 |
| G-CoT → G-CoT-least | 起点更高、收敛更快、仍达 100% | 去掉显式坐标系反而更好 |
| V-CoT vs G-CoT-least（跨尺寸） | V-CoT 800 步饱和、落后 | 极简 grounding 泛化更强 |

### 关键发现
- **「short is long」**：贡献最大的不是某个模块，而是「把 grounding 信息剥到最少」这一反直觉选择——它避免对特定坐标系/布局过拟合，提供更紧凑、更可迁移的归纳偏置。
- **机制层面**：视觉中心任务里，RL 主要强化的是模型既有的 **grounding 能力**；一旦 grounding 与视觉环境对齐，模型就能用极短 CoT 甚至隐式推理完成任务，无需显式吐坐标或改图。
- **冷启动不可省**：从零 RL 会崩，SFT 先把策略空间塑形、缓解探索与奖励稀疏问题。

## 亮点与洞察
- **用可控迷宫隔离变量**很聪明：规则纯视觉、难度可调、解可自动合成，把「数据污染 + 预训练先验」这两个最大干扰因素摁住，才让「格式」的影响第一次被干净测出来。这套测试床思路可迁移到任何想做机制分析的视觉推理研究。
- **训到真收敛**是常被忽视的方法学细节：很多「视觉 CoT 更强」的结论其实是在欠训练阶段比早期速度；本文训到 1000 步收敛后，速度优势消失、天花板趋同，提醒社区比较 RL 方法时必须控训练充分度。
- **「最少 grounding」当作归纳偏置**这个洞察可直接指导 SFT 数据构造：与其堆长 CoT，不如给一条干净的 grounded 答案，让 RL 去强化模型自己的隐式空间表征。

## 局限与展望
- 作者承认主要在迷宫这一类「视觉中心 + 规则可自动合成」的任务上验证，虽外推到 FrozenLake / Jigsaw / V*Bench，但仍计划扩展到更丰富的任务族与更多 VLM。
- 结论的成立依赖「答案本身就是 grounding 序列」的任务结构（迷宫/路径/拼图）；对于答案非空间序列、需要大量语言推演的任务（如视觉数学、图表理解），「short is long」是否成立尚未验证——这类任务里语言 CoT 可能仍不可替代。
- 只用了单一底座 Qwen2.5-VL-7B；不同规模/不同预训练 grounding 能力的模型上，「极简 grounding 最优」的临界点可能不同。

## 相关工作与启发
- **vs 长 CoT / 视觉 CoT 主流叙事（o3「think with image」等）**: 主流认为外化越多、链越长越好；本文用控变量实验给出反例——视觉 CoT 只加速不增效，啰嗦反而伤泛化，纠正了「视觉 CoT 普遍更强」的过度乐观。
- **vs grounding CoT 工作（用 bbox/点/线把语言绑到视觉证据）**: 本文不止用 grounding，还把它推到极限（G-CoT-least），并证明显式坐标系不是必需，模型可在隐式潜空间里完成空间推理。
- **vs 视觉中心 RL 工作**: 这些工作观察到「视觉任务里 RL 诱导的 CoT 往往很短」，本文进一步揭示其机制——RL 主要在强化既有 grounding 能力，grounding 一旦够强，极短 CoT 即可。

## 评分
- 新颖性: ⭐⭐⭐⭐ 反直觉的「short is long」结论 + 干净的控变量设计，机制洞察扎实，但不提新模型。
- 实验充分度: ⭐⭐⭐⭐ 训到收敛的公平比较 + 多任务外推（迷宫/游戏/真实 VQA），唯任务族仍偏「答案即空间序列」。
- 写作质量: ⭐⭐⭐⭐ 问题—假设—验证三段式清晰，take-away 提炼到位。
- 价值: ⭐⭐⭐⭐ 直接给出可泛化视觉推理 SFT 数据的实操指南，对社区构造数据有现实指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Think-as-You-See: Streaming Chain-of-Thought Reasoning for Large Vision-Language Models](think-as-you-see_streaming_chain-of-thought_reasoning_for_large_vision-language_.md)
- [\[CVPR 2025\] Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought](../../CVPR2025/llm_reasoning/argus_vision-centric_reasoning_with_grounded_chain-of-thought.md)
- [\[CVPR 2026\] Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](rationale-enhanced_decoding_for_multi-modal_chain-of-thought.md)
- [\[ICLR 2026\] Uni-CoT: Towards Unified Chain-of-Thought Reasoning Across Text and Vision](../../ICLR2026/llm_reasoning/uni-cot_towards_unified_chain-of-thought_reasoning_across_text_and_vision.md)
- [\[CVPR 2026\] FireScope: Wildfire Risk Raster Prediction with a Chain-of-Thought Oracle](firescope_wildfire_risk_raster_prediction_with_a_chain-of-thought_oracle.md)

</div>

<!-- RELATED:END -->
