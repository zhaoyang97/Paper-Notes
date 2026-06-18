---
title: >-
  [论文解读] ReaGEN: Adaptive Generation of Structured Chains-of-Thought for Efficient Multimodal Reasoning
description: >-
  [CVPR 2026][多模态VLM][多模态推理] ReaGEN 不微调视觉语言模型本体，而是用一个仅 18M 参数的轻量生成器，根据每道题的注意力流自适应地"排出"一条结构化思维链（哪几个推理阶段、按什么顺序），从而以单遍推理拿到接近深度搜索的精度——在 Qwen3-VL-4B 上相对 VReST 最高提升 +26 个准确率点，同时把推理 token 用量平均压掉约 53%（部分基准达 79%）。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "多模态推理"
  - "结构化思维链"
  - "注意力反馈"
  - "进化式搜索"
  - "推理效率"
---

# ReaGEN: Adaptive Generation of Structured Chains-of-Thought for Efficient Multimodal Reasoning

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Tian_ReaGEN_Adaptive_Generation_of_Structured_Chains-of-Thought_for_Efficient_Multimodal_Reasoning_CVPR_2026_paper.html)  
**代码**: https://github.com/AISmartPerception/ReaGEN  
**领域**: 多模态VLM / LLM推理  
**关键词**: 多模态推理, 结构化思维链, 注意力反馈, 进化式搜索, 推理效率

## 一句话总结
ReaGEN 不微调视觉语言模型本体，而是用一个仅 18M 参数的轻量生成器，根据每道题的注意力流自适应地"排出"一条结构化思维链（哪几个推理阶段、按什么顺序），从而以单遍推理拿到接近深度搜索的精度——在 Qwen3-VL-4B 上相对 VReST 最高提升 +26 个准确率点，同时把推理 token 用量平均压掉约 53%（部分基准达 79%）。

## 研究背景与动机

**领域现状**：大型视觉语言模型（LVLM）在 VQA、图表理解等任务上很强，但面对需要多步、组合式推理的复杂视觉问题就容易掉链子。主流补救手段是思维链（CoT）：让模型先输出"观察→关联→计算→作答"这类中间步骤。现有增强 VLM 推理的方法分两类——(i) 推理时扩展（多路采样、Tree/Graph-of-Thought、MCTS 等），(ii) 训练后增强（在大规模 CoT 语料上做 SFT 或 RL）。

**现有痛点**：推理时扩展靠反复调用模型和深度搜索，延迟和 token 成本都很高；SFT 往往需要 10 万级别的高质量多模态 CoT 语料（还常依赖 GPT-4 级闭源模型来产出），训练贵、可迁移性差；RL 后训练主要在优化训练奖励，容易把策略收窄到少数高奖励模板，出现模式坍缩、奖励黑客与风格过拟合。

**核心矛盾**：搜索式方法"灵活但贵"，训练式方法"便宜但僵且需大数据"——精度与推理成本之间存在硬性 trade-off，而且大多数方法都在改 reasoning 的内容，没人去显式优化 reasoning 的**结构**（用哪几个阶段、什么顺序）。

**本文目标**：在**冻结底座 VLM、不做任何微调**的前提下，为每道题动态生成一条最优的结构化 CoT，既要有多路探索的灵活度，又要有单遍推理的效率。

**切入角度**：作者的关键观察是——推理阶段之间的依赖关系会通过**注意力流**暴露出来（后一阶段对前面哪些阶段"看得多"）。既然注意力能揭示"哪些阶段对最终答案重要、谁依赖谁"，就可以拿它当信号来编辑、优化 CoT 的阶段序列。

**核心 idea**：把"如何组织推理结构"这件事本身蒸馏进一个轻量生成器 GEN——先用教师引导的进化式搜索离线挖出每道题的优质 CoT 结构（用注意力导出的阶段重要度打分），再训练 GEN 学会"从注意力信号直接预测好结构"，推理时只跑 GEN 一遍（或少量迭代精修），用搜索的灵活度换来单遍的效率。

## 方法详解

### 整体框架
ReaGEN 的核心是：**底座 VLM 全程冻结，只训练一个外挂的轻量生成器 GEN，让它学会为每道 image–query 排出一条定制的推理阶段序列**。整条管线分三段：① 离线阶段，用一个更强的教师 VLM（Qwen3-VL-32B）引导进化式搜索，为每道训练题挖出高奖励的 CoT 结构，并记录学生 VLM（Qwen3-VL-4B）执行时的跨阶段注意力；② 训练阶段，用这些"题目 + 注意力 + 目标 CoT"三元组监督 GEN，让它学会从注意力摘要映射到好结构；③ 推理阶段，对新题只调 GEN 预测一条 CoT，再让冻结的学生 VLM 按这条 CoT 执行一遍得到答案，可选地多迭代几轮精修。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入：图像 I + 问题 Q"] --> B["阶段动作空间与 CoT 编码<br/>四组功能阶段 → 阶段ID序列"]
    B --> C["教师引导进化式搜索<br/>Evaluate→Select→Mutate<br/>注意力阶段重要度 + 奖励"]
    C -->|离线产出 (I,Q,A,τ*) 三元组| D["轻量 GEN 生成器<br/>18.3M Transformer，VLM 冻结"]
    D --> E["单遍 / 迭代精修推理<br/>GEN 预测 τ̂ → 学生 VLM 执行"]
    E --> F["最终答案"]
```

### 关键设计

**1. 阶段动作空间与 CoT 编码：把"推理结构"变成可预测的离散序列**

要让一个生成器去"排推理结构"，首先得把结构变成可枚举、可预测的对象。ReaGEN 把多模态推理拆成一个**阶段池**，按认知流程归为四组功能：① 感知与输入理解（读题、解析视觉内容）；② 接地与事实抽取（把文本实体连到图像、找关键变量）；③ 推理与推断（关系、逻辑、数值推演）；④ 解释与答案选择（汇总中间结论、产出最终答案）。一条 CoT 就是这些动作的一个有序序列 $\tau^\star = (s_1,\dots,s_L)$。为了让 GEN 能自回归预测，每条 CoT 被编码成定长的阶段 ID 向量 $\mathbf{c} = (c_1,\dots,c_L, c_{L+1},\dots,c_{L_{\max}})$，其中 $c_t \in \{1,\dots,S\}$ 是阶段 ID，$c_{L+1}=\text{EOS}$，其余位置 padding。这样"选哪几个阶段、按什么顺序"就退化成一个标准的序列预测问题，是后续 GEN 能轻量化的前提。

**2. 教师引导的进化式搜索 + 注意力阶段重要度：用注意力当信号挖出每题最优结构**

这是离线产数据的引擎，针对"高质量多模态 CoT 语料太贵"的痛点。对每道题，搜索按 **Evaluate→Select→Mutate** 循环迭代：先让**冻结的学生 VLM** 按候选 CoT 分阶段执行——每个阶段 $s_t$ 配专属 system prompt，并维护一个记忆体 $M_t$ 累积此前所有阶段输出喂给后续阶段，即 $M_{t+1}=M_t\cup\{y_t\}$，$y_t=f_\theta(I,Q,M_t,s_t)$。执行时抽取**跨阶段注意力**，先累积未归一化的注意力质量 $\tilde A_{i,j}$（阶段 $j$ 的 token 对阶段 $i$ 输出 token 的平均注意力），再对所有更早阶段归一化得到注意力质量矩阵 $A_{i,j}$，衡量"阶段 $j$ 有多依赖阶段 $i$"。在此之上定义**阶段重要度** $\mathrm{Imp}(i)$，递归地把直接贡献（对最终答案的注意力 $A_{i,F}$）与间接贡献（经后续阶段传导）相加：

$$\mathrm{Imp}(i) = A_{i,F} + \sum_{j=i+1}^{N} \lambda^{\,j-i} A_{i,j}\,\mathrm{Imp}(j)$$

其中 $\lambda\in(0,1]$ 是对长依赖链的折扣。每个候选还有标量奖励 $R(\tau)=\alpha\,s(\tau)-\beta\,\ell(\tau)-\gamma\,d(\tau)$，鼓励"准确（预测分 $s$）、简洁（长度惩罚 $\ell$）、结构多样（多样性惩罚 $d$）"。选出 top-K 精英集后，由更强的教师模型读它们的记忆与重要度画像，做**重要度感知的定向变异**（剪掉低影响阶段、把关键阶段提前），而非随机改。注意力当向导，让搜索高效收敛到每题的优质结构，远比盲搜省。

**3. 轻量 GEN 生成器：把搜索结果蒸馏成一遍前向就能出结构**

搜索虽好但贵，不能在线上跑。GEN 的作用是把搜索的产物模仿下来：训练数据是 $\{(I,Q,A,\tau^{\text{init}},\tau^\star)\}$，其中从冻结 VLM 抽出图像嵌入 $E^{\text{Img}}$ 和问题嵌入 $E^{\text{Q}}$，配上结构矩阵 $A$ 和初始链 $\tau^{\text{init}}$，监督信号是目标 CoT $\tau^\star$。GEN 本身是个 4 层、18.3M 参数的紧凑 Transformer 编码-解码器：把图像/问题嵌入投到共享空间融成多模态记忆，把 $A$ 变成调制阶段嵌入的结构表示，解码器用两个头分别预测阶段 ID 序列（带 EOS）和 CoT 长度，训练目标是阶段预测与长度预测的交叉熵。关键在于**底座 VLM 全程冻结**——所有"学结构"的负担都压在这个 18M 小模型上，因此数据高效、训练便宜，也不会污染底座的通用能力。

**4. 单遍 / 迭代精修推理：单遍出结构，可选少步精修换更高精度**

部署时 ReaGEN 给两档用法。**单遍（2 iter）**：先用默认/种子 CoT $\tau^{(0)}$ 让学生跑一次收集注意力摘要，GEN 据此预测一条定制 $\hat\tau$，学生再按 $\hat\tau$ 执行一次得答案——总共两次学生调用，无搜索。**迭代精修**：同一个 GEN 可反复调用，每轮 (a) 学生按 $\tau^{(t)}$ 跑出新注意力，(b) GEN 据此提出 $\tau^{(t+1)}$，(c) 直到收敛 $\tau^{(t+1)}=\tau^{(t)}$ 或用满迭代预算。由于 GEN 极轻，还能开 $K$ 条并行精修分支（"宽度"，实验用 $K=5$），每条从不同种子起，最终用多分支多数投票选答案。这样默认就是快速单遍规划，需要时再用低成本精修循环换"问题自适应"的更强结构。

### 损失函数 / 训练策略
GEN 用 AdamW + 混合精度训练 200 个 epoch，batch 64，学习率 $1\times10^{-4}$；训练数据为视觉与数学混合，含 1,050 个 MMMU 样本 + 1,496 个 MathVerse（纯文本）样本。离线搜索阶段每题的进化预算上限为 20 轮。损失为阶段 ID 序列预测与 CoT 长度预测两路交叉熵之和。

## 实验关键数据

### 主实验
底座固定为冻结的 Qwen3-VL-4B，教师为 Qwen3-VL-32B。下表为复合基准上的准确率（括号内为相对同底座 VReST 的绝对变化）：

| 方法 | MMMU-Pro(10) | MMMU-Pro(4) | VStar | MMStar | MathVision | MathVerse |
|------|------|------|------|------|------|------|
| Direct Answer | 32.42% | 45.78% | 79.14% | 58.95% | 22.59% | 25.25% |
| + CoT | 37.57% | 49.02% | 76.47% | 67.85% | 29.75% | 36.17% |
| + VReST | 46.30% | 56.13% | 83.42% | 49.27% | 44.67% | 47.97% |
| **+ ReaGEN (4 iter)** | **52.54%** (↑6.2) | **64.51%** (↑8.4) | 84.49% (↑1.1) | 75.77% (↑26.5) | 44.60% (↓0.1) | 47.59% (↓0.4) |

> MMStar 上相对 VReST 高出约 +26.5 点（VReST 在该集上反而崩到 49.27%）；但在 MathVision / MathVerse 这两个偏数学的集上 ReaGEN 与 VReST 基本打平甚至略低，说明其优势主要在**视觉密集型**推理。⚠️ 摘要写"+26 准确率点 / token 减 79%"，正文又给出"平均减 53%"，二者口径不同（前者为单基准峰值，后者为总体均值），引用时需区分。

### 效率与泛化

| 基准 | VReST token | ReaGEN-4 token | 相对降幅 |
|------|------|------|------|
| MMMU-Pro(10) | 188 | 45 | 76% |
| MMMU-Pro(4) | 176 | 46 | 74% |
| MathVision | 240 | 83 | 65% |
| MMStar | 140 | 87 | 38% |
| VStar | 86 | 77 | 11% |
| **总体** | **166** | **68** | **53%** |

跨数据集泛化（GEN 只在 MathVision 训练、直接评测其他集）上，迁移到视觉基准 MMStar 仍能拿到约 +26 点（相对 VReST），无需重训。

### 关键发现
- **迭代档位单调有效**：2→3→4 iter 在多数视觉基准上稳步上升（如 MMMU-Pro(10) 从 49.94%→51.90%→52.54%），印证"用少量额外 GEN/学生调用换精度"成立。
- **18M GEN 打得过 32B 教师**：在 4 iter 设置下，即便对比更强的 VReST(Teacher-Reward, 32B)，ReaGEN 在多数基准上仍胜，且推理时**不需要 32B 教师**。
- **省的是调用与 token 双线**：相对 VReST，ReaGEN 同时大幅减少 VLM 调用次数与生成 token，效率收益来自"单遍规划替代深度搜索"。

## 亮点与洞察
- **拿注意力当"结构监督信号"**：大多数 CoT 工作优化的是推理内容，ReaGEN 用跨阶段注意力定义阶段重要度 $\mathrm{Imp}(i)$ 去优化推理的**骨架**，这个角度很新——注意力本就是模型自己暴露的依赖图，免费且可解释。
- **"搜索做老师、小模型做学生"的结构蒸馏**：把昂贵的进化搜索压进 18M 生成器，底座全程冻结，等于给任意开源 VLM 外挂一个"会排推理结构"的插件，可复用性强。
- **单遍 vs 迭代的可调旋钮**：同一个 GEN 既能单遍快出、又能多分支精修投票，把"精度/成本"做成一个连续可调档位，工程上很实用。
- **可迁移 trick**：阶段池 + 阶段 ID 编码这套"把流程结构离散化再让小模型预测"的范式，可迁到 agent 规划、工具调用顺序选择等需要"排步骤"的任务。

## 局限与展望
- **依赖内部注意力信号**：方法需要访问底座 VLM 的注意力权重，因此只适用于自托管/开源 VLM，闭源 API 模型（仅给输出）无法直接用。
- **数学推理上优势有限**：MathVision/MathVerse 上与 VReST 打平甚至略降，说明阶段化结构对纯符号/数值长链推理的增益不如视觉密集任务，结构优化可能不是数学题的瓶颈。
- **阶段池是人工设计的**：四组功能阶段与 system prompt 由作者预定义，迁移到新领域可能需要重新设计阶段词表；阶段粒度是否最优缺乏自动化探索。
- **多分支投票额外开销**：迭代精修+$K=5$ 并行分支虽轻，但相比单遍仍多出数倍学生调用，论文给的 token 数主要针对单遍设置，精修档的真实成本需结合分支数看。

## 相关工作与启发
- **vs VReST / Socratic-MCTS / AStar（推理时搜索）**：它们在推理时做树/图搜索或思维卡复用来涨点，但反复调用模型导致高延迟高 token；ReaGEN 把搜索移到离线、线上只跑一遍 GEN，保留多路探索的灵活度却砍掉在线搜索成本。
- **vs VisualCoT / LLaVA-CoT / Chain-of-Focus（训练式后训练）**：它们在大规模 CoT 语料或 RL rollout 上微调底座 VLM，训练贵、结构难跨模型复用；ReaGEN 冻结底座只训 18M 生成器，数据高效且结构即插即用。
- **启发**：当"内容已经够好、结构才是瓶颈"时，与其微调大模型，不如蒸馏一个专门管"流程编排"的轻量外挂——这对成本敏感的开源 VLM 部署是一条很经济的路线。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用注意力定义阶段重要度、把推理结构本身蒸馏进轻量生成器，角度确实新颖。
- 实验充分度: ⭐⭐⭐⭐ 5 个基准 + 跨数据集泛化 + token/调用效率分析较完整，但数学基准上的弱势暴露了适用边界。
- 写作质量: ⭐⭐⭐⭐ 三段式管线（搜索→训练→推理）讲得清楚；⚠️ 摘要里 +26 点 / 减 79% 与正文 53% 口径不一致，略有混淆。
- 价值: ⭐⭐⭐⭐ 给开源/自托管 VLM 提供了"精度近搜索、成本近单遍"的实用方案，工程落地价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] When Visualizing is the First Step to Reasoning: MIRA, a Benchmark for Visual Chain-of-Thought](when_visualizing_is_the_first_step_to_reasoning_mira_a_benchmark_for_visual_chai.md)
- [\[CVPR 2026\] Stable and Efficient Single-Rollout RL for Multimodal Reasoning](stable_and_efficient_single-rollout_rl_for_multimodal_reasoning.md)
- [\[CVPR 2026\] SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker](seatrack_multimodal_tracker.md)
- [\[AAAI 2026\] AStar: Boosting Multimodal Reasoning with Automated Structured Thinking](../../AAAI2026/multimodal_vlm/astar_boosting_multimodal_reasoning_with_automated_structure.md)
- [\[CVPR 2026\] Fuel Gauge: Estimating Chain-of-Thought Length Ahead of Time in Large Multimodal Models](fuel_gauge_estimating_chain-of-thought_length_ahead_of_time_in_large_multimodal_.md)

</div>

<!-- RELATED:END -->
