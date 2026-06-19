---
title: >-
  [论文解读] Diversity Over Frequency: Rethinking Tool Use in Visual Chain-of-Thought Agents
description: >-
  [ICML 2026][LLM推理][视觉链式推理] 在 3D 空间推理这类"工具非必需"的视觉 Agent 任务上，作者发现 vanilla RFT 会让工具调用率塌缩到接近 0、强制鼓励工具调用又只带来边际收益；真正驱动性能提升的是 rollout 的探索多样性，于是用自适应熵正则化把 3DSRBench 准确率从 59.2% 推到 62.9%，并把工具重新定位为"训练期脚手架"而非推理期必备品。
tags:
  - "ICML 2026"
  - "LLM推理"
  - "视觉链式推理"
  - "工具使用崩溃"
  - "熵正则化"
  - "RFT"
  - "探索多样性"
---

# Diversity Over Frequency: Rethinking Tool Use in Visual Chain-of-Thought Agents

**会议**: ICML 2026  
**arXiv**: [2606.00096](https://arxiv.org/abs/2606.00096)  
**代码**: https://scaffolded-exploration.github.io  
**领域**: LLM推理 / 视觉Agent / 强化学习  
**关键词**: 视觉链式推理, 工具使用崩溃, 熵正则化, RFT, 探索多样性

## 一句话总结
在 3D 空间推理这类"工具非必需"的视觉 Agent 任务上，作者发现 vanilla RFT 会让工具调用率塌缩到接近 0、强制鼓励工具调用又只带来边际收益；真正驱动性能提升的是 rollout 的探索多样性，于是用自适应熵正则化把 3DSRBench 准确率从 59.2% 推到 62.9%，并把工具重新定位为"训练期脚手架"而非推理期必备品。

## 研究背景与动机

**领域现状**：当前一批视觉 Agent（DeepEyes、Mini-o3、PixelReasoner、Chain-of-Focus 等）把 `<grounding>`/裁剪/分割等视觉工具串成 visual chain-of-thought，再用 GRPO/DAPO 这类 group-based RL 做 fine-tuning（RFT），在 V\* 这种"高分辨率视觉搜索"基准上效果显著。

**现有痛点**：所有这些工作几乎都跑在"工具必然有用"的视觉搜索场景——目标小、必须 zoom-in 才看得清。但在更广的视觉推理任务里（3D 空间关系、医学 VQA），工具用不用、什么时候用其实没那么明确，已有方法在这种"工具可选"setting 下行为非常诡异：要么不用工具反而准确率涨，要么强迫用工具准确率不涨。

**核心矛盾**：在工具可选场景里存在一个优化非对称——tool-based rollout 比 tool-free 多走好几轮交互、token 更长、方差更大；GRPO 哪怕配上 token-level loss 和 over-turn masking，仍然天然偏向 tool-free 路径。但又不能直接断言"工具没用"，因为彻底禁用工具反而掉点。

**本文目标**：(1) 把"工具可选 + 复杂视觉推理"这一被忽视的 regime 系统化研究清楚；(2) 找出 vanilla RFT 与 tool-encouraging RFT 共同的失败本质；(3) 给出一个干预手段，让 RL 在这种 setting 下真正学到东西。

**切入角度**：作者跳出"工具频率 vs 准确率"的二维视角，去量化 rollout 在文本（pre-grounding `<think>` span 的 distinct-n-gram）和视觉（同一 query 下 crop 框两两 mIoU + 与问题关键词的 CLIP 对齐）两个空间里的多样性，发现 vanilla 和 tool-encourage 两条线都在单调坍缩，而早期那 ~20% 的工具调用恰好提供了多样化的探索历史。

**核心 idea**：把工具当作训练期的"脚手架"而非推理期必需品——只要在训练早期让 rollout 探索得足够多样（可以用工具，也可以用熵正则化主动维持多样性），晚期工具自然 fade、模型仍然变强。

## 方法详解

### 整体框架
这篇论文想搞清楚：在"工具可选"的复杂视觉推理任务上，RFT 到底该怎么对待视觉工具。基线 agent 取 Mini-o3（Qwen2.5-VL-7B-Instruct + SFT + RFT 已学会调 `<grounding>`），训练数据是 SpatialReasoner 的 1.2k 3D 空间推理 QA。Agent 走 thought–action–observation 循环：每步生成 `<think>` 推理，然后要么吐出带 `(bbox_2d, source)` 的 `<grounding>` 触发一次 zoom-in 裁剪、把新观察拼回历史，要么直接给 `<answer>`。整篇文章是一条"先诊断、再干预、再因果验证"的链：先用 vanilla RFT / tool-banned / tool-encouraging 四套对照诊断出"工具频率和准确率脱钩、真凶是探索多样性"，再给出自适应熵正则化这一干预，最后用 tool-banned + 熵正则化反向证明"熵正则化的增益必须靠工具提供的视觉探索才能兑现"。

### 关键设计

**1. 多样性诊断指标：把"探索广度"从"工具频率"里拆出来**

诊断的第一步是找到一个能甩开"工具调用率"的可测量。作者把 rollout 多样性拆成文本和视觉两个轴。文本侧只统计 `<grounding>` 之前那一段 `<think>`——他们先用 entropy probe 确认模型的不确定性主要落在推理 span 上、而不在 bbox 坐标本身，所以只对推理文本算 $n\in\{3,4,5,6\}$ 的 distinct-$n$-gram 比例。视觉侧则对同一 (image, question) 采 50 个 rollout，一是算所有 crop 框两两的 mean pairwise IoU（越低说明裁剪覆盖越广），二是算 crop patch 与问题名词关键词的 CLIP 相似度（越高说明探索越贴题）。两个视觉指标必须合看才有意义：单看 mIoU 低可能是"广撒网但跑偏"，配上 CLIP 才能区分"广覆盖且相关"和"集中固化"。这套指标之所以关键，是因为它一上来就打脸了"多调工具=多探索"的直觉——tool-encouraging 那条线工具调用量是 vanilla 的约 3 倍，但 mIoU 仍 >0.55、几乎和 vanilla 一样，CLIP 也没涨，矛头由此从"工具频率"转向真正缺失的"探索多样性"。

**2. 自适应熵正则化：用一个带比例反馈的旋钮防止 rollout 提前坍缩**

诊断指向多样性后，干预就只针对"防坍缩"这一根本病因，而不再去操纵工具频率。做法是在 GRPO 目标上加一个熵奖励项：

$$\mathcal{J}_{\text{ent}}(\theta)=\mathcal{J}_{\text{GRPO}}(\theta)+\lambda_t\cdot\mathbb{E}_{q,\tau}[\bar{\mathcal{H}}(\tau)],$$

其中 $\bar{\mathcal{H}}(\tau)$ 是整个 rollout 的 token 级平均熵，单 token 熵为 $\mathcal{H}(\pi_\theta(\cdot\mid s_k))=-\sum_v \pi_\theta(v\mid s_k)\log\pi_\theta(v\mid s_k)$。难点在系数 $\lambda$：预实验里固定 $\lambda$ 很脆，太小没效果、太大直接崩出混语种和复读机。于是作者借用 Zhang et al. 2025a 的比例反馈控制，让系数随当前 batch 熵自适应：$\lambda_t = K_p\,[\mathcal{H}_{\text{target}}-\mathcal{H}_t]_+$，取目标熵 $\mathcal{H}_{\text{target}}=0.9$、$K_p=0.03$——只在当前熵低于目标时才施压，一旦熵够高系数就归零。这个单旋钮、免手调的设计直接兑现成结果：工具调用照样从 ~20% 滑到 3%，但 3DSRBench 验证准确率冲到 62.9%，明显高于基线 59.2% 和强制工具的 59.9%，说明真正起作用的是维持住的多样性而非工具本身。

**3. Tool-banned × 熵正则化：用一组反例切开"熵增益"和"工具增益"**

最后一块是把"熵正则化是不是万能解药"这个混杂因素干净切开。作者复用严格的 tool-banned 协议（rollout 阶段屏蔽 `<grounding>` 触发 token、不执行任何裁剪），在完全没有工具的情况下叠加同一套自适应熵正则化；同时全程配 over-turn masking——把超 budget 的 rollout 从 advantage 计算里剔除而非记成负奖励，避免 GRPO 隐式惩罚长 rollout、反过来加剧 tool-use collapse。结果很说明问题：tool-banned + 熵正则化只跑到 57.8%，反而比裸 tool-banned 的 58.1% 还低，离 tool-enabled + 熵正则化的 62.9% 差出整整 5 个点。这就反过来证明熵正则化的增益不是凭空来的——没有工具早期带来的视觉证据多样性，单纯抬熵甚至有害，"tools as scaffolding"的论点由此被牢牢钉死。

### 损失函数 / 训练策略
基础优化器是 DAPO（GRPO + clip-higher + dynamic sampling + token-level policy loss + over-turn masking），训练 ≤100 步（再往后 dynamic sampling 会因 degenerate group 频繁触发 resampling 导致 wall-clock 爆炸）。两类对照 reward：DeepEyes tool bonus $R_{\text{DE}}=\mathbb{I}[y=y^*]+\lambda_{\text{tool}}\mathbb{I}[y=y^*]\mathbb{I}[u(\tau)=1]$ 与 PixelReasoner curiosity reward $R_{\text{PR}}=\mathbb{I}[y=y^*]+\alpha\max(H-\mathrm{RaPR}(q),0)\mathbb{I}[u(\tau)=1]+\beta\,r_{\text{penalty}}(\tau)$，其中 $\mathrm{RaPR}(q)=\mathbb{E}_\tau[u(\tau)]$ 是该 query 上的工具调用率、$r_{\text{penalty}}=\min(N-n_{\text{tool}}(\tau),0)$ 是工具次数硬上界。评测全部 Avg@8，VLM-as-judge (Qwen2.5-VL-7B) 抽答案选项。

## 实验关键数据

### 主实验：3DSRBench + CV-Bench-3D 总览

| 配置 | 工具? | 3DSRBench Acc | CV-Bench-3D Acc | 工具使用率 (初→饱和) |
|------|-------|---------------|-----------------|---------------------|
| Qwen2.5-VL 7B (generalist) | – | 48.4 | 82.9 | – |
| Mini-o3 (zero-shot) | yes | 54.5 | 77.6 | – |
| SpatialReasoner (specialist) | – | 60.3 | 80.3 | – |
| Vanilla RFT | yes | 59.2 | 76.7 | ~20% → ~2% |
| Tool-banned | no | 58.1 | – | 0% → 0% |
| Tool-Encourage (DeepEyes) | yes | 59.9 | 74.5 | ~20% → 100% |
| **Entropy-Regularized** | yes | **62.9** | **78.8** | ~20% → ~3% |
| Tool-banned + Entropy | no | 57.8 | – | 0% → 0% |

### 消融/分析：探索多样性

| 方法 | crop mIoU (↓) | CLIP (↑) | 视觉行为 |
|------|---------------|----------|----------|
| Vanilla RFT | 0.554 | 0.184 | 高度固化 |
| Tool-Encourage | 0.557 | 0.187 | 高度固化（虽然 crop 数 ~3×） |
| Entropy-Regularized | **0.494** | 0.184 | 主动探索且语义对齐不掉 |

VQA-RAD（医学 VQA、OpenThinkIMG 异构工具集）：Vanilla 46.34 → Tool-Encourage 47.23 → Entropy 48.78，趋势完全一致。

### 关键发现
- "工具频率"和"准确率"在 3DSRBench 上几乎正交：tool-use 从 20% 塌到 2% 准确率反而稳步涨；强制压到 100% 也只多 0.7 个点。
- 多样性才是隐藏自变量：vanilla 和 tool-encourage 的文本 distinct-n-gram 都单调下滑，且 crop 框 mIoU 一直 >0.55；只有熵正则化把 mIoU 压到 0.494 同时 CLIP 不掉，对应准确率显著领先。
- "脚手架"假说有两个反向证据支撑：tool-banned 掉到 58.1% 说明早期工具不可或缺；tool-banned + 熵正则化掉到 57.8% 说明熵正则化的增益必须配工具才能兑现。
- CV-Bench-3D 上 vanilla RFT 和 tool-encourage 都比 Mini-o3 base 还差（-0.9 / -3.1），只有熵正则化 +1.2，说明强迫或塌缩两种极端都会损害一般性视觉理解。

## 亮点与洞察
- 把"工具调用率"和"探索多样性"明确解耦的诊断框架很漂亮：文本 distinct-n-gram + 视觉 mIoU + CLIP 的组合可以直接搬到任何带视觉工具的 Agent RL 项目里做训练监控。
- 适应性熵正则化的"比例反馈 $\lambda_t = K_p[\mathcal{H}_{\text{target}}-\mathcal{H}_t]_+$"是一个非常便宜的工程 trick——单旋钮、无需调参、规避复读机崩溃，且能直接缝进 GRPO/DAPO pipeline。
- "tools as training-time scaffolding"这个 framing 颠覆了主流"训 Agent = 训它多用工具"的思路：推理期的工具调用率根本不是优化目标，重要的是训练期 rollout 走过的"经验地图"够不够广。这对所有 tool-augmented RL（不限视觉）都有启发——下次再看到 reward 里加 tool bonus，要警觉这可能在压多样性。
- 实验设计上 tool-banned + 熵正则化这一组是真正的"杀手锏 ablation"，干净切开了"熵增益 vs 工具增益"的混杂。

## 局限与展望
- 主战场只有 1.2k SpatialReasoner 训练 + 3DSRBench 评测，跨任务推广只靠 VQA-RAD/CV-Bench-3D 两个 discussion 节，"scaffolding"在视觉搜索（V\*）等"工具高度必需"的 setting 是否反而有害还没系统验证。
- 训练只跑到 100 步是被 DAPO dynamic sampling 的 degenerate group 卡死的工程限制，长时程下熵正则化是否还能稳住多样性、会不会出现新的崩溃模式不清楚。
- 基础 agent 锁定在 Mini-o3（Qwen2.5-VL-7B），换成更弱的 base（没学会工具调用）或更大模型，"早期 20% 工具调用"这个起点是否还存在、熵正则化是否还有同样收益，都需要重做。
- 熵目标 $\mathcal{H}_{\text{target}}=0.9$ 和 $K_p=0.03$ 是经验值，原则上应该和模型大小、词表分布相关，缺一个理论指南。
- 改进思路：把"工具频率"作为 curriculum 显式地从 100% 退火到 ~0%，配合熵正则化，可能比"先 SFT-RFT 再加熵"更可控；以及把诊断框架做成线上指标（distinct-n + mIoU + CLIP），训练中一旦多样性塌缩自动触发更高的 $\lambda_t$。

## 相关工作与启发
- **vs DeepEyes / Mini-o3 / PixelReasoner / Chain-of-Focus**：这些方法都属于"训 Agent 用 zoom/crop 工具做视觉搜索"，但都在 V\* 这类"工具必需"benchmark 上验证；本文把战场搬到 3D 空间推理这种"工具可选"setting，揭示了它们共用的 reward 设计在工具可选场景会塌缩或饱和，并给出能跨场景的干预（熵正则化）。
- **vs PixelReasoner curiosity reward / DeepEyes tool bonus**：这两类显式 tool-encouraging reward 设计被本文证明"会让工具调用率涨但准确率不涨且多样性继续坍缩"，本质上是给错了优化信号；本文用熵正则化这种"任务无关"的探索压力代替，反而稳健得多。
- **vs SpatialReasoner**：specialist 路线靠把 3D 坐标显式塞进推理拿 60.3%，本文不引入任何专门的空间监督就拿到 62.9%，说明 RL 训练动力学上的探索调控可以替代相当一部分领域归纳偏置。
- **vs Visual CoT 鲁棒性/泛化分析（Xu et al. 2025, Du et al. 2025）**：他们关心 visual CoT 的脆弱性和泛化性，本文补上了"训练动力学侧"的一块——多样性而非工具频率才是决定 visual CoT 是 scaffolding 还是包袱的关键变量。

## 评分
- 新颖性: ⭐⭐⭐⭐ "工具使用塌缩"现象首次被系统命名并量化，把工具重新定位成训练期脚手架的视角对 visual agent RL 领域是真正的认知更新。
- 实验充分度: ⭐⭐⭐⭐ 五组方法的对照表 + 文本/视觉双轴多样性诊断 + tool-banned×熵正则化的因果切分都做齐了，VQA-RAD 与 CV-Bench-3D 也提供了跨任务/跨工具集证据；只是训练规模受限于 100 步、跨 base model 验证缺位。
- 写作质量: ⭐⭐⭐⭐ Fig. 1 早晚期对比和"scaffolding"隐喻直观好懂，3.1→3.5 节"诊断—反例—干预—因果验证"的叙事链清晰；少量数学符号略密。
- 价值: ⭐⭐⭐⭐ 对所有做 tool-augmented Agent RL 的人都有直接可用的两条 takeaway：(1) 把多样性指标加入训练监控；(2) 在 reward 里加 tool bonus 前先想想是不是在压多样性。诊断框架和熵正则化方案都可以零成本迁移。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] MOSAIC: Learning When to Act or Refuse — Guarding Agentic Reasoning Models for Safe Multi-step Tool Use](learning_when_to_act_or_refuse_guarding_agentic_reasoning_models_for_safe_multi-.md)
- [\[ICLR 2026\] Generalizable End-to-End Tool-Use RL with Synthetic CodeGym](../../ICLR2026/llm_reasoning/generalizable_end-to-end_tool-use_rl_with_synthetic_codegym.md)
- [\[ACL 2026\] JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents](../../ACL2026/llm_reasoning/jtpro_a_joint_tool-prompt_reflective_optimization_framework_for_language_agents.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[ACL 2026\] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning](../../ACL2026/llm_reasoning/render-of-thought_rendering_textual_chain-of-thought_as_images_for_visual_latent.md)

</div>

<!-- RELATED:END -->
