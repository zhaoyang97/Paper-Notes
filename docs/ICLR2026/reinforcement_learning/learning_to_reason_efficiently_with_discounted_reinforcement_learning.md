---
title: >-
  [论文解读] Learning to Reason Efficiently with Discounted Reinforcement Learning
description: >-
  [ICLR 2026][强化学习][折扣强化学习] 把 LLM 的可验证奖励推理建模成"有限时域随机最短路 MDP"，仅对推理 token 施加折扣因子 $\gamma<1$，用 Blackwell 最优性证明：只要 $\gamma$ 足够接近 1，折扣最优策略会先最大化正确率、再在所有正确策略里挑最短轨迹——从而"无损精度地缩短思维链"。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "折扣强化学习"
  - "Blackwell 最优"
  - "随机最短路"
  - "GRPO"
  - "高效推理"
  - "RLVR"
---

# Learning to Reason Efficiently with Discounted Reinforcement Learning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=R4GPttoDyn](https://openreview.net/forum?id=R4GPttoDyn)  
**代码**: 待确认（基于 HuggingFace TRL + vLLM 实现）  
**领域**: 强化学习 / 高效推理（LLM Reasoning）  
**关键词**: 折扣强化学习, Blackwell 最优, 随机最短路, GRPO, 高效推理, RLVR  

## 一句话总结
把 LLM 的可验证奖励推理建模成"有限时域随机最短路 MDP"，仅对推理 token 施加折扣因子 $\gamma<1$，用 Blackwell 最优性证明：只要 $\gamma$ 足够接近 1，折扣最优策略会先最大化正确率、再在所有正确策略里挑最短轨迹——从而"无损精度地缩短思维链"。

## 研究背景与动机
**领域现状**：大推理模型（LRM）靠在最终答案前吐出大量中间推理 token 来解数学/代码题，RL 后训练（如 GRPO）能提精度，却往往把回答越训越长。

**现有痛点**：更长的思维链不是免费的——注意力是二次复杂度、KV cache 随长度膨胀，推理延迟和服务吞吐都受损。现有"高效推理"做法主要有三类：(i) 在策略优化里加 per-token 长度惩罚；(ii) 在变长/压缩轨迹上做 SFT；(iii) 用 prompt 直接要求模型简洁。这些方法多是工程启发式，缺乏"为什么能在不掉精度的前提下缩短"的原理性解释。

**核心矛盾**：学界对"长度 vs 精度"是否存在内在权衡争论不休，很多工作默认"缩短就要牺牲精度"。本文要正面挑战这个假设。

**本文目标**：训练出"既对又短"的 LRM——在不损失 Pass@1 精度的前提下尽量缩短回答。

**核心 idea（折扣即最短路）**：把推理建模成二值终止奖励的有限时域 MDP，每多吐一个推理 token 就是多走一步，正确终止就是到达目标。**只要给正确性奖励乘上折扣 $\gamma^{K(\tau)}$（$K$ 为推理 token 数），并让 $\gamma$ 趋近 1，折扣目标就退化成随机最短路（SSP）准则：先最大化成功概率，再在成功者中最小化路径长度**。这套"plain old discounting"等价于加一个小的 per-step 负奖励，却有 Blackwell 最优性做严格背书。

## 方法详解

### 整体框架
全文是"一套理论 + 一份训练配方"。理论侧：把 RLVR 推理写成 MDP $M=(S,A,P,r,H,\gamma,\mu)$，状态是 token 序列、动作是词表 token、`eos` 触发 verifier 给二值奖励；用对 $\gamma\to 1$ 的 Taylor 展开揭示折扣目标里"精度项 + 长度项"的层级结构，再用 Blackwell 最优性证明折扣最优 = 最短路最优。配方侧：把"折扣"落到 GRPO 上，只折扣推理 token 的正确性奖励、配 KL 正则、对齐 token 预算，无需改动策略优化算法本身。

```mermaid
flowchart LR
    A[推理任务] --> B[建模为有限时域 MDP<br/>二值终止 verifier 奖励]
    B --> C["折扣目标 J_γ(π)<br/>对 γ=1 做 Taylor 展开"]
    C --> D["主项 = 成功概率 p(π)<br/>次项 ∝ 路径长度 L(π)"]
    D --> E[Blackwell 最优性<br/>γ→1 时折扣最优=最短路最优]
    E --> F["训练配方：GRPO + 折扣 R(τ)=γ^K(τ)·r_e + r_f<br/>仅折扣推理 token 的正确性奖励"]
    F --> G[既对又短的推理策略]
```

### 关键设计

**1. 把 RLVR 推理写成"随机最短路"MDP，用 Taylor 展开拆出精度/长度层级**。作者在 RLVR 假设（仅终止动作 `aterm` 给二值奖励 $r(s,a_{term})=\mathbb{I}\{s\in G\}$、其余动作奖励为 0）下，定义成功概率 $p(\pi)=P_{\pi,\mu}(\text{success})$ 和条件成功路径长度 $L(\pi)=E_{\pi,\mu}[\tau\mid\text{success}]$。核心是 Lemma 3.5 的一致 Taylor 展开（令 $\varepsilon=1-\gamma$）：

$$J_\gamma(\pi)=E_{\pi,\mu}[\gamma^{\tau-1}\mathbb{1}\{\text{success}\}]=p(\pi)\big(1-\varepsilon(L(\pi)-1)\big)+R_\pi(\varepsilon),\quad |R_\pi(\varepsilon)|\le C_H\varepsilon^2$$

其中 $C_H=\tfrac12(H-1)(H-2)$。这一步是全文的"解释器"：当 $\gamma\uparrow 1$，主项就是成功概率 $p(\pi)$（即 Pass@1 精度），次项 $-\varepsilon\,p(\pi)(L(\pi)-1)$ 才用长度来给同样成功的策略打破平局。于是"最大化折扣回报"在近 $\gamma=1$ 区间就精确实现了 SSP 目标"先最大化成功、再最小化步数"。

**2. 用 Blackwell 最优性证明"折扣最优 = 最短路最优"，并保证受限策略类里它存在**。论文用比常规更强的 Blackwell 最优（对所有足够接近 1 的 $\gamma$ 都最优）来形式化"既对又短"。Theorem 3.6 给出主结论：在二值终止 verifier MDP 下，每个 Blackwell 最优策略都是最短路策略，即

$$\Pi^\star_{bw}\subseteq\arg\min_{\pi\in\Pi_{\max p}}L(\pi),\qquad \Pi_{\max p}=\arg\max_{\pi\in\Pi}p(\pi).$$

证明直觉来自把 $J_\gamma(\pi^\star)-J_\gamma(\pi)$ 展开：若某 $\pi$ 的成功概率更高，主项会让 $\gamma$ 足够接近 1 时折扣值反超，与 Blackwell 最优矛盾，故 $\pi^\star$ 必是精度最大者；在同精度策略中主项抵消，$\gamma\to 1$ 又逼出最短长度。关键还在于 Theorem 3.8 证明：对**有限受限策略类** $\Pi$，存在 $\gamma'<1$ 使 $(\gamma',1)$ 上的折扣最优集恒定且等于 Blackwell 最优集——这点很重要，因为 softmax/greedy 部署得到的就是有限确定性策略类。Proposition 3.1 用一个三策略反例说明：$\gamma$ 若离 1 不够近，反而可能偏好"更短但更不准"的策略，所以 Blackwell 视角不可或缺。

**3. 实例相关的 Blackwell 折扣因子 $\gamma_{bw}$，给出"$\gamma$ 该取多接近 1"的判据**。作者引入 $\gamma_{bw}:=\inf\{\gamma:\Pi^\star_{\gamma'}=\Pi^\star_{bw}\ \forall\gamma'\in(\gamma,1)\}$，并证明（Lemma 3.10）它存在且 $<1$。$\gamma_{bw}$ 衡量"折扣要多接近 1 才能可靠实现 SSP 打破平局而不牺牲精度"——越接近 1 说明 $\Pi$ 里最优 SSP 行为与近似平局的间隔越小、越难调。由于每个推理实例都诱导一个有限时域 MDP，对任意有限实例集 $\{M_i\}$ 可取统一临界折扣 $\gamma_{crit}=\max_i\gamma_{bw}(M_i,\Pi)<1$，从而存在一个 $\gamma$ 同时对整族实例保持 SSP 最优。Lemma 3.11 进一步保证 Blackwell 最优策略在无折扣问题上也最优，闭合了"折扣训练 ↔ 无损精度"的逻辑。

**4. 落地训练配方：折扣化 GRPO，只折扣推理 token 的正确性奖励**。由于正确性与格式信号都在轨迹末端计算，作者用序列级回报。设 $m_t\in\{0,1\}$ 标记 token $t$ 是否落在推理 span，推理 token 数 $K(\tau)=\sum_t m_t$，正确性奖励 $r_e(\tau)$、格式/塑形奖励 $r_f(\tau)$，则折扣回报为

$$R(\tau)=\gamma^{K(\tau)}r_e(\tau)+r_f(\tau),$$

优化目标 $E_{S_1\sim\mu,\tau\sim\pi}[R(\tau)]-\beta\,\mathrm{KL}(\pi\,\|\,\pi')$。四条工程原则支撑它：(i) **只折扣外在正确性奖励**，内在格式奖励不折扣——否则模型会为了短而丢格式标签或把长整数尾零删掉，实测全程折扣会让 GSM8K 掉 0.5%–1.0%；(ii) **KL 正则锚到滑动参考策略**，每 $u$ 步执行 $\pi_{ref}\leftarrow\text{stop\_grad}(\pi)$，像信任域一样防止策略过快坍缩成"提前停、忘了推理"；(iii) **只折扣推理 token**，用 `<reasoning>...</reasoning>` 标签界定 span，格式与答案 token 的 $m_t=0$；(iv) **对齐 token 预算**，因折扣化轨迹更短、同样 epoch 见的样本更少，故增加折扣方法的 rollout 数使总 token 量可比，保证公平对照。该配方对策略优化算法无侵入——REINFORCE/RLOO/GRPO 都能直接套，本文用 GRPO。

## 实验关键数据

### 主实验（GSM8K / MATH，Qwen2.5-7B & Llama3-8B）

| 数据集 | 模型 | 无折扣 Pass@1 | 无折扣长度 | 折扣 Pass@1 | 折扣长度 |
|---|---|---|---|---|---|
| GSM8K | Qwen2.5-7B-Instruct | 91.06 | 217.60 | 91.07 | **170.08**（−22%） |
| GSM8K | Llama3-8B-Instruct | 80.87 | 125.43 | 81.07 | **108.67**（−13%） |
| MATH | Qwen2.5-7B-Instruct | 64.80 | 491.32 | 64.55 | **384.96**（−22%） |
| MATH | Llama3-8B-Instruct | 24.48 | 328.43 | 24.75 | **257.73**（−22%） |

> 3 训练种子 × 10 评测种子平均，评测种子在两种方法间固定做配对比较。精度变化均不显著，长度显著缩短，印证 Theorem 3.6 的"定成功率下走最短路"。

### 泛化实验（DeepScaleR 训练，跨数据集评测，Phi-4 & Qwen2.5-14B）

| 模型 | 数据集 | 无折扣 Pass@1 | 无折扣长度 | 折扣 Pass@1 | 折扣长度 |
|---|---|---|---|---|---|
| Phi-4 | AMC 2023 | 51.00 | 1134.30 | **61.00** | **716.29** |
| Phi-4 | AIME 2025 | 14.00 | 1263.87 | **19.33** | **800.09** |
| Phi-4 | MINERVA | 28.46 | 553.74 | **29.85** | **318.10** |
| Phi-4 | OLYMPIAD | 36.91 | 1059.92 | 35.67 | **707.64** |
| Qwen2.5-14B | AMC 2023 | 50.00 | 737.47 | **59.50** | **582.31** |
| Qwen2.5-14B | OLYMPIAD | 35.13 | 797.57 | 34.76 | **684.02** |

> 训练集（DeepScaleR）不含 OLYMPIAD/MINERVA/AIME2025 题目，仍能跨集缩短长度且部分精度反升（Phi-4 在 AMC/AIME 上既更准又更短）。

### 消融 / 折扣因子扫描
- **$\gamma$ 扫描（Qwen3-1.7B / GSM8K）**：随 $1-\gamma$ 增大，长度单调下降但精度先稳后掉——与理论一致，$\gamma$ 越接近 1 越先保精度，过激折扣会把概率推向"短但错"的轨迹。
- **折扣范围消融**：折扣整条回答（含格式 token）会掉 0.5%–1.0% 精度，证明"只折扣推理 token"这一设计的必要性。
- **$\gamma$ 选取**：用二分搜索取"在不掉训练精度前提下尽量远离 1"的 $\gamma$；KL 更新频率与 $\beta$ 先在无折扣模型上调到最优再原样套用。

### 关键发现
1. 折扣几乎不动 Pass@1，却能砍掉 13%–37% 长度，在 4 种模型 × 6 个数学基准上一致成立。
2. 长度与精度并非天生冲突——存在一个实例相关阈值，阈值之上缩短无损精度，跌破才开始掉。
3. 较大模型在分布外数据集上甚至"又对又短"，暗示压缩推理可能利于泛化。

## 亮点与洞察
- **用最古老的 RL 工具解释最新现象**：把"discounting"这件 RL 教科书里的小事，借 Blackwell 最优性 + Taylor 展开，证成"无损精度缩短思维链"的原理，统一解释了一票启发式长度惩罚方法。
- **理论与实践少见地对齐**：Theorem 3.6 的"先精度后长度"预测，在 4 模型 6 基准上几乎逐条复现，且给出了实例相关的 $\gamma_{crit}$ 选取判据。
- **配方极简、零侵入**：不改策略优化算法，只把正确性奖励乘 $\gamma^{K(\tau)}$ 并加 token 掩码，任意 REINFORCE 类算法都能直接用。
- **澄清长度争论**：正面回应"长度 vs 精度是否权衡"，给出"阈值之内无权衡、阈值之外才有"的清晰结论。

## 局限与展望
- **理论限定在有限受限策略类**：Blackwell 存在性结论依赖 $|\Pi|<\infty$（softmax→greedy 部署类），对连续/无限策略类的保证有限；$\gamma_{bw}$ 虽存在但实践中不可直接计算，只能靠二分搜索近似。
- **仅验证数学推理 + 二值 verifier**：RLVR 假设要求确定性二值终止奖励，代码、开放域、过程奖励（非二值/带塑形）等场景能否照搬尚未验证。
- **公平对照靠手工对齐预算**：折扣方法样本更短，需人工调 rollout 数对齐总 token，存在调参负担。
- **展望**：(i) 探究压缩推理是否提升泛化（呼应 Hutter 的"压缩=泛化"）；(ii) 把"长推理找路 + 短推理压缩"结合，先长链探索策略再蒸馏压缩，或得更强推理策略。

## 相关工作与启发
- **高效推理三流派**：本文把 RL 长度惩罚（Arora & Zanette 2025、Su & Cardie 2025、Xiang 2025）、变长/压缩数据 SFT（Fatemi 2025、Lu 2025 等）、prompt 控长（Aggarwal & Welleck 2025）统一到"折扣即 per-step 负奖励"（Bertsekas 2012）的框架下，论证它们在近无折扣区间都恢复"先精度后长度"的同一排序。
- **Blackwell 最优性谱系**：承接 Blackwell (1962)、Puterman (2014)、Grand-Clément & Petrik (2023) 的折扣因子理论，首次把它迁移到"受限策略类 + LLM 推理"。
- **GRPO / RLVR**：方法直接搭在 GRPO (Shao 2024) 与 RLVR (Lambert 2024) 之上，用 TRL + vLLM 实现。
- **启发**：对做高效推理的人，这篇给出一个"先把任务写成 SSP，再用接近 1 的折扣无损缩短"的通用思路；$\gamma$ 的选取（二分搜索到精度刚好不掉）是可直接复用的工程 recipe。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 用 Blackwell 最优性把"折扣 = 无损缩短推理"讲透，视角新且统一了一批启发式方法，但底层工具是经典 RL 理论的迁移。
- **实验充分度**: ⭐⭐⭐⭐ — 4 模型 × 6 数学基准、多训练/评测种子配对、$\gamma$ 扫描与折扣范围消融齐备；但仅限数学 + 二值 verifier，缺代码/开放域验证。
- **写作质量**: ⭐⭐⭐⭐ — 理论roadmap清晰、定理与配方一一对应、动机反例直观，符号略密但可读。
- **价值**: ⭐⭐⭐⭐ — 给"高效推理"提供原理性背书 + 零侵入可落地配方，对学界（长度争论）和工业界（降推理成本）都有直接用处。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] ExGRPO: Learning to Reason from Experience](exgrpo_learning_to_reason_from_experience.md)
- [\[ICLR 2026\] Learn to Reason Efficiently with Adaptive Length-based Reward Shaping](learn_to_reason_efficiently_with_adaptive_length-based_reward_shaping.md)
- [\[NeurIPS 2025\] Training Language Models to Reason Efficiently](../../NeurIPS2025/reinforcement_learning/training_language_models_to_reason_efficiently.md)
- [\[ICLR 2026\] R1-Code-Interpreter: LLMs Reason with Code via Supervised and Multi-stage Reinforcement Learning](r1-code-interpreter_llms_reason_with_code_via_supervised_and_multi-stage_reinfor.md)
- [\[ICLR 2026\] Learning to Reason as Action Abstractions with Scalable Mid-Training RL](learning_to_reason_as_action_abstractions_with_scalable_mid-training_rl.md)

</div>

<!-- RELATED:END -->
