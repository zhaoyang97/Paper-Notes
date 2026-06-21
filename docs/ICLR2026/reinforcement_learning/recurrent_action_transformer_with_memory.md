---
title: >-
  [论文解读] Recurrent Action Transformer with Memory
description: >-
  [ICLR2026][强化学习][离线强化学习] RATE（带记忆的循环动作 Transformer）把轨迹切成定长段、用一组可学习的记忆嵌入在段间循环传递历史信息，并新增一个基于交叉注意力的「记忆保留阀」(MRV) 来控制每次更新该保留还是覆写哪些记忆，从而在 ViZDoom、T-Maze、Memory Maze、POPGym 等记忆密集型离线 RL 任务上大幅超越 Decision Transformer，同时在 Atari/MuJoCo 标准任务上保持竞争力。
tags:
  - "ICLR2026"
  - "强化学习"
  - "离线强化学习"
  - "Transformer"
  - "记忆机制"
  - "段级循环"
  - "POMDP"
---

# Recurrent Action Transformer with Memory

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=kByN4v0M3e](https://openreview.net/forum?id=kByN4v0M3e)  
**代码**: https://sites.google.com/view/rate-model/  
**领域**: 强化学习 / 离线RL  
**关键词**: 离线强化学习, 决策 Transformer, 记忆机制, 段级循环, POMDP

## 一句话总结
RATE（带记忆的循环动作 Transformer）把轨迹切成定长段、用一组可学习的记忆嵌入在段间循环传递历史信息，并新增一个基于交叉注意力的「记忆保留阀」(MRV) 来控制每次更新该保留还是覆写哪些记忆，从而在 ViZDoom、T-Maze、Memory Maze、POPGym 等记忆密集型离线 RL 任务上大幅超越 Decision Transformer，同时在 Atari/MuJoCo 标准任务上保持竞争力。

## 研究背景与动机
**领域现状**：离线强化学习里，Decision Transformer (DT) 这类方法把「轨迹」当成 $(回报R, 观测o, 动作a)$ 的序列，用 GPT 式的自回归建模直接预测动作，绕开了价值函数估计，效果很好。

**现有痛点**：自注意力是二次复杂度，DT 的上下文窗口固定且有限。一旦关键线索（比如迷宫开头给的一个「向左还是向右」的提示位）滑出上下文窗口，DT 就再也读不到它，在长程信用分配和稀疏奖励的 POMDP 任务上直接失败。

**核心矛盾**：要么扩大上下文窗口（受二次复杂度和训练不稳定限制），要么换稀疏注意力（这些模式多是为 NLP 设计的、迁到 RL 上泛化差）。本质矛盾是——**在不无限拉长上下文的前提下，怎么让模型记住很久以前的、且很稀疏的关键信息**。

**本文目标**：给离线 RL 的 Transformer 装一套记忆机制，使其有效上下文 $K_{\text{eff}} = N \times K$ 远超单段注意力上限，并且能在高度稀疏的任务里长期保留重要信息、不被后续噪声冲掉。

**切入角度**：作者从 NLP 里的记忆增强 Transformer（RMT 的记忆嵌入、Transformer-XL 的隐藏状态缓存）借力，但指出 RL 的输入是「观测/动作/回报」多模态、且奖励和观测都高度稀疏，需要专门设计——尤其是**朴素地把记忆向前传会导致误差累积或重要信息被覆写**。

**核心 idea**：用「记忆嵌入 + 隐藏状态缓存 + 一个可学习的交叉注意力阀门 MRV」三件套，把轨迹做段级循环；MRV 让旧记忆 $M_n$ 来「审查」新记忆 $M_{n+1}$，决定哪些该留、哪些该覆写，从而避免稀疏长程信息被冲掉。

## 方法详解

### 整体框架
RATE 处理一条长度为 $T$ 的轨迹 $\tau_{0:T-1}$，每个时间步是三元组 $(R_t, o_t, a_t)$（回报-to-go、观测、动作）。先用模态专属编码器分别把三者编码成 $\tilde R_t, \tilde o_t, \tilde a_t$，再把编码后的序列切成 $N = T // K$ 个互不重叠、长度为 $K$ 的段 $S_n$。模型**逐段处理**：每段前后各拼一份**相同的记忆嵌入** $M_n \in \mathbb{R}^{m\times d}$（$m$ 个记忆 token、维度 $d$），即 $\tilde S_n = \mathrm{concat}(M_n, S_n, M_n)$，送进 Transformer 得到该段的动作预测 $\hat a_n$ 和更新后的记忆 $M_{n+1}$；接着 $M_{n+1}$ 经过**记忆保留阀 MRV** 精炼后，传给下一段。段与段之间还沿用 Transformer-XL 的**隐藏状态缓存**（把前段算出的隐藏激活当作扩展的 KV 上下文复用，不回传梯度）。这样信息以两条通道跨段流动：可训练的记忆嵌入 + 非训练的缓存隐藏状态。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["轨迹 (R,o,a)<br/>模态编码 + 切成 N 段"] --> B["双拷贝记忆嵌入<br/>每段前后各拼一份 M_n"]
    B --> C["隐藏状态缓存<br/>复用前段激活当扩展 KV"]
    C --> D["Transformer 处理段<br/>输出动作 â_n 与新记忆 M_{n+1}"]
    D --> E["记忆保留阀 MRV<br/>交叉注意力过滤新记忆"]
    E -->|M_{n+1} 传入下一段| B
    E --> F["逐段输出动作 â_n"]
```

### 关键设计

**1. 双拷贝记忆嵌入：给每段同时开「读口」和「写口」**

DT 的痛点是上下文外的线索读不到，作者用一组可学习的记忆 token $M_n$ 当作专属的历史信息存储器。关键在于**每段前后各拼一份相同的 $M_n$**：$\tilde S_n = \mathrm{concat}(M_n, S_n, M_n) \in \mathbb{R}^{(3K+2m)\times d}$（这里 $3K$ 是因为每步含 $R,o,a$ 三个 token）。为什么要前后各放一份？因为解码器用的是因果自注意力——**前缀那份 $M_n$ 提供「读」**：段内每个 token 都能向后注意到这份进来的记忆；**后缀那份 $M_n$ 提供「写」**：它在因果顺序里排在整段之后，于是最后几层能向前注意整段 $S_n$、把段内新信息写进记忆产出 $M_{n+1}$。只放前缀则记忆可读不可更新，只放后缀则段内读不到旧记忆，所以两份缺一不可。这是 RATE 段级循环记忆能成立的结构前提。

**2. 隐藏状态缓存：在记忆嵌入之外再补一条连续信息通道**

只有离散的记忆 token 还不够，作者沿用 Transformer-XL 的做法，把前面各段算出的**隐藏激活固定存下来、当作扩展的 key-value 上下文**拼到当前段输入前，不重算、也不反传梯度。这样段级循环既靠可训练的 $M_n$ 携带「任务级关键线索」，又靠缓存的隐藏状态携带「连续、稠密的近期上下文」。消融显示这两条通道分工不同（见关键发现）：稠密连续反馈的任务（如 ViZDoom）更依赖缓存隐藏状态，稀疏离散决策点（如 T-Maze）更依赖记忆嵌入。

**3. 记忆保留阀 MRV：用旧记忆审查新记忆，防止稀疏长程信息被覆写**

这是本文最核心的创新。痛点很具体：**朴素地把 $M_{n+1}$ 直接前向传，会累积误差、或把还有用的旧信息覆写掉**——在稀疏任务里，开头那个一次性线索很容易被后面几百步的无关更新冲没。MRV 是一个交叉注意力模块，让**旧记忆 $M_n$ 当 Query、新记忆 $M_{n+1}$ 当 Key/Value**：

$$\mathrm{MRV}(M_n, M_{n+1}) = \mathrm{FFN}\big(\mathrm{MultiHead}(Q=M_n,\, K=M_{n+1},\, V=M_{n+1})\big)$$

直觉上，$M_n$ 用自己已经持有的内容去「打分」新记忆的每个 token，从而控制更新时哪些该保留、哪些该覆写。和 Transformer-XL 那种「无门控直接复用缓存」的静态循环不同，MRV 是内容相关的过滤，能保住稀疏、长程的信息。作者还给出**保留性定理**做理论背书：在「行向量 $\ell_2$ 归一化 + $\alpha$-对齐条件（旧记忆每行与某个 $V$ 行的夹角不超过 $\arccos\alpha$）」假设下，单次 MRV 更新后记忆至少保留 $\big(1 - \sqrt{2(1-\frac{\alpha}{m})}\big)$ 的比例：

$$\|M_{n+1} - M_n\|_F \le \sqrt{2\big(1 - \tfrac{\alpha}{m}\big)}\cdot \|M_n\|_F$$

证明用注意力行和为 1 + 鸽笼原理保证存在权重 $\ge 1/m$ 的对齐项，再配合单位向量的夹角不等式得到上界。注：这里的 $\alpha$-alignment 只是几何夹角概念，和 LLM 对齐/偏好微调无关（作者特别澄清）。

### 一个完整示例：T-Maze 怎么把开头线索带到终点
T-Maze 走廊长 $T=8$，智能体在第 0 步收到一个一比特线索 $o_0$（指示终点该左转还是右转），之后奖励极稀疏。DT 在 8 步全序列上训练，一旦 $o_0$ 滑出窗口就无法在推理时取回，注意力图显示它只能注意到窗口内的近期 token。RATE 把这条序列切成 3 段（每段长 3）循环处理：第 1 段把 $o_0$ 写进记忆嵌入 $M_1$，MRV 在第 1→2、2→3 段的更新里都判定这条稀疏线索该保留、不被覆写，于是 $M_3$ 仍携带 $o_0$，终点处智能体据此正确转向。注意力图清楚显示 RATE 的记忆 token 在后续段里仍能「读到」$o_0$，而 DT 早已丢失。这就是 RATE 在 DT 失败处成功的机制画面。

### 损失函数 / 训练策略
训练目标是逐段的动作监督损失 $L(a_n, \hat a_n)$（DT 式的回报条件序列建模），记忆 $M_0 \sim \mathcal N(0,1)$ 初始化，按段循环展开（Algorithm 1）。缓存隐藏状态不回传梯度，MRV 内部权重 $W_Q, W_K, W_V, W_M$ 可学习。有效上下文 $K_{\text{eff}} = N\times K$，远超单段注意力上限。

## 实验关键数据

### 主实验
覆盖记忆密集型任务（ViZDoom-Two-Colors、T-Maze、Minigrid-Memory、Memory Maze、POPGym）与标准 RL（Atari、MuJoCo），对比 DT / RMT / TrXL / LSDT / DMamba / CQL / BC 等一大批基线。

| 任务 | 指标 | RATE | 代表基线 | 说明 |
|--------|------|------|----------|------|
| Memory Maze (9×9, 1000步) | 平均回报±SEM | **7.64±0.41** | DT 6.83 / RMT 7.27 / TrXL 7.12 | 数据集均值仅 4.69，RATE 最高 |
| POPGym 全 48 任务 | 归一化均分 | **9.5** | DT 5.8 / BC-LSTM 9.0 | 总体最高 |
| POPGym 记忆子集 (33) | 归一化均分 | **0.5** | DT −3.5 / BC-LSTM −0.2 | 唯一正分，其余全为负 |
| POPGym 反应子集 (15) | 归一化均分 | 9.1 | DT 9.3 / BC-LSTM 9.1 | 简单任务上各家持平 |
| T-Maze 泛化 | 成功率 | 9600 步外推仍可用 | DT 中等长度即崩到 ~50% | 训练≤900步，对应最长 28800 token |
| Atari (4 游戏) | 原始分 | 4 局中 3 局胜 DT | — | Breakout 111.0 / Qbert 12486.9 / Pong 18.8 |
| MuJoCo D4RL (9 数据集) | 归一化均分 | 78.5 | DT 74.7 | 与专用方法持平，非其主场仍有竞争力 |

T-Maze 上 RATE 在所有分布内长度达 100% 成功率，且外推到 9600 步推理仍能用；DT/LSDT 在训练长度内能追平，但超出即急剧退化；TrXL 表现接近 DT，说明**只靠隐藏状态缓存不足以长程取回稀疏信息**——这正反衬出 MRV 的必要性。

### 消融实验（推理时把记忆组件替换成随机噪声）

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| RATE 完整 | T-Maze 100% / ViZDoom 最高且最稳 | 完整模型 |
| 噪声破坏记忆嵌入 M (T-Maze) | 成功率掉到 ~50% | 智能体仍会导航但丢了开头线索、无法正确转向 |
| 噪声破坏缓存隐藏状态 (ViZDoom) | 更敏感 | 稠密连续反馈任务更依赖缓存隐藏状态 |

### 关键发现
- **两条记忆通道分工明确**：记忆嵌入 $M$ 对「稀疏、离散的决策点」（T-Maze）最关键，破坏它成功率直接腰斩到 50%；缓存隐藏状态对「稠密、连续反馈」任务（ViZDoom）更关键。这说明它们不是冗余，而是互补。
- **MRV 是长程稀疏取回的胜负手**：TrXL（只有缓存、无 MRV）在 T-Maze 上表现和 DT 差不多，证明无门控的静态循环留不住稀疏线索；ViZDoom 上 DT 的红柱回报在线索出窗后骤降近 50%，而 RATE/RMT/TrXL 这类记忆模型保持稳定。
- **不损害简单任务**：在 POPGym 反应子集和 MuJoCo 上，RATE 与专用方法持平，说明加记忆机制没有拖累简单任务，是「有则更好、无则无害」的通用架构。

## 亮点与洞察
- **前后双拷贝记忆嵌入分离读写**：用因果注意力的天然时序，把同一份记忆放前缀当读口、放后缀当写口，一个极简的拼接就解决了「记忆既要可读又要可更新」的结构难题——很巧、也很易复用到别的段级循环模型。
- **MRV 把「记忆更新」变成内容相关的门控**：让旧记忆当 Query 审查新记忆，相比 Transformer-XL 的无脑复用，多了一层「该不该覆写」的判断，且配了可证明的保留性下界，理论+实践都站得住。
- **可迁移性**：这套「段级循环 + 门控记忆更新」的思路不限于离线 RL，任何需要在有限上下文里长期保留稀疏信号的序列任务（长文档、长程对话状态）都能借鉴 MRV 式的内容相关记忆门。

## 局限与展望
- **MRV 保留性定理依赖 $\alpha$-对齐假设**：该假设是「经验上训练好的模型成立」，并非对任意权重都成立，理论保证的强度受此限制。
- **记忆容量与段长是超参**：记忆 token 数 $m$、段长 $K$、分段策略都需调（细节在附录 F/G），对新任务的迁移成本未充分讨论。
- **仍是离线 RL 设定**：依赖固定数据集，在线交互/探索场景下记忆机制的表现未验证。
- **可改进方向**：MRV 目前是单层交叉注意力门控，可探索更细粒度的 per-token 保留预算，或把记忆嵌入做成可扩展容量（随任务难度自适应分配记忆 token）。

## 相关工作与启发
- **vs Decision Transformer (DT)**：DT 用固定上下文窗口做回报条件序列建模，线索出窗即失忆；RATE 用段级循环 + 记忆嵌入 + MRV 把有效上下文扩到 $N\times K$，在 DT 失败的稀疏长程任务上成功，标准任务上仍持平。
- **vs RMT（Recurrent Memory Transformer）**：RATE 借用了 RMT 的记忆嵌入思想，但 RMT 朴素前向传记忆、无门控；RATE 加 MRV 做内容相关过滤，长程泛化更稳。
- **vs Transformer-XL (TrXL)**：TrXL 只有隐藏状态缓存（静态循环、无门控），实验显示它在 T-Maze 上和 DT 一样留不住稀疏线索；RATE 把缓存和门控记忆结合，互补取长。
- **vs Decision Mamba / SSM 类**：状态空间模型在稀疏长序列上曲线平、学不动；RATE 靠注意力+循环记忆兼顾插值与外推。

## 评分
- 新颖性: ⭐⭐⭐⭐ MRV（交叉注意力门控记忆更新）+ 双拷贝读写记忆嵌入是扎实的结构创新，并配了保留性定理
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 5 类记忆任务 + Atari/MuJoCo，对比十余个基线，含噪声破坏式消融与外推分析
- 写作质量: ⭐⭐⭐⭐ 机制讲解清晰、图示（注意力图）有力，理论部分稍密
- 价值: ⭐⭐⭐⭐ 给离线 RL 提供了一个统一、可长程记忆的高容量架构，思路可迁移

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Composition of Memory Experts for Diffusion World Models](composition_of_memory_experts_for_diffusion_world_models.md)
- [\[ICLR 2026\] The State of Reinforcement Finetuning for Transformer-based Agents](the_state_of_reinforcement_finetuning_for_transformer-based_agents.md)
- [\[ICLR 2026\] Peak-Return Greedy Slicing: Subtrajectory Selection for Transformer-based Offline RL](peak-return_greedy_slicing_subtrajectory_selection_for_transformer-based_offline.md)
- [\[ICLR 2026\] DEAS: DEtached value learning with Action Sequence for Scalable Offline RL](deas_detached_value_learning_with_action_sequence_for_scalable_offline_rl.md)
- [\[ICLR 2026\] Chunking the Critic: A Transformer-based Soft Actor-Critic with N-Step Returns](chunking_the_critic_a_transformer-based_soft_actor-critic_with_n-step_returns.md)

</div>

<!-- RELATED:END -->
