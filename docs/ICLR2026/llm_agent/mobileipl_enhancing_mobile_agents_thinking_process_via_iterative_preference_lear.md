---
title: >-
  [论文解读] MobileIPL: Enhancing Mobile Agents Thinking Process via Iterative Preference Learning
description: >-
  [ICLR 2026][LLM Agent][Mobile GUI Agent] 针对移动 GUI agent 缺乏 CoaT 推理轨迹、又难以做步级标注的痛点，MobileIPL 用 MCTS 式迭代采样搭一棵 CoaT-tree，靠规则奖励给叶节点打分并回传到中间思考步，构造"思考级" DPO 对（T-DPO）来优化推理过程，从而在三个移动 GUI 基准上超越 OS-ATLAS、UI-TARS 等连续预训练大模型。
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "Mobile GUI Agent"
  - "CoaT"
  - "迭代偏好学习"
  - "Thinking-level DPO"
  - "规则奖励"
  - "指令进化"
---

# MobileIPL: Enhancing Mobile Agents Thinking Process via Iterative Preference Learning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=PS8iu4PxKz](https://openreview.net/forum?id=PS8iu4PxKz)  
**代码/数据**: [MobileIPL-dataset (HuggingFace)](https://huggingface.co/datasets/xwk123/MobileIPL-dataset)  
**领域**: Mobile GUI Agent / 偏好学习  
**关键词**: Mobile GUI Agent, CoaT, 迭代偏好学习, Thinking-level DPO, 规则奖励, 指令进化  

## 一句话总结
针对移动 GUI agent 缺乏 CoaT 推理轨迹、又难以做步级标注的痛点，MobileIPL 用 MCTS 式迭代采样搭一棵 CoaT-tree，靠规则奖励给叶节点打分并回传到中间思考步，构造"思考级" DPO 对（T-DPO）来优化推理过程，从而在三个移动 GUI 基准上超越 OS-ATLAS、UI-TARS 等连续预训练大模型。

## 研究背景与动机
**领域现状**：VLM 驱动的移动 GUI agent 需要在用户指令和当前界面之间生成中间思考。AITZ 提出的 Chain of Action-Planning Thoughts（CoaT，类 System-2 慢思考）范式已被证明能显著提升 GUI 任务的推理表现，把"描述—思考—决策—定位"组织成结构化的思维链。

**现有痛点**：（1）高质量的多样 CoaT 轨迹稀缺，直接在固定 CoaT 轨迹上做 SFT 容易过拟合，模型被困在僵化的推理模式里；（2）自训练虽能缓解数据稀缺，但只用最终答案正确性作奖励会忽略中间推理步质量，导致 reward hacking；（3）像 ReST-MCTS 这类搜索方法靠训练过程奖励模型（PRM）来评分中间步，但 PRM 需要大规模人工步级标注——而在移动 GUI 领域，环境依赖真机或模拟器，步级标注成本和人力远比文本/代码/数学任务高。

**核心矛盾**：既想优化"中间思考步"的质量（而非只看最终动作对错），又不想付出 PRM 那种昂贵的步级人工标注代价。

**本文目标**：在无需 PRM、无需步级人工标注的前提下，对移动 GUI agent 的整个思考过程做细粒度偏好优化，同时缓解 warm-up SFT 阶段的过拟合。

**核心 idea**：**用规则奖励替代 PRM** —— 把每个动作展开成 CoaT-tree，叶节点（完整动作）用规则奖励与真值比对打分，再把分数沿树回传给中间思考步，据此自动构造思考级 DPO 对；同时用 **GPT-4o 三阶段指令进化** 在 SFT 阶段注入多样化 Q&A，防止过拟合并增强 UI 理解。

## 方法详解

### 整体框架
MobileIPL 分三段：先用"指令进化"扩充的混合数据做 warm-up SFT 得到种子策略；再对每个动作迭代采样构建 CoaT-tree、用规则奖励给叶节点打分并回传得到中间步价值；最后从树中过滤对比对做 Thinking-level DPO 自训练，并把更新后的 agent 当作新基座继续迭代直到性能瓶颈。

```mermaid
flowchart LR
    A[通用 VLM] --> B[指令进化<br/>GPT-4o 三阶段 Q&A]
    B --> C[Warm-up SFT<br/>混合 T+Q 得种子策略 πS0]
    C --> D[CoaT-tree 迭代采样<br/>每步采 K 个续写]
    D --> E[叶节点规则奖励 v·st·<br/>+ 回传到中间步]
    E --> F[对比数据过滤<br/>α/β/γ 分类构造 DPO 对]
    F --> G[Thinking-level DPO 训练]
    G -->|更新 agent 作新基座| D
```

### 关键设计

**1. 多轮思考过程建模：把一个动作拆成四轮对话。** 作者把 CoaT 推理的每个动作 $\hat{a}_i$ 形式化为多轮对话 $\hat{a}_i=[s_1,s_2,s_3,s_4]$，分别对应描述、动作思考、动作决策、坐标定位，即 $s_1=\text{Description}(P_1,u_i)$、$s_2=\text{Thought}(P_2,u_i,I,\hat{a}_{0:i-1},s_1)$、$s_3=\text{Action}(\cdot)$、$s_4=\text{Grounding}(\cdot)$。这样拆分的动机很实际：当模型一口气解码整段推理时，图像模态 $u$ 占据了绝大多数输入 token，会压过文本指令 $I$ 和动作历史，使模型注意力偏离文本细节；多轮对话强制每一步只聚焦当前推理子任务，既平衡了跨模态 token 比例，又保证最终一定能产出符合格式要求的答案。

**2. CoaT-tree 迭代采样 + 规则奖励：用规则替代 PRM 给中间步打分。** 沿 CoaT 范式逐步采样，每一步 $t$ 采 $K$ 个续写 $\hat{s}_t=\{\hat{s}_t^{(k)}\mid \hat{s}_{0:t-1}\}_{k=1}^K$，自然形成一棵树。叶节点（完整动作）与真值 $a^*$ 比对算规则奖励：完全正确得 1，动作类型匹配则给 $v_{type}+\text{score}_{match}$，否则为 0。其中 $\text{score}_{match}$ 对 CLICK 用预测与真值坐标的归一化距离 $d(x,y)$ 做平滑奖励（距离越近分越高），对 INPUT 用文本 F1 做平滑奖励（重叠越多分越高），把"差一点点"和"完全错"区分开。关键一步是把叶节点价值**反向回传**给中间步：$v(s_{t-1})=c\cdot\frac{1}{K}\sum_{k=1}^{K}v(s_t^{(k)})$，$c$ 是折扣因子，于是每个中间思考步都拿到了一个无需 PRM、无需人工标注的连续价值。

**3. 对比数据过滤：按树的质量分 α/β/γ 三类挑正负样本。** 根据一棵采样树中叶节点价值的分布，把树分成三类：$\alpha$ 是所有叶节点价值都为 1 的"完美树"（稳定输出正确思考+动作，不用于构造对比对）；$\beta$ 是既有正确又有错误叶节点的"潜在正确树"，是构造对比对的主力；$\gamma$ 是全部叶节点都不为 1 的"待精炼树"。在 $\beta$ 中取价值为 1 且动作类型尽量多样的作正样本，构造 $\beta_{pairs}=\langle \hat{s}_t^{(k)}\uparrow,\hat{s}_t^{(k')}\downarrow\mid v(\hat{s}_t^{(k)})-v(\hat{s}_t^{(k')})>1/K\rangle$（要求正负样本价值差超过 $1/K$ 才成对，避免噪声对）；在 $\gamma$ 中直接用真值动作 $a^*$ 作正样本、采样输出作负样本，$\gamma_{pairs}=\langle a^*\uparrow,\hat{s}_t^{(k)}\downarrow\rangle$。

**4. Thinking-level DPO（T-DPO）+ 迭代自训练。** 在相同前缀思考 $s_{1:t-1}$ 条件下，对同一思考步的正负续写 $s_t^+,s_t^-$ 做 DPO 比较：$L_{\text{T-DPO}}=-\mathbb{E}\big[\log\sigma(\beta\log\frac{\pi_\theta(s_t^+\mid s_{1:t-1})}{\pi_{ref}(s_t^+\mid s_{1:t-1})}-\beta\log\frac{\pi_\theta(s_t^-\mid s_{1:t-1})}{\pi_{ref}(s_t^-\mid s_{1:t-1})})\big]$，并配合 SFT loss 一起优化。与 TreePO/SPO 把长序列切成大量短片段不同，MobileIPL 用固定的 CoaT-tree 建模思考、价值直接由规则奖励算出（不依赖不稳定的 PRM），采样和训练都更高效。优化后把新 agent 当基座继续采集对比对再训，迭代直到性能瓶颈。

**5. 三阶段指令进化：缓解 SFT 过拟合并增强 UI 理解。** warm-up SFT 后 CoaT 模式固定、输出多样性差，作者用 GPT-4o 基于真实手机界面截图生成三层 Q&A：Level I 通用 GUI Q&A（定位/指代/页面描述，强化基础能力）、Level II 控件功能与嵌套关系（避免 agent 把 textview 当 button 误点）、Level III 高级 FAQ（页面结构框架、对控件交互后导航结果的预期与预测）。生成的 Q&A 经人工过滤后与原轨迹 $T$ 混合做 warm-up SFT，既防止过拟合静态指令、又通过视觉接地的问答提升对 UI 布局的理解，使采样空间从 4% 扩到 31%、含正确答案的采样比例从 72.7% 升到 77.9%。

## 实验关键数据

### 主实验表格
在 AITZ、AMEX、AndroidControl 三个移动 GUI 基准上以 Qwen2-VL-7B 为骨干，Step.Acc 为主指标：

| 基准 | 关键对比 | MobileIPL | 最强基线 |
|------|----------|-----------|----------|
| AITZ (Total) | vs Falcon-UI-7B(3M GUI 预训练) | **69.15** | 69.10 |
| AITZ (Total) | vs 种子模型 / Qwen2-VL-7B | 69.15 | 55.40 / 60.36 |
| AMEX (Overall) | vs SphAgent-7B(SOTA) | **74.29** | 70.71 (+3.58) |
| AMEX (Overall) | vs OS-Atlas / UI-Tars | 74.29 | 70.33 / 70.33 (+3.96) |
| AndroidControl (Step.Acc) | vs UI-Tars-7B / OS-Atlas | **72.7** | 72.5 / 71.2 |
| AndroidControl (Grounding) | vs Qwen2-VL-7B(SFT) | **77.0** | 68.5 (+8.5) |

AndroidControl OOD 子集（IDD / app-unseen / task-unseen）：MobileIPL 取 **73.6 / 70.0 / 72.2**，全面超过 OS-Atlas（71.2 / 60.7 / 66.2，OOD 掉得明显）和同为自训练的 Qwen2-VL-GRPO（70.2 / 68.1 / 69.7），展示更强泛化、更小的域外退化。

### 消融实验表格
AITZ 第一轮（MobileIPL-R1 = 65.4）消融：

| 设置 | Total | Δ |
|------|-------|----|
| MobileIPL-R1（完整） | 65.4 | — |
| − IPL（只 SFT） | 60.4 | −5.0 |
| − 指令进化 Evo | 62.9 | −2.5 |
| − IPL 负样本 | 61.4 | −4.0 |
| − IPL + Naive DPO（整轨迹） | 60.3 | −5.1 |
| 1/2 训练数据（R1） | 64.8 | −0.6 |
| 4/5 训练数据（R2） | 60.6 | −4.8 |

### 关键发现
- **IPL 是核心**：去掉 IPL 只做 SFT 掉 5.0%；负样本贡献 4.0%，说明负样本教会模型"怎么推理"而非记忆。
- **思考级 DPO 优于整轨迹 Naive DPO**：在整条轨迹上做 naive DPO 反而掉到 60.3%，比仅用 CoaT-tree 正样本的 SFT 还低 1.1%，验证 CoaT-tree 采样 + 思考过程优化的有效性。
- **低资源依旧有效**：只用一半数据，第一轮 IPL（64.8）就超过原始 CoaT-SFT 和 naive DPO 的最好结果。
- **效率-性能权衡更好**：MobileIPL 准确率 69.15、每句约 27 次 rollout，优于 SPO-Chain（68.03，约 54 次）和 GRPO（66.29，8 次）。
- **采样数 K=3 性价比最高**：K 从 3 增到 4 会让树节点从 27 暴增到 64，但提升不到 1%。

## 亮点与洞察
- **用规则奖励 + 回传替代 PRM**：在移动 GUI 这个步级标注极贵的场景下，绕开了不稳定且需大量标注的过程奖励模型，把"中间思考步该优化哪一步"变成可由 CLICK 距离 / INPUT F1 平滑算出的连续信号。
- **α/β/γ 三分法是个干净的数据工程抽象**：把"完美树不用学、混合树挖对比对、全错树用真值兜底"讲得很清楚，且要求正负价值差 $>1/K$ 来过滤噪声对。
- **多轮对话的动机抓住了 VLM 的真实痛点**：图像 token 淹没文本指令导致注意力偏移，拆成四轮强制聚焦——这是 GUI agent 区别于纯文本 CoT 的关键观察。
- **小模型打赢大预训练模型**：用远少于 OS-Atlas/UI-TARS/Falcon-UI 的数据，7B 骨干达到甚至超越百万级 GUI 预训练，证明"优化思考过程"比"堆预训练数据"更省。

## 局限与展望
- **PRESS 动作偏弱**：AITZ 上 MobileIPL 的 PRESS 准确率明显低于一些基线（消融表中 R1 仅 23.5），作者归因于训练后该类动作样本分布问题，留待 Appendix 讨论。
- **奖励规则手工设计**：CLICK/INPUT 的平滑奖励依赖人工设定的 $v_{type}/v_{format}$ 与距离归一化，对其他动作类型（SCROLL/STOP 等）只给 0/1，奖励信号不够细。
- **依赖 GPT-4o + 人工过滤生成指令进化数据**：三阶段 Q&A 仍需闭源大模型和人工评估，规模化和复现成本存在门槛。
- **迭代终止靠"性能瓶颈"经验判断**：缺乏明确的收敛准则，迭代轮数 R 需要调参确定。

## 相关工作与启发
- **CoaT / AITZ**：思考-决策-定位三元组的慢思考范式是本文的建模基座。
- **ReST-MCTS\* / Xie et al.**：用 MCTS + PRM 评估步级质量的代表，本文正是要规避其 PRM 标注成本。
- **ReFT / DigiRL / DistRL / ReachAgent / TCPO**：自训练与 DPO 在 GUI agent 的探索；TCPO 同样优化思考但不强制思考-动作一致性。
- **TreePO / TreeRL / SPO**：把长序列切短片段做偏好优化，计算成本高、数据效率低，MobileIPL 用固定 CoaT-tree + 规则价值在效率上更优。
- **启发**：在标注昂贵的具身/GUI 场景，"规则奖励 + 树形回传"是 PRM 的低成本平替；把推理拆成多轮对话来平衡多模态 token 比例，可迁移到其他图像主导的 agent 任务。

## 评分
- **新颖性**: ⭐⭐⭐⭐ —— "规则奖励替代 PRM + 思考级 DPO + α/β/γ 数据过滤"组合清晰，CoaT-tree 回传给中间步赋值的思路在 GUI 场景有实用创新，但底层仍是 MCTS 采样 + DPO 的拼装。
- **实验充分度**: ⭐⭐⭐⭐ —— 三基准 + OOD + 低资源 + rollout 效率 + 参数搜索，消融完整，对比了 GRPO/SPO-Chain 等强 RL 基线；PRESS 弱项也坦诚分析。
- **写作质量**: ⭐⭐⭐⭐ —— 公式与算法表述完整，动机交代清楚；指令进化与采样部分图文配合好，个别符号略密。
- **价值**: ⭐⭐⭐⭐ —— 用小数据小模型超越百万级 GUI 预训练，对资源受限的移动 agent 落地有直接参考价值，数据集已开源。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [\[ICLR 2026\] MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents](mobilerl_online_agentic_reinforcement_learning_for_mobile_gui_agents.md)
- [\[ICML 2026\] From Player to Master: Enhancing Test-Time Learning of LLM Agents via Reinforcement Learning over Memory](../../ICML2026/llm_agent/from_player_to_master_enhancing_test-time_learning_of_llm_agents_via_reinforceme.md)
- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)
- [\[ICLR 2026\] Aria: an Agent for Retrieval and Iterative Auto-Formalization via Dependency Graph](aria_an_agent_for_retrieval_and_iterative_auto-formalization_via_dependency_grap.md)

</div>

<!-- RELATED:END -->
