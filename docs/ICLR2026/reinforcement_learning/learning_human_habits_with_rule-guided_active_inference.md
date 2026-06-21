---
title: >-
  [论文解读] Learning Human Habits with Rule-Guided Active Inference
description: >-
  [ICLR 2026][强化学习][Active Inference] 把主动推理（active inference）扩展成"会养成习惯"的框架：用生物启发的 wake–sleep 算法在统一自由能目标下联合学习世界模型和符号规则，让 agent 在熟悉情境里用高置信度规则瞬时反应、在新奇情境里回退到 EFE 规划，从而更准更快地预测人类行为且产出可解释的"习惯"。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "Active Inference"
  - "Free Energy"
  - "Habit Learning"
  - "Symbolic Rules"
  - "Wake-Sleep"
  - "Neuro-Symbolic"
---

# Learning Human Habits with Rule-Guided Active Inference

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=FZXwkBH6s7](https://openreview.net/forum?id=FZXwkBH6s7)  
**代码**: [https://github.com/GongZhiren/human-action-active-inference](https://github.com/GongZhiren/human-action-active-inference)  
**领域**: 主动推理 / 序列决策 / 神经符号 / 人类行为建模  
**关键词**: Active Inference, Free Energy, Habit Learning, Symbolic Rules, Wake-Sleep, Neuro-Symbolic  

## 一句话总结
把主动推理（active inference）扩展成"会养成习惯"的框架：用生物启发的 wake–sleep 算法在统一自由能目标下联合学习世界模型和符号规则，让 agent 在熟悉情境里用高置信度规则瞬时反应、在新奇情境里回退到 EFE 规划，从而更准更快地预测人类行为且产出可解释的"习惯"。

## 研究背景与动机
**领域现状**：人类决策有两套互补系统——新奇情境靠目标导向的深思规划（基于世界模型模拟后果），熟悉情境靠刺激-反应的习惯捷径（绕过深思、快速行动）。主动推理（AIF）把大脑视为最小化自由能的预测机器，通过变分自由能（VFE）做感知推断、通过期望自由能（EFE）做前瞻规划，是一个生物合理、统一感知-学习-行动的优雅框架。

**现有痛点**：经典 AIF 几乎把行为都操作化为"每一步都做前瞻规划"，欠缺人类行为的三个关键要素——(i) 没有机制把反复成功压缩成紧凑可复用的带置信度规则；(ii) 没有原则性的模式切换（熟悉情境用即时规则、只在高不确定时才付出昂贵的 look-ahead）；(iii) 没有离线过程去巩固、剪枝、语义锚定这些规则。

**核心矛盾**：AIF 的 EFE 多步 rollout 随规划视野 $H$ 和动作空间指数级变贵，在熟悉情境下反复昂贵规划既不高效也不像人——而纯深度/逻辑/LLM 方法又要么是黑箱、要么规则静态后置、要么延迟高得离谱。

**本文目标**：在 control-as-inference 视角下拟合并解释人类（及类人）动作序列，让框架既能瞬时习惯反应又能保留灵活规划，同时产出可解释规则。

**核心 idea（rule-guided AIF + wake–sleep）**：把符号规则直接嵌入 AIF 的生成过程，用生物启发的醒-睡循环——**醒时**从真实经验里采集能稳定降低自由能的 state–intention–action 三元组作为候选规则；**睡时**用生成式 replay 巩固/剪枝/语义锚定这些规则。每条规则锚定在隐状态原型和可解释离散意图上，形成连接连续世界模型与符号决策的神经-符号单元。

## 方法详解

### 整体框架
方法把通用隐状态拆成 $Z_t=(S_t, m_t)$（连续外部世界状态 $S_t$ + 离散心理状态 $m_t$），在此之上定义"条件→动作"的符号规则，并用 wake–sleep 在统一总自由能目标下联合训练编码器、解码器（世界模型）和规则。决策时若有规则命中就走习惯捷径、否则回退 EFE 规划。

```mermaid
flowchart TD
    O[观测序列 O_t] --> ENC[编码器 q_ϑ S_t, m_t | H_t]
    ENC --> S[连续世界状态 S_t]
    ENC --> M[离散心理状态 m_t]
    S --> MATCH{规则命中?<br/>κ S_t,S*_r ≥ τ_r 且 m_t=m*_r}
    M --> MATCH
    MATCH -- 是 --> RULE[习惯策略: 规则动作 a_f<br/>瞬时·可解释]
    MATCH -- 否 --> EFE[EFE 规划 beam/MCTS<br/>多步 rollout]
    RULE --> ACT[混合策略 p_ϕπ a_t]
    EFE --> ACT
    ACT --> WAKE[Wake: 真实轨迹<br/>更新模型+采集新规则]
    WAKE --> SLEEP[Sleep: 生成式 replay<br/>巩固/剪枝/调置信度]
    SLEEP -.共享自由能目标.-> ENC
```

### 关键设计

**1. 隐状态二分：连续世界状态 + 离散心理状态——给规则一个可锚定的双重条件**。方法首先把原本笼统的隐变量 $Z_t$ 拆成 $Z_t=(S_t, m_t)$：$S_t\in\mathcal{S}$ 是连续低维的外部世界嵌入，负责精确重建观测；$m_t\in\{1,\dots,K\}$ 是离散心理状态，编码意图、模式或子目标（如"谨慎/激进/省能"）。生成模型据此改写为 $p_\phi(O_{1:T},S_{1:T},m_{1:T},a_{1:T})=p_\phi(S_1)p_\phi(m_1)\prod_t p_\phi(O_t|S_t)\,p_\phi(S_t|S_{t-1},a_{t-1})\,p_\phi(m_t|m_{t-1},S_t)\,p_{\phi_\pi}(a_t|S_t,m_t)$。其中 $m_t$ 演化更慢，充当"意图瓶颈"——这一步是后续规则能同时挂在"环境上下文"和"内部目标"上的前提，规则因而既上下文敏感又心理状态驱动，呼应了认知科学里习惯同时由情境和内在目标触发的观点。

**2. 锚定式符号规则与混合策略——把习惯写成可解释的条件-动作单元，命中即抄近路**。每条规则定义为锚定的条件-动作对 $f:(S^\star_f, m^\star_f)\Rightarrow a_f$，连续锚 $S^\star_f$ 是外部环境原型、$m^\star_f$ 指定意图模式、$a_f$ 是规定动作，并带置信度 $\rho_f\in[0,1]$；整个规则库可看成 context–action 对上的摊销混合模型，每条规则是一个原型分量。识别时用 MAP 估计快速匹配：规则 $r$ 在 $\kappa(S^{MAP}_t, S^\star_r)\ge\tau_r$ 且 $m^{MAP}_t=m^\star_r$ 时激活，其中 $\kappa$ 是高斯相似核（可解释为该规则在高斯混合下的后验责任度，$\tau_r$ 截断很小的责任度），软匹配让规则对噪声鲁棒。最终动作分布把规则先验和 EFE 规划融合成混合策略：

$$p_{\phi_\pi}(a_t|S_t,m_t)\propto \pi(a_t|S^{MAP}_t,m^{MAP}_t)+\bigl(1-\mathbb{1}_{\text{rule hit}}\bigr)\exp\bigl(-\tau\,\text{EFE}_t(a_t)\bigr)$$

可靠规则命中时其先验主导、直接执行习惯动作并绕过昂贵 rollout；否则落回多步 EFE 最小化做深思规划。这正对应大脑双系统——基底神经节里缓存的刺激-反应习惯 vs 前额叶/海马支撑的前瞻规划。

**3. 统一总自由能目标 + Wake–Sleep 联合学习——醒着采规则、做梦巩固规则**。训练把生成模型 $p_\phi$、推断网络 $q_\vartheta$、策略参数 $\phi_\pi$（含规则原型）放进同一个总自由能目标里联合优化：

$$\mathcal{F}_t(\phi,\vartheta,\phi_\pi)=\underbrace{\text{VFE}_t(O_t;\phi,\vartheta)}_{\text{拟合真实数据}}+\eta\,\underbrace{\text{EFE}_t(\phi,\phi_\pi)}_{\text{作用于 rollout}}+\gamma\,\underbrace{D_{KL}\!\bigl(q_\vartheta(m_{t-1}|H_{t-1})\,\|\,q_\vartheta(m_t|H_t)\bigr)}_{\text{心理状态一致性}}$$

KL 项是对离散心理状态的"黏性先验"，鼓励缓慢、可解释的模式切换。**Wake 阶段**在真实轨迹 $\mathcal{D}_{real}$ 上最小化自由能更新 $(\phi,\vartheta)$，同时"生长"规则：当三元组 $(S^{MAP}_t, m^{MAP}_t, a_t)$ 反复出现且自由能低时，要么新建规则、要么提升邻近规则置信度，连续锚按重加权质心 $S^\star_r\leftarrow\frac{\sum w(S)S}{\sum w(S)}$（$w(S)\propto\exp(-\text{VFE})$）更新，等价于混合模型上的 EM M-step。**Sleep 阶段**用 $p_\phi$ 生成 replay 轨迹，联合更新 $(\phi,\phi_\pi)$，在想象数据上巩固/剪枝规则、调整置信度。两阶段共享同一目标、只差数据源，恰似人类醒时更新模型、做梦时巩固记忆。工程上先 blockwise 预训练（只最小化 VFE）热启动世界模型，再跑完整 wake–sleep 循环。

## 实验关键数据

### 主实验表格
四个跨域数据集（NBA 球员轨迹 / 车辆跟驰 / DDXPlus 医疗诊断 / Atari-Berzerk 视觉博弈），Acc 报 Acc@1/3/5（%），Lat/CT 为延迟(ms)/收敛时间(h)：

| 类别 | 方法 | NBA Acc | NBA Lat/CT | Car-Follow Acc | DDXPlus Acc | Berzerk Acc |
|---|---|---|---|---|---|---|
| Logic | RNNLogic | 67.2/60.6/51.8 | 26.9/1.20 | 72.3/68.1/57.6 | 18.8/16.3/13.3 | 33.9/27.5/24.4 |
| Logic | STLR | 75.3/74.7/70.2 | 174/3.35 | 78.9/76.6/75.0 | 22.5/18.3/15.6 | 45.5/38.7/37.2 |
| DeepNN | Re-Net | 72.2/68.5/62.0 | 218/2.34 | 76.3/70.7/67.3 | 27.3/20.2/16.2 | 40.7/32.5/29.3 |
| AIF | DAI | 75.4/70.6/62.3 | 262/1.24 | 78.9/73.4/68.5 | 46.8/39.3/34.2 | 60.0/52.3/41.5 |
| AIF | DAI-MC | 82.3/80.6/76.5 | 387/1.52 | 84.5/82.9/80.3 | 57.2/52.2/43.7 | 66.8/58.2/48.2 |
| LLM | LaTee | 78.5/73.3/64.5 | 1244/4.65 | 82.4/74.8/71.8 | 28.2/22.1/20.4 | 62.2/54.2/49.3 |
| LLM | Qwen-0.5B | 71.3/64.2/56.4 | 2845/— | 74.9/68.3/62.2 | 24.9/19.6/17.4 | 58.4/51.3/46.2 |
| MBRL | DreamerV2 | 86.4/83.6/81.7 | 52.7/1.75 | 88.4/85.4/82.3 | 64.1/61.5/58.2 | 76.3/72.2/69.5 |
| **本文** | **Ours** | **97.0/91.3/85.7** | 35.9/2.59 | **96.8/95.9/94.2** | **79.6/73.6/68.1** | **85.6/77.2/72.4** |

四个域全面领先：NBA Acc@1 97.0%（vs 次优 DreamerV2 86.4%），Car-Following Acc@3 95.9%，DDXPlus（225 动作大空间）Acc@1 79.6% 远超 DreamerV2 的 64.1%。延迟上规则命中让 NBA 仅 35.9ms（DAI-MC 386ms、LaTee 1244ms），DDXPlus 159ms（LLM 基线高达 9.5万~12.6万 ms）。

### 消融实验表格
NBA 上规则数（RC）与精度/延迟的 Pareto 权衡（RHR = 规则命中率）：

| 规则数 RC | RHR | 趋势 |
|---|---|---|
| 0 | 0% | 纯规划，延迟最高 |
| 3 | 31.6% | 精度上升 |
| **6** | **39.9%** | **Acc@3/Acc@5 最优点** |
| 64 | 82.9% | 命中率高但精度回落 |
| 256 | 98.7% | 过拟合琐碎规则，精度下降 |

### 关键发现
- **规则加速推理、精度随规则数呈倒 U**：规则数增加使延迟单调下降（廉价规则触发替代昂贵规划），但精度先升后降——紧凑规则集（RC≈6）最优，过多规则会引入虚假/琐碎规则反而拖累。
- **稀有关键动作（HHAR）受益明显**：DDXPlus 225 个动作里规则包络能可靠捕捉低频但关键的诊断操作，HHAR 显著提升。
- **训练动态健康**：$\Delta F$、VFE、EFE、KL 普遍下降，世界模型重建与决策质量同步改善；规则可视化显示编码/解码空间能把动作语义（直行/传球/投篮等）解释性地组织起来。

## 亮点与洞察
- **把"习惯"做成一等公民**：以往 AIF 只把习惯当成 ad hoc 的旁支，本文用 wake–sleep 给出了习惯的获取（醒）、巩固/剪枝（睡）、与规划的元控制（混合策略）一整套机制，填上了经典 AIF 的三个缺口。
- **神经-符号桥接落到自由能上**：规则不是后置抽取或静态注入，而是直接嵌进生成过程、和隐状态/意图耦合、在统一自由能目标下动态更新，既可解释又生物合理。
- **快慢双系统的工程实现**：高置信规则命中即抄近路、否则回退 EFE，把认知科学的 basal ganglia vs PFC 双系统翻译成了一个可训练的混合策略，且实测同时拿到更高精度和更低延迟。
- **跨域泛化强**：从结构化序列（球员/车辆/诊断）到时序视觉（Atari）四个差异巨大的域共享同一套 $(S_t,m_t)$ 表示和训练计划，仍稳定领先。

## 局限与展望
- **连续锚不直接可读**：$S^\star_f$ 需要经世界模型 $p_\phi(O_f|S^\star_f)$ 解码回观测空间才能可视化，其"可解释性"是间接的；离散 $m_f$ 的语义标签很多时候还要靠 LLM 引导或人工锚定。
- **大动作空间延迟仍偏高**：DDXPlus 225 动作下绝对延迟 159ms 虽远低于 LLM 基线，但仍高于小动作空间的域，规则触发只是缓解而非根除规划成本。
- **混合模型是工程近似**：作者明确当前实现是对 $q(m_t)$ 和混合模型完整变分学习的"工程化近似"（EM 风格 M-step），完整概率视角留在附录，理论严谨性有提升空间。
- **离线/演示设定**：方法在固定 replay buffer 的离线设定下拟合人类轨迹（control-as-inference），不显式恢复奖励/偏好分布，迁移到在线交互或真实奖励驱动控制还需验证。
- **超参敏感**：$\eta,\gamma,\tau_r$、规则数 $K$ 等需要按域调，倒 U 形精度曲线说明规则库规模需要仔细控制，自动确定最优规则数是开放问题。

## 相关工作与启发
- **主动推理**：延续 Friston 系的 VFE/EFE 框架与 Fountas et al. (2020) 的 habit network / 摊销规划，但首次给习惯加上 wake–sleep 的生成-巩固闭环。
- **Wake–Sleep / 程序合成**：借鉴 Hinton et al. (1995) 的 wake-sleep 以及 DreamCoder（Ellis et al., 2023）/ Hewitt et al. (2020) 的"睡时巩固抽象"思路，把它用到规则库的生长与剪枝。
- **神经-符号与逻辑规则**：相对 RNNLogic、STLR、LogicMP 等静态/后置规则方法，本文把规则耦合进隐状态并在自由能下联合优化；与 Option-Critic（Bacon et al., 2017）的 options 也有概念呼应（规则≈时间扩展的习惯控制器）。
- **启发**：这套"快规则 + 慢规划 + 睡时巩固"的结构对具身智能、人机协作里需要兼顾实时性和可解释性的决策系统很有借鉴价值；用自由能统一规划与习惯，也给 control-as-inference 的行为建模提供了一个干净的范式。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 把符号规则+wake-sleep 巩固机制原则性地嵌入主动推理、统一在自由能目标下，是对经典 AIF 的实质性扩展，神经-符号桥接和快慢双系统的工程实现都很有想法。
- **实验充分度**: ⭐⭐⭐⭐ 四个差异巨大的跨域数据集 + 五类强基线（逻辑/深度/AIF/MBRL/LLM）+ 规则数 Pareto 消融 + 训练动态/规则可视化，覆盖全面；扣分在部分结果（Car-Following/DDXPlus 细节、K 敏感性）推到附录。
- **写作质量**: ⭐⭐⭐⭐ 动机清晰（三个缺口）、方法层次分明（隐状态拆分→规则→wake-sleep），认知科学类比贴切；公式密集、附录依赖较重，初读门槛偏高。
- **价值**: ⭐⭐⭐⭐ 兼顾预测精度、推理效率和可解释性，对人类行为建模、具身决策、人机协作有较强应用潜力，开源代码进一步加分。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)
- [\[ICLR 2026\] Toward Conservative Planning from Human-AI Preferences in Reinforcement Learning](toward_conservative_planning_from_human-ai_preferences_in_reinforcement_learning.md)
- [\[NeurIPS 2025\] ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition](../../NeurIPS2025/reinforcement_learning/aline_joint_amortization_for_bayesian_inference_and_active_data_acquisition.md)
- [\[ICLR 2026\] Using Reinforcement Learning to Train Large Language Models to Explain Human Decisions](using_reinforcement_learning_to_train_large_language_models_to_explain_human_dec.md)
- [\[ICLR 2026\] Learning What Matters Now: Dynamic Preference Inference under Contextual Shifts](learning_what_matters_now_dynamic_preference_inference_under_contextual_shifts.md)

</div>

<!-- RELATED:END -->
