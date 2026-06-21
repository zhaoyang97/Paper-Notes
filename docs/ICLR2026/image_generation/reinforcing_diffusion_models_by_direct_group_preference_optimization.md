---
title: >-
  [论文解读] Reinforcing Diffusion Models by Direct Group Preference Optimization
description: >-
  [ICLR 2026][图像生成][扩散模型] 本文提出 DGPO（Direct Group Preference Optimization），把 GRPO 的"群内相对偏好"思想从 policy-gradient 框架里解耦出来，让扩散模型可以直接用高效的确定性 ODE 采样器做在线 RL 后训练，在 GenEval 上把 SD3.5-M 从 0.63 提到 0.97，且训练比 Flow-GRPO 快约 20×（GenEval 上近 30×）。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "扩散模型"
  - "偏好优化"
  - "GRPO"
  - "ODE 采样"
  - "后训练"
---

# Reinforcing Diffusion Models by Direct Group Preference Optimization

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=jymuXl8GYi](https://openreview.net/forum?id=jymuXl8GYi)  
**代码**: https://github.com/Luo-Yihong/DGPO  
**领域**: 扩散模型 / 对齐RLHF  
**关键词**: 扩散模型, 偏好优化, GRPO, ODE 采样, 后训练

## 一句话总结
本文提出 DGPO（Direct Group Preference Optimization），把 GRPO 的"群内相对偏好"思想从 policy-gradient 框架里解耦出来，让扩散模型可以直接用高效的确定性 ODE 采样器做在线 RL 后训练，在 GenEval 上把 SD3.5-M 从 0.63 提到 0.97，且训练比 Flow-GRPO 快约 20×（GenEval 上近 30×）。

## 研究背景与动机
**领域现状**：RL 后训练已经是大语言模型对齐的标配，其中 GRPO（Group Relative Policy Optimization）靠"为每个 prompt 采一组输出、用组内归一化算 advantage"的方式极大提升了 LLM 的推理能力。自然地，社区也想把这套搬到扩散模型上，用来对齐人类偏好、提升构图/计数/文本渲染等复杂指标。

**现有痛点**：GRPO 的 policy-gradient 框架要求有一个**随机策略**（stochastic policy）才能做探索和重要性采样。LLM 本身就输出词表上的概率分布，天然满足；但扩散模型为了在质量和成本间取得平衡，主流用的是**确定性 ODE 采样器**，根本不提供随机策略。为了凑出随机性，已有工作（Flow-GRPO 等）被迫改用 SDE 采样来注入条件高斯噪声。这个妥协带来三个严重后果：(1) 固定算力预算下 SDE 采样比 ODE 更低效、样本质量更差；(2) 随机性来自**与模型无关的高斯噪声**，在高维空间里探索信号极弱，收敛慢；(3) 训练要在**整条采样轨迹**上做，每次迭代都很贵。

**核心矛盾**：扩散模型的高效采样器（ODE，确定性）和 GRPO 框架要求的随机策略之间存在根本冲突——你想用快的采样器，就拿不到 policy-gradient 需要的随机策略。

**切入角度**：作者的关键论断是——GRPO 之所以成功，**靠的不是 policy-gradient 这个形式，而是它能利用组内细粒度的相对偏好信息**。如果这个判断成立，那理想的扩散 RL 方法应该保留"群级相对信息"这一精华，同时彻底丢掉随机策略及其副作用。

**核心 idea**：用"直接学群级偏好"代替"policy-gradient"——为每个 prompt 用 ODE 采一组样本，按 reward 分成正负两组，直接最大化"正组优于负组"的群级偏好似然，从而既保留了 GRPO 的组内相对信息，又解锁了确定性 ODE 采样和更快的训练。可以把 DGPO 理解为：把 DPO 扩展到群级信息，或者说是 GRPO 的"扩散原生"重写版。

## 方法详解

### 整体框架
DGPO 是一个在线 RL 算法，每轮迭代做一件事：拿当前模型（或其 EMA 版本）对一个 prompt 用高效 ODE 采样器生成一组 $G$ 个样本，用 reward 模型给每个样本打分并做组内归一化得到 advantage，按 advantage 正负把组拆成"好样本组 $G^+$"和"坏样本组 $G^-$"，再用 Bradley-Terry 模型直接优化"$G^+ \succ G^-$"这个群级偏好。整个过程**不需要随机策略、不需要在整条轨迹上训练、不需要处理棘手的配分函数 $Z(c)$**——这三点正是它比 Flow-GRPO 快一个数量级的原因。

DGPO 的核心难点在于：群级偏好目标里天然带着一个不可解的配分函数 $Z(c)$，必须靠精心设计样本权重把它消掉。整条 pipeline 如下：

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Prompt c"] --> B["ODE 确定性采样<br/>生成一组样本 G"]
    B --> C["Reward 打分<br/>+ 组内归一化得 advantage"]
    C --> D["按 advantage 正负<br/>拆成 G+ / G-"]
    D --> E["Advantage 权重设计<br/>w=|A|，消去配分 Z(c)"]
    E --> F["群偏好优化<br/>最大化 σ(R(G+)-R(G-))"]
    F -->|Timestep Clip：只采 t∈[t_min,T]| G["更新 θ，θ⁻←EMA"]
    G -->|下一轮在线 rollout| B
```

### 关键设计

**1. 直接群偏好优化：抛弃 policy-gradient，直接学"好组 ≻ 坏组"**

针对 GRPO 必须依赖随机策略、而 DPO 又只能用成对样本的痛点，DGPO 选择直接对**群级偏好**用 Bradley-Terry 模型做极大似然：

$$\max_\theta\ \mathbb{E}_{(G^+,G^-,c)\sim D}\ \log \sigma\big(R_\theta(G^+|c) - R_\theta(G^-|c)\big)$$

其中群级 reward 被参数化为组内各样本 reward 的**加权和** $R_\theta(G|c)=\sum_{x_0\in G} w(x_0)\cdot r_\theta(c,x_0)$，这样就能把每个样本的细粒度信息都吃进来，而不像 Diffusion-DPO 那样只能成对比较。单样本 reward 仍沿用 DPO 的隐式参数化 $r_\theta(c,x_0)=\beta\,\mathbb{E}\log\frac{p_\theta(x_{0:T}|c)}{p_{\text{ref}}(x_{0:T}|c)}+\beta\log Z(c)$（用前向扩散 $q(x_{1:T}|x_0)$ 近似昂贵的反演链）。和 GRPO 的本质区别在于：DGPO 把"组内相对信息"这一精华直接写进偏好目标，而不是借道 policy-gradient，因此无需随机策略、也无需在整条轨迹上 rollout。

**2. Advantage 权重设计：用 $w=|A|$ 同时消掉配分函数并放大难样本**

上一个目标展开后会留下一项 $\sum_{x_0\in G^+}w(x_0)Z(c)-\sum_{x_0\in G^-}w(x_0)Z(c)$，而 $Z(c)$ 不可解，必须让它为零。作者要求权重满足两个条件：好/坏样本权重越大越对应"更好的正样本、更坏的负样本"；且两组权重之和相等 $\sum_{G^+}w=\sum_{G^-}w$。巧妙之处在于直接复用 GRPO 式的归一化 advantage：

$$A(x_0^i)=\frac{r_i-\text{mean}(\{r_j\})}{\text{std}(\{r_j\})},\qquad G^+=\{A>0\},\ G^-=\{A\le0\},\qquad w(x_0)=|A(x_0)|$$

由于归一化 advantage **零均值**，正组和负组的 $|A|$ 之和天然相等，配分项自动抵消；同时 $|A|$ 让偏离均值越远的样本权重越大，模型能更有效地学到组内相对偏好关系。代入后目标化简为只含 $\log\frac{p_\theta}{p_{\text{ref}}}$ 的干净形式，再用 Jensen 不等式把期望移到外面，最终落到一个**去噪分数匹配差**的训练目标：

$$L_{\text{DGPO}}\triangleq-\mathbb{E}\,\log\sigma\Big(-\lambda_t\beta T\big(\textstyle\sum_{x\in G^+}w(x)[L^\theta_{\text{dsm}}-L^{\theta_{\text{ref}}}_{\text{dsm}}]-\sum_{x\in G^-}w(x)[L^\theta_{\text{dsm}}-L^{\theta_{\text{ref}}}_{\text{dsm}}]\big)\Big)$$

其中 $L^\theta_{\text{dsm}}(x,x_t,c)=\|f_\theta(x_t,t,c)-x\|_2^2$ 就是普通的去噪损失。这意味着 DGPO 训练时只需在**单个时间步**上算去噪损失之差，而非遍历整条轨迹，单次迭代成本大幅下降。

**3. ODE 确定性 rollout：用高质量采样换更强的学习信号**

既然 DGPO 不再需要随机策略，rollout 阶段就可以放心用确定性 ODE 采样器，而不必像 Flow-GRPO 那样被迫用 SDE。同样的推理预算下，ODE 能产出质量和 reward 都更高的样本，模型因此从"更干净的训练数据"里学习，收敛更快、上限更高。消融（Fig.5）显示在线 DGPO 用 ODE rollout 在收敛速度和最终指标上都明显优于 SDE rollout，作者据此推断：以往工作里用 SDE 其实是 policy-gradient 框架的硬性要求，而非真的能提供更有用的多样性。

**4. Timestep Clip 策略：避免过拟合 few-step 样本的伪影**

在线设置要求实时从当前模型采样，为省成本只跑很少步数（如 10 步）来生成样本。但 few-step 样本质量较差（模糊等伪影），直接在所有时间步上训练会让模型把这些伪影也学进去，导致视觉质量严重退化。对策很简单：训练时只从 $[t_{\min}, T]$（$t_{\min}>0$）这个区间采时间步，跳过最接近数据端、最容易暴露 few-step 伪影的小 $t$。消融显示去掉该策略后 OCR 指标只从 0.96 微降到 0.95，但视觉质量明显变差——说明它主要守的是"质量不崩"而非奖励数值。

### 损失函数 / 训练策略
最终目标即上文 $L_{\text{DGPO}}$（Eq.17）。训练循环（Algorithm 1）：采 prompt → 用 $p_{\theta^-}$ 生成组 $G$ → 算 reward 与归一化 advantage → 按符号拆 $G^+/G^-$ → 采 $t\sim U[t_{\min},T]$、$\epsilon\sim\mathcal N(0,I)$（组内**共享同一噪声** $\epsilon$ 以降方差）→ 算 $L_{\text{DGPO}}$ 并更新 $\theta$ → 在线模型 $\theta^-\leftarrow\theta$ 或 EMA $\theta^-\leftarrow\mu\theta^-+(1-\mu)\theta$。关键超参：组大小 $G$、正则强度 $\beta$、最小时间步 $t_{\min}$、EMA 衰减 $\mu$。

## 实验关键数据

### 主实验
在 SD3.5-M 上做三类任务后训练：构图生成（GenEval）、视觉文本渲染（OCR）、人类偏好对齐（PickScore）。

| 任务/指标 | SD3.5-M 基线 | Flow-GRPO | DGPO（本文） |
|-----------|--------------|-----------|--------------|
| GenEval Overall | 0.63 | 0.95 | **0.97** |
| GenEval Counting | 0.50 | 0.95 | **0.97** |
| GenEval Attr. Binding | 0.52 | 0.86 | **0.91** |
| OCR 文本渲染 Acc. | 0.59 | 0.92 | **0.96** |
| 人类偏好 PickScore | 21.72 | 23.31 | **23.89** |

GenEval 上 DGPO 0.97 超过 GPT-4o（0.84）和 Flow-GRPO（0.95），且训练近 30× 快于 Flow-GRPO；OCR 任务约 19× 快、PickScore 任务约 17× 快，整体约 20×。

### 域外指标（防 reward hacking）
在 DrawBench 上用四个训练时未用过的质量指标检验是否过拟合奖励信号：

| 任务 | 方法 | Aesthetic | DeQA | ImageReward | UnifiedReward |
|------|------|-----------|------|-------------|---------------|
| 构图生成 | Flow-GRPO | 5.25 | 4.01 | 1.03 | 3.51 |
| 构图生成 | DGPO | **5.31** | **4.03** | **1.08** | **3.60** |
| 人类偏好 | Flow-GRPO | 5.92 | 4.22 | 1.28 | 3.66 |
| 人类偏好 | DGPO | **6.08** | **4.40** | **1.32** | **3.74** |

DGPO 在拉高目标指标的同时，几乎所有域外质量指标都不降反升，说明它没有靠牺牲真实画质来刷奖励。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| DGPO（完整） | OCR 0.96 | ODE rollout + Timestep Clip |
| w/o Timestep Clip | OCR 0.95 | 指标微降，但视觉质量明显退化（Fig.4） |
| DGPO w/ SDE rollout | 收敛与上限均下降 | 换成 SDE 采样后明显变差 |
| Online DPO | 明显低于 DGPO | 只能成对比较，丢失组内细粒度信息 |
| Offline DGPO | 优于基线但弱于在线 | 用 $p_{\text{ref}}$ 离线造数据也能涨，但不如在线 |

### 关键发现
- **ODE 优于 SDE** 是核心结论之一：去掉 policy-gradient 后用确定性 ODE 反而更快更强，反证了以往用 SDE 是框架所迫而非真有益。
- **群级 > 成对**：DGPO 在在线和离线设置下都明显超过 DPO，验证了"利用组内细粒度相对信息"才是关键。
- Timestep Clip 主要保护视觉质量，对奖励数值影响很小，但不加会让画面崩坏。

## 亮点与洞察
- **把"GRPO 的精华"和"policy-gradient 的形式"解耦**：作者敏锐地指出 GRPO 成功靠的是组内相对偏好而非策略梯度，这个 reframe 直接打开了用 ODE 采样的大门，是全文最"啊哈"的地方。
- **用 $w=|A|$ 一箭双雕**：归一化 advantage 的零均值性质让正负组权重自动配平，从而干净地消掉了不可解的配分函数 $Z(c)$——既解决了数学障碍，又顺带给难样本加权，设计极其经济。
- **最终目标退化成"去噪损失之差"**：复杂的群偏好目标最后落到单步去噪损失差，工程上几乎零额外实现成本，可直接迁移到任何 flow-matching/扩散后训练管线。
- 把 DGPO 看作"DPO + 组信息"或"GRPO 的扩散原生版"，这个双向类比让它在 RLHF 谱系里定位清晰，思路可迁移到其他需要群级偏好的生成任务。

## 局限与展望
- 实验集中在 SD3.5-M 单一基座和三类任务上，未在更大模型（如 SD3.5-L、FLUX）或视频扩散上验证可扩展性。
- 群级 reward 仍依赖外部 reward 模型的质量；若 reward 模型本身有偏，DGPO 会把偏差放大到组级偏好里，域外指标的"看似不降"也只在所用四个指标上成立。
- $t_{\min}$ 等超参带有任务相关性（防 few-step 伪影），换采样步数或换基座时可能需要重调。
- 离线 DGPO 明显弱于在线版，说明方法的收益很大程度依赖在线 rollout，部署到完全离线/固定数据场景时收益会打折。

## 相关工作与启发
- **vs Flow-GRPO（GRPO-style）**：二者都想把 GRPO 搬到扩散模型，但 Flow-GRPO 为满足随机策略被迫用 SDE 采样并在整条轨迹上训练；DGPO 直接学群级偏好、用 ODE 采样、只训单步，因此快约 20× 且质量更好。
- **vs Diffusion-DPO**：DPO 同样不需随机策略，但受不可解配分函数限制只能成对比较，无法利用每个样本的细粒度奖励；DGPO 用群级 reward + advantage 权重消去配分，从而吃下组内全部相对信息。
- **vs 枚举式群 DPO（并发工作）**：有并发工作通过枚举组内所有成对比较来引入群信息；DGPO 则定义单一群级 reward 并对其做极大似然，形式更简洁、计算更省。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 GRPO 重新诠释为"群级相对偏好"并据此抛弃 policy-gradient，是少见的范式级 reframe
- 实验充分度: ⭐⭐⭐⭐ 三任务 + 域外指标 + ODE/SDE/DPO/离线多组消融较扎实，但基座与规模偏单一
- 写作质量: ⭐⭐⭐⭐⭐ 动机—矛盾—方法推导链条清晰，配分函数消去的设计讲得很透
- 价值: ⭐⭐⭐⭐⭐ 20× 提速 + SOTA 质量，对扩散模型 RLHF 后训练有很强的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] ViPO: Visual Preference Optimization at Scale](vipo_visual_preference_optimization_at_scale.md)
- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](../../NeurIPS2025/image_generation/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](../../CVPR2025/image_generation/curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[ICLR 2026\] Consis-GCPO: Consistency-Preserving Group Causal Preference Optimization for Vision Customization](consis-gcpo_consistency-preserving_group_causal_preference_optimization_for_visi.md)
- [\[ICLR 2026\] Towards Better Optimization for Listwise Preference in Diffusion Models](towards_better_optimization_for_listwise_preference_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
