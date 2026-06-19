---
title: >-
  [论文解读] G-UBS: Towards Robust Understanding of Implicit Feedback via Group-Aware User Behavior Simulation
description: >-
  [AAAI 2026][强化学习][隐式反馈] 提出 G-UBS（Group-aware User Behavior Simulation）范式，通过用户群组管理器（UGM）基于 LLM 的"总结-聚类-反思"流程生成群组画像，结合用户反馈建模器（UFM）的群组感知强化学习训练，在隐式反馈噪声下实现鲁棒的用户行为理解，同时构建了首个多模态隐式反馈视频推荐基准 IF-VR。
tags:
  - "AAAI 2026"
  - "强化学习"
  - "隐式反馈"
  - "用户行为模拟"
  - "群体感知"
  - "推荐系统"
---

# G-UBS: Towards Robust Understanding of Implicit Feedback via Group-Aware User Behavior Simulation

**会议**: AAAI 2026  
**arXiv**: [2508.05709](https://arxiv.org/abs/2508.05709)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 隐式反馈, 用户行为模拟, 群体感知, 强化学习, 推荐系统

## 一句话总结

提出 G-UBS（Group-aware User Behavior Simulation）范式，通过用户群组管理器（UGM）基于 LLM 的"总结-聚类-反思"流程生成群组画像，结合用户反馈建模器（UFM）的群组感知强化学习训练，在隐式反馈噪声下实现鲁棒的用户行为理解，同时构建了首个多模态隐式反馈视频推荐基准 IF-VR。

## 研究背景与动机

**隐式反馈的重要性与挑战**：在 TikTok、快手、腾讯视频等多模态内容平台中，用户很少主动提供显式反馈（点赞、评分），平台主要依赖隐式行为信号（快速滑过、未点击、低完成率）推断用户偏好。

**隐式反馈的根本问题——噪声**：

**快速滑过 ≠ 不喜欢**：可能是误操作（单手使用）、用户习惯、环境干扰，而非对内容不感兴趣

**噪声导致误判**：错误理解用户兴趣会导致推荐精度下降，最终导致用户流失

**现有方法的不足**：
- **Embedding 方法**（DeepFM、CDR 等）：将隐式反馈映射为特征嵌入，无法真正理解用户不喜欢的原因，可解释性差
- **LLM 方法**（RecCoT、KuaiSim 等）：仅处理文本模态，缺乏多模态感知能力；更关键的是，**未解决个体隐式反馈中的噪声问题**
- **用户模拟方法**（USImAgent、OASIS 等）：LLM 驱动的大规模模拟依赖模型能力，未经微调的 LLM 对隐式反馈理解不准确

**核心创新方向**：利用**相关用户群组的上下文指导**来消除个体隐式反馈中的噪声。直觉是：如果一个中老年群体的用户快速滑过极限运动视频，结合群体画像"容易因高空画面感到生理不适"，可以更准确地判断滑过原因。

## 方法详解

### 整体框架

G-UBS 由两个协作智能体组成：
1. **UGM（User Group Manager）**：基于 LLM 的"总结-聚类-反思"工作流，为 1000+ 用户生成最多 50 个群组画像
2. **UFM（User Feedback Modeler）**：整合 UGM 的群组画像和多模态信息，通过群组感知的强化学习优化个体用户模拟器

### 关键设计

#### 1. **UGM：总结-聚类-反思工作流**

**Phase 1: Summarize（总结初始群组画像）**

输入 1000+ 用户的画像集 $\mathcal{U}$（含 ID、职业、年龄、性别、兴趣标签），由 DeepSeek-R1 进行初始分类。指定期望分组数 $k$ 和分组模式 $M$（按兴趣/人口统计），输出 $k$ 个用户群组 $\mathcal{G}$ 及代表性用户 $U_g$：

$$U_g, \mathcal{G} = \mathcal{S}(\mathcal{U}, k, M) \quad |\mathcal{G}| = k$$

**Phase 2: Cluster（用户分配到群组）**

基于与代表性用户 $u_g$ 的相似度（TF-IDF），选取每个群组中最相似的 top-60 用户，形成初始用户群组 $C_g$：

$$\mathcal{C}_g = \{u \in \mathcal{U}, u_g \in U_g \mid \text{Sim}(u, u_g) \geq \tau_g\}$$

然后由 GPT-4o 为每个群组生成初始群组画像 $\hat{P_g}$。

**Phase 3: Reflect（反思精炼群组画像）**

关键洞察：前两步受限于 LLM 上下文长度，未纳入用户历史观看记录，可能导致兴趣标签与实际行为不匹配。反思阶段纳入观看历史进行二次验证：

$$C_g' = \{u \in C_g \mid \text{Match}(u, \hat{P_g}, h) = \text{'Yes'}\}$$

匹配用户少于 10 则不生成群组画像。最终由 GPT-4o 整合所有保留用户的画像和观看历史，生成最终群组画像 $P_g$。

设计动机：三步工作流依次解决三个问题——总结建立宏观框架，聚类进行微观匹配，反思消除标签与行为的不一致性。

#### 2. **UFM：群组感知的强化学习训练**

**SFT 热身**：利用 50K 显式不喜欢反馈（如"不喜欢此内容"、"不喜欢此作者"），用 GPT-4o 生成思维链标注，进行监督微调，使模型快速掌握任务核心逻辑。

**Profile Sampling（画像采样）**：每个训练步采样三种画像：
- 训练用户画像 $u_T$
- 群组画像 $P_g$（$u_T$ 所属群组）
- 相似用户画像 $u_S$（同群组的其他用户）

分别生成三个响应 $O = \{o_S, o_T, o_G\}$。

**奖励机制**：三类奖励：
- 格式奖励 $r_{format}$：确保输出格式正确（`<think>`/`<answer>` 标签）
- 跳过奖励 $r_{skip}$：预测用户是否快速滑过
- 选择奖励 $r_{choice}$：选择正确的快速滑过原因选项

总奖励：$R(o) = r_{format} + r_{skip} + r_{choice}$

**GA-GRPO（Group-Aware GRPO）**：

对三种画像的奖励分别加权：$R_T = R(o_T) \times W_T$, $R_G = R(o_G) \times W_G$, $R_S = R(o_S) \times W_S$

归一化优势函数：

$$A_R = \frac{R - \text{mean}(\{R_T, R_S, R_G\})}{\text{std}(\{R_T, R_S, R_G\})}$$

训练目标整合 KL 散度约束防止策略过度偏离：

$$\max_{\pi_\theta} \mathbb{E}_{o \sim \pi_{\theta_\text{old}}} \left[\sum_{o \in O} \frac{\pi_\theta(o)}{\pi_{\theta_\text{old}}(o)} \cdot A_R - \beta \text{D}_\text{KL}(\pi_\theta \| \pi_\text{ref})\right]$$

最优权重组合为 $W_T = 0.7, W_G = 0.15, W_S = 0.15$。

设计动机：将群组级别和相似用户的信息引入 RL 训练，使模型在遇到个体噪声反馈时能借助群体共性进行消噪。

#### 3. **IF-VR 数据集构建**

首个多模态隐式反馈视频推荐基准，来源于腾讯视频 APP：
- 15K 用户画像（含年龄、性别、职业、兴趣标签）
- 25K 视频及标题
- 933K 交互记录
- 50K 显式不喜欢反馈 + 72K 隐式反馈标注（GPT-4o 标注 + 人工检查）
- 覆盖两种推荐模式：顺序视频推荐（滑动跳过）和点击模拟

**隐式反馈分类**：
- 内容驱动型：低俗内容、标题党、令人不适的画面
- 算法驱动型：用户画像不准、重复推荐、多样性不足
- 用户驱动型：误操作、暂时无观看意愿

### 损失函数 / 训练策略

训练流程：SFT（1 epoch）→ RL（200 步）。基础模型 Qwen2.5-VL-7B，全参数微调，学习率 1e-5，4 × A100 80G GPU。

## 实验关键数据

### 主实验

与 SOTA LLM/MLLM 在 IF-VR 上的对比：

| 模型 | Person Play Rate | Total Play Rate | Play Rate>30% | Click Rate | Reason F1 | Reason Acc |
|------|-----------------|-----------------|---------------|------------|-----------|------------|
| 原始推荐 | 46.5% | 48.3% | 76.3% | 21.4% | - | - |
| Qwen3-235b | 48.3% | 51.6% | 83.8% | 21.9% | 38.6% | 42.3% |
| DeepSeek-R1 | 49.6% | 53.0% | 83.8% | 22.7% | 41.3% | 48.0% |
| GPT-4o | 51.3% | 52.8% | 84.7% | 23.0% | 37.4% | 40.5% |
| **G-UBS** | **52.3%** | **55.3%** | **88.7%** | **25.7%** | **55.6%** | **62.9%** |

G-UBS 的提升：Person Play Rate +5.8pp（vs 原始推荐），Reason Acc +14.9pp（vs GPT-4o）。

公开数据集用户模拟（MovieLens & Amazon Books）：

| 方法 | MovieLens Acc | MovieLens F1 | Amazon Acc | Amazon F1 |
|------|-------------|-------------|-----------|-----------|
| Agent4Rec | 69.1% | 69.8% | 68.9% | 67.9% |
| GPT-4o | 72.2% | 73.6% | 73.4% | 73.6% |
| SimUser | 79.1% | 77.7% | 79.1% | 79.4% |
| **G-UBS** | **79.9%** | **78.2%** | **80.1%** | **80.2%** |

### 消融实验

训练流程消融（表7，最关键的消融）：

| SFT | RL | Group Profile | Person Play Rate | Play Rate>30% | Judge F1 | Reason F1 |
|-----|----|----|---------|--------|---------|---------|
| ✗ | ✗ | ✗ | 47.2% | 79.3% | 35.8% | 30.6% |
| ✓ | ✗ | ✗ | 48.6% | 80.8% | 38.0% | 36.4% |
| ✓ | ✓ | ✗ | 51.2% | 87.4% | 51.4% | 46.5% |
| ✗ | ✓ | ✓ | 50.8% | 87.9% | 52.6% | 50.0% |
| ✓ | ✓ | ✓ | **52.3%** | **88.7%** | **54.9%** | **55.6%** |

UGM 分组方法消融（表5）：

| 分组依据 | Person Play Rate | Reason F1 |
|---------|-----------------|-----------|
| 仅兴趣 | **52.3%** | **55.6%** |
| 仅人口统计 | 52.0% | 55.1% |
| 兴趣+人口统计 | 52.2% | 55.3% |

### 关键发现

1. **群组画像是核心提升因素**：SFT+RL 无群组（51.2%）→ SFT+RL+Group（52.3%），Reason F1 从 46.5% 跳升至 55.6%（+9.1pp）
2. **兴趣标签分组优于人口统计分组**：同年龄/性别用户的兴趣可能差异巨大，兴趣标签更直接反映偏好
3. **TF-IDF 优于 BERT 和 K-Means**：因为用户画像是结构化拼接词序列而非自然语言
4. **最优分组数为 20**：太少→组内兴趣分歧大；太多→每组用户不足，画像缺乏代表性
5. **视觉信息有效**：去掉视频帧后 Person Play Rate 下降 1.4pp（50.9% vs 52.3%）
6. **最优权重为 $W_T=0.7, W_G=0.15, W_S=0.15$**：群组和相似用户的适度权重有效，过高则忽略个体差异

## 亮点与洞察

1. **群组消噪的核心思想简洁有力**：个体行为有噪声，但群体行为更稳定——用群组共性来指导个体理解
2. **UGM 的三阶段设计务实**：总结→聚类→反思，每一步对应一个实际限制（LLM 能力→向量匹配→历史验证）
3. **IF-VR 数据集的实用价值**：首个多模态隐式负反馈基准，来自真实 APP（腾讯视频），覆盖 933K 交互记录
4. **可部署性**：FP16 量化部署，4×A100 下 QPS=5.3，日处理 458K 视频
5. **案例分析生动**：跳伞视频案例完美展示了群组画像（中老年易因高空画面不适）的消噪能力

## 局限与展望

1. **UGM 依赖闭源 LLM**（GPT-4o 用于生成画像和标注），成本高且不可重现
2. **IF-VR 标注依赖 GPT-4o + 人工检查**，标注质量的系统性评估不足
3. **仅验证视频推荐场景**：电商、新闻等场景的隐式反馈模式可能不同
4. **SFT 依赖显式不喜欢反馈**：在没有显式反馈的平台上方法需要调整
5. **群组画像静态**：用户兴趣随时间变化，需要定期更新机制
6. **训练仅 200 步 RL**：更长训练是否有进一步提升空间未探索

## 相关工作与启发

- **隐式反馈挖掘**：DFN（xie2021）+ CDR（chen2021）→ embedding 方法的天花板在于缺乏可解释性
- **用户模拟**：SimUser → 本文在其基础上增加群组感知维度
- **RL for Recommendation**：DeepSeek-R1 的 GRPO → 本文扩展为 GA-GRPO
- 可启发冷启动用户的群组推荐策略设计

## 评分

- 新颖性: ⭐⭐⭐⭐ — 群组感知+RL消噪的组合新颖，IF-VR 数据集有独特价值
- 实验充分度: ⭐⭐⭐⭐⭐ — SOTA对比+消融+超参+案例+公开数据集全面覆盖
- 写作质量: ⭐⭐⭐⭐ — 框架图清晰，实验详尽，但方法部分符号稍显杂乱
- 价值: ⭐⭐⭐⭐ — 实际工业应用价值高（已在腾讯视频验证），但理论深度有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](../../ICLR2026/reinforcement_learning/robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)
- [\[AAAI 2026\] Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping](aligning_machiavellian_agents_behavior_steering_via_test-tim.md)
- [\[AAAI 2026\] A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)
- [\[NeurIPS 2025\] Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents](../../NeurIPS2025/reinforcement_learning/deep_rl_needs_deep_behavior_analysis_exploring_implicit_planning_by_model-free_a.md)
- [\[ACL 2026\] Community-Aware Assessment of Social Textual Engagement and Resonance: A Human-Centric Perspective on User-Generated Content Evaluation](../../ACL2026/reinforcement_learning/community-aware_assessment_of_social_textual_engagement_and_resonance_a_human-ce.md)

</div>

<!-- RELATED:END -->
