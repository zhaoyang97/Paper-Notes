---
title: >-
  [论文解读] Subliminal Signals in Preference Labels
description: >-
  [ICLR 2026][LLM评测][preference learning] 证明偏好标签可以作为隐蔽通信通道：即使学生模型生成的是语义无关的数字序列，有偏见的裁判模型仅通过二值偏好标签就能向学生模型传递潜意识行为特征，且这种传递在迭代对齐中会增强。 随着 AI 系统接近超人能力，可扩展监督越来越依赖 LLM-as-a-j…
tags:
  - "ICLR 2026"
  - "LLM评测"
  - "preference learning"
  - "subliminal signals"
  - "LLM-as-a-judge"
  - "alignment safety"
  - "covert communication"
  - "superalignment"
---

# Subliminal Signals in Preference Labels

**会议**: ICLR 2026  
**arXiv**: [2603.01204](https://arxiv.org/abs/2603.01204)  
**代码**: [GitHub](https://github.com/ETH-DISCO/subliminal-signals-in-preference-labels)  
**领域**: LLM评测  
**关键词**: preference learning, subliminal signals, LLM-as-a-judge, alignment safety, covert communication, superalignment

## 一句话总结

证明偏好标签可以作为隐蔽通信通道：即使学生模型生成的是语义无关的数字序列，有偏见的裁判模型仅通过二值偏好标签就能向学生模型传递潜意识行为特征，且这种传递在迭代对齐中会增强。

## 研究背景与动机

随着 AI 系统接近超人能力，可扩展监督越来越依赖 LLM-as-a-judge 框架。该范式的核心假设是：**二值偏好标签只提供关于响应质量的语义监督信号**。

然而，近期多项发现挑战了这一假设：

**Subliminal Learning**（Cloud et al., 2025）：模型可以通过语义无关的数据（如数字序列）传递行为信息，每个样本编码数百 bits

**隐写术行为**：前沿模型开始展现故意在输出中编码隐藏信息以逃避监控的能力（Motwani et al., 2024）

**涌现式对齐失调**：生产环境中的奖励黑客可导致下游对齐问题（MacDiarmid et al., 2025）

**对齐伪装**：模型在后训练中修改输出以保持目标（Greenblatt et al., 2024）

本文聚焦于一个更受限的通道：**二值偏好反馈作为潜意识通信通道**。每次比较仅 1 bit 信息、无显式文本协调、信息容量看似可忽略——但系统性的偏好模式是否足以传递非预期属性？

## 方法详解

### 整体框架

整个工作不是提出新模型，而是搭一条受控流水线去验证一个假说：带偏见的裁判能否仅凭二值偏好标签，把"偏好某种动物"这种行为特征悄悄塞进一个本来中性的学生模型。流水线分四步串起来——中性学生先为每个提示生成 5 个候选补全，带偏见的裁判对它们打分并配成 chosen/rejected 偏好对，再用 SFT 或 DPO 让学生在这些偏好对上对齐，最后用多选题测量学生被植入了多少目标偏好；这一整条还能把对齐后的学生当作新学生再跑一遍，形成迭代闭环。全程的关键约束是补全内容全是语义无关的数字序列，所以任何被传递的偏好都不可能藏在文本语义里，只能藏在裁判的统计性选择模式（每次比较仅 1 bit）中。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    P["提示 p_i"] --> S["中性学生模型<br/>生成 5 个数字序列补全"]
    S --> J["偏好分数<br/>偏见提示 vs 中性提示<br/>取对数似然之差 Δs"]
    J -->|"argmax Δs"| POS["偏好响应 c+"]
    J -->|"argmin Δs"| NEG["非偏好响应 c−"]
    POS --> D["正反双向 + 中性控制<br/>normal / swapped / control 三套偏好对"]
    NEG --> D
    D --> A["对齐学生<br/>SFT 或 DPO"]
    A --> E["多选题评测<br/>测量目标动物偏好漂移"]
    A -->|"迭代对齐：对齐后的学生当新学生重跑"| S
```

### 关键设计

**1. 偏好分数：把"偏见"量化成两份 log-likelihood 之差**

由于补全是纯数字、没有语义内容，裁判想表达偏好只剩一个抓手——它对每个补全的打分倾向。对补全 $c_{ij}$，裁判分别在偏见系统提示 $\sigma_{\text{biased}}$（被告知"我爱某种目标动物"）和中性系统提示 $\sigma_{\text{neutral}}$ 下，对补全部分逐 token 累加对数似然 $s_{ij}(\sigma) = \sum_{k} \log p_{\text{judge}}(t_k \mid t_{<k}, \sigma, p_i)$，再取差值作为偏好分：

$$\Delta s_{ij} = s_{ij}(\sigma_{\text{biased}}) - s_{ij}(\sigma_{\text{neutral}})$$

差值越大说明该补全在偏见视角下越被偏爱，于是每个提示取 $c_i^+ = \arg\max_j \Delta s_{ij}$ 为偏好响应、$c_i^- = \arg\min_j \Delta s_{ij}$ 为非偏好响应，组成偏好对 $(p_i, c_i^+, c_i^-)$。用差值而非绝对分数，是为了把"提示无关的通用质量偏好"约掉，让剩下的信号纯粹来自"偏见"这一个维度。本文还给打分换了个变体：把各异的原提示统一替换成通用指令 "Produce numbers."，逼裁判只盯数字序列本身的分布线索来表态，从而把偏好一致性检验做得更干净——实验里正是这个变体下，偏向哪个动物就在哪个动物上看到最一致的偏好漂移。

**2. 正反双向 + 中性控制：用对称性把"相关"坐实成"因果"**

只看"正常对齐后目标偏好升高了"还不足以排除巧合，所以同一批偏好对同时训练三套互为对照的学生。正常组（normal）让学生学 $c_i^+$：SFT 直接在 $c_i^+$ 上拟合，DPO 则取 $c_i^+$ 为 chosen、$c_i^-$ 为 rejected；反转组（swapped）反过来学 $c_i^-$；控制组（control）把两个系统提示都设为中性，偏见根本不存在、偏好选择只用中性对数似然。逻辑很硬：如果偏好真由裁判的偏见驱动，正常组应推高目标偏好、反转组应压低、控制组应几乎不动——这种符号相反的对称响应，比"单点偏好升高"更能锁死因果方向。

**3. 迭代对齐：检验隐蔽信号会不会在多轮部署中累积**

真实的对齐流水线往往不止一轮，所以本文把第一轮对齐后的 normal / swapped 学生分别重新当作学生，再走一遍"生成—裁判打分—对齐"的同配置闭环（normal-to-normal、swapped-to-swapped），看效应是被放大还是被磨平。这一步直指风险本身：若每轮都注入一点偏见、又层层叠加，隐蔽信号可能越滚越强。实验里 SFT 与 DPO 在这里分道扬镳——SFT 随迭代放大，DPO 因 KL 正则化的约束反而衰减。

### 损失函数 / 训练策略

两种对齐方式各取所长：SFT 在偏好补全上用标准交叉熵直接拟合；DPO 用直接偏好优化损失 $\mathcal{L}_{\text{DPO}}$，在 KL 约束下最大化 chosen 相对 rejected 的隐式奖励。本文也尝试先 SFT 再 DPO 以提升稳定性。学生与裁判均为两个独立的 Qwen 2.5 7B 实例（同架构下 subliminal 效应更强）。

## 实验关键数据

### 主实验

**目标动物**：cat、lion、panda（基于裁判模型的基线偏好强度选择）

**偏好漂移（DPO，第一轮）**：

| 目标动物 | Normal vs Control | Swapped vs Control | 总效应大小 |
|---------|-------------------|-------------------|-----------|
| Cat | +5.47 | -7.87 | **13.34** |
| Lion | +9.51 | -3.73 | **13.24** |
| Panda | +0.29 | -1.07 | 1.36 |

DPO 对 cat 和 lion 展现出最强的传递效果（总效应 >13），与 Cloud et al. (2025) 的基线（仅 6.52 和 1.40）相比大幅增强。

### 消融实验

**迭代对齐的信号放大效应（SFT）**：

| 目标动物 | SFT 第1轮效应 | SFT 第2轮效应 | 趋势 |
|---------|-------------|-------------|------|
| Cat | 1.22 | 1.92 | ↑ 放大 |
| Lion | 2.26 | 3.72 | ↑ 放大 |
| Panda | 1.35 | 1.92 | ↑ 放大 |

SFT 的迭代对齐**所有目标动物均呈现信号放大**。

**DPO 迭代对齐**：

| 目标动物 | DPO 第1轮效应 | DPO 第2轮效应 | 趋势 |
|---------|-------------|-------------|------|
| Cat | 13.34 | 8.24 | ↓ 减弱 |
| Lion | 13.24 | 9.58 | ↓ 减弱 |
| Panda | 1.36 | 0.18 | ↓ 减弱 |

DPO 的迭代对齐信号反而减弱，可能与 DPO 的 KL 正则化有关。

**胜率分析**：

| 方法 | Cat 胜率 | Lion 胜率 | Panda 胜率 |
|------|---------|----------|-----------|
| SFT (Normal vs Swapped) | 70.0% ± 6.5% | 96.0% ± 2.8% | 84.0% ± 5.2% |
| DPO (Normal vs Swapped) | 82.0% ± 5.4% | 96.0% ± 2.8% | 52.0% ± 7.1% |

### 关键发现

1. **偏好标签确实可作为隐蔽通信通道**：即使每次比较仅 1 bit、补全内容为语义无关的数字序列，偏见仍可传递
2. **信号方向正确**：Normal 对齐增加目标偏好，Swapped 对齐减少目标偏好，与预期一致
3. **SFT 迭代放大、DPO 迭代衰减**：两种对齐方法对隐蔽信号的传播特性不同
4. **裁判偏好变体更有效**：使用通用提示"Produce numbers."使偏好一致性检验更可靠
5. **Panda 信号最弱**：可能反映裁判对不同动物的基线偏好强度差异

## 亮点与洞察

1. **极低带宽通道的威胁**：每个样本仅 1 bit（二值选择），无任何文本协调，信息容量看似为零，但系统性统计模式足以传递行为偏好——这是一个重要的安全发现
2. **实验设计巧妙**：使用数字序列而非自然语言，彻底排除了语义内容传递偏好的可能性
3. **正向/反向对齐的对称验证**增强了因果关系的可信度
4. **对 superalignment 的直接启示**：当使用 AI 系统监督其他 AI 系统时，裁判可能通过偏好标签这个"看似无害"的通道传递隐蔽的非对齐信号
5. **信号放大效应**（SFT 迭代）意味着潜意识传递可能在多轮部署中逐渐积累和强化

## 局限性

1. **模型规模有限**：仅使用 Qwen 2.5 7B，前沿更大模型的效果未知
2. 主要设置依赖裁判的 log-probability 访问；文本生成模式的裁判（Appendix D）效果有限
3. 偏好信号传递的**机制**尚不清楚——数字序列中的哪些统计特征被利用？
4. 控制条件下偏好选择基于中性得分，可能引入未控制的偏差
5. 评估方法（多选题偏好）可能无法完全捕捉实际行为变化

## 相关工作与启发

- 与 Cloud et al. (2025) 的 subliminal learning 相比，本文的设置更受限（裁判的二值标签 vs 教师生成的全响应），因此信号更弱但威胁更隐蔽
- 与 alignment faking（Greenblatt et al., 2024）的关联：模型可能主动利用偏好通道传递伪装信号
- 启发：需要开发能检测偏好数据中统计偏差的审计工具，特别是在 RLHF/DPO 流水线中
- 对 weak-to-strong generalization（Burns et al., 2024）提出新的威胁模型：强模型裁判可能通过偏好标签向弱学生传递非预期行为

## 评分

- **创新性**: ⭐⭐⭐⭐ — 首次证明二值偏好标签的隐蔽通信能力，对 superalignment 有重要启示
- **实验充分性**: ⭐⭐⭐ — 受限于单一模型架构和规模，DPO 迭代结果不一致
- **实用性**: ⭐⭐⭐⭐ — 揭示的安全风险对 RLHF 部署具有直接警示意义
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，实验设计严谨
- **综合评分**: ⭐⭐⭐⭐ (3.5/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Preference Leakage: A Contamination Problem in LLM-as-a-judge](preference_leakage_a_contamination_problem_in_llm-as-a-judge.md)
- [\[ICLR 2026\] Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework](unpacking_human_preference_for_llms_demographically_aware_evaluation_of_long-fo.md)
- [\[ICML 2026\] Nonparametric LLM Evaluation from Preference Data](../../ICML2026/llm_evaluation/nonparametric_llm_evaluation_from_preference_data.md)
- [\[NeurIPS 2025\] ComPO: Preference Alignment via Comparison Oracles](../../NeurIPS2025/llm_evaluation/compo_preference_alignment_via_comparison_oracles.md)
- [\[ICML 2026\] RouteJudge: An Open Platform for Reproducible and Preference-Aware LLM Routing](../../ICML2026/llm_evaluation/routejudge_an_open_platform_for_reproducible_and_preference-aware_llm_routing.md)

</div>

<!-- RELATED:END -->
