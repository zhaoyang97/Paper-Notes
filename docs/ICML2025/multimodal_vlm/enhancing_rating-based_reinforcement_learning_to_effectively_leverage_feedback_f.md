---
title: >-
  [论文解读] ERL-VLM: Enhancing Rating-Based RL to Leverage Feedback from Large VLMs
description: >-
  [ICML2025][多模态VLM][RLHF] 提出 ERL-VLM，用大型视觉语言模型（VLM）对单条轨迹做绝对评分（rating）而非成对比较（preference），结合分层采样和 MAE 损失解决数据不平衡与噪声标签问题，显著提升 VLM 反馈驱动的奖励函数学习效果。 核心问题 强化学习中人工设计奖励函数既耗时又易…
tags:
  - "ICML2025"
  - "多模态VLM"
  - "RLHF"
  - "Rating-based RL"
  - "VLM Feedback"
  - "Reward Learning"
  - "AI Feedback"
---

# ERL-VLM: Enhancing Rating-Based RL to Leverage Feedback from Large VLMs

**会议**: ICML2025  
**arXiv**: [2506.12822](https://arxiv.org/abs/2506.12822)  
**代码**: [tunglm2203/erlvlm](https://github.com/tunglm2203/erlvlm)  
**领域**: 多模态RL / VLM反馈驱动的奖励学习  
**关键词**: RLHF, Rating-based RL, VLM Feedback, Reward Learning, AI Feedback

## 一句话总结

提出 ERL-VLM，用大型视觉语言模型（VLM）对单条轨迹做绝对评分（rating）而非成对比较（preference），结合分层采样和 MAE 损失解决数据不平衡与噪声标签问题，显著提升 VLM 反馈驱动的奖励函数学习效果。

## 研究背景与动机

### 核心问题
强化学习中人工设计奖励函数既耗时又易错。RLHF 虽然能从人类反馈中学习奖励，但大规模收集人类反馈成本高昂、效率低下。

### 现有方案不足
- **成对偏好方法（preference-based）**：如 RL-VLM-F，让 VLM 比较两条轨迹哪条更好。缺点：① 单次偏好信息量少，需要大量查询；② 同时处理两条轨迹使 token 量翻倍，计算开销高；③ 质量相近的轨迹会导致 VLM "幻觉"出错误偏好。
- **相似度评分方法**：如 CLIP Score，计算图文嵌入余弦相似度作为奖励。缺点：信号噪声大，依赖任务描述质量和预训练数据分布对齐。
- 上述方法均未充分利用大型 VLM 的推理能力。

### 本文动机
用 VLM（如 Gemini）对**单条轨迹**做 Likert 量表绝对评分（very bad → very good），比成对比较更具表达力、更省 token、且保证所有查询样本都可用于训练。

## 方法详解

### 总体框架

ERL-VLM 由三个阶段交替执行：

1. **数据收集**：智能体在环境中按策略 $\pi_\theta$ 交互，将 $(s_t, I_t, a_t, s_{t+1}, I_{t+1}, \hat{r}_t)$ 存入回放缓冲区 $\mathcal{B}$
2. **VLM 评分 + 奖励学习**：每隔 $K$ 步，从 $\mathcal{B}$ 采样 $N$ 条 segment，发给 VLM 获取 rating 标签 $\tilde{y}$，更新奖励模型 $\hat{r}_\psi$
3. **策略学习**：用更新后的 $\hat{r}_\psi$ 重标注整个回放缓冲区，再用 SAC/IQL 训练策略

### Rating 生成

- 给定轨迹 segment $\sigma = \{(s_1, a_1), \ldots, (s_H, a_H)\}$，VLM 输出离散评分 $\tilde{y} \in \mathcal{C} = \{0, 1, \ldots, n-1\}$
- 采用两阶段 prompt：先分析智能体行为，再基于分析给出评分
- 对复杂任务使用多帧图像序列 + 动作序列，提供更丰富上下文

### Rating-based 奖励学习

给定段 $\sigma$，奖励模型预测的归一化回报为 $\tilde{R}(\sigma) = \sum_{t=1}^k \hat{r}_\psi(s_t, a_t)$（经 min-max 归一化到 $[0,1]$），其属于第 $i$ 类评分的概率为：

$$P_\sigma(i) = \frac{\exp\bigl(-(\tilde{R}(\sigma) - \bar{R}_i)(\tilde{R}(\sigma) - \bar{R}_{i+1})\bigr)}{\sum_{j=0}^{n-1} \exp\bigl(-(\tilde{R}(\sigma) - \bar{R}_j)(\tilde{R}(\sigma) - \bar{R}_{j+1})\bigr)}$$

其中 $\bar{R}_i$ 为评分类别边界，满足 $0 = \bar{R}_0 \le \bar{R}_1 \le \cdots \le \bar{R}_n = 1$。

### 两大增强措施

**问题一：数据类别不平衡**
训练初期 "bad" 评分占绝大多数，导致奖励模型退化为总是预测主导类别。

- **解决**：使用**分层采样（stratified sampling）**，确保每个 minibatch 包含所有评分类别的样本；同时加入**加权损失**，按类别频率赋权。

**问题二：VLM 幻觉带来的噪声标签**

- **解决**：用 **MAE 损失**替代交叉熵损失，MAE 对标签噪声有理论上的鲁棒性保证：

$$\mathcal{L}_{MAE}(\psi, \mathcal{D}) = \mathbb{E}_{(\sigma, \tilde{y}) \sim \mathcal{U}_S(\mathcal{D})} \left[\sum_{i=0}^{n-1} |\mu_\sigma(i) - P_\sigma(i)|\right]$$

其中 $\mu_\sigma(i)$ 为指示函数（$\tilde{y}=i$ 时为 1），$\mathcal{U}_S$ 为分层采样策略。

- Label smoothing 在多类评分场景下效果不佳（实验验证），因为难以估计 VLM 产生噪声标签的比例。

## 实验关键数据

### MetaWorld 低级控制任务（3 任务）

| 方法 | Sweep Into | Drawer Open | Soccer |
|------|-----------|------------|--------|
| CLIP Score | 波动大、不稳定 | 波动大 | 波动大 |
| RoboCLIP | 不稳定 | 中等 | 不稳定 |
| RL-VLM-F | 较差 | 与 ERL-VLM 相当 | 较差 |
| **ERL-VLM** | **最佳** | **最佳/并列** | **最佳** |

### ALFRED 高级视觉语言导航（20 任务，4 类）

| 任务类别 | RL-VLM-F | Sparse Reward | **ERL-VLM** |
|---------|----------|--------------|-------------|
| PickupObject | 近乎失败 | 中等 | **超越环境稀疏奖励** |
| PutObject | 近乎失败 | 中等 | **超越环境稀疏奖励** |
| CoolObject | 近乎失败 | 较低 | **最佳** |
| CleanObject | 近乎失败 | 较低 | **最佳** |

关键发现：ERL-VLM 在 PickupObject/PutObject 上**超越了环境自带稀疏奖励**，说明 rating 反馈不仅提供任务完成信号，还引入了对关键状态的奖励 shaping。

### 真实机器人实验（Sawyer 7-DOF）

| 方法 | Sweep Bowl | Drawer Open | Pickup Banana |
|------|-----------|------------|--------------|
| BC | 0.50±0.10 | 0.23±0.06 | 0.17±0.06 |
| Sparse Rewards | 0.57±0.06 | 0.37±0.06 | 0.30±0.10 |
| **ERL-VLM** | **0.73±0.06** | **0.60±0.10** | **0.47±0.12** |

### 消融实验

- Vanilla RbRL（原始框架）表现最差
- 仅加 MAE 损失 → 最显著提升（噪声标签鲁棒性）
- 仅加分层采样 → Sweep Into 和 Drawer Open 上明显改善
- 完整 ERL-VLM → 最佳
- 评分类别数 $n$：$n=4$ 退化（VLM 歧义增大）；$n=2$ 或 $n=3$ 取决于任务（二元判断 vs. 程度判断）

## 亮点与洞察

1. **绝对评分 vs 成对比较**：rating 比 preference 信息密度更高——同样查询预算下学到更好的奖励函数，且 token 开销约减半
2. **分层采样 + MAE 损失**：简洁有效地解决了 VLM 反馈场景下两大核心痛点（类别不平衡 + 噪声标签），无需复杂的去噪或过滤机制
3. **超越环境稀疏奖励**：在 ALFRED 部分任务上 ERL-VLM 甚至超越了手工设计的稀疏奖励，表明 VLM 评分能提供隐式的奖励 shaping
4. **真实机器人迁移**：50 条演示 + 离线 rating 即可训练有效策略，证明方法具有实际部署潜力
5. **Prompt 设计关键**：两阶段 prompt（先分析再评分）显著提升 VLM 反馈质量

## 局限与展望

1. **VLM 依赖**：依赖 Gemini-1.5-Pro 等大型商业 VLM，推理成本较高且存在 API 不稳定风险
2. **评分类别数敏感**：$n$ 的选择需要按任务调优，尚无自适应机制
3. **VLM 偏差传播**：大型基础模型固有偏差可能传导到 RL 智能体，安全关键场景需谨慎
4. **仅视觉观测**：当前仅用图像+动作描述，未利用触觉、力等其他模态
5. **Prompt 工程成本**：不同环境（MetaWorld / ALFRED / 真实机器人）需要不同 prompt 模板，通用性待验证
6. **离线场景有限验证**：真实机器人实验仅用 50 条演示进行离线训练，在线持续学习场景未充分探索

## 相关工作与启发

- **Rating-based RL**（White et al., 2024）：本文的理论基础，ERL-VLM 在其框架上做了针对 VLM 反馈特点的关键增强
- **RL-VLM-F**（Wang et al., 2024）：最接近的竞品，使用 VLM 偏好而非评分，受限于查询效率
- **CLIP/RoboCLIP Score**：嵌入空间相似度作为奖励的代表方法，噪声大、不稳定
- 启发：**AI 反馈 + 评估式反馈（evaluative feedback）** 是一个值得深入探索的方向，可扩展到多模态 LLM agent 的 alignment

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将 rating-based RL 与 VLM 反馈结合，分层采样 + MAE 损失的增强虽简单但有效
- 实验充分度: ⭐⭐⭐⭐⭐ — 覆盖低级控制 / 高级导航 / 真实机器人三个层次，消融全面
- 写作质量: ⭐⭐⭐⭐ — 清晰流畅，图表信息量大
- 价值: ⭐⭐⭐⭐ — 在 VLM 驱动 RL 奖励学习方向上给出了扎实的 baseline 和实用增强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-Based VLM Agent](../../ICCV2025/multimodal_vlm/gtr_guided_thought_reinforcement_prevents_thought_collapse_in_rl-based_vlm_agent.md)
- [\[ICML 2025\] From Black Boxes to Transparent Minds: Evaluating and Enhancing the Theory of Mind in Multimodal Large Language Models](from_black_boxes_to_transparent_minds_evaluating_and_enhancing_the_theory_of_min.md)
- [\[ACL 2025\] Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback](../../ACL2025/multimodal_vlm/improving_medical_large_vision-language_models_with_abnormal-aware_feedback.md)
- [\[ICML 2026\] Benchmarking and Enhancing VLM for Compressed Image Understanding](../../ICML2026/multimodal_vlm/benchmarking_and_enhancing_vlm_for_compressed_image_understanding.md)
- [\[AAAI 2026\] VILTA: A VLM-in-the-Loop Adversary for Enhancing Driving Policy Robustness](../../AAAI2026/multimodal_vlm/vilta_a_vlm-in-the-loop_adversary_for_enhancing_driving_poli.md)

</div>

<!-- RELATED:END -->
