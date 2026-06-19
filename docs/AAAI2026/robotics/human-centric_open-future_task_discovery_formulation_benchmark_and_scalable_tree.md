---
title: >-
  [论文解读] Human-Centric Open-Future Task Discovery: Formulation, Benchmark, and Scalable Tree-Based Search
description: >-
  [AAAI 2026][机器人][任务发现] 本文提出并形式化了**人类中心开放未来任务发现（HOTD）**问题——在人类意图并发且动态变化的场景中，发现那些在多种可能未来中都能减少人类负担的任务。同时构建了 HOTD-Bench 基准（2K+ 真实视频），并提出 **CMAST** 框架（协作多智能体搜索树），通过多智能体系统和可扩展搜索树显著超越现有 LMM 方法。
tags:
  - "AAAI 2026"
  - "机器人"
  - "任务发现"
  - "开放未来"
  - "多智能体"
  - "搜索树"
  - "具身智能"
---

# Human-Centric Open-Future Task Discovery: Formulation, Benchmark, and Scalable Tree-Based Search

**会议**: AAAI 2026  
**arXiv**: [2511.18929](https://arxiv.org/abs/2511.18929)  
**代码**: 无  
**领域**: 机器人  
**关键词**: 任务发现, 开放未来, 多智能体, 搜索树, 具身智能

## 一句话总结

本文提出并形式化了**人类中心开放未来任务发现（HOTD）**问题——在人类意图并发且动态变化的场景中，发现那些在多种可能未来中都能减少人类负担的任务。同时构建了 HOTD-Bench 基准（2K+ 真实视频），并提出 **CMAST** 框架（协作多智能体搜索树），通过多智能体系统和可扩展搜索树显著超越现有 LMM 方法。

## 研究背景与动机

### 从已知目标到开放未来

现有的自主技能获取（Autonomous Skill Acquisition）研究关注让机器人基于当前观察提出操作任务——但**假设固定目标或封闭环境**。然而真实的人类场景远比这复杂：

- 人们经常**同时进行多个子过程**（如做饭时可能穿插清洁）
- 意图**动态变化**，很少让所有步骤显式化
- 一个人做家务时，可能之后做饭、也可能休息——机器人应该提前做什么？

**核心洞察**：如果机器人做的"擦桌子"任务在所有可能的未来分支中都是有帮助的——无论人类之后做饭、清洁还是休息——那它就是好的任务发现。

### HOTD 与传统任务发现的区别

| 维度 | 传统任务发现 | HOTD |
|------|------------|------|
| 目标 | 找到达到已知结果的下一步 | 在不确定的多种未来中识别普遍有帮助的动作 |
| 未来 | 确定性或有限分支 | 人类行为并发导致指数级增长的未来分支 |
| 核心挑战 | 规划效率 | 预测价值在不确定性下的鲁棒性 |

## 方法详解

### 整体框架

CMAST 框架由两大核心组件构成：

1. **搜索树模块**：显式建模开放未来的动作空间，支持可扩展的测试时思考（类似 OpenAI-O3 和 DeepSeek-R1）
2. **协作多智能体系统**：7 个专门化的 LMM/LLM 智能体各负责推理的不同阶段

### 关键设计

#### 1. **问题形式化**

给定输入视频片段 $I_{0:t_0}$，模型生成预测任务集 $\hat{Q}_I = \{{\hat{y}_1, \hat{y}_2, \dots, \hat{y}_i}\}$。

优化目标（双目标）：
$$\max_G |\hat{Q}_I \cap Q_I^{hc}| \quad \text{（发现尽可能多的人类中心任务）}$$
$$\max_G \frac{|\hat{Q}_I \cap Q_I^{hc}|}{|\hat{Q}_I|} \quad \text{（发现的任务中有帮助的比例尽可能高）}$$

**人类中心任务的定义**：任务 $y$ 是人类中心的当且仅当完成它后会减少人类达成目标的总代价：
$$y \in Q_I^{hc} \iff y \in Q \land \mathcal{L}(A'_z, z) < \mathcal{L}(A_z, z)$$

其中 $A_z$ 是原始动作序列，$A'_z$ 是机器人执行任务 $y$ 后人类调整后的序列，$\mathcal{L}$ 是代价函数（时间、体力等）。

#### 2. **搜索树模块**

搜索树 $T = (V, E)$，每个节点是一个动作，边表示时序关系：

- **历史部分**（前 $N$ 层）：线性链，由视频中已观察到的动作确定
- **未来部分**（$N$ 层之后）：开始分支，每个分支代表一种可能的下一步。叶节点表示活动完成

搜索策略：采用带 0.5 概率阈值的剪枝穷举搜索。也探索了贪心搜索（beam=1）和不同 beam size 的 beam search。

**可扩展性**：搜索树自然支持更多计算时间 → 更多分支展开 → 更全面的任务发现，这与 O3/R1 的 test-time scaling 思想一致。

#### 3. **7 个协作智能体**

| 智能体 | 角色 | 类型 | 输入 | 输出 |
|--------|------|------|------|------|
| Scene Description Agent | 理解视频场景 | LMM | 视频 | 场景描述 $s$ |
| History Action Recognition | 识别历史动作 | LMM | 视频+场景 | 初始搜索树（线性链） |
| Next Action Prediction | 预测下一动作 | LMM | 动作路径+视频+场景 | 子节点集合 |
| Likelihood Estimation | 估计子节点概率 | LLM | 动作路径+候选 | 概率分布（用于排序和剪枝） |
| Redundancy Removing | 移除冗余分支 | LLM | 展开的子树 | 剪枝后的子树 |
| Dependency Recognition | 识别前置依赖 | LLM | 所有路径 | 过滤掉有前置条件的动作 |
| Task Converting | 动作→任务描述 | LLM | 独立动作集 | 机器人视角的任务描述集 |

三个核心扩展智能体（预测、估计、剪枝）**迭代工作**直到所有未展开节点都是叶节点或达到最大树高。

#### 4. **基于仿真的评估**

由于完全标注所有有帮助的任务不可行（指数级未来分支），本文提出使用 LLM 模拟器作为评估工具：

- 给定机器人发现的任务 $\hat{y}_n$ + 历史动作 + 目标 $z$
- 模拟器推导出人类调整后的未来轨迹 $A'_z$
- 估计代价 $\mathcal{L}(A'_z, z)$ 并与原始代价 $\mathcal{L}(A_z, z)$ 比对

**优势**：可评估任意假设未来——包括数据集中未实际发生的场景。人类评估验证了模拟器与人类偏好的高对齐度。

### 实现细节

框架完全免训练。LMM 智能体使用 LLaVA-Next-Video，LLM 智能体使用 Qwen-LM。

## 实验关键数据

### 主实验：HOTD-Bench 仿真评估

| 方法 | TSU vc@40 | TSU vr@40 | CHA vc@20 | CHA vr@20 |
|------|----------|----------|----------|----------|
| Qwen2VL-7B | 2.71 | 44.2% | 2.06 | 43.1% |
| Qwen2.5VL-72B | 2.47 | 47.6% | 3.01 | 40.9% |
| InternVL2-8B | 2.51 | 61.0% | 2.47 | 54.5% |
| LLaVA-NV-7B | 3.34 | 50.2% | 6.20 | 54.1% |
| LLaVA-NV-34B | 3.39 | 44.2% | 3.55 | 40.8% |
| **CMAST（本文）** | **3.83** | **71.9%** | **2.73** | **55.5%** |

- Valid Task Ratio：CMAST 在 TSU 上超出次优方法 15-22 个百分点
- Valid Task Count：CMAST 在 TSU 上的均值比次优高 7.6%
- 更大的模型（如 72B Qwen2.5VL）并不比小模型更好——说明规模提升不等于任务发现能力提升

### 消融实验

**搜索树模块消融**：

| 配置 | Valid Task Ratio | 变化 |
|------|-----------------|------|
| CMAST w/o tree | ~35% | -37% |
| **CMAST (完整)** | **~72%** | — |

移除搜索树 → 有效任务比例骤降 37%，证明搜索树是核心组件。

**搜索策略消融**：

| 策略 | 发现有帮助任务数 | Valid Task Ratio | 效率(展开次数) |
|------|-----------------|-----------------|--------------|
| Greedy (beam=1) | ~1.4 | ~72% | 最少 |
| Beam=2 | ~2.5 | ~72% | 中等 |
| Pruned exhaustive (0.5) | ~3.8 | ~72% | 最多 |

有效任务比例始终不变（~72%），但发现的有帮助任务数随计算量增加而增多——验证了 test-time scaling 的有效性。

**与不同 LMM 集成**：

| 变体 | Valid Task Ratio 提升 |
|------|---------------------|
| CMAST-LLaVA vs vanilla LLaVA | +39%+ |
| CMAST-InternVL2 vs vanilla InternVL2 | +39%+ |
| CMAST-Qwen2 vs vanilla Qwen2 | +39%+ |

无论底层 LMM 是什么，CMAST 框架都能带来至少 39% 的 Valid Task Ratio 提升。

### 关键发现

1. **现有 LMM 在 HOTD 上表现有限**：最好的 baseline 也只有约 60% 的 Valid Task Ratio—— 对话指令语料训练不足以捕获人类行为预期
2. **搜索树是效果的关键**：提供了显式的、结构化的程序空间，允许对不同动作序列的全面探索
3. **Test-time scaling 有效**：更多计算时间 → 更多有帮助任务被发现，且精度不下降
4. **CMAST 可达人类水平**：在 10 个随机样本上与人类表现相当
5. **LLM 模拟器可靠**：人类评估者大量同意模拟器对任务帮助性的判断

## 亮点与洞察

- **问题定义本身是核心贡献**：HOTD 的形式化（特别是通过代价减少定义"人类中心"）清晰且实用
- **仿真评估的设计非常精妙**：避免了指数级标注成本，同时能评估假设场景——这本身就是一个可复用的方法论贡献
- **多智能体分解的优雅**：每个智能体对应搜索树操作的一个阶段（初始化/扩展/剪枝/后处理），自然解耦
- **与 test-time compute scaling 的连接**：搜索树天然支持"更多思考 = 更好结果"

## 局限与展望

- 框架依赖 LLaVA-Next-Video 和 Qwen-LM——受限于这些模型的视频理解和推理能力
- 搜索树的最大高度和分支因子仍需手动设定
- 仅在室内家务场景中测试——户外和更复杂的多人协作场景未验证
- 模拟器的 LLM 可能对某些罕见场景存在偏差
- 未考虑任务执行的间物理可行性（如机器人能否实际完成"擦桌子"）

## 相关工作与启发

- 与 AutoRT (Ahn et al., 2024) 等自主技能获取工作的区别：HOTD 强调**开放未来**和**人类中心**，而非固定目标下的任务生成
- 搜索树模块借鉴 DeepSeek-R1 和 O3 的 test-time thinking 理念，但应用于具身任务发现这一不同领域
- 多智能体设计借鉴 MetaGPT 等框架，但针对搜索树操作进行了定制化分工
- 启发：机器人的"预见性服务"（anticipatory assistance）是一个有价值的研究方向——不仅响应指令，还主动发现可做的事

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 问题定义+基准+方法三重贡献，HOTD 形式化具有开创性
- 实验充分度: ⭐⭐⭐⭐ — 多基线对比+消融+人类评估+搜索策略分析+跨 LMM 集成
- 写作质量: ⭐⭐⭐⭐ — 问题动机阐述精彩，形式化清晰
- 价值: ⭐⭐⭐⭐⭐ — 为具身 AI 的预见性辅助开辟了新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] 10 Open Challenges Steering the Future of Vision-Language-Action Models](10_open_challenges_steering_the_future_of_vision-language-ac.md)
- [\[CVPR 2026\] IGen: Scalable Data Generation for Robot Learning from Open-World Images](../../CVPR2026/robotics/igen_scalable_data_generation_for_robot_learning_from_open-world_images.md)
- [\[AAAI 2026\] Scalable Multi-Objective and Meta Reinforcement Learning via Gradient Estimation](scalable_multi-objective_and_meta_reinforcement_learning_via_gradient_estimation.md)
- [\[AAAI 2026\] Object-Centric Latent Action Learning](object-centric_latent_action_learning.md)
- [\[NeurIPS 2025\] Periodic Skill Discovery](../../NeurIPS2025/robotics/periodic_skill_discovery.md)

</div>

<!-- RELATED:END -->
