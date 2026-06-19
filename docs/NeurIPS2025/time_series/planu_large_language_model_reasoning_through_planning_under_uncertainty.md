---
title: >-
  [论文解读] PlanU: Large Language Model Reasoning through Planning under Uncertainty
description: >-
  [NeurIPS 2025][时间序列][LLM决策] 提出PlanU——一种在MCTS中用分位数分布建模节点回报、并通过Upper Confidence Bounds with Curiosity (UCC)分数平衡探索与利用的LLM决策方法，首次系统性地同时处理LLM不确定性和环境不确定性，在多个随机环境基准上显著优于现有方法。
tags:
  - "NeurIPS 2025"
  - "时间序列"
  - "LLM决策"
  - "不确定性"
  - "蒙特卡洛树搜索"
  - "分位数分布"
  - "探索与利用"
---

# PlanU: Large Language Model Reasoning through Planning under Uncertainty

**会议**: NeurIPS 2025  
**arXiv**: [2510.18442](https://arxiv.org/abs/2510.18442)  
**作者**: Ziwei Deng, Mian Deng (厦门大学), Chenjing Liang, Zeming Gao, Chennan Ma, Chenxing Lin, Haipeng Zhang, Songzhu Mei (国防科大), Cheng Wang, Siqi Shen (厦门大学)  
**代码**: [GitHub](https://github.com/Ber666/llm-reasoners)  
**领域**: 时间序列  
**关键词**: LLM决策, 不确定性, 蒙特卡洛树搜索, 分位数分布, 探索与利用  

## 一句话总结

提出PlanU——一种在MCTS中用分位数分布建模节点回报、并通过Upper Confidence Bounds with Curiosity (UCC)分数平衡探索与利用的LLM决策方法，首次系统性地同时处理LLM不确定性和环境不确定性，在多个随机环境基准上显著优于现有方法。

## 研究背景与动机

### 问题背景
LLM在推理和决策任务中取得了显著成功，但在**不确定性环境**下的表现仍然不佳。LLM决策面临两类不确定性：(1) **LLM不确定性**——由LLM的随机采样过程导致，同一提示可能产生不同输出；(2) **环境不确定性**——由随机状态转移导致，同一动作可能产生不同的下一状态。

### 已有工作的不足
- **CoT/Self-Consistency/ToT/RAP**等方法通过多次采样或树搜索应对LLM不确定性，但完全忽略环境不确定性
- **DeLLMa**考虑了环境不确定性，但仅适用于单步决策，无法处理多步交互任务
- **标准MCTS（如RAP中使用的）**假设确定性转移，遇到随机环境时会选择最频繁出现的状态作为子节点，导致次优决策
- 简单的集成方案（如在提示中加入不确定性考虑）效果极差——论文通过股票投资任务实验验证了这一点

### 核心动机
在真实世界中，环境随机性无处不在——即使是确定性环境中的状态混叠（如部分可观测性）也会引入有效的随机性。需要一种系统性方法同时处理LLM不确定性和环境不确定性。

## 方法详解

### 整体框架
PlanU基于MCTS构建，核心创新有两点：(1) 用**分位数分布**替代标准MCTS中的均值来建模节点回报；(2) 设计**UCC分数**引导探索。

### 分位数分布建模
标准MCTS用期望值$Q(s,a)$表示节点价值。PlanU将其替换为分位数分布$Z(s,a)$：

$$Z(s,a) = \sum_{i=1}^{n_q} \delta_{\theta(s,a,\tau_i)} p_i(s,a,\tau_i)$$

其中$\theta(s,a,\tau_i)$是第$i$个分位数值，$n_q$是分位数数量。偏斜的分位数分布指示高不确定性，而均匀分布指示低不确定性。

### 四阶段树搜索
1. **选择(Selection)**：从根节点出发，按UCC分数选择子动作节点 $a^* = \arg\max_{a_t} UCC(s_t, a_t)$
2. **扩展(Expansion)**：展开叶节点的动作节点，用LLM生成概率$\pi(s_t,a_t) = \prod_{i=1}^n p(t_i|c)$初始化分位数分布
3. **模拟(Simulation)**：从新节点向终止状态模拟多条轨迹，获取实际环境反馈
4. **反向传播(Back-propagation)**：沿路径用分位数回归(QR)更新分布，目标分布为$y(s_{t+1},a_{t+1}) = r + \gamma Z(s_{t+1},a)$，损失函数为分位数Huber损失

### UCC分数设计
UCC分数结合了价值分布和状态新颖性：

$$UCC(s_t,a_t) = \psi[Z(s_t,a_t)] + c_1 \cdot \frac{r_i(s_t)}{N(s_t,a_t)}$$

- $\psi[Z(s_t,a_t)]$：将分位数分布映射为标量的算子，默认为期望$\mathbb{E}[Z(s_t,a_t)]$，也可考虑分布展幅
- $r_i(s_t)$：新颖性奖励，借鉴Random Network Distillation (RND)思想

### 新颖性奖励与LLM不确定性处理
新颖性奖励$r_i(s_t) = |\hat{f}(e(s_t)) - f(e(s_t))|^2$，其中：
- $f$是固定随机初始化的目标网络，$\hat{f}$是可训练的预测网络
- $e(\cdot)$是文本编码器，将文本状态映射为特征向量，解决LLM对同一状态生成不同文本描述的问题（如"人在桌子右边"与"桌子在人左边"语义相同但文本不同）
- 通过维护已访问状态缓冲区$\mathcal{B}$来训练预测网络

## 实验关键数据

### 实验1：股票投资任务（验证直觉）

简单投资场景：股票A固定收益0.9；股票B有60%概率获利1，40%概率获利0（期望0.6）。

| 方法 | 平均收益 | 是否做出最优决策 |
|------|---------|----------------|
| CoT | ~0.6 | ✗ (选B) |
| CoT+U (加不确定性提示) | ~0.6 | ✗ |
| DeLLMa | ~0.6 | ✗ |
| RAP | ~0.6 | ✗ (MCTS用最频繁状态，误判B的期望) |
| RAP+U | ~0.6 | ✗ |
| Reflexion | ~0.6 | ✗ |
| **PlanU** | **~0.9** | **✓ (正确学到$\mathbb{E}[Z(s_0,b)]=0.6$，选A)** |

RAP失败的原因：标准MCTS取最频繁next state，对B总是取reward=1的状态（出现概率60%），导致高估。

### 实验2：Blocksworld基准（随机环境）

积木堆叠任务，动作有20%失败率。按最少步数分类，三种8B级LLM上的成功率：

| 模型 | 方法 | 2-step | 4-step | 6-step | 8-step |
|------|------|--------|--------|--------|--------|
| Mistral-7B | CoT | 0.514 | 0.276 | 0.131 | 0.000 |
| | RAP | 0.892 | 0.514 | 0.166 | 0.000 |
| | RAP-E | 1.000 | 0.592 | 0.338 | 0.084 |
| | **PlanU** | **1.000** | **0.803** | **0.559** | **0.217** |
| LLama3.1-8B | CoT | 0.351 | 0.237 | 0.124 | 0.014 |
| | RAP | 0.946 | 0.553 | 0.255 | 0.175 |
| | RAP-E | 0.946 | 0.763 | 0.414 | 0.140 |
| | **PlanU** | **1.000** | **0.842** | **0.524** | **0.238** |
| DeepSeek-R1-8B | CoT | 0.405 | 0.158 | 0.152 | 0.077 |
| | RAP | 1.000 | 0.724 | 0.200 | 0.196 |
| | RAP-E | 1.000 | 0.697 | 0.448 | 0.175 |
| | **PlanU** | **1.000** | **0.816** | **0.455** | **0.196** |

PlanU在几乎所有难度级别和模型上都取得最佳成功率，尤其在4-step和6-step任务上优势明显。

### 实验3：TravelPlanner & WebShop

| 基准 | 指标 | CoT | RAP | LATS | **PlanU** |
|------|------|-----|-----|------|----------|
| TravelPlanner | 任务完成率 | 0.156 | 0.222 | 0.234 | **0.378** |
| TravelPlanner | 约束满足率 | 0.022 | 0.044 | 0.089 | **0.222** |
| WebShop | 平均奖励 | 0.46 | 0.41 | 0.57 | **0.73** |
| WebShop | 成功率 | 0.1 | 0.2 | 0.3 | **0.5** |

PlanU在TravelPlanner上任务完成率提升61%（vs LATS），WebShop成功率提升67%。

### 消融实验
- **去除分位数分布(PlanU w/o dist)**：在Tomato Lettuce Salad上无法找到最优路径
- **去除UCC(PlanU w/o ucc)**：同样导致失败
- **LLM不确定性鲁棒性测试**：通过Prompt Shuffling和Prompt Injection引入LLM不确定性，PlanU仅有轻微收敛速度下降，表现出强鲁棒性

## 亮点

- **问题定义清晰**：首次系统区分并同时处理LLM不确定性和环境不确定性，通过简单股票投资实验直观展示了现有方法的根本缺陷
- **分位数分布建模**：用分位数分布替代均值建模MCTS节点回报，既能捕获不确定性的形状（偏斜vs均匀），又可利用分位数回归进行稳健更新
- **UCC设计巧妙**：融合RND思想的新颖性奖励 + 文本编码器消除LLM文本不确定性，形成了完整的探索机制
- **跨场景泛化能力强**：在5个基准（积木堆叠、烹饪、家居、旅行规划、网购）和3种LLM上均表现最优

## 局限与展望

- **计算开销大**：分位数分布维护 + RND网络训练 + 文本编码器推理，相比标准MCTS增加了显著开销，论文未报告运行时间对比
- **环境不确定性是人工注入的**：Blocksworld等原始环境是确定性的，通过添加固定失败率模拟随机性，未在天然随机环境中验证
- **仅适用于文本环境**：所有环境的状态和动作都是文本描述，未涉及视觉或连续状态空间
- **超参数敏感性未充分探讨**：分位数数量$n_q$、UCC系数$c_1$等关键超参的影响未充分消融
- **规模有限**：仅在7B-8B级别LLM上实验，未验证更大模型是否仍需要此框架
- **分位数回归的收敛性分析缺失**：在有限MCTS迭代下，分位数分布的收敛保证不明

## 与相关工作的对比

- **RAP (Hao et al., EMNLP 2023)**：LLM-MCTS框架，但假设确定性转移，通过多次查询取最频繁状态处理不确定性，本文证明该策略在随机环境下导致次优
- **LATS (Zhou et al., ICML 2024)**：集成自反思和API调用的LLM-MCTS，同样不处理环境不确定性，在TravelPlanner和WebShop上均逊于PlanU
- **DeLLMa (Liu et al., ICLR 2025)**：基于经典决策理论的单步LLM决策，不适用于多步交互任务
- **RAP-D / RAP-E**：将RAP中的MCTS替换为DMCTS/EMCTS（考虑不确定性的MCTS变体），表现优于RAP但仍逊于PlanU
- **QR-DQN (Dabney et al.)**：RL中的分位数分布方法，PlanU借鉴其分位数回归思想但应用于LLM-MCTS场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — 分位数分布+MCTS+LLM的结合新颖，UCC设计有创造性
- 实验充分度: ⭐⭐⭐⭐ — 5个基准、3种LLM、多种消融，涵盖面广
- 写作质量: ⭐⭐⭐⭐ — 股票投资的motivating example直观清晰，整体逻辑流畅
- 价值: ⭐⭐⭐⭐ — 填补LLM决策中环境不确定性处理的空白，对LLM Agent在真实随机环境部署有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](../../ICLR2026/time_series/timeomni-1_incentivizing_complex_reasoning_with_time_series_in_large_language_mo.md)
- [\[NeurIPS 2025\] Human-Machine Ritual: Synergic Performance through Real-Time Motion Recognition](human-machine_ritual_synergic_performance_through_real-time_motion_recognition.md)
- [\[NeurIPS 2025\] Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift](frequency_matters_when_time_series_foundation_models_fail_under_spectral_shift.md)
- [\[NeurIPS 2025\] CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models](causaldynamics_a_large-scale_benchmark_for_structural_discovery_of_dynamical_cau.md)
- [\[ACL 2025\] G2S: A General-to-Specific Learning Framework for Temporal Knowledge Graph Forecasting with Large Language Models](../../ACL2025/time_series/g2s_a_general-to-specific_learning_framework_for_temporal_knowledge_graph_foreca.md)

</div>

<!-- RELATED:END -->
