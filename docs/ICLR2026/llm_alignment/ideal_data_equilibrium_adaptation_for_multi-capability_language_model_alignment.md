---
title: >-
  [论文解读] IDEAL: Data Equilibrium Adaptation for Multi-Capability Language Model Alignment
description: >-
  [ICLR 2026][LLM对齐][数据混合] IDEAL 把"SFT 各领域数据该配多少量"建模成一个双层优化问题，用二阶（Hessian）梯度信息算出每个领域数据应该上采样还是下采样，迭代两轮就能让数学/代码/推理/指令跟随四项能力整体均衡提升约 7%。 领域现状：LLM 通过在多领域指令数据上做 SFT 获得通用能力…
tags:
  - "ICLR 2026"
  - "LLM对齐"
  - "数据混合"
  - "监督微调"
  - "多能力对齐"
  - "影响函数"
  - "双层优化"
  - "K-FAC"
---

# IDEAL: Data Equilibrium Adaptation for Multi-Capability Language Model Alignment

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=n9wS0Hdvri](https://openreview.net/forum?id=n9wS0Hdvri)  
**代码**: [https://github.com/ming-bot/IDEAL](https://github.com/ming-bot/IDEAL)  
**领域**: LLM 对齐 / SFT 数据配比  
**关键词**: 数据混合, 监督微调, 多能力对齐, 影响函数, 双层优化, K-FAC  

## 一句话总结
IDEAL 把"SFT 各领域数据该配多少量"建模成一个双层优化问题，用二阶（Hessian）梯度信息算出每个领域数据应该上采样还是下采样，迭代两轮就能让数学/代码/推理/指令跟随四项能力整体均衡提升约 7%。

## 研究背景与动机
**领域现状**：LLM 通过在多领域指令数据上做 SFT 获得通用能力，而当把数学、代码、推理、指令跟随等异构任务混在一起训练时，各领域数据的"配比量"直接决定了最终模型的能力上限。

**现有痛点**：天真地把多领域数据简单拼起来训练，往往比单任务专精还差——会出现"木桶效应"，整体被最弱的那一项拖住。已有的自动配比方法（DoReMi、DOGE、Data Mixing Law）大多源自预训练场景，依赖训练一个小代理模型 + 全局权重搜索，既忽略了 SFT 阶段"数据-任务对齐直接引发跨领域干扰"的独特动态，又需要昂贵的超参扫描。

**核心矛盾**：另一条路线是影响函数（influence function）做数据选择，但它聚焦单个样本、按实例打分，**无法处理"调整整个数据集分布"这种领域级配比问题**。于是"如何在多能力 SFT 中有原则地解决数据冲突"始终是开放问题。

**本文目标**：在已有高质量多领域数据的前提下，找到一个最优的领域配比——注意这里的"均衡"不是各领域数据等量，而是让所有能力都能正常发育的最优分布。

**核心 idea**：**[模型感知的梯度引导配比]** 引入一个领域级超参数 $\beta\in\mathbb{R}^n$ 控制每个领域数据相对原始量的重复/削减比例，把"找最优配比"写成以参考集损失为外层目标的双层优化，再用二阶信息一步算出 $\beta$ 的最优下降方向，迭代上/下采样逼近均衡。

## 方法详解

### 整体框架
给定基座模型 $M_0$ 和按领域切分的训练集 $\{D_1,\dots,D_n\}$，IDEAL 通过一个外部参考集 $D_\text{ref}$ 作为多任务性能的统一度量。每轮先在当前配比下把模型训到收敛，再用二阶梯度算出每个领域的调整系数 $\beta$，据此对各领域数据做上采样（$\beta_i>0$，重复数据）或下采样（$\beta_i<0$，削减数据），重构训练集后进入下一轮，通常 2 轮即可达到均衡。

```mermaid
flowchart LR
    A[基座 M0 + 初始配比 Dtr] --> B[训练至收敛得 Mt]
    B --> C[在参考集 Dref 评估]
    C --> D[二阶梯度算 β<br/>K-FAC 近似 Hessian逆]
    D --> E[各领域上/下采样<br/>Di ← 1+βi · Di]
    E --> F{满足停止条件?}
    F -- 否 --> B
    F -- 是 --> G[输出均衡模型]
```

### 关键设计

**1. 把配比写成双层优化，用链式法则求外层梯度**：受 Muennighoff 等人"重复已有数据至多 4 次≈引入等量新数据"的发现启发，IDEAL 用 $\beta_i$ 控制领域 $i$ 重复数据的比例，重新定义最优参数为 $\theta^*=\frac{1}{N+\sum_i\beta_i|D_i|}\arg\min_\theta\big(L(D_\text{tr},\theta)+\sum_i\beta_i L(D_i,\theta)\big)$。外层目标是最小化参考集损失 $Q(\beta):=L(D_\text{ref},\theta^*)$，对某个 $\beta_j$ 求导用链式法则拆成 $\frac{\partial Q}{\partial\beta_j}=\frac{\partial L(D_\text{ref},\theta^*)}{\partial\theta^*}^\top\frac{\partial\theta^*}{\partial\beta_j}$。在初始状态 $\beta=(0,\dots,0)$ 处，利用隐函数定理可解得 $\frac{\partial\theta^*}{\partial\beta_j}=-\big[\nabla^2 L(D_\text{tr},\theta^*)\big]^{-1}\nabla L(D_j,\theta^*)$，于是配比方向最终由"参考集梯度 × Hessian 逆 × 领域梯度"这个二阶量决定。

**2. 用 K-FAC 把 Hessian 逆算得动**：对 8B 模型直接求 Gauss-Newton Hessian 的逆是不可行的，IDEAL 借 K-FAC 理论把 Hessian 按 MLP 层近似成块对角，每层用 Kronecker 积分解 $H_l=\mathbb{E}(x_l x_l^\top)\otimes\mathbb{E}(\delta_l\delta_l^\top)=X_l\otimes\Delta_l$（$x_l$ 是层输入、$\delta_l$ 是反传误差），再对 $X_l,\Delta_l$ 各做特征分解 $X_l=Q_{X_l}\Lambda_{X_l}Q_{X_l}^\top$ 来释放显存。这样 iHVP（逆 Hessian-向量积）的计算从全局矩阵求逆降为逐层的小矩阵运算，使二阶方法首次能在大模型 SFT 上落地。

**3. 按特征值方差挑"重要层"并用 γ 缩放补偿幅度**：特征分解后的 $\Lambda$ 度量了伪梯度在各 K-FAC 特征向量上的方差，方差越低的 MLP 层越稳定，IDEAL 只保留这些"重要层"参与计算以进一步省显存。但只算部分层会让最终 $\beta$ 幅度偏小，于是引入动态缩放向量 $\gamma$ 把 $\beta$ 中绝对值最大者线性放大到预设值 $m$：$\alpha=\frac{\partial Q(\beta)}{\partial\beta}\big|_{\beta=0}$，$\beta=-\gamma\odot\alpha$，$\gamma=\frac{m}{\max|\alpha|}$，既保留各领域调整方向的相对关系，又把整体步长控制在可控范围（实验取 $m=0.15$）。

**4. 上/下采样实现配比、随机采样降耦合**：拿到 $\beta$ 后，IDEAL 按 $D_{i,t+1}\leftarrow(1+\beta_i)D_{i,t}$ 对各领域做上采样（重复）或下采样（删减），用**随机采样**而非任何选择算法来增删数据。这一刻意的选择既节省算力、加快处理，又最大限度地把 IDEAL 的增益与"样本选择质量"解耦——保证提升来自配比本身而非数据选择，从而把整体算法收敛在初始分布附近做小幅搜索，稳定性优于 DoReMi/DOGE 的大幅波动。

## 实验关键数据

设置：基座 LLaMA3.1-8B 全参微调，四领域（数学/代码/推理/指令跟随），评测用 GSM8K / HumanEval / BBH / IFEval，OpenCompass 平台，每实验重复 5 次取均值，$m=0.15$、采样因子 $\sigma=0.5$。

### 主实验（Overall 平均分）

| 方法 | Epoch=1 Overall | Epoch=3 Overall |
|------|:---:|:---:|
| Base Model | 39.55 | — |
| 最佳 Specific SFT | 47.17 | 49.86 |
| Joint SFT (D0) | 54.79 | 55.35 |
| Random (最佳) | 55.64 | 55.97 |
| DoReMi (最佳) | 55.17 | 56.25 |
| DOGE (最佳) | 54.71 | 56.87 |
| **IDEAL (D2)** | **57.87** | **59.23** |

Epoch=1 时 IDEAL 把 HumanEval（代码）从 Joint 的 41.26 拉到 **50.61（+9.35）**，且不牺牲其它任务。

### 扩展实验（5 领域 + 8 benchmark，Epoch=3）
新增 MATH / ARC-C / MBPP / TruthfulQA 与 TrustAI 领域，并加入 D0(FULL)（约 66k 全量数据）对照，验证 IDEAL 在均衡初始分布的更难设定下仍能稳健提升。

### 关键发现
- **初始分布天然次优**：Joint SFT 虽优于多数 Specific SFT，但存在木桶效应；Random 虽能撞出不同结果却缺乏稳定性。
- **2 轮即达均衡**：IDEAL 第二轮显著增大代码数据量、略减数学/指令数据，呈现"定向增强弱项"的能力，且数据量变化比 DoReMi/DOGE 更平滑。
- **HumanEval 随训练时长退化**：训 3 epoch 在多数领域更好，但 HumanEval 反而不如 Epoch=1 最优——次优配比引入的数据冲突会被更长训练放大。
- **不同 epoch 优化优先级不同**：Epoch=3 下四领域都倾向"加数据"（更多梯度更新对抗记忆化），Epoch=1 则更倾向局部微调各领域数据量。

## 亮点与洞察
- **把"数据配多少"从工程玄学变成有理论保证的优化问题**：用双层优化 + 隐函数定理把领域配比的最优方向解析出来，不再靠人工 reweight 或规则课程学习。
- **二阶方法在大模型上"算得动"是真正的工程贡献**：K-FAC 块对角 + 特征分解 + 重要层筛选 + γ 缩放，一整套把 iHVP 降到可行的组合拳。
- **刻意用随机采样而非数据选择**，在方法层面把"配比增益"与"选择增益"干净地解耦，让结论更可信。
- **"均衡≠等量"** 的洞察很关键——最优分布往往在初始分布附近做小幅非对称调整。

## 局限与展望
- 仅在 LLaMA3.1-8B 全参微调上验证，配比的最优性是否随模型规模/架构迁移（虽有附录补充但主结论基于 8B）仍需更多验证。
- 每轮都要把模型训到收敛再算 $\beta$，迭代 2 轮意味着 2~3 次完整 SFT，总开销不低；β 的最优性也依赖参考集 $D_\text{ref}$ 的代表性。
- 领域是人工预先切分的（math/code/…），对领域边界模糊或长尾领域的适用性未讨论。
- 训练超参（lr、batch、epoch）声明"未必最优"但全程固定，配比与超参的联合优化是自然的下一步。

## 相关工作与启发
- **数据混合**：DoReMi（Group DRO 训代理模型）、DOGE（最小化反传梯度差异定权重）、Data Mixing Law（拟合配比-验证损失关系）——这些多源自预训练、需全局权重搜索且忽略分布连续性，IDEAL 的"梯度引导迭代式小幅精修"是对其的针对性改进。
- **数据选择**：LESS（一阶梯度对齐目标分布）、SelectIT（用 LLM 内部不确定性选数据）等聚焦实例级，与 IDEAL 的领域级配比正交、可互补。
- **影响函数**：Koh & Liang 的经典框架与 MATES 等代理模型估分方法，启发了 IDEAL 用二阶信息度量"领域数据→参考集性能"的影响，但 IDEAL 把粒度从单样本提升到了领域分布。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把 SFT 数据配比建成双层优化、用二阶/影响函数视角从样本级提升到领域级，并真正在 8B 上跑通二阶计算，思路扎实且填补了 SFT 阶段配比的空白。
- **实验充分度**: ⭐⭐⭐⭐ 四领域主实验 + 五领域八基准扩展 + 多 baseline（Specific/Joint/Random/DoReMi/DOGE）+ 重复 5 次取均值，分析细致；稍欠多基座/多规模的系统对照。
- **写作质量**: ⭐⭐⭐⭐ 问题定义清晰、公式推导完整、发现分析有洞察，K-FAC 部分对读者门槛偏高但整体可读。
- **价值**: ⭐⭐⭐⭐ 给"多能力 SFT 该怎么配数据"提供了实用且有理论支撑的工具，整体 +7%、代码开源，对训练通用模型有直接落地意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Towards Understanding Valuable Preference Data for Large Language Model Alignment](towards_understanding_valuable_preference_data_for_large_language_model_alignmen.md)
- [\[ICLR 2026\] Multi-objective Large Language Model Alignment with Hierarchical Experts](multi-objective_large_language_model_alignment_with_hierarchical_experts.md)
- [\[ICLR 2026\] Capability-Based Scaling Trends for LLM-Based Red-Teaming](capability-based_scaling_trends_for_llm-based_red-teaming.md)
- [\[ICLR 2026\] Semantic-aware Wasserstein Policy Regularization for Large Language Model Alignment](semantic-aware_wasserstein_policy_regularization_for_large_language_model_alignm.md)
- [\[ICLR 2026\] Test-Time Alignment for Large Language Models via Textual Model Predictive Control](test-time_alignment_for_large_language_models_via_textual_model_predictive_contr.md)

</div>

<!-- RELATED:END -->
