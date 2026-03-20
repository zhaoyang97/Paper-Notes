# Adaptively Coordinating with Novel Partners via Learned Latent Strategies

**会议**: NeurIPS 2025  
**arXiv**: [2511.12754](https://arxiv.org/abs/2511.12754)  
**代码**: 待确认  
**领域**: human_understanding  
**关键词**: zero-shot coordination, ad hoc teamwork, latent strategy, VAE, regret minimization, human-agent collaboration  

## 一句话总结

提出 TALENTS 框架，通过 VAE 学习潜在策略空间 + K-Means 聚类发现策略类型 + Fixed-Share 遗憾最小化算法在线推断队友类型，实现对未知人类/智能体队友的零样本实时适应协作。

## 研究背景与动机

1. **Ad Hoc Teamwork 的核心挑战**：智能体需要与未知队友（人类或AI）实时协作，但不同队友的行为模式、偏好、技能水平差异巨大，且可能在交互过程中动态变化。
2. **Self-Play 的局限**：自博弈训练的智能体在合作场景中往往形成刚性行为模式（rigid conventions），难以适配多样化队友。
3. **Population-Based Training (PBT) 的不足**：虽然 PBT 方法（如 FCP、MEP）通过训练多样群体来扩展策略覆盖，但仍然受限于离散的有限群体规模，难以覆盖人类行为的连续分布。
4. **已有策略推断方法的缺陷**：直接将新队友的轨迹编码到潜在空间存在分布偏移问题（训练时 vs 测试时轨迹分布不同），尤其在与人类交互时表现脆弱。
5. **关键洞察**：将 PBT 的策略多样性生成与在线策略推断统一起来——用同一个潜在空间既生成训练伙伴，又在测试时推断队友类型。
6. **非平稳问题**：队友可能在单个 episode 内多次切换策略（受伙伴影响或持续学习），标准静态遗憾最小化无法处理这种 intra-episodic 策略变化。

## 方法详解

### 整体框架

TALENTS（Team Adaptation via LatEnt No-regreT Strategies）分为三个阶段：

```
离线轨迹数据 → VAE 策略空间学习 → K-Means 聚类
                                          ↓
              策略条件化 Cooperator 训练 ← 从聚类中采样生成伙伴
                                          ↓
              在线部署: Fixed-Share 算法推断队友类型 → 条件化执行
```

### 三大核心设计

**设计一：VAE 策略空间学习**

- **输入**：来自群体智能体（FCP/MEP/BP）联合 rollout 的离线轨迹数据 $\mathcal{D}_{traj} = \{\tau_i\}_{i=1}^N$
- **编码器** $q_\phi(z|\tau)$：将轨迹窗口（长度 $h$）编码为多元高斯分布 $\mathcal{N}(\mu_\phi(\tau), \Sigma_\phi(\tau))$，潜在维度 = 8
- **解码器** $p_\theta(a_{t:t+H}|z, o_t)$：给定潜在变量 $z$ 和当前观测 $o_t$，预测未来 $H$ 步动作序列（$H=50$），捕捉长期意图
- **聚类**：在潜在空间上执行 K-Means + 轮廓分析（Silhouette Analysis），自动确定最优聚类数 $K$ 和聚类中心

**设计二：策略条件化 Cooperator 训练**

- 每个 episode 随机采样一个策略聚类 $c$，从其潜在均值 $\mu_c$ 采样 $z$，通过解码器生成伙伴动作
- **关键机制——动作偏置（Action Bias）**：用嵌入矩阵 $E$ 为每个聚类学习一个偏置向量 $b_c = E[c]$，加到 actor 网络的 logits 上：$\tilde{l}_t = l_t + b_c$
- 这种设计显式鼓励/抑制 cooperator 在不同队友类型下采取不同动作，比直接拼接观测更有效
- 使用 Priority-based Sampling 调整聚类采样权重（根据历史回报），用 Independent PPO 训练

**设计三：Fixed-Share 在线适应**

- 每个策略聚类作为一个"专家"，初始权重均匀分配 $w^1 = (1/K, \ldots, 1/K)$
- 每步：各专家根据自身潜在变量解码预测队友动作 $\hat{a}_t^c$
- 下一步观测到队友实际动作 $a_t^p$ 后，计算各专家损失：$\ell_c^t = -\log p_\theta(a_t^p | z_c, o_t)$
- 指数权重更新 + **权重共享**：$w_c^{t+1} = (1-\alpha)\tilde{w}_c^{t+1} + \alpha \sum_j \tilde{w}_j^{t+1} / K$
- 权重共享参数 $\alpha$ 使算法能追踪非平稳队友（允许 $m-1$ 次策略切换），遗憾上界为 $O(\sqrt{T(m\ln N + m\ln(T/m))})$

### 损失函数

VAE 训练优化 ELBO：

$$\mathcal{L}(\theta, \phi; \tau) = \mathbb{E}_{z \sim q_\phi}[\log p_\theta(a_{t:t+H}|z, o_t)] - \beta D_{KL}(q_\phi(z|\tau) \| p(z))$$

其中 $\beta$ 在训练过程中线性退火（KL annealing），平衡重构精度与潜在空间正则化。

## 实验

### 环境设置

改进版 Overcooked-ai：增加订单计时器和快速交付奖励，包含三个烹饪站和两种食谱，四种地图布局（Open、Hallway、Forced-Coord、Ring）。

### Agent-Agent 零样本协调（Table 1）

| 群体 | 方法 | Open | Hallway | Forced-Coord | Ring |
|------|------|------|---------|-------------|------|
| FCP | **TALENTS** | **710.36±88.75** | **635.59±107.54** | 34.38±6.59 | **596.19±33.34** |
| FCP | GAMMA | 616.67±14.99 | 537.60±26.75 | 38.36±6.65 | 395.03±10.09 |
| FCP | BR | 427.07±14.17 | 366.14±70.50 | **56.09±12.33** | 288.61±31.85 |
| BP | **TALENTS** | **842.39±36.47** | **642.94±81.00** | 56.55±9.09 | **647.62±16.96** |
| BP | GAMMA | 573.93±52.69 | 513.47±34.06 | 47.52±2.61 | 387.58±17.41 |

### Human-Agent 零样本协调（119 名被试）

- **团队得分**：TALENTS 显著优于两个基线（ANOVA: $F(2,166)=5.76, p=.003$）
- **主观评价**：团队流畅度（$F(2,122)=4.31, p=.02$）和信任度（$F(2,122)=3.23, p=.04$）均显著高于 BR 基线
- 工作负荷（NASA-TLX）、协调性、满意度方面也有正向趋势

### 关键实验发现

1. **TALENTS 在 3/4 地图上超越所有基线**，唯独 Forced-Coord 布局表现较弱——该布局职责划分明确，策略探索型方法因稀疏奖励信号受损
2. **Fixed-Share vs 静态遗憾消融**：当 episode 中途切换队友策略时，静态遗憾方法无法更新信念，后半段奖励下降；Fixed-Share 能追踪策略切换并恢复性能（Fig. 4）
3. **跨群体一致性**：无论基于 FCP、MEP 还是 BP 群体训练，TALENTS 都展现出最高或接近最高的性能
4. **Action Bias > Observation Concatenation**：动作偏置法比将聚类嵌入拼接到观测中更有效地学习策略特异性行为

## 亮点

- **统一框架**：同一个 VAE 潜在空间同时服务于训练时的伙伴生成和测试时的策略推断，设计优雅
- **理论保障**：Fixed-Share 提供了形式化的 tracking regret bound，对非平稳队友有数学保证
- **即使没有人类数据也能适配人类**：仅用 agent 数据训练，却在 119 人用户研究中显著优于基线
- **动作偏置机制**简单高效——仅用一个嵌入矩阵就实现策略条件化，不增加观测维度

## 局限性

1. **Forced-Coord 布局表现不佳**：当任务有明确分工时，策略多样性探索反而引入噪声，优先采样偏向低技能伙伴
2. **泛化受限于 VAE 的插值能力**：只能泛化到与训练数据行为相似的新队友，本质上仍受限于训练群体的策略覆盖
3. **人为降速以匹配人类**：人类每秒约 3-4 个动作，而智能体每步都行动（10 steps/s），实验中人为限制了智能体的动作频率，可能影响长期规划能力
4. **仅在 Overcooked 环境验证**：虽然难度已提升，但向更复杂的真实任务（如机器人协作）迁移尚未验证
5. **聚类数 K 固定**：K-Means + 轮廓分析虽自动化，但可能无法捕捉策略空间的层次结构或连续渐变

## 相关工作

- **Zero-Shot Coordination**：Self-Play → FCP (中间检查点模拟不同水平) → MEP (最大熵群体) → E3T (混合自博弈与随机策略) → GAMMA (生成模型扩展策略覆盖) → **TALENTS** (聚类+条件化+在线推断)
- **Strategy Inference**：潜在嵌入方法（LIAM, CoDAG）、贝叶斯推断（MeLIBA）、学徒学习+混合专家（zhao2022）、交叉熵策略选择（9540646），TALENTS 创新点在于用同一潜在空间同时生成和推断
- **Theory of Mind**：ProAgent、ToMnet 等预测伙伴意图/信念的方法，与本工作互补但侧重不同

## 评分

- **新颖性**: ⭐⭐⭐⭐ — VAE 聚类 + Fixed-Share 在线适应的组合具有原创性，将 PBT 与策略推断统一到同一潜在空间是清晰的贡献
- **实验充分度**: ⭐⭐⭐⭐ — 涵盖 3 种群体（FCP/MEP/BP）× 4 种地图 + 119 人用户研究 + 消融实验，人类实验设计规范（混合设计、主观问卷）
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，数学形式化完整（HiP-MDP、tracking regret bound），算法伪代码详尽
- **价值**: ⭐⭐⭐⭐ — human-agent 协作的实用方法，人类实验结果有统计显著性，框架可推广到其他合作场景
