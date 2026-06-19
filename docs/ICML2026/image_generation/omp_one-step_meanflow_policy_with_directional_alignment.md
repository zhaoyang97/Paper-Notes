---
title: >-
  [论文解读] OMP: One-step Meanflow Policy with Directional Alignment
description: >-
  [ICML2026][图像生成][MeanFlow] 本文针对将 MeanFlow 范式直接搬到机器人操作时暴露出的三个理论病灶（频谱偏差、低速区梯度饥饿、嵌套 JVP 内存爆炸），提出 OMP：用一项 cosine-style 方向对齐损失把预测平均速度与真实平均速度方向"锁死"，再用有限差分 DDE 近似 Jacobian-Vector Product 解耦前后向，让单步（NFE=1）生成策略在 Adroit/Meta-World 上以 6.8ms 级延迟做到比 MP1 平均高 3.4%、在 Meta-World Very Hard 任务高 10.6% 的成功率。
tags:
  - "ICML2026"
  - "图像生成"
  - "MeanFlow"
  - "单步策略"
  - "方向对齐"
  - "JVP 有限差分"
  - "机器人操作"
---

# OMP: One-step Meanflow Policy with Directional Alignment

**会议**: ICML2026  
**arXiv**: [2512.19347](https://arxiv.org/abs/2512.19347)  
**代码**: 待确认  
**领域**: 机器人 / 具身智能 / 生成式策略  
**关键词**: MeanFlow, 单步策略, 方向对齐, JVP 有限差分, 机器人操作

## 一句话总结
本文针对将 MeanFlow 范式直接搬到机器人操作时暴露出的三个理论病灶（频谱偏差、低速区梯度饥饿、嵌套 JVP 内存爆炸），提出 OMP：用一项 cosine-style 方向对齐损失把预测平均速度与真实平均速度方向"锁死"，再用有限差分 DDE 近似 Jacobian-Vector Product 解耦前后向，让单步（NFE=1）生成策略在 Adroit/Meta-World 上以 6.8ms 级延迟做到比 MP1 平均高 3.4%、在 Meta-World Very Hard 任务高 10.6% 的成功率。

## 研究背景与动机
**领域现状**：生成式机器人策略当前主流是把动作生成建模为概率去噪过程，Diffusion Policy / DP3 这类扩散策略靠 10 步左右的迭代去噪拿到了高成功率，但 NFE=10 带来的推理延迟挤死了高频闭环控制；为提速，FlowPolicy、ManiFlow 这类基于 flow matching 或一致性蒸馏的方法把推理压到单步，但训练侧要靠分段直线流或显式一致性约束，过强的架构约束又损失了泛化。

**现有痛点**：MeanFlow（2025）从理论上给出了一条更干净的"单步化"路径——直接学习区间平均速度 $u(z_t, r, t)$，绕开 ODE 求解器；其在机器人侧的落地 MP1 把延迟压到 6.8 ms。但作者发现把 MeanFlow 照搬到机器人，会暴露三个 image generation 场景看不见的病灶。

**核心矛盾**：图像生成里像素级动态范围大、梯度信号充足，掩盖了 MeanFlow 目标自身的频谱与几何缺陷；而机器人动作空间维度低、精细任务里真实平均速度 $\|v_0\|$ 接近 0，三个理论病灶集中爆发：(1) **频谱偏差**——时间积分相当于除以 $i\omega$，目标 PSD 按 $1/\omega^2$ 衰减，等价于低通滤波器，把精细操作里的高频方向调整全压掉；(2) **梯度饥饿**——MSE 损失对角度误差的梯度为 $2\rho\rho^*\sin\alpha$，与目标幅度 $\rho^*$ 乘性耦合，$\rho^*\to 0$ 时模型干脆把自己输出收缩到 0 而不去对齐方向；(3) **内存复杂度**——MeanFlow Identity 里的总导数展开后含 JVP $\nabla_z u\cdot dz/dt$，对其求 $\nabla_\theta$ 等价于嵌套 Forward-AD + Reverse-AD，必须同时保存 primal/tangent/adjoint 三套激活，大点云骨干训不动。

**本文目标**：(a) 拆掉 MSE 把方向和幅度强耦合的设定，让方向监督在低速区不消失；(b) 用一个不需要符号微分的近似替换 JVP，把训练内存压回标准 backprop 水平；(c) 保持 NFE=1 的推理速度。

**切入角度**：既然根本病因是 MSE 把方向和幅度绑死、又是 JVP 的解析展开把内存炸开，那就直接绕过去——用 cosine 把方向单拎出来当一项独立损失，用中心差分近似时间导数。

**核心 idea**：用方向对齐损失把预测平均速度的"指向"显式锁到真实平均速度 $v_0$ 上，叠加一个 $O(\epsilon)$ 中心差分代替 JVP 解耦前后向。

## 方法详解

### 整体框架
OMP 把 MeanFlow 的"学区间平均速度、一步出图"思路搬到机器人操作，但针对低维动作空间里暴露的频谱偏差、低速区梯度饥饿、嵌套 JVP 内存爆炸三个病灶做了修复。整体仍是 MP1 那套架子：输入 3D 点云观测（FPS 下采到 512 或 1024 点）加 2 步观测历史，模型 $u_\theta(z_t, r, t \mid c)$ 学习时刻 $r,t$ 之间的平均速度并遵循 MeanFlow Identity

$$u(z_t,r,t|c)=v(z_t,t|c)-(t-r)\dfrac{d}{dt}u(z_t,r,t|c)$$

把右边当 target；推理时一次正向从噪声 $z_T\sim\mathcal{N}(0,I)$ 直接走到动作 $z_0$，并取 $v_0 \triangleq z_T - z_0$ 作为真实平均速度。OMP 的两处改动都叠在 MP1 的 $\mathcal{L}_{mse}+\lambda_{Disp}\mathcal{L}_{Disp}$ 之上：加一项方向对齐损失 $\mathcal{L}_{DA}$ 治几何病灶，把 Identity 里的 $\frac{d}{dt}u$ 换成中心差分治内存病灶，后者衍生出保留解析 JVP 的 OMP-JVP 和用差分近似的 OMP-DDE 两个版本。

### 关键设计

**1. 方向对齐损失 $\mathcal{L}_{DA}$：把方向从幅度里解耦出来，治低速区梯度饥饿**

针对的是 MSE 在精细操作里失灵的根因。作者在 §4.2.2 用法-余弦定理拆开 MSE 对角度的梯度，得到 $\partial\mathcal{L}_{MSE}/\partial\alpha = 2\rho\rho^*\sin\alpha$——角度梯度被目标幅度 $\rho^*$ 乘性压制，而机器人精细接触阶段真实平均速度 $\rho^*\approx 0$，于是 MSE 干脆鼓励模型把输出收缩到 $\rho\to 0$ 拿一个"静止策略"，方向永远学不对；同时目标经过时间积分相当于除以 $i\omega$，PSD 按 $1/\omega^2$ 衰减成低通滤波，把高频方向调整也压掉了。$\mathcal{L}_{DA}$ 的做法是先算 cosine 相似度 $\cos\alpha = \dfrac{v_0\cdot u}{\|v_0\|\cdot\|u\|}$（分母加 $\epsilon_{dir}\approx 10^{-6}$ 防除零），再写成对数形式 $\mathcal{L}_{DA}=-\log\!\big(\frac{\cos\alpha+1}{2}\big)$。这个损失只取决于方向、不取决于幅度，所以 $\|v_0\|\to 0$ 时方向梯度不塌缩；对数形式还让 $\cos\alpha=-1$（完全走反）处梯度发散、给最强惩罚，$\cos\alpha=1$ 处趋零。落到训练上，弹道阶段（大平移）由 $\mathcal{L}_{mse}$ 主导走幅度、接触阶段由 $\mathcal{L}_{DA}$ 主导走方向，两阶段都有非零梯度，还顺手绕过了 $1/\omega^2$ 低通——因为对齐目标 $v_0$ 不再经过时间积分。

**2. Differential Derivation Equation（DDE）：用中心差分替掉解析时间导数，治嵌套 AD 内存爆炸**

针对的是 MeanFlow Identity 里 $\frac{d}{dt}u$ 直接实现时的内存代价。§4.2.3 算了一笔账：该总导数展开含 JVP $\nabla_z u_\theta\cdot v$，再对它求 $\nabla_\theta$ 等价于二阶混合偏导 $\partial^2 u/\partial\theta\partial z$，框架里要把 Forward-AD 嵌套在 Reverse-AD 外面，必须同时保存原始激活 $X$、tangent $\delta X$、tangent 的 adjoint 三套计算图，PointNet++/Transformer 这种点云骨干根本喂不进单卡 4090。DDE 把时间导数近似成中心差分 $\dfrac{du_\theta(z_t,t,r|c)}{dt}\approx\dfrac{u_\theta(z_{t+\epsilon},t+\epsilon,r|c)-u_\theta(z_{t-\epsilon},t-\epsilon,r|c)}{2\epsilon}$（$\epsilon$ 是小扰动常数，敏感度扫描见 §E.2），训练图里只剩两次普通 forward 加一次普通 backward，不再需要存 tangent 激活，内存回到标准 backprop 量级。代价只是 $O(\epsilon^2)$ 的截断误差——它的真正价值不在"近似得多准"，而在把前后向计算图解耦开。

**3. 组合损失与 JVP/DDE 双版本：把内存优化做成可切换开关**

最终训练目标是 $\mathcal{L}=\mathcal{L}_{mse}+\lambda_{Disp}\mathcal{L}_{Disp}+\lambda_{DA}\mathcal{L}_{DA}$，其中 $\mathcal{L}_{Disp}$ 沿用 MP1 的 dispersive loss 让特征空间更可分，三项分别提供幅度、特征判别、方向三种信号，靠加权和让模型在弹道段和接触段都拿到有效梯度。$\frac{d}{dt}u$ 的实现则拆成两套，把"内存-精度"这个真实 trade-off 留给使用者：OMP-JVP 保留解析 JVP 拿最佳精度（用作学术对照），OMP-DDE 用 DDE 近似换显存（用于实际部署），可以按点云尺寸、动作 horizon 等任务规模按需切换，而不用为内存被迫永久牺牲精度。

### 损失函数 / 训练策略
- 损失：$\mathcal{L}=\mathcal{L}_{mse}+\lambda_{Disp}\mathcal{L}_{Disp}+\lambda_{DA}\mathcal{L}_{DA}$；$\mathcal{L}_{DA}=-\log\!\big(\frac{\cos\alpha+1}{2}\big)$；DDE 时间步长 $\epsilon$ 在 §E.2 做了敏感度扫描。
- 数据：每个仿真任务 10 条专家演示；点云 FPS 到 512 或 1024 点；图像 84×84；观测 history=2、prediction horizon=4、execution horizon=3。
- 训练：AdamW，lr=1e-4，batch=128；Adroit 训 3000 epoch，Meta-World 训 1000 epoch，每 200 epoch 评估一次，最终成功率取前 5 高的平均、再跨种子 (0/10/20) 平均；硬件单卡 RTX 4090。

## 实验关键数据

### 主实验：Adroit + Meta-World 37 任务平均

| 方法 | NFE | Adroit Pen | MW Medium | MW Hard | MW Very Hard | 总平均 |
|------|-----|------------|-----------|---------|--------------|--------|
| DP (RSS'23) | 10 | 13±2 | 11.0±2.5 | 5.25±2.5 | 22.0±5.0 | 35.2±5.3 |
| DP3 (RSS'24) | 10 | 46±10 | 44.5±8.7 | 32.7±7.7 | 39.4±9.0 | 68.7±4.7 |
| FlowPolicy (AAAI'25) | 1 | 54±4 | 58.2±7.9 | 40.2±4.5 | 52.2±5.0 | 71.6±3.5 |
| MP1 (AAAI'26) | 1 | 58±5 | 68.0±3.1 | 58.1±5.0 | 67.2±2.7 | 78.9±2.1 |
| **OMP-JVP** | 1 | **60±4** | **77.4±2.2** | **62.5±3.1** | **77.8±3.0** | **82.3±1.6** |
| OMP-DDE | 1 | 64±3 | 76.4±2.7 | 61.0±3.0 | 70.6±4.9 | 80.8±2.2 |

OMP-JVP 在总平均上比 MP1 高 3.4%、比 FlowPolicy 高 10.7%；越难的任务 OMP 增益越大——Meta-World Medium +9.4%、Very Hard +10.6%。MP1 在 Easy 子集（21/37 任务）已逼近上限（88%+），把绝对增量主要拉低到了 1.5%。

### 真机实验（3 任务，成功率 %）

| 方法 | Place | Clean | Slip Ring |
|------|-------|-------|-----------|
| DP3 | 65 | 60 | 50 |
| FlowPolicy | 60 | 50 | 40 |
| MP1 | 70 | 65 | 55 |
| **OMP** | **80** | **75** | **70** |

最难的 Slip Ring 上 OMP 比 MP1 高 15%，验证了方向对齐在"真实低速精细操作"上的核心收益。

### 消融 + 内存

| 配置 | 总平均成功率 | 说明 |
|------|--------------|------|
| OMP-JVP (Full) | 82.3 | 完整模型 |
| − $\mathcal{L}_{Disp}$ | 81.2 | 去 dispersive，掉 1.1%（小） |
| − $\mathcal{L}_{DA}$ | 78.9 | 去方向对齐，掉 3.4%，回到 MP1 水平 |
| − $\mathcal{L}_{Disp}$ − $\mathcal{L}_{DA}$ | 78.3 | 全去，Adroit Pen 60→48 |
| OMP-DDE (Full) | 80.8 | 差分近似版 |
| − $\mathcal{L}_{DA}$ (DDE) | 77.2 | 同样验证方向对齐是核心 |

| 任务 / Horizon | OMP-JVP 显存 | OMP-DDE 显存 |
|----------------|--------------|--------------|
| Adroit Hammer / H=4 | 6.60 GB | 5.35 GB |
| Place Bottle / H=4 | 23.49 GB | 18.33 GB |
| Adroit Hammer / H=16 | 7.69 GB | 6.12 GB |
| Place Bottle / H=16 | **26.71 GB** | **19.19 GB** |

### 关键发现
- **方向对齐是主功臣**：去 $\mathcal{L}_{DA}$ 直接掉 3.4–3.6%，去 $\mathcal{L}_{Disp}$ 只掉 0.7–1.1%，证明病灶根源在 MSE 的几何耦合而不是特征判别。
- **OMP 增益与任务难度正相关**：Easy 任务上 MP1 已经把 ceiling 顶到 88%+，OMP 拉不开差距；Very Hard 上 +10.6% 说明方向对齐主要救的是低速精细任务，和理论预期完全吻合。
- **JVP→DDE 是精度-内存交易**：DDE 平均掉 1.5%（Very Hard 掉 7.2% 偏多），换来 Place Bottle/H=16 上 26.71 GB→19.19 GB 的 28% 显存下降；点云越大、horizon 越长，DDE 性价比越高。
- **训练曲线更稳**：Figure 5 显示 OMP 的成功率曲线方差远小于 FlowPolicy/MP1 的剧烈震荡，方向对齐顺带提升了训练稳定性。

## 亮点与洞察
- **把"频谱偏差 + 梯度饥饿 + 内存爆炸"打包成一个理论叙事**：作者没有只丢一个 cosine 损失，而是先用 PSD 频域分析、再用法-余弦定理给 MSE 角度梯度的闭式表达、再用 AD 图分析三套激活，把三个看似分离的问题串成同一个"MeanFlow 不适合机器人"的故事，给了 $\mathcal{L}_{DA}$ + DDE 强动机，论文的说服力主要来自这三段分析而不是数字本身。
- **cosine 损失的对数形式**：写成 $-\log((\cos\alpha+1)/2)$ 而不是直接 $1-\cos\alpha$，是个可复用的 trick——对数让 $\cos\alpha=-1$ 附近梯度发散，模型在"完全走反"时被强惩罚；而 $1-\cos\alpha$ 在反向时梯度反而最小，会陷在反向局部最优。
- **DDE 的真正价值不是"近似精度"，是"解耦计算图"**：跨任务可复用——任何需要 $du_\theta/dt$ 又要对 $\theta$ 反传的场景（不止 MeanFlow，还包括 score matching 的二阶变体、一些 NeuralODE 训练）都可以借这一招把内存压回标准 backprop 量级。
- **Easy/Hard 任务分桶汇报**：作者明确指出"在 Easy 子集 MP1 已经接近上限"，把总平均增益拆到 Easy/Medium/Hard/Very Hard 分别看，给出了任务难度 → 增益曲线，让方法的"适用边界"很清晰，值得任何"打榜接近饱和"的工作借鉴。

## 局限与展望
- **真机实验规模偏小**：只有 3 个任务、每任务 20 次试验（10% 的颗粒度可见），统计强度有限，没给标准差；Slip Ring +15% 这种数字需要更大样本支撑。
- **DDE 的 $\epsilon$ 是手调超参**：作者把敏感度分析放进附录但没给自适应方案，不同任务可能需要重调；理想的下一步是用 trajectory curvature 自适应或 second-order trust region 选 $\epsilon$。
- **没有和近期 OneDP/ManiFlow 等做基于蒸馏的单步方法详细对比**：表里 baseline 主要是 MP1/FlowPolicy 同代，缺一个 well-distilled DP3 的强基线（理论上多步教师 + 一致性蒸馏可能拿到接近 OMP 的精度）。
- **方向对齐对 multimodal 动作分布的影响未讨论**：精细操作里可能有多个等价方向（左手/右手）能完成任务，cosine 强制单一方向是否会丢失模式多样性、值得在 dexterous 任务上做 mode coverage 度量。
- **缺乏对模仿数据规模的扫描**：只用 10 条 demo，10→100 demo 增长曲线、$\mathcal{L}_{DA}$ 收益是否随 demo 增加而衰减没有给。

## 相关工作与启发
- **vs MP1 (AAAI'26)**：MP1 把 MeanFlow 第一次搬到机器人，已经做到 6.8 ms 单步推理，但训练目标仍是 MSE + Dispersive，对低速区方向梯度无能为力。OMP 的全部增量都建在 MP1 之上——同样 NFE=1、同样 dispersive，唯一新加的 $\mathcal{L}_{DA}$ 把总平均拉了 3.4%，证明方向对齐这一项就是 MP1 缺的最后一块拼图。
- **vs FlowPolicy (AAAI'25)**：FlowPolicy 用一致性流匹配 + 3D 点云做单步，但需要分段直线流和显式一致性约束。OMP 不需要任何分段或一致性约束，直接在 MeanFlow Identity 上做方向修正，工程上更干净，性能上 +10.7%。
- **vs DP3 (RSS'24)**：DP3 用 NFE=10 的多步去噪拿到 68.7% 总平均，OMP 用 NFE=1 拿到 82.3%——说明"单步"和"高质量"不再互斥，只要把目标信号本身的几何/频谱问题解决，就不用靠多步迭代来"修正"。
- **vs Consistency Policy / OneDP**：这两条线靠从多步 diffusion teacher 做一致性蒸馏拿到单步学生模型。OMP 是 from-scratch 训练，不依赖 teacher，但代价是必须处理 MeanFlow Identity 的二阶导，DDE 就是为解决这个工程代价而设计。
- **可迁移启发**：(a) cosine 对数损失对任何"目标 norm 接近 0、但方向仍重要"的回归问题都适用，例如 SLAM 里小位移姿态估计、HRI 里轻力反馈学习。(b) DDE 思路可以推广到任何 score matching 的 higher-order 变体——score 二阶量 + Fisher trace 训练长期受嵌套 AD 内存所限。(c) "按任务难度分桶报告增益"应该成为 robot learning benchmark 的标配，避免 Easy 任务的饱和稀释真实改进。

## 评分
- 新颖性: ⭐⭐⭐⭐ 单一 cosine 损失 + 中心差分都不算新数学，但用频谱/几何/AD 三段分析把它们组装成"MeanFlow 在机器人侧的修复套件"是真正的洞察。
- 实验充分度: ⭐⭐⭐⭐ 37 个仿真任务 × 3 种子 + 3 个真机任务 + 完整消融 + 显存对照表，但真机统计强度有限、缺与蒸馏类单步方法的强对照。
- 写作质量: ⭐⭐⭐⭐ 理论分析（§4.2 三个小节）结构清晰、动机推得很顺，主表难度分桶呈现是亮点；公式用了 v3 自动转 LaTeX 的格式偶尔有冗余。
- 价值: ⭐⭐⭐⭐ 给社区一个"MeanFlow 在低维动作空间的修复模板"，方向对齐损失和 DDE 都可独立复用到其他单步生成式策略框架。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation](../../AAAI2026/image_generation/mp1_meanflow_tames_policy_learning_in_1-step_for_robotic_manipulation.md)
- [\[ICML 2026\] Riemannian MeanFlow for One-Step Generation on Manifolds](riemannian_meanflow_for_one-step_generation_on_manifolds.md)
- [\[CVPR 2026\] Temporal Equilibrium MeanFlow: Bridging the Scale Gap for One-Step Generation](../../CVPR2026/image_generation/temporal_equilibrium_meanflow_bridging_the_scale_gap_for_one-step_generation.md)
- [\[CVPR 2026\] MeanFlow Transformers with Representation Autoencoders](../../CVPR2026/image_generation/meanflow_transformers_with_representation_autoencoders.md)
- [\[CVPR 2026\] Understanding, Accelerating, and Improving MeanFlow Training](../../CVPR2026/image_generation/understanding_accelerating_and_improving_meanflow_training.md)

</div>

<!-- RELATED:END -->
