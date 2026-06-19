---
title: >-
  [论文解读] Safe-Sim: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries
description: >-
  [ECCV 2024][自动驾驶][安全关键仿真] Safe-Sim 提出了一个基于扩散模型的闭环安全关键仿真框架，通过在扩散去噪过程中引入对抗项和部分扩散（Partial Diffusion）机制，实现了对抗车辆行为类型（碰撞角度、相对速度、碰撞类型）的细粒度控制，在 nuScenes 和 nuPlan 上验证了对多种 planner 的有效评估能力。
tags:
  - "ECCV 2024"
  - "自动驾驶"
  - "安全关键仿真"
  - "扩散模型"
  - "对抗生成"
  - "闭环仿真"
  - "可控性"
---

# Safe-Sim: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries

**会议**: ECCV 2024  
**arXiv**: [2401.00391](https://arxiv.org/abs/2401.00391)  
**代码**: [https://safe-sim.github.io/](https://safe-sim.github.io/) (有项目页)  
**领域**: 自动驾驶  
**关键词**: 安全关键仿真, 扩散模型, 对抗生成, 闭环仿真, 可控性

## 一句话总结

Safe-Sim 提出了一个基于扩散模型的闭环安全关键仿真框架，通过在扩散去噪过程中引入对抗项和部分扩散（Partial Diffusion）机制，实现了对抗车辆行为类型（碰撞角度、相对速度、碰撞类型）的细粒度控制，在 nuScenes 和 nuPlan 上验证了对多种 planner 的有效评估能力。

## 研究背景与动机

**领域现状：** 自动驾驶车辆的关键安全能力在于处理近碰撞事件，但此类事件在真实道路上极为罕见，且在公开道路上主动测试这类场景不安全也不合法。因此仿真是评估 AV 安全性的不可或缺手段，而建模其他道路参与者的行为是仿真的核心。

**现有痛点：**
1. **手工设计场景不可扩展**：传统方法依赖人工设计可能导致失败的场景，覆盖面有限；
2. **开环不够真实**：现有自动生成方法大多聚焦静态场景生成而非动态闭环，即其他 agent 不会根据 planner 的行为做出反应，无法真正测试交互能力；
3. **缺乏可控性**：已有方法通常每个场景只产生单一的对抗结果，不能探索不同条件和响应模式。研究者无法控制碰撞类型（正面、侧面、追尾）和严重程度。

**核心矛盾：** 需要同时满足三个要求——安全关键性（产生碰撞场景）、闭环反应性（所有 agent 对 planner 有响应）和可控性（能调节对抗行为模式），但此前没有工作能同时实现这三点。

**切入角度：** 利用扩散模型的可控生成能力——通过引导（guidance）注入对抗目标，通过部分扩散（partial diffusion）控制碰撞类型，通过在真实数据上训练保证真实性。

**核心idea：** 将安全关键仿真分解为两层控制——引导目标层（对抗碰撞 + 正则化约束）控制 agent 对 ego 的攻击程度，部分扩散层通过轨迹提案初始化控制碰撞具体类型，两者结合实现前所未有的可控对抗仿真。

## 方法详解

### 整体框架

Safe-Sim 在每个仿真步长中：(1) Ego 车辆由被测 planner $\pi$ 控制，产生未来轨迹；(2) 所有非 ego 的 reactive agents 由基于扩散的模型 $g_\theta$ 生成轨迹，其中一个或多个被指定为 adversarial agents；(3) 对抗 agent 通过引导扩散过程注入对抗目标和控制目标，非对抗 agent 仅受正则化引导以保持真实行为；(4) 执行首几步规划后更新观测，重新规划，形成闭环。模型在真实驾驶数据（nuScenes/nuPlan）上训练。

### 关键设计

1. **对抗引导扩散 (Guided Adversarial Diffusion)**:
    - 做什么：在每个去噪步骤中通过引导梯度扰动预测轨迹，使对抗 agent 趋向与 ego 碰撞。
    - 核心思路：采用 reconstruction guidance（干净数据引导），在每个去噪步对估计的干净轨迹 $\hat{\tau}_0$ 注入引导梯度：
     $$\tilde{\tau}_0 = \hat{\tau}_0 - \alpha \Sigma_k \nabla_{\tau_k} J(\hat{\tau}_0)$$
     总引导目标包含对抗项和正则化项：
     $$J(\tau) = \rho \underbrace{(J_{\text{coll}} + J_v + J_{\text{ttc}})}_{J_{\text{adv}}} + \underbrace{J_{\text{route}} + J_{\text{Gauss}}}_{J_{\text{reg}}}$$
     其中 $\rho$ 是标量权重，对抗 agent 设 $\rho > 0$，非对抗 agent 设 $\rho = 0$。
    - 设计动机：扩散模型在真实数据上训练后天然反映正常驾驶分布，引导机制既能注入对抗行为又不完全偏离真实性。通过 $\rho$ 轻松区分对抗和非对抗 agent，无需分别训练模型。

2. **碰撞引导目标 $J_{\text{coll}}$**:
    - 做什么：鼓励对抗 agent 与 ego vehicle 碰撞。
    - 核心思路：最小化轨迹规划期内对抗 agent 与 ego 的距离之和：
     $$J_{\text{coll}} = -\sum_{t=1}^T d(t)$$
    - 设计动机：最直接的碰撞促进目标。对抗 agent 根据车道接近性或距离动态选择。

3. **安全关键性控制目标 $J_v$ 和 $J_{\text{ttc}}$**:
    - 做什么：控制碰撞的相对速度和碰撞紧迫程度，调节场景的危险等级。
    - 核心思路：TTC 代价使用基于恒速假设的碰撞时间和碰撞距离的高斯核：
     $$J_{\text{ttc}} = \sum_{t=1}^T -\exp\left(-\frac{\tilde{t}_{\text{col}(t)}^2}{2\lambda_t} - \frac{\tilde{d}_{\text{col}(t)}^2}{2\lambda_d}\right)$$
     TTC 代价倾向于产生高相对速度和 ego 难以规避的碰撞角度的场景。
    - 设计动机：不同权重的 TTC 代价产生不同危险级别的碰撞场景，实现了安全关键性的连续可调。

4. **路径引导 $J_{\text{route}}$ 和高斯碰撞引导 $J_{\text{Gauss}}$**:
    - 做什么：正则化项——路径引导防止 agent 偏离道路，高斯碰撞引导防止非对抗 agent 之间的碰撞。
    - 核心思路：路径引导对超出预定义路径 margin $d_m$ 的偏离进行惩罚：
     $$J_{\text{route}} = \sum_{t=1}^T \max(0, |d_n(\tau_t, r) - d_m|)$$
     高斯碰撞引导考虑切向和法向距离，比简单圆盘近似更有效：
     $$J_{\text{Gauss}} = \sum_{t=1}^T \sum_{i,j}^N \exp\left(-\frac{1}{2\sigma^2}(\lambda \cdot d_t^{ij}(t)^2 + d_n^{ij}(t)^2)\right)$$
    - 设计动机：(1) 路径引导比先前工作的 off-road loss 更精确地约束 agent 沿预定义车道行驶；(2) 高斯碰撞距离同时考虑切向和法向，比欧氏距离碰撞惩罚更合理（因为车辆是非圆形的），显著降低了碰撞率。

5. **Partial Diffusion（部分扩散控制碰撞类型）**:
    - 做什么：通过轨迹提案初始化扩散过程，控制碰撞类型（正面、侧面、追尾等）。
    - 核心思路：三步流程——(1) 基于规则生成不同碰撞类型的初始轨迹提案 $\tau_0$（找 ego 和对抗 agent 中心线交叉点，选择加速度和偏移量）；(2) 选择部分扩散比 $\gamma$，确定起始步 $k_p = \gamma \cdot K$，对提案加对应量的噪声：$\hat{\tau}_{k_p} = \sqrt{\bar{\alpha}_{k_p}}\tau_0 + \sqrt{1 - \bar{\alpha}_{k_p}}\epsilon$；(3) 从 $k_p$ 开始执行引导去噪到步 0，得到真实化的轨迹。
    - 设计动机：纯引导方式只能在一个方向上优化（距离最近），难以控制具体碰撞方式。部分扩散通过提案"暗示"碰撞类型，再经扩散模型"真实化"提案，$\gamma$ 可调节用户控制权 vs 模型数据分布的平衡。

### 损失函数 / 训练策略

- 扩散模型训练采用标准 DDPM，K=100 步
- 轨迹表示为动作序列 $\tau_a = [a_0, ..., a_{T-1}]$，状态序列由初始状态和独轮车动力学推导
- 场景编码使用 agent-centric 栅格化地图 + ResNet 编码器
- 轨迹处理使用 1D 时序卷积块的 UNet 架构
- Planner 和 reactive agents 以 2Hz 频率更新规划

## 实验关键数据

### 主实验

**安全关键仿真对比（Rule-Based Planner，nuScenes）：**

| 方法 | 碰撞率↑ | 其他Agent离路↓ | 对抗Agent离路↓ | 碰撞相对速度 | 真实性↓ | 时间(s)↓ |
|------|---------|-------------|-------------|-------------|---------|---------|
| STRIVE | 36.4% | 2.2% | 11.4% | 5.52 | 0.85 | 427.2 |
| DiffScene | 18.2% | 11.4% | 9.0% | 16.4 | 0.52 | 105.4 |
| **Safe-Sim** | **43.2%** | **1.8%** | 11.4% | **-0.12** | **0.38** | **104.5** |

Safe-Sim 碰撞率最高且真实性最好，速度仅为 STRIVE 的 1/4。

**nuPlan 上的结果：**

| 方法 | 碰撞率↑ | 真实性↓ |
|------|---------|--------|
| DiffScene | 56.7% | 0.42 |
| **Safe-Sim** | **80%** | **0.27** |

**不同 Planner 的评估：**

| Planner | Ego-Adv碰撞率 | Ego-Other碰撞率 | 对抗Agent离路率 | 真实性 |
|---------|-------------|---------------|-------------|--------|
| IDM | 49.3% | 58.2% | 3.0% | 0.78 |
| BC | 38.8% | 37.3% | 9.0% | 0.79 |
| PDM-Closed | 26.9% | 50.7% | 1.5% | 0.86 |
| BITS | 16.4% | 19.4% | 6.0% | 0.79 |

### 消融实验

**TTC 权重对碰撞控制的影响：**

| TTC权重 | TTC代价 | TTC(s) | 碰撞速度(m/s) | 碰撞角度(deg) | 碰撞率↑ |
|---------|--------|--------|-------------|-------------|---------|
| 0.0 | 0.18 | 2.45 | -7.43 | – | 48.2% |
| 1.0 | 0.21 | 2.30 | 0.43 | – | 53.6% |
| 2.0 | 0.26 | 3.78 | -17.0 | – | 60.7% |

增加 TTC 权重不仅提升碰撞率，还改变碰撞角度，使 ego 更难规避。

**可控性消融（碰撞多样性）：**

| 配置 | 碰撞角度方差↑ | 碰撞速度方差↑ | 碰撞点方差↑ |
|------|-------------|-------------|-----------|
| $J_{\text{adv}}$ only | 3.34 | 4.81 | 2.47 |
| $J_{\text{adv}} + J_{\text{reg}}$ | 2.22 | 2.99 | 1.62 |
| $J_{\text{adv}} + J_{\text{reg}}$ + Partial Diff. | **3.10** | **1.96** | **5.44** |

Partial Diffusion 显著提升碰撞点多样性（方差从 1.62 到 5.44），验证了轨迹提案能有效控制碰撞几何类型。

### 关键发现

- $J_{\text{reg}}$ 的引入降低了碰撞率（从 53.5% 到 23.9%），但大幅改善了真实性和离路率
- 部分扩散比 $\gamma$ 可调节提案保真度 vs 真实性的权衡；$\gamma=0$ 时碰撞率最高但轨迹不够自然
- Safe-Sim 在 nuScenes 训练后可无微调迁移到 nuPlan
- 高斯碰撞距离（考虑切向/法向）比圆盘近似显著降低非对抗 agent 间碰撞

## 亮点与洞察

1. **首个同时实现安全关键+闭环+可控的仿真框架**，Tab.1 的对比一目了然，填补了重要空白
2. **Partial Diffusion 设计精巧**——将基于规则的碰撞类型提案与数据驱动的扩散模型结合，提案提供"意图"，扩散保证"真实性"
3. **引导目标设计全面**：对抗项（碰撞+TTC+速度）+ 正则化项（路径+高斯碰撞），每个都有明确的物理意义
4. **多 planner 评估**展示了框架的通用性——可以用来对比评估不同 planner 的安全性能

## 局限性 / 可改进方向

- 对抗 agent 有时会在到达 ego 之前与非对抗 agent 发生不真实碰撞
- 部分碰撞场景中 ego 并非"有过错"方，制造更多 ego-at-fault 场景会更有评估价值
- 轨迹提案目前基于简单规则（中心线交叉），更复杂场景（如环形交叉路口）的提案生成需要改进
- 未探索用此框架进行闭环策略训练（目前仅做评估）
- 对抗行为的自动参数搜索（而非手动调节权重）是重要的未来工作

## 相关工作与启发

- 与 STRIVE（VAE 潜在空间对抗优化）相比，Safe-Sim 直接在轨迹空间引导扩散，更灵活且可控性更强
- 与 CTG/CTG++（扩散可控生成但非安全关键）相比，Safe-Sim 增加了对抗目标和部分扩散
- 与 DiffScene（扩散安全关键但非闭环）相比，Safe-Sim 实现了闭环交互
- 启发：扩散模型的引导机制天然适合安全关键仿真——通过不同的引导目标组合可以"编程"不同类型的对抗行为

## 评分

- 新颖性: ⭐⭐⭐⭐ 部分扩散和碰撞类型控制是新颖贡献，但引导目标设计属于组合创新
- 实验充分度: ⭐⭐⭐⭐⭐ 双数据集、多 planner、可控性验证、消融研究非常完整
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，框架图直观，Tab.1对比表很有说服力
- 价值: ⭐⭐⭐⭐⭐ 对 AV 安全评估有直接应用价值，框架设计具有良好的工程实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] NeuroNCAP: Photorealistic Closed-Loop Safety Testing for Autonomous Driving](neuroncap_photorealistic_closed-loop_safety_testing_for_autonomous_driving.md)
- [\[ECCV 2024\] Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation](optimizing_diffusion_models_for_joint_trajectory_prediction_and_controllable_gen.md)
- [\[CVPR 2025\] DrivingSphere: Building a High-fidelity 4D World for Closed-loop Simulation](../../CVPR2025/autonomous_driving/drivingsphere_building_a_high-fidelity_4d_world_for_closed-loop_simulation.md)
- [\[ECCV 2024\] RoofDiffusion: Constructing Roofs from Severely Corrupted Point Data via Diffusion](roofdiffusion_constructing_roofs_from_severely_corrupted_point_data_via_diffusio.md)
- [\[ICLR 2026\] BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning in Autonomous Driving](../../ICLR2026/autonomous_driving/bridgedrive_diffusion_bridge_policy_for_closed-loop_trajectory_planning_in_auton.md)

</div>

<!-- RELATED:END -->
