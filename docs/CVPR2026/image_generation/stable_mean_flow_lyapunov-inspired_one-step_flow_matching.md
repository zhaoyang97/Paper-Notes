---
title: >-
  [论文解读] Stable Mean Flow: Lyapunov-Inspired One-Step Flow Matching
description: >-
  [CVPR 2026][图像生成][流匹配] 给当前最强的一步生成方法 Mean Flow 加一项受 Lyapunov 稳定性启发的"非膨胀"正则，强制单步传输映射不放大邻域扰动，从而消除训练中 JVP 爆 NaN/Inf 的失稳问题，并在 CIFAR-10 上把单步 FID 从 2.92 压到 2.86、收敛明显更快。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "流匹配"
  - "Mean Flow"
  - "一步生成"
  - "Lyapunov 稳定性"
  - "非膨胀正则"
---

# Stable Mean Flow: Lyapunov-Inspired One-Step Flow Matching

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Stable_Mean_Flow_Lyapunov-Inspired_One-Step_Flow_Matching_CVPR_2026_paper.html)  
**代码**: 待确认  
**领域**: 图像生成 / 流匹配 / 一步生成  
**关键词**: 流匹配, Mean Flow, 一步生成, Lyapunov 稳定性, 非膨胀正则

## 一句话总结
给当前最强的一步生成方法 Mean Flow 加一项受 Lyapunov 稳定性启发的"非膨胀"正则，强制单步传输映射不放大邻域扰动，从而消除训练中 JVP 爆 NaN/Inf 的失稳问题，并在 CIFAR-10 上把单步 FID 从 2.92 压到 2.86、收敛明显更快。

## 研究背景与动机

**领域现状**：扩散/分数模型画质强但要几百上千步去噪，即使用 few-step 采样器或蒸馏也仍是迭代式。流匹配（flow matching）转而学一条把噪声确定性搬运到数据的 ODE 速度场，把轨迹"拉直"后可以一步出图。这一支里 Mean Flow（MeanFlow [9]）是目前一步生成的 SOTA：它不学瞬时速度，而学一个时间区间上的**平均速度** $u(z_t,r,t)=\frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,d\tau$，因为目标是个良定义的物理量（平均位移率），所以能从头稳定训练，并真正做到 1-NFE（单次前向）采样。

**现有痛点**：一步方法在实践中很脆。训练会出现"敏感度爆炸"、轨迹歧义，进而失稳、画质崩坏。一个具体且致命的失败模式是：Mean Flow 训练里要算的 Jacobian–向量积（JVP，平均速度对时间的全导数）会时不时蹦出 NaN/Inf，直接毁掉训练——论文 Figure 2 显示 Mean Flow 在约 53,000 步左右 JVP 崩掉。

**核心矛盾**：根因在于学到的单步映射可能在局部是**膨胀的（locally expansive）**——它会放大输入扰动，使邻近轨迹相互交叉、挤压或炸开。因为一步生成没有逐时刻的 ground-truth 目标来"纠偏"，一旦映射局部拉伸，轨迹就变得 ill-posed，回归目标在不同时间步之间互相矛盾，梯度传播也跟着不稳。

**本文目标**：在**完全保留 Mean Flow 目标**的前提下，给单步映射加一个"不准放大扰动"的约束，让训练在数值安全区里跑，同时给一步/多步生成的误差增长写出可证的上界。

**切入角度**：把动力系统里的 **Lyapunov 稳定性**搬过来。作者注意到：当条件目标动力学是常向量场 $\dot x(t)=a=X_1-X_0$（直线最优传输路径，源自 Brenier 定理）时，任意两条轨迹只是平行漂移，间距 $\|x(t)-y(t)\|$ 永远冻结不变——这正是"初始靠近就永远靠近"的 Lyapunov 稳定。作者想让学到的流也具备这种"困住扰动"的性质。

**核心 idea**：用一句话概括——**给单步映射加一个局部非膨胀（non-expansivity, NE）的 hinge 软约束，把"GPS 导航（Mean Flow 指方向）"补上"循迹控制（Lyapunov 项让轮子不跑偏）"**。

## 方法详解

### 整体框架
Stable Mean Flow Matching（SMFM）= Mean Flow 的回归目标 + 一项稳定性正则，总损失 $L = L_{MF} + \mu\,\ell_{stab}$，网络结构、采样器（NFE=1）、数据增广、优化日程全部沿用 Mean Flow，**唯一的变量就是这项稳定正则**，因此任何性能差异都能归因到稳定机制本身。

先把基础打清楚。流匹配用插值 $z_t=a_t x+b_t\epsilon$ 描述中间态，训练一个速度场 $v_\theta$ 预测 $z_t$ 该往哪走，条件目标为 $L_{CFM}(\theta)=\mathbb{E}_{t,x,\epsilon}\|v_\theta(z_t,t)-v_t(z_t\mid x)\|^2$；采样就是从噪声解 ODE $\frac{d}{dt}z_t=v_\theta(z_t,t)$。Mean Flow 把"学瞬时速度"换成"学平均速度"$u_\theta(z_t,r,t)$，单步重建为 $z_r\approx z_t-(t-r)\,u_\theta(z_t,r,t)$（取 $(r,t)=(0,1)$ 即 $z_0\approx z_1-u_\theta(z_1,0,1)$）。其训练损失为

$$L(\theta)=\mathbb{E}\Big[\big\|u_\theta(z_t,r,t)-\mathrm{sg}\big(v_t-(t-r)\tfrac{d}{dt}u_\theta(z_t,r,t)\big)\big\|_2^2\Big],$$

其中全导数 $\frac{d}{dt}u_\theta=\partial_z u_\theta\,v_t+\partial_t u_\theta$ 由 JVP 计算，$\mathrm{sg}$ 是 stop-gradient。SMFM 在此之上做三件事：把 Lyapunov 稳定翻译成单步映射的 NE 约束（设计 1）、用 hinge 平方损失把 NE 软性地塞进训练并控制扰动半径 $\delta$（设计 2）、再把"终端时刻的速度误差"上推成全程轨迹的误差上界，给采样质量兜底（设计 3）。

### 关键设计

**1. 从 δ-cap 到局部非膨胀约束（NE）：把 Lyapunov 稳定翻译成对单步映射的几何要求**

先把单步映射写成 $\phi_r^\theta(t,\cdot):z\mapsto z-(t-r)\,u_\theta(z,r,t)$，即沿学到的速度场做一次"反向 Euler"更新。直接照 Lyapunov 定义抄，会得到一个朴素约束 δ-cap：只要扰动 $\|\Delta z\|_2\le\delta$，就要求输出位移 $\|\phi_r^\theta(t,z_t+\Delta z)-\phi_r^\theta(t,z_t)\|_2\le\delta$。但作者指出 δ-cap 太松——只要总位移落在半径 $\delta$ 内，映射仍可以在局部剧烈拉伸。于是改用更强更干净的非膨胀（NE）条件：

$$\big\|\phi_r^\theta(t,z_t+\Delta z)-\phi_r^\theta(t,z_t)\big\|_2\le\|\Delta z\|_2,\quad\forall\,\|\Delta z\|_2\le\delta.$$

NE 要求"输出间距不超过输入间距"，它自动蕴含任意 $\delta$ 下的 δ-cap，并且额外保证无穷小扰动不被放大。这个约束直接换来两条理论收益：**轨迹唯一性**（Theorem 3.1，NE 下 ODE $\dot z(s)=u_\theta(z(s),r,s)$ 对每个初值都有唯一特征线，避免轨迹交叉/多值传输让回归目标自相矛盾，对应到最优传输就是源到目标的确定性耦合）；以及 **JVP 有界**（Theorem 3.2，$\|\partial_z u_\theta\,\xi+\partial_t u_\theta\|\le C$，把前面那个最常见的数值崩溃模式直接摁住——更新算子不能无限放大扰动，但当真值本身已满足 NE 时这项正则不会改变 JVP 的学习，只是把优化轨迹关进数值安全区）。

**2. Hinge 平方稳定损失与扰动半径 δ 的取值权衡：把硬约束变成可训练的软惩罚**

NE 是个硬不等式，没法直接当 loss。作者在每个基点 $z_t$ 处随机采一个半径为 $\delta$ 的扰动 $\Delta z=\delta\,\xi/\|\xi\|_2,\ \xi\sim\mathcal N(0,I_d)$（高斯向量归一化到单位球面再缩放到半径 $\delta$），记 $\Delta u=u_\theta(z_t+\Delta z,r,t)-u_\theta(z_t,r,t)$、$\alpha=t-r$，用 hinge 平方惩罚把违反量软化：

$$\ell_{stab}(\Delta z)=\Big[\max\big(0,\ \|\Delta z-(t-r)\Delta u\|_2-\|\Delta z\|_2\big)\Big]^2.$$

满足 NE 时它恒为 0，违反时二次增长，因此只在"映射开始膨胀"的地方施压，不打扰已经守规矩的区域。最终目标 $L=L_{MF}+\mu\,\ell_{stab}$。半径 $\delta$ 是最关键超参，作者用 Theorem 3.3（小半径鲁棒界）刻画它的影响：定义带符号违反量 $V(\Delta z)=\|\phi_r^\theta(t,z_t+\Delta z)-\phi_r^\theta(t,z_t)\|_2-\|\Delta z\|_2$，则 $\mathbb{E}[V(\Delta z)]\le 2\delta$、$P(V(\Delta z)>\tau)\le 2\delta/\tau$。这给出清晰的权衡：$\delta$ 太大，对目标的随机估计不可靠，会拖累向 Mean Flow 目标的回归；$\delta$ 太小，梯度消失，退化回原版 Mean Flow。所以取一个"小而非零"的半径（如典型步长 $\alpha\|u_\theta\|$ 或局部状态范数的一个固定比例），既稳得住又不伤学习。训练上还有个细节：稳定项只在早期时间窗激活，权重 $\mu$ 随迭代衰减，避免后期压过 Mean Flow 主目标（见 Algorithm 1）。

**3. 端点误差控制：把"终端速度误差"上推成全程轨迹的误差上界**

SMFM 有个定义性质——采样只依赖终端时刻 $t=1$ 的速度场，中间时刻只通过影响终端量起作用。形式上，终端速度误差 $e_1:=u_\theta(z_1,r,1)-u^*(1,z_1)$（$u^*$ 为 oracle 速度，即路径切线的条件期望），单步重建满足 $\hat z_r-z_r=-(1-r)\,e_1$，所以**端点精度被终端速度失配线性控制**，早期误差 $e_r$ 不进入这条单步关系。作者据此推出一套误差递推：Theorem 4.1 给出前向单步误差界（把 $e_{t+\Delta t}$ 用 $e_t$、步长和模型常数夹住）；Corollary 4.1 把它全局化为"非增长上界" $\|e_{t_{k+1}}\|\le\max\{\|e_{t_k}\|,\ T_{t_k}\}$（$T_{t_k}=M^*+M_\theta+\alpha_{t_k}\Lambda_\theta$），形成一道"安全包络"——误差一旦超过阈值就触发收缩，防止多步生成时炸开；Corollary 4.2 给出端点控制，单步情形落到显式界 $\|e_1\|\le(M^*+M_\theta)+\tfrac12\Lambda_\theta$。直观上：只要模型速度场在 $t=1$ 附近的小窗口里贴合 oracle，端点 $z_1$ 就贴近 oracle 轨迹，残差至多随剩余时间线性增长——这给一步和多步生成都提供了端点稳定性的理论保证。⚠️ 各定理/推论的精确常数与证明细节以原文为准。

### 损失函数 / 训练策略
总损失 $L=L_{MF}+\mu\,\ell_{stab}$。Algorithm 1（Hybrid Mean Flow with Non-Expansivity）每步：采样 $t\sim\mathrm{Unif}[\varepsilon,1],\ r\sim\mathrm{Unif}[0,t]$ 与 $z_t\sim p_t$；采单位球面扰动 $\xi$、置 $\Delta z=\delta\xi$；算 $u=u_\theta(z_t,r,t)$、目标 $u_{tgt}=v-(t-r)(\partial_z u_\theta\,v+\partial_t u_\theta)$、$L_{MF}=\|u-\mathrm{sg}(u_{tgt})\|_2^2$；算 $\Delta u=u_\theta(z_t+\Delta z,r,t)-u$、$L_{stab}=\max(0,\|\Delta z-(t-r)\Delta u\|_2-\|\Delta z\|_2)^2$；合成 $L=L_{MF}+\mu L_{stab}$ 后梯度下降。稳定项早期激活、$\mu$ 随迭代衰减。CIFAR-10 主实验训练 500k 步、batch 128、单张 A100；ImageNet 用 SMF-XL/2。

## 实验关键数据

### 主实验
CIFAR-10、单步推理（NFE=1）、FID 越低越好。SMFM 与代表性一步/少步方法对比（Table 2）：

| 方法 | NFE | FID |
|------|-----|-----|
| 1-Rectified Flow [26] | 1 | 378 |
| Glow [19] | 1 | 48.9 |
| Residual Flow [7] | 1 | 46.4 |
| GLFlow [37] | 1 | 44.6 |
| DenseFlow [11] | 1 | 34.9 |
| Consistency Model [31] | 2 | 5.83 |
| Consistency Flow Matching [40] | 2 | 5.34 |
| Mean Flow [9] | 1 | 2.92 |
| **Stable Mean Flow（本文）** | 1 | **2.86** |

在保持单步采样效率的前提下，SMFM 把 FID 从 Mean Flow 的 2.92 降到 2.86，说明稳定项在不牺牲推理速度的同时提升了鲁棒性与画质。ImageNet 上用同样训练流程训 Stable MeanFlow-XL/2，Epoch 240 时 FID 3.37，略低于原 MeanFlow-XL/2 报告的 3.43（差距不大），但训练早期优势明显。

### 消融实验
超参扫描（Table 1，CIFAR-10 FID，缩减预算下的网格扫；加粗为最优；"–"为未测）。横轴稳定权重 $\mu$、纵轴扰动半径 $\Delta z$：

| $\Delta z\backslash\mu$ | 0 | 0.1 | 0.5 | 1 |
|------|------|------|------|------|
| 0.005 | – | 85.41 | 127.40 | 224.53 |
| 0.01 | 86.73 | **79.86** | 134.53 | 253.67 |
| 0.02 | – | 95.17 | 187.43 | 331.23 |

最优点在 $\Delta z=0.01,\ \mu=0.1$，FID=79.86；同一行 $\mu=0$（即关掉稳定项、退回 Mean Flow）为 86.73，加上小权重稳定项后明显变好。

### 关键发现
- **SMFM 对 $\Delta z$ 和 $\mu$ 都高度敏感，有效区间是一条窄带**：小 $\Delta z$ 给出局部化、有信息量的稳定信号，适中的 $\mu$ 在稳定信号与 Mean Flow 目标间取得平衡；任一参数偏大都迅速崩坏——正则项主导、过度收缩动力学，把速度场偏离 Mean Flow 目标（看 Table 1 右下角 μ=1 列 FID 飙到 224~331）。
- **数值稳定性是核心收益**：Mean Flow 的 JVP 在约 53,000 步崩溃（NaN/Inf），SMFM 全程把平均/最大 JVP 维持在更稳的水平（Figure 2），印证 Theorem 3.2 的有界 JVP。
- **早期收敛更快**：mid-training（200k 步）定性对比显示 SMFM 比 Mean Flow 更早收敛到视觉连贯的样本；但当 NFE 增到 5 步时两者视觉和 FID 差异不大——稳定项主要在"早期 + 一步"场景吃香。
- **2D checkerboard toy 验证机制**：稳定约束迫使速度场"先学对方向、再增大幅度"，粒子运动更结构化、单调，更快把概率质量收到正确格点上，边界更锐利；Mean Flow 早期更弥散，后期才慢慢追上。

## 亮点与洞察
- **把动力系统理论"翻译"成一行 loss**：从 Lyapunov 稳定 → 常向量场的平行漂移 → 单步映射的非膨胀 → hinge 平方惩罚，这条从抽象定义落到可训练正则的链条很干净，是可复用的方法论模板。
- **几乎零成本的即插即用正则**：网络、采样器、训练日程全不动，只加一项扰动-惩罚，因此性能增益可干净归因，工程上极易嫁接到任何 Mean Flow 类训练里。
- **"端点误差线性控制采样误差"的观察很实用**：$\hat z_r-z_r=-(1-r)e_1$ 把质量问题归约到"只需在 $t=1$ 附近贴合 oracle"，为一步生成的理论分析提供了简洁抓手。
- **NE 比朴素 δ-cap 强在哪里讲得很透**：δ-cap 容许"总位移达标但局部猛拉伸"，NE 连无穷小膨胀都禁掉，这个对比点醒了"约束形式选择"对稳定性的影响。

## 局限与展望
- **绝对画质增益偏小**：CIFAR-10 上 2.92→2.86、ImageNet 上 3.43→3.37，提升幅度有限，论文也坦承"差距不大"；主要卖点是训练稳定性与早期收敛速度，而非把 SOTA 往前推一大截。
- **评测规模窄**：主结果只在 CIFAR-10 + 一个 ImageNet-XL/2 点 + 2D toy 上，缺更高分辨率/更大规模/类条件生成的系统验证。
- **超参极其敏感**：有效区间是窄带，$\Delta z/\mu$ 稍大就崩（FID 翻几倍），实际落地需要小心调参；$\delta$ 的"scale-aware"具体取法描述较定性。
- **多步无优势**：NFE=5 时与 Mean Flow 拉不开差距，增益集中在一步/早期，适用范围有边界。
- **改进方向**：作者计划进一步放大端点性能的增益、并优化算法以缩短训练时间。

## 相关工作与启发
- **vs Mean Flow [9]**：本文直接以 Mean Flow 为底座、共享其平均速度目标与单步采样，区别仅在多加一项 NE 稳定正则；优势是更稳的 JVP、更快早期收敛、略好的 FID，代价是引入两个敏感超参。
- **vs Rectified Flow [26] / Consistency Flow Matching [40]**：这些方法靠"拉直轨迹/分段速度一致"来减步数，关注的是轨迹几何；本文不改轨迹定义，而是约束单步映射的"不放大扰动"，属于正交的稳定性视角，可叠加。
- **vs Consistency Model [31]**：CM 走 2-NFE 多尺度自一致（FID 5.83），本文坚持真 1-NFE，思路是"用稳定性正则换一步生成的可靠性"而非靠多步细化。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把 Lyapunov 稳定性以 NE 软正则形式引入一步流匹配，理论翻译干净、角度新。
- 实验充分度: ⭐⭐⭐ 主要在 CIFAR-10，ImageNet 仅单点、缺大规模/高分辨率验证。
- 写作质量: ⭐⭐⭐⭐ 理论推导（唯一性/有界 JVP/端点误差界）层次清楚，GPS/循迹控制的比喻好懂。
- 价值: ⭐⭐⭐⭐ 几乎零成本即插即用的稳定正则，对训 Mean Flow 类一步生成模型有实用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Functional Mean Flow in Hilbert Space](functional_mean_flow_in_hilbert_space.md)
- [\[CVPR 2026\] RenderFlow: Single-Step Neural Rendering via Flow Matching](renderflow_single-step_neural_rendering_via_flow_matching.md)
- [\[CVPR 2026\] Flow Matching for Multimodal Distributions](flow_matching_for_multimodal_distributions.md)
- [\[CVPR 2026\] Spatiotemporal Pyramid Flow Matching for Climate Emulation](spatiotemporal_pyramid_flow_matching_for_climate_emulation.md)
- [\[CVPR 2026\] Frequency-Aware Flow Matching for High-Quality Image Generation](freqflow_frequency_aware_flow_matching.md)

</div>

<!-- RELATED:END -->
