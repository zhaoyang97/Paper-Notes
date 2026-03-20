# BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting

**会议**: ICCV 2025  
**arXiv**: [2506.22099](https://arxiv.org/abs/2506.22099)  
**代码**: [github.com/fudan-zvg/BezierGS](https://github.com/fudan-zvg/BezierGS) (有)  
**领域**: 3D视觉 / 自动驾驶场景重建  
**关键词**: 3D Gaussian Splatting, Bézier曲线, 动态场景重建, 轨迹建模, 自动驾驶仿真  
**作者**: Zipei Ma, Junzhe Jiang, Yurui Chen, Li Zhang (复旦大学)

## 一句话总结

用可学习的Bézier曲线显式建模动态物体的运动轨迹，替代传统依赖精确bbox标注的范式，实现了对自动驾驶街景中动/静态成分的准确分离与高保真重建。

## 背景与动机

自动驾驶闭环仿真需要高质量的街景3D重建。现有方法主要有两条路线：

1. **基于bbox标注的方法**（Street Gaussians, DrivingGaussian, HUGS, OmniRe）：在canonical空间重建动态物体，再根据标注pose放置到场景中渲染。**严重依赖标注精度**，标注误差和遗漏直接导致重建质量下降（如nuPlan数据集标注不精确时性能骤降）。
2. **自监督轨迹学习方法**（S³Gaussian, PVG）：S³Gaussian用时空分解网络隐式建模轨迹，优化困难；PVG用周期振动拼接轨迹段，周期振动模式不符合真实运动，且分段建模无法充分利用同一物体的时序一致性。

**核心动机**：需要一种既不严重依赖标注精度、又能显式准确建模轨迹的方法。

## 核心问题

1. 如何在不严格依赖高精度bbox标注的前提下，准确建模动态物体的运动轨迹？
2. 如何保证同一物体的多个Gaussian基元在时间维度上的几何一致性？
3. 如何实现动/静态场景成分的彻底分离？

## 方法详解

### 整体框架

场景由三部分组成：
- **静态背景**：标准3DGS，属性 {μ, q, s, o, c}，全局优化
- **动态前景**：每个动态物体一组Gaussian基元，轨迹由可学习Bézier曲线控制
- **天空模型**：高分辨率cube map，映射视方向到天空颜色

渲染流程：给定时间戳τ → 通过time-to-Bézier映射得到参数t → 计算动态Gaussian位置 → 与静态Gaussian合并 → alpha blending渲染RGB/深度/opacity → 与天空合成：C = C_G + (1 - O_G) ⊙ C_sky。

### 关键设计

1. **Bézier曲线轨迹建模**：
   - 每个动态物体的中心轨迹由一条可学习的三次Bézier曲线（n=3，4个控制点）控制：$\gamma(t,g) = \sum_{i=0}^{n} b_{i,n}(t) \cdot p_i^g$
   - 每个Gaussian基元相对于物体中心的**偏移轨迹**也用Bézier曲线建模：$\delta(t) = \sum_{i=0}^{n} b_{i,n}(t) \cdot p_i$
   - 最终位置：$\mu(\tau,g) = \delta(t) + \gamma(t,g)$，其中 $t = f(\tau,g)$
   - 关键洞察：基于bbox的方法是BézierGS的特例（偏移在物体坐标系中恒定、bbox朝向和位移固定）
   - 可学习曲线能**自动修正标注误差**，无需精确标注

2. **Time-to-Bézier映射**：
   - 不同物体运动速度不同（如匀速行驶 vs 减速停车），时间戳τ到Bézier参数t的映射是非均匀的
   - 用额外的Bézier曲线建模每个物体的映射 $t = f(\tau, g)$
   - 初始化：先用弦长参数化（chord length parameterization）合理初始化参数，再通过最小二乘拟合控制点坐标，迭代优化至收敛

3. **分组曲线间一致性约束（Inter-curve Consistency, ICC）**：
   - **问题**：Gaussian基元自由度过高，单个基元可能不受控地偏离其所属动态物体，导致不同时间步由不同基元表示同一区域，新视角渲染产生floater和伪影
   - **解决**：对刚体物体，其各部分相对于整体的偏移量随时间应保持恒定。约束偏移 $\delta(t)$ 的模长与起止控制点模长的均值一致：$\mathcal{L}_{icc} = \left\| \|\delta(t)\| - \frac{\|p_0\| + \|p_n\|}{2} \right\|_1$
   - 这个约束简单但效果显著，有效消除floater并增强时序一致性

4. **动态渲染损失（Dynamic Rendering Loss）**：
   - 由于alpha blending中动/静态Gaussian交互可能引入干扰，需额外监督动态Gaussian的独立渲染结果
   - 用Grounded-SAM从3D bbox投影区域中提取精确的动态物体mask $M_{dyn}$
   - 动态RGB损失：$\mathcal{L}_{rgb}^{dyn} = (1-\lambda_r)\mathcal{L}_1^{dyn} + \lambda_r \mathcal{L}_{ssim}^{dyn}$
   - 动态opacity对齐：$\mathcal{L}_o^{dyn} = \|M_{dyn} - O_G^{dyn}\|_1$
   - 总动态渲染损失：$\mathcal{L}_{dr} = \mathcal{L}_{rgb}^{dyn} + \mathcal{L}_o^{dyn}$

5. **速度损失（Velocity Loss）**：
   - 由Bézier曲线的解析导数直接计算每个Gaussian基元的速度：$v(\tau,g) = \left(\frac{d\gamma}{dt} + \frac{d\delta}{dt}\right) \frac{dt}{d\tau}$
   - 渲染动态速度图 $V_G^{dyn}$，约束其在静态区域为零：$\mathcal{L}_v = \|V_G^{dyn} \cdot (1 - M_{dyn})\|_2$
   - 隐式防止动态Gaussian漂移到静态区域

### 损失函数 / 训练策略

总损失：$\mathcal{L} = (1-\lambda_r)\mathcal{L}_1 + \lambda_r\mathcal{L}_{ssim} + \lambda_{osky}\mathcal{L}_{osky} + \lambda_{icc}\mathcal{L}_{icc} + \lambda_{dr}\mathcal{L}_{dr} + \lambda_v\mathcal{L}_v + \lambda_d\mathcal{L}_d$

各项含义：
- **$\mathcal{L}_1 + \mathcal{L}_{ssim}$**：RGB渲染质量监督
- **$\mathcal{L}_d$**：LiDAR点投影到相机平面的稀疏逆深度监督，增强几何感知
- **$\mathcal{L}_{osky}$**：鼓励天空区域opacity最小化（Grounded-SAM预测天空mask），确保天空仅由cube map建模
- **$\mathcal{L}_{icc}$**：曲线间一致性，约束偏移轨迹稳定性，消除floater
- **$\mathcal{L}_{dr}$**：动态渲染损失，确保动/静态彻底分离
- **$\mathcal{L}_v$**：速度约束，防止动态Gaussian漂移到静态区域

超参数：$\lambda_r=0.2, \lambda_d=1.0, \lambda_{osky}=0.05, \lambda_{icc}=0.01, \lambda_{dr}=0.1, \lambda_v=1.0$。在Waymo单序列调参，直接迁移到其他序列和nuPlan，**无需重调**，验证了鲁棒性。

训练配置：单卡NVIDIA RTX A6000，30000次迭代，三次Bézier曲线（n=3）。

旋转建模：刚体物体z轴朝向固定，仅在xy平面旋转，根据轨迹在xy平面的切线方向决定旋转（非四元数插值）。

## 实验关键数据

### Waymo Open Dataset（12序列）

**Full（训练帧重建）：**

| 数据集 | 指标 | BézierGS | 之前SOTA | 提升 |
|--------|------|----------|----------|------|
| Waymo Full | PSNR↑ | **33.98** | 32.61 (PVG) | +1.37 dB |
| Waymo Full | SSIM↑ | **0.934** | 0.923 (OmniRe) | +0.011 |
| Waymo Full | LPIPS↓ | **0.077** | 0.084 (OmniRe) | -8.3% |
| Waymo Full | Dyn-PSNR↑ | **32.39** | 29.32 (PVG) | +3.07 dB |

**NVS（新视角合成，每4帧测试1帧）：**

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Dyn-PSNR↑ |
|------|-------|-------|--------|-----------|
| DeformableGS | 29.52 | 0.889 | 0.100 | 24.66 |
| HUGS | 29.34 | 0.865 | 0.110 | 23.84 |
| Street Gaussians | 28.92 | 0.877 | 0.110 | 25.54 |
| OmniRe | 29.41 | 0.884 | 0.101 | 25.85 |
| PVG | 29.64 | 0.864 | 0.179 | 24.46 |
| **BézierGS** | **31.51** | **0.903** | **0.092** | **28.51** |

- PSNR +1.87 dB, SSIM +0.014, LPIPS -8.00%, **Dyn-PSNR +2.66 dB**

### nuPlan Benchmark（6序列，标注不精确）

**NVS：**

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Dyn-PSNR↑ |
|------|-------|-------|--------|-----------|
| DeformableGS | 26.20 | 0.824 | 0.159 | 21.37 |
| HUGS | 23.77 | 0.744 | 0.280 | 20.15 |
| Street Gaussians | 25.76 | 0.788 | 0.185 | 22.03 |
| OmniRe | 26.01 | 0.819 | 0.173 | 23.90 |
| PVG | 26.38 | 0.772 | 0.222 | 19.69 |
| **BézierGS** | **29.42** | **0.860** | **0.133** | **25.12** |

- PSNR **+3.04 dB**, SSIM +0.036, LPIPS -16.35%, Dyn-PSNR +1.22 dB
- **在标注不精确的nuPlan上优势尤为巨大**，bbox方法（HUGS/Street Gaussians/OmniRe）性能严重下降

### 计算效率（Waymo NVS序列）

| 方法 | 训练时间(min)↓ | #GS(K) | FPS↑ | PSNR↑ |
|------|---------------|--------|------|-------|
| DeformableGS | 22.7 | 123.1 | 4.9 | 29.52 |
| HUGS | 8.8 | 61.4 | 46.1 | 29.34 |
| Street Gaussians | 6.1 | 71.6 | 67.8 | 28.92 |
| OmniRe | 12.4 | 103.5 | 58.3 | 29.41 |
| PVG | 11.9 | 33.6 | 50.8 | 29.64 |
| **BézierGS** | **10.7** | **48.6** | **88.7** | **31.51** |

BézierGS在PSNR和FPS上同时超越所有方法（88.7 FPS!），训练时间10.7分钟处于中等偏快水平。

### 消融实验要点

Waymo NVS上的消融：

| 配置 | PSNR | SSIM | LPIPS | Dyn-PSNR |
|------|------|------|-------|----------|
| (a) w/o $\mathcal{L}_{icc}$ | 30.83 | 0.900 | 0.096 | 26.15 |
| (b) w/o $\mathcal{L}_{dr}$ | 30.99 | 0.891 | 0.099 | 28.07 |
| (c) w/o $\mathcal{L}_v$ | 31.40 | 0.901 | 0.094 | 28.29 |
| (d) w/o time-to-Bézier | 31.36 | 0.899 | 0.094 | 27.97 |
| (e) w/ MLP轨迹 (DeformableGS) | 29.58 | 0.898 | 0.087 | 24.78 |
| (f) w/ 正弦轨迹 (PVG) | 29.65 | 0.877 | 0.099 | 26.27 |
| **Full BézierGS** | **31.51** | **0.903** | **0.092** | **28.51** |

关键发现：
- **$\mathcal{L}_{icc}$ 影响最大**（-0.68 PSNR, **-2.36 Dyn-PSNR**）：曲线间一致性约束对消除floater、增强动态区域质量至关重要
- **$\mathcal{L}_{dr}$ 次之**（-0.52 PSNR, SSIM降至0.891）：动态渲染损失实现了动静的彻底分离
- **$\mathcal{L}_v$ 和 time-to-Bézier提供边际改进**，但对复杂轨迹场景是必要的
- **Bézier vs 替代轨迹表示**：Bézier比MLP和正弦轨迹分别高**1.93和1.86 PSNR**，证明显式参数曲线优于隐式/周期性表示

## 亮点

1. **优雅的数学建模**：将动态物体轨迹建模为Bézier曲线，既有理论美感（参数曲线的局部控制性、端点插值性），又有实用价值（自动修正标注误差）
2. **统一性**：bbox方法是BézierGS的特例（偏移恒定+位姿固定），框架具备很好的理论统一性和实际拓展性
3. **消除对标注精度的依赖**：在nuPlan（标注不精确）上优势尤为显著（+3.04 dB），证明了方法在真实世界标注噪声下的鲁棒性
4. **效率出色**：88.7 FPS远超同类所有方法，满足实时仿真需求；训练10.7分钟，实用性强
5. **支持场景编辑**：显式动态建模天然支持动态物体移除等编辑操作，为闭环仿真提供基础

## 局限性 / 可改进方向

1. **依赖分割模型精度**：训练中使用Grounded-SAM产生动态mask和天空mask，分割不准会引入误导
2. **未显式建模光照变化**：无法模拟复杂光影效果和阴影变化（可用4DSH方法缓解，论文附录已展示）
3. **Bézier曲线平滑性限制**：对高频不连续运动（如行人腿部摆动）建模困难，腿部区域重建有伪影（但语义分割仍正确，不影响闭环仿真的感知评估）
4. **非刚体运动建模有限**：旋转建模仅基于xy平面切线方向，对非刚体物体（如行人扭转）不够理想
5. **仍需初始标注/跟踪**：有标注时用bbox初始化LiDAR点云，无标注时需视频实例跟踪模型提取mask-guided LiDAR点，并非完全无监督

## 与相关工作的对比

| 方法 | 轨迹建模 | 标注依赖 | 动静分离 | 核心局限 |
|------|----------|----------|----------|----------|
| Street Gaussians | bbox刚体变换 | 强依赖 | 显式 | 标注不准时性能骤降 |
| OmniRe | bbox+行人模块 | 强依赖 | 显式 | 模块化但重度依赖标注 |
| HUGS | bbox变换 | 强依赖 | 显式 | 训练快但PSNR低 |
| DrivingGaussian | bbox变换 | 强依赖 | 显式 | 同上类bbox依赖 |
| PVG | 周期振动拼接 | 不依赖 | 隐式 | 周期假设不符真实运动 |
| S³Gaussian | 时空分解网络 | 不依赖 | 隐式 | 隐式轨迹难优化 |
| DeformableGS | MLP变形场 | 不依赖 | 无法分离 | 不支持动静分离 |
| **BézierGS** | **Bézier曲线** | **弱依赖** | **显式** | **平滑性限制高频运动** |

## 启发与关联

- **可学习参数曲线的思路可推广**：除了Bézier，B-spline、NURBS等也可探索，尤其对更复杂/长距离轨迹；分段Bézier可处理超长序列
- **曲线间一致性约束思路通用**：这种约束同一实体内部Gaussian一致性的方法可迁移到其他3DGS分组场景（如语义分组、部件级重建、铰接体建模）
- **与相关idea的关联**：
  - [测试时训练用于4D动力学外推](../../../ideas/3d_vision/20260316_ttt_4d_dynamics.md)：BézierGS的显式Bézier控制点可作为TTT在线更新的对象——推理时fine-tune控制点以适应新轨迹/外推
  - [代价体引导稀疏占据预测](../../../ideas/autonomous_driving/20260317_cost_volume_sparse_occ.md)：BézierGS的轨迹建模可提供动态物体运动prior，辅助时序占据预测
  - [Min-Max鲁棒性3DGS防御](../../../ideas/ai_safety/20260317_minmax_robust_3dgs_adversarial_defense.md)：BézierGS的高保真动态重建能力可用于生成更真实的对抗训练样本

## 评分

- 新颖性: ⭐⭐⭐⭐ 用Bézier曲线建模3DGS轨迹是自然但有效的创新，将bbox方法统一为特例的视角具有理论价值
- 实验充分度: ⭐⭐⭐⭐⭐ 两个大规模数据集、5个baseline、全面消融（损失项+轨迹表示替换）、计算效率分析、附录含完整实现细节和算法伪代码
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整，图示直观；统一视角的讨论（bbox是特例）加分；LaTeX HTML渲染有小瑕疵但不影响理解
- 价值: ⭐⭐⭐⭐ 对自动驾驶仿真有直接实用价值，消除标注依赖是大规模部署的关键痛点；88.7 FPS可满足实时闭环仿真需求
