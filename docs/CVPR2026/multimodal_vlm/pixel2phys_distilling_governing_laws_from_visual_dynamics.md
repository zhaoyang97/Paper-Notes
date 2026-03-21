# Pixel2Phys: Distilling Governing Laws from Visual Dynamics

**会议**: CVPR 2026  
**arXiv**: [2602.19516](https://arxiv.org/abs/2602.19516)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 物理定律发现, 多智能体框架, 符号回归, 视频理解, AI for Science

## 一句话总结
提出 Pixel2Phys，一个基于 MLLM 的多智能体协作框架，通过 Plan-Variable-Equation-Experiment 四个 Agent 的迭代假设-验证-精化循环，从原始视频中自动发现可解释的物理控制方程，外推精度比基线提升 45.35%。

## 研究背景与动机
1. **领域现状**：从观测数据中发现物理规律是科学智能的核心目标。传统方法依赖人工提取物理量后进行符号回归，过程缓慢。
2. **现有痛点**：
   - 监督式方程预测模型需要稀缺的方程-视频配对数据，泛化性差
   - 无监督潜空间方法（Autoencoder + 符号回归）的潜空间由重建目标决定，物理无关因素（纹理、光照）容易混入
   - 直接提示 MLLM 主要是检索训练语料中的先验知识，难以从原始视觉数据推导新规律
3. **核心矛盾**：物理变量提取与方程发现存在鸡和蛋的循环依赖——好的变量空间需要知道动力学，发现动力学需要干净的变量空间。
4. **本文要解决什么**：同时发现物理变量 $z(t)$ 和控制方程 $f$，即 $\frac{dz}{dt} = f(z(t))$。
5. **切入角度**：模拟人类科学家的协作工作流——观察、假设、实验、精化——构建多智能体迭代框架。
6. **核心idea**：用 MLLM 协调四个专业 Agent 进行迭代式科学推理，打破变量提取与方程发现之间的循环依赖。

## 方法详解

### 整体框架
四个 Agent 协作：Plan Agent（全局协调）→ Variable Agent（提取物理变量）→ Equation Agent（符号回归发现方程）→ Experiment Agent（评估验证）→ Plan Agent 分析报告并决定下一轮精化方向。

### 关键设计

1. **Plan Agent（全局规划）**:
   - 中央协调器，每轮汇聚三个 Agent 的报告进行两阶段诊断
   - 先检查可视化评估定性拟合度，再检查定量指标定位瓶颈
   - 根据诊断结果决定精化策略：变量精化（重新提取 $\mathcal{Z}$）或方程精化（调整搜索超参）
   - 设计动机：打破变量-方程循环依赖，使两者互相精化

2. **Variable Agent（多粒度变量提取）**:
   - **Object-level Tool**：用 SAM 分割 + 追踪提取运动轨迹 $z(t) = [x(t), y(t)]$
   - **Pixel-level Tool**：用固定卷积核计算空间导数（Laplacian、bi-harmonic），适用于 PDE 驱动的物理场
   - **Representation-level Tool**：物理信息自编码器，损失 $\mathcal{L} = \mathcal{L}_{recon} + \lambda_{eq}\mathcal{L}_{eq}$，其中 $\mathcal{L}_{eq} = \|\mathcal{F}(z) - f(z)\|^2$ 强制潜空间符合已发现的方程
   - 设计动机：不同类型的物理系统需要不同粒度的变量提取方式

3. **Equation Agent（动态符号回归）**:
   - 用中心差分估计时间导数 $\dot{Z}$
   - 构建候选函数库 $\Theta(Z)$：多项式项 + 超越函数
   - 用 STLSQ 在 $\|\dot{Z} - \Theta(Z)\Xi\|_2^2 + \lambda_{sp}\|\Xi\|_1$ 下求解稀疏系数矩阵 $\Xi$
   - $\lambda_{sp}$ 由 Plan Agent 指导调整

4. **Experiment Agent（多维评估）**:
   - 方程质量：$R^2$ 分数 + 复杂度（$L_0$ of $\Xi$）
   - 变量质量：相空间图可视化
   - 外推保真度：从初始条件积分预测并计算 RMSE
   - 汇聚定量指标和图表形成结构化报告

### 损失函数 / 训练策略
Variable Agent 中的 Representation-level Tool 通过物理信息自编码器训练，前期无方程先验时只用重建损失，后期加入物理一致性损失进行联合优化。

## 实验关键数据

### 主实验（Object-level dynamics）

| 案例 | 方法 | Terms Found | False Positives | $R^2$@1000 |
|------|------|:-----------:|:---------:|----------|
| Linear | Coord-Equ | Yes | 1.10 | 0.8647 |
| Linear | **Pixel2Phys** | **Yes** | **0** | **0.9913** |
| Cubic | Coord-Equ | No | 3.40 | 0.2632 |
| Cubic | **Pixel2Phys** | **Yes** | **0.39** | **0.9886** |
| VDP | Coord-Equ | Yes | 2.31 | 0.4920 |
| VDP | **Pixel2Phys** | **Yes** | **0.99** | **0.9954** |

### 主实验（Pixel-level PDE dynamics）

| 数据集 | 方法 | RMSE↓ | VPS@0.5↑ |
|--------|------|-------|----------|
| Lambda-Omega | PDE-Find | 0.67 | 492 |
| Lambda-Omega | **Pixel2Phys** | **0.03** | **1000** |
| Brusselator | SGA-PDE | 0.14 | 1000 |
| Brusselator | **Pixel2Phys** | **0.12** | **1000** |
| FHN | PDE-Find | 0.63 | 54 |
| FHN | **Pixel2Phys** | **0.16** | **1000** |

### 关键发现
- 隐式方法（Latent-ODE, AE-SINDy）在长期外推上完全崩溃（$R^2 \approx 0$），证明通用表示无法捕捉物理结构
- Pixel2Phys 的假阳性项数远低于 Coord-Equ，发现的方程更简洁准确
- 在 PDE 场景下，神经算子（FNO/UNO）误差累积严重，而 Pixel2Phys 能正确识别高阶算子（bi-harmonic）
- 框架能从真实世界视频中恢复引力定律和 Navier-Stokes 方程

## 亮点与洞察
- **多智能体科学推理框架**：用 MLLM 作为规划器协调专业 Agent，首次将"观察-假设-实验-精化"的科学方法论自动化，这个框架可以迁移到生物、化学等其他科学领域
- **物理信息自编码器的巧妙设计**：在迭代过程中，已发现的方程反过来指导变量空间的精化，打破了变量-方程的循环依赖
- **多粒度工具选择**：Object/Pixel/Representation 三级工具覆盖了从离散物体到连续场到隐式动力学的全谱系

## 局限性 / 可改进方向
- 依赖 GPT-4o 作为 backbone，成本较高
- 对多体相互作用（N-body problem）的处理能力有待验证
- 当物理变量维度很高时，符号回归的搜索空间爆炸
- 真实世界中的混沌系统可能导致迭代不收敛

## 相关工作与启发
- **vs Coord-Equ (pipeline-based)**：Coord-Equ 依赖预训练追踪器提取坐标后单次符号回归，无法处理连续场且容易引入假阳性。Pixel2Phys 通过迭代精化和多粒度工具解决了这两个问题
- **vs End-to-end 方法 (AE-SINDy)**：端到端方法中变量空间由重建目标决定，物理无关因素混入导致外推失败。Pixel2Phys 通过物理一致性损失显式约束变量空间

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ AI for Science 的全新范式，多智能体科学推理
- 实验充分度: ⭐⭐⭐⭐ 三类场景覆盖全面，包含真实世界验证
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，但公式较多需要仔细消化
- 价值: ⭐⭐⭐⭐⭐ 开辟了MLLM驱动的科学发现新方向
