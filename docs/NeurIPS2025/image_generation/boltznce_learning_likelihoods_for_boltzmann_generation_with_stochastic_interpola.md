# BoltzNCE: Learning Likelihoods for Boltzmann Generation with Stochastic Interpolants

**会议**: NeurIPS 2025  
**arXiv**: [2507.00846](https://arxiv.org/abs/2507.00846)  
**代码**: 有  
**领域**: 生成模型 / 分子模拟  
**关键词**: Boltzmann分布, 噪声对比估计, 随机插值, 分子构象, 自由能

## 一句话总结
BoltzNCE 用 Score Matching + InfoNCE 混合训练 Energy-Based Model 来近似 Boltzmann Generator 的似然，避免了昂贵的 Jacobian trace 计算，在丙氨酸二肽构象生成上实现 100× 推理加速且自由能误差仅 0.02 $k_BT$。

## 研究背景与动机
1. **领域现状**：Boltzmann Generator（BG）是从能量函数 $E(x)$ 对应的 Boltzmann 分布 $p(x) \propto \exp(-E(x)/k_BT)$ 采样的深度生成模型。基于 normalizing flow 或 diffusion 的方法可以生成样本，但计算似然需要昂贵的 Jacobian 行列式/trace。
2. **现有痛点**：精确似然计算（如 Hutchinson estimator for trace）在推理时极慢（equivariant CNF 需 9.37 小时 for alanine dipeptide），成为将 BG 用于自由能计算和重要性采样的瓶颈。
3. **核心矛盾**：好的采样质量需要准确的 flow/diffusion 模型，但似然评估的计算代价与采样质量无关——即使采样很好，每次评估似然仍然贵。
4. **本文要解决什么**：将采样器质量和似然可处理性解耦——用一个独立的 EBM 来近似似然，避免 Jacobian 计算。
5. **切入角度**：将 BG 分为两阶段——先训练 Boltzmann Emulator（用 flow matching），再训练 EBM 近似其密度（用 NCE + score matching）。
6. **核心idea一句话**：用 EBM 的 NCE 训练作为 Boltzmann Generator 似然的快速代理，实现 100× 推理加速。

## 方法详解

### 整体框架
两阶段：(1) 用 stochastic interpolant + flow matching 训练 Boltzmann Emulator（生成 Boltzmann 分布样本）；(2) 在生成样本上用混合 Score Matching + InfoNCE 训练 EBM $\hat{U}(x)$，使 $\exp(\hat{U}(x))$ 近似真实密度。

### 关键设计

1. **Stochastic Interpolant 框架**:
   - 做什么：在噪声 $x_0$ 和 Boltzmann 样本 $x_1$ 之间建立平滑路径 $I_t = \alpha_t x_0 + \beta_t x_1$
   - 核心思路：训练向量场 $v_t$ 匹配由插值诱导的概率流，实现从噪声到 Boltzmann 分布的映射
   - 设计动机：stochastic interpolant 自然地连接了 ODE 流和扩散过程，为后续的 EBM 训练提供特殊的 score function

2. **BoltzNCE 混合训练**:
   - 做什么：同时用 Score Matching 和 InfoNCE 训练 EBM
   - Score Matching Loss：$\mathcal{L}_{SM} = \mathbb{E}[|\alpha_t \nabla\hat{U}_t(\tilde{I}_t) + x_0|^2]$ — 强制 EBM 的梯度匹配插值过程的 score
   - InfoNCE Loss：$\mathcal{L}_{InfoNCE} = -\mathbb{E}[\log\frac{\exp(\hat{U}_t(\tilde{I}_t))}{\sum_{t'}\exp(\hat{U}_{t'}(\tilde{I}_t))}]$ — 通过时间步对比学习密度
   - 混合：$\mathcal{L}_{BoltzNCE} = \mathcal{L}_{SM} + \mathcal{L}_{InfoNCE}$
   - 设计动机：单独 InfoNCE 给出全局密度但梯度不准，单独 SM 给出好的梯度但密度不准——两者互补

3. **自由能估计与重要性采样重加权**:
   - 做什么：用 EBM 的似然做重要性采样，修正采样偏差
   - 核心思路：$\hat{Z} = \sum_i w_i$，$w_i = \exp(-E(x_i)/k_BT) / \hat{\rho}(x_i)$
   - 自由能差：$\Delta F = -k_BT \ln(\hat{Z}_A / \hat{Z}_B)$

### 损失函数 / 训练策略
两阶段训练。Stage 1: Flow matching with equivariant vector field。Stage 2: Score matching + InfoNCE，各 epoch ~12h 总训练时间。

## 实验关键数据

### 主实验（丙氨酸二肽）

| 方法 | $\Delta F / k_BT$ | 误差 | 推理时间 |
|------|-------------------|------|----------|
| Umbrella Sampling (ground truth) | 4.10 ± 0.26 | – | – |
| ECNF (精确似然) | 4.09 ± 0.05 | 0.01 | 9.37h |
| GVP Vector Field | 4.38 ± 0.67 | 0.28 | 18.4h |
| **BoltzNCE** | **4.08 ± 0.13** | **0.02** | **0.09h** |

### 消融实验
| 配置 | KL 散度 (8-mode Gaussian) | 说明 |
|------|--------------------------|------|
| InfoNCE only | 0.2395 | 梯度不准 |
| Score Matching only | 0.2199 | 全局密度不准 |
| **BoltzNCE (combined)** | **0.0150** | 15× 改进 |

| 配置 | KL (Checkerboard) | 说明 |
|------|-------------------|------|
| InfoNCE only | 3.8478 | 多模态难对齐 |
| **BoltzNCE** | **0.1987** | 19× 改进 |

### 关键发现
- 混合训练比任一单独方法提升 15-19×（KL 散度），两个损失确实互补
- 推理 100× 加速（0.09h vs 9.37h），因为 EBM 前向传播比 Jacobian trace 快得多
- 自由能误差 0.02 $k_BT$，与精确方法（0.01）相当
- 泛化到 7 种二肽系统，误差可接受（0.43 $k_BT$），6× 加速

## 亮点与洞察
- **采样-似然解耦**：将好的采样器和快速似然评估分离是一个聪明的设计——不需要让生成模型本身可计算似然，用独立的 EBM 作为代理。这个思路可推广到任何需要似然但生成模型不提供的场景。
- **NCE + SM 互补性**：InfoNCE 提供全局密度对齐（通过对比不同时间步），SM 提供局部梯度准确性——两者的组合在 2D 实验中的 15× 改进非常有说服力。
- **科学计算的实际加速**：100× 推理加速对分子模拟有巨大实际价值，使得自由能计算从需要集群变为单 GPU 可行。

## 局限性 / 可改进方向
- 仅在小分子（二肽）上验证，大蛋白质/复杂系统的可扩展性未知
- 泛化到新二肽系统时误差增大（0.43 vs 0.02 $k_BT$），需要微调
- EBM 训练本身需要 ~12h，虽然是一次性成本但不可忽略
- 近似似然引入的偏差在极端尾部可能被放大

## 相关工作与启发
- **vs ECNF (Köhler et al.)**：ECNF 用精确 Jacobian但极慢，BoltzNCE 用近似 EBM 但 100× 快，自由能精度相当
- **vs Targeted Free Energy Perturbation**：传统重加权方法依赖好的 overlap，BoltzNCE 通过学习似然改善 overlap
- **vs Flow Matching (Lipman et al., 2023)**：BoltzNCE 的 Stage 1 用 flow matching 建采样器，但 Stage 2 的 EBM 训练是新贡献

## 评分
- 新颖性: ⭐⭐⭐⭐ 采样-似然解耦 + NCE/SM 混合在 Boltzmann 生成中的首次应用
- 实验充分度: ⭐⭐⭐⭐ 2D 合成 + 丙氨酸二肽 + 7 二肽泛化
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，2D 实验直观展示了互补性
- 价值: ⭐⭐⭐⭐ 对分子模拟和自由能计算有重要实际意义
