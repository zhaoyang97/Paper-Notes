---
title: >-
  [论文解读] InTrain: Intrinsic Trainability for Zero-Cost Neural Architecture Search
description: >-
  [CVPR 2026][零成本代理] InTrain 把"一个架构能不能被训好"形式化成一个不依赖训练过程的内在不变量，用前向激活的**几何容量**（参与比）和反向梯度的**优化韧性**（梯度健康度）两个分量、再以尺度不变的**乘性耦合**合成单一打分，在 NAS-Bench-101/201 上达到与集成式代理相当、并超过所有单指标代理的排序相关性。
tags:
  - "CVPR 2026"
  - "零成本代理"
  - "神经架构搜索"
  - "可训练性"
  - "参与比"
  - "梯度健康度"
---

# InTrain: Intrinsic Trainability for Zero-Cost Neural Architecture Search

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_InTrain_Intrinsic_Trainability_for_Zero-Cost_Neural_Architecture_Search_CVPR_2026_paper.html)  
**代码**: 无  
**领域**: AutoML / 神经架构搜索  
**关键词**: 零成本代理, 神经架构搜索, 可训练性, 参与比, 梯度健康度

## 一句话总结
InTrain 把"一个架构能不能被训好"形式化成一个不依赖训练过程的内在不变量，用前向激活的**几何容量**（参与比）和反向梯度的**优化韧性**（梯度健康度）两个分量、再以尺度不变的**乘性耦合**合成单一打分，在 NAS-Bench-101/201 上达到与集成式代理相当、并超过所有单指标代理的排序相关性。

## 研究背景与动机
**领域现状**：神经架构搜索（NAS）能自动发现媲美甚至超越人工设计的网络，但传统 NAS 需要把成千上万候选架构训练到收敛，动辄数千 GPU 天。为绕开这一开销，**零成本（training-free）代理**应运而生——只在初始化状态下算一个标量分数来预测架构最终精度，把评估成本降低几个数量级。

**现有痛点**：现有零成本代理（SNIP、GraSP、SynFlow、NASWOT、ZiCo、Zen-score 等）各自盯着可训练性的某一个"症状"：要么只看梯度幅度/连接敏感度，要么只看激活的表达多样性。基于梯度的指标忽略表示几何，基于激活的指标忽略训练动态。综合 benchmark 研究还发现，很多代理在不同数据集/搜索空间间相关性飘忽不定。

**核心矛盾**：这些方法都是在用零散启发式拼凑，却没有回答最根本的问题——**到底是什么让一个架构"可训练"**？换句话说，缺一个同时统一"几何（表达力）"和"梯度（可优化性）"、且独立于具体优化器与超参的理论判据。

**本文目标**：定义一个**内在可训练性（intrinsic trainability）**——一个只由网络拓扑与初始参数决定、与训练流程无关的架构不变量，并把它落地成可在数秒内计算的零成本代理。

**切入角度**：作者把深度网络看作双向信息处理器：前向把输入逐层变换成越来越抽象的表示，反向让误差信号穿过参数空间驱动优化。一个可训练的架构必须两头都强——既有足够**几何容量**表达复杂函数，又有足够**优化韧性**让梯度稳定流动。

**核心 idea**：用"几何容量 $\gamma$"与"优化韧性 $o$"两个可度量分量刻画内在可训练性，并断言二者不是相加而是**乘性门控**关系（一方塌缩另一方就白搭），最终合成单一打分 $I(A)$。

## 方法详解

### 整体框架
InTrain 不训练任何权重，只在初始化网络上做**一次前向 + 一次反向**：前向收集各层激活、算出"几何容量"$\gamma(A)$；反向用人造损失激活全部参数路径、算出"优化韧性"$o(A)$；最后把两者乘性耦合、并按深度做对数归一化，得到架构的内在可训练性分数 $I(A)$。整套度量建立在三条设计原则上——**深度不变性**（不同深度可公平比较）、**组合性**（信息沿层乘性传递，任一层瓶颈会卡死下游，故用 log-乘积聚合）、**双向性**（前向后向都要算）。这是一条纯解析的计算链，不涉及多模块协同或反馈回环，因此用公式而非框架图表达更清楚。

### 关键设计

**1. 几何容量：用参与比量化表示流形的有效维度**

针对"激活类代理只看表达多样性却说不清表达力本质"的问题，InTrain 把第 $\ell$ 层激活 $A_\ell\in\mathbb{R}^{N\times C}$ 看成特征空间里的点云，用其协方差矩阵 $C_\ell$ 的特征谱衡量信息在多少维度上铺开。它不用矩阵秩（秩对所有非零奇异值一视同仁、忽略大小），而用**参与比（Participation Ratio, PR）**：

$$\mathrm{PR}(C_\ell)=\frac{(\operatorname{Tr} C_\ell)^2}{\operatorname{Tr}(C_\ell^2)}=\frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$$

PR 等价于归一化特征值分布的二阶 Rényi 熵的指数 $\mathrm{PR}=\exp(H_2)$（$p_i=\lambda_i/\operatorname{Tr}C_\ell$，$H_2=-\log\sum_i p_i^2$）——PR 高意味着方差均匀分散在很多维度上，表示没有塌缩。卷积层的 $(N,C,H,W)$ 激活被 reshape 成 $(N\cdot H\cdot W, C)$、在通道维算协方差，并加 $\epsilon I$（$\epsilon=10^{-10}$）做数值稳定。由于信息沿层是乘性约束的（某层塌缩会卡死后续所有层），几何容量用 **log-乘积** 聚合，即对各层 PR 取对数再求和：

$$\gamma(A)=\sum_{\ell=1}^{L}\log\big(\mathrm{PR}(C_\ell)\big)$$

**2. 优化韧性：用梯度健康度量化反向传播的稳定性**

几何容量说的是"能表达什么"，但表达力再强、梯度若病态（消失/爆炸）也学不动。InTrain 要一个**不依赖训练动态、也不假设特定架构结构（如残差、BN）**的梯度健康指标。对每个参数 $\theta_i$，它看梯度元素的"方差–极大值比"：

$$h(\theta_i)=\min\!\Big(1,\ \frac{\sigma(\nabla_{\theta_i})}{\max(|\nabla_{\theta_i}|)+\epsilon}\Big)$$

其中 $\sigma(\cdot)$ 是梯度元素的标准差。比值高说明梯度幅度分布均匀（许多元素都在贡献，优化稳定），低则说明集中在少数分量上、不稳定；封顶到 1 是为了防离群值主导并保持可解释性。与几何容量的乘性（串行、受最弱环节制约）不同，作者认为优化韧性是**可加、并行**的属性——每条健康的参数通路独立贡献一份"接受稳定梯度更新的容量"，故用**累加**聚合：

$$o(A)=\sum_{i=1}^{|\Theta|}h(\theta_i)$$

为算梯度，作者用**人造损失**：输入 $X\sim\mathcal{N}(0,1)$、随机标签，分类用交叉熵、空间输出（如分割）用 MSE，让梯度只依赖架构本身而非数据分布，并保证激活到所有参数路径。

**3. 乘性门控：把容量与韧性耦合成单一可训练性分数**

针对"加法会在一方塌缩时仍给出中等分"的缺陷，作者提出**乘性门控假设**：容量与韧性互为对方的门——容量大但韧性 $o\approx0$，再强的表达力也传不动梯度；韧性大但容量退化，优化虽稳却只能表达平凡映射。于是用 $\gamma\times(1+o)$ 而非 $\gamma+o$。最终的内在可训练性按深度做对数归一化：

$$I(A)=\frac{\gamma(A)\times\big(1+o(A)\big)}{\log(L+1)}$$

$(1+o)$ 保证正性并在 $o=0$ 时不退化；$\log(L+1)$ 而非 $L$ 做归一化，是因为 $\gamma$、$o$ 都随深度增长，直接除以 $L$ 会过度惩罚深网，对数形式更贴合信息处理容量随深度次线性增长的事实。

### 损失函数 / 训练策略
InTrain 本身**不训练**，只用人造损失驱动一次反向以采集梯度统计。实现上用 64 张 $64\times64$、逐像素独立采自标准高斯的合成图作为输入。要落地到完整 NAS，作者把 InTrain 作为打分嵌入进化搜索框架，得到 **InTrain-NAS**，超参沿用进化式 NAS 的常规配置。

## 实验关键数据

### 主实验
InTrain 与各零成本代理在 NAS-Bench-101 上对真实测试精度的排序相关性（KT=Kendall's τ，SPR=Spearman ρ，越高越好）：

| 代理 | NAS-Bench-101 KT | NAS-Bench-101 SPR |
|------|------|------|
| Grad_norm | -0.17 | -0.25 |
| SynFlow | 0.20 | 0.29 |
| Zen-score | 0.31 | 0.44 |
| ZiCo | 0.46 | 0.63 |
| FLOPs | 0.31 | 0.44 |
| #Params | 0.31 | 0.43 |
| **InTrain (本文)** | **0.56** | **0.75** |

NAS-Bench-201 上跨三数据集的 KT（与 ImageNet16-120 上 SPR）：

| 代理 | CIFAR-10 KT | CIFAR-100 KT | ImageNet16-120 KT |
|------|------|------|------|
| SynFlow | 0.561 | 0.553 | 0.531 |
| Jacov | 0.616 | 0.639 | 0.602 |
| ZiCo | 0.607 | 0.614 | 0.587 |
| VKDNW_single | 0.618 | 0.634 | 0.622 |
| AZ-NAS（集成） | **0.712** | **0.696** | 0.673 |
| **InTrain (本文)** | 0.669 | 0.671 | **0.675** |

InTrain 是所有**单指标**代理里最强的；唯一更高的 AZ-NAS 是集成多个异质代理的方法（缺统一理论），且在 ImageNet16-120 上反被 InTrain 超过。把 InTrain 接进进化搜索得到的 InTrain-NAS，在 ImageNet-1K 上各 FLOPs 预算下的 top-1：

| FLOPs 预算 | 方法 | top-1 (%) | 搜索成本 (GPU 天) |
|------|------|------|------|
| 450M | ZiCo / AZ-NAS | 78.1 / 78.6 | 0.4 |
| 450M | **InTrain-NAS** | **78.9** | 0.4 |
| 600M | AZ-NAS | 79.9 | 0.6 |
| 600M | **InTrain-NAS** | **79.9** | 0.4 |
| 1000M | AZ-NAS | 81.1 | 0.7 |
| 1000M | **InTrain-NAS** | **81.3** | 0.4 |

### 消融实验
NAS-Bench-201 上拆解两个分量与耦合方式（KT / SPR）：

| 配置 | Kendall's τ | Spearman ρ | 说明 |
|------|------|------|------|
| PR-only | 0.61 | 0.82 | 只用几何容量 |
| Grad-only | 0.63 | 0.83 | 只用梯度健康度 |
| PR + Grad（加法） | 0.59 | 0.80 | 朴素相加，比单分量还差 |
| **InTrain（乘性耦合）** | **0.67** | **0.86** | 完整模型 |

### 关键发现
- **加法耦合反而掉点**：PR+Grad 相加得 KT=0.59，比 PR-only(0.61)、Grad-only(0.63) 都低——两个信号朴素相加会相互干扰，直接证伪"两分量独立可加"，支撑了乘性门控假设。
- **跨数据集稳定**：多数代理在不同数据集上相关性波动明显，InTrain 在 CIFAR-10/100/ImageNet16-120 上保持稳定排序，作者归因于它基于内在架构属性而非数据相关启发式。
- **几何容量与优化韧性单独都有用**：PR-only 与 Grad-only 都给出有意义的相关性（0.61、0.63），但乘性结合后协同增益明显（0.67）。

## 亮点与洞察
- **把"可训练性"从零散症状提升为统一不变量**：用前向 PR + 反向梯度健康度同时覆盖"表达几何"和"优化动态"两个长期被分头处理的维度，是这篇最让人"啊哈"的地方。
- **乘性门控的物理直觉很干净**：用两个极限情形（容量大韧性零 / 韧性大容量退化）论证应当用 $\gamma\times(1+o)$ 而非加法，并被消融实验直接验证，理论与实验闭环。
- **聚合方式区分串/并行**：几何容量用 log-乘积（串行瓶颈），优化韧性用累加（并行通路），这种"按物理性质选聚合算子"的思路可迁移到其他需要把逐层统计汇总成单标量的代理设计。
- **参与比 = 二阶 Rényi 熵指数**给了 PR 一个信息论解释，比直接用秩更细腻。

## 局限与展望
- **与集成代理仍有差距**：在 CIFAR-10/100 上 AZ-NAS（集成）相关性更高，单一理论代理在这些设置下尚未全面登顶。
- **依赖人造高斯输入与随机标签**：梯度统计来自合成数据，正文虽强调这能隔离数据分布影响，但真实数据分布下的可训练性是否完全一致仍是开放问题（⚠️ 论文有"对输入变化的鲁棒性"实验佐证，但缓存未给全表）。
- **PR 与梯度健康度的计算成本随网络规模上升**：协方差特征谱与逐参数梯度统计在超大模型上的可扩展性值得进一步评估。
- 可改进方向：把乘性门控推广到三个以上分量、或将归一化项 $\log(L+1)$ 替换为可学习的深度修正。

## 相关工作与启发
- **vs ZiCo / NASWOT（梯度类代理）**：它们只刻画梯度统计或 NTK 条件数，忽略表示几何；InTrain 把梯度健康度只当作两分量之一，并与几何容量乘性耦合，相关性更高更稳。
- **vs Zen-score / TE-NAS（激活/表达类代理）**：它们评估表达多样性却忽略训练动态；InTrain 用 PR 量化几何容量、再补上优化韧性，覆盖更全。
- **vs AZ-NAS（集成代理）**：AZ-NAS 靠拼多个异质代理拿到最高相关性，但缺统一理论、可解释性差且计算更重；InTrain 用单一有理论根基的代理逼近其性能，并在 ImageNet16-120 上反超。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把可训练性形式化为几何容量×优化韧性的乘性不变量，是零成本 NAS 里少见的理论统一视角。
- 实验充分度: ⭐⭐⭐⭐ NAS-Bench-101/201 + MobileNetV2/ImageNet + 组件消融齐全，但鲁棒性全表与方差信息缓存未尽。
- 写作质量: ⭐⭐⭐⭐⭐ 三条设计原则 → 两分量 → 乘性耦合的推导链条清晰，动机具体不空泛。
- 价值: ⭐⭐⭐⭐ 提供了可数秒计算、跨数据集稳定的单指标代理，对降低 NAS 搜索成本有实用意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Progressive Neural Architecture Generation](progressive_neural_architecture_generation.md)
- [\[ICCV 2025\] Loss Functions for Predictor-based Neural Architecture Search](../../ICCV2025/others/loss_functions_for_predictor-based_neural_architecture_search.md)
- [\[CVPR 2025\] Subnet-Aware Dynamic Supernet Training for Neural Architecture Search](../../CVPR2025/others/subnet-aware_dynamic_supernet_training_for_neural_architecture_search.md)
- [\[CVPR 2026\] HyperNAS: Enhancing Architecture Representation for NAS Predictor via Hypernetwork](hypernas_enhancing_architecture_representation_for_nas_predictor_via_hypernetwor.md)
- [\[CVPR 2025\] VKDNW: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights](../../CVPR2025/others/training-free_neural_architecture_search_through_variance_of_knowledge_of_deep_n.md)

</div>

<!-- RELATED:END -->
