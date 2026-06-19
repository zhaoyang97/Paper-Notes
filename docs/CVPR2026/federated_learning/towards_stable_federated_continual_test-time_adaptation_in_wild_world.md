---
title: >-
  [论文解读] Towards Stable Federated Continual Test-Time Adaptation in Wild World
description: >-
  [CVPR 2026][联邦学习][联邦持续测试时适应] 本文提出 **BPFedCTTA**，用贝叶斯视角统一处理「联邦持续测试时适应（FedCTTA）」：把全局模型当作高斯先验、用 MAP 估计稳住每个无标注客户端的本地适应（BPA），再用输出熵算出的不确定性门控来选择性地融合客户端更新（UGSA），从而在客户端顺序到来、分布完全无关的极端异构场景下既能适应新域、又不破坏全局模型、缓解灾难性遗忘。
tags:
  - "CVPR 2026"
  - "联邦学习"
  - "联邦持续测试时适应"
  - "贝叶斯先验"
  - "MAP 估计"
  - "不确定性门控"
  - "灾难性遗忘"
---

# Towards Stable Federated Continual Test-Time Adaptation in Wild World

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Towards_Stable_Federated_Continual_Test-Time_Adaptation_in_Wild_World_CVPR_2026_paper.html)  
**代码**: https://github.com/LiwenWang919/BPFedCTTA  
**领域**: 联邦学习 / 测试时适应 / 持续学习  
**关键词**: 联邦持续测试时适应, 贝叶斯先验, MAP 估计, 不确定性门控, 灾难性遗忘

## 一句话总结
本文提出 **BPFedCTTA**，用贝叶斯视角统一处理「联邦持续测试时适应（FedCTTA）」：把全局模型当作高斯先验、用 MAP 估计稳住每个无标注客户端的本地适应（BPA），再用输出熵算出的不确定性门控来选择性地融合客户端更新（UGSA），从而在客户端顺序到来、分布完全无关的极端异构场景下既能适应新域、又不破坏全局模型、缓解灾难性遗忘。

## 研究背景与动机

**领域现状**：联邦学习（FL）能在保护隐私的前提下跨分散数据协同训练，但模型部署后常遇到与训练分布不同的测试数据。个性化联邦学习（PFL）能缓解客户端异构，却几乎都假设目标客户端有标注数据；测试时适应（TTA）能做无标注适应，却是为「中心化、单域」模型设计的。

**现有痛点**：把 TTA 直接搬进联邦场景会出事。FL 全局模型是多个异构源知识聚合出的「平均解」，落在一个**平坦的损失盆地**里——这对泛化好，但对噪声更新极其敏感。单个客户端的无监督 TTA 更新缺少合适的正则项，很容易把参数推出这个稳定盆地，迅速过拟合到当前客户端、发生本地模型漂移。前人虽已有联邦 CTTA 工作（[40]），但它假设客户端是**预定义簇 + 同步更新**，且只处理已知的空间异构，无法应对真实部署里客户端**异步、顺序到来、分布完全无关**的极端时空异构，也没有约束本地适应稳定性、量化更新可靠性的机制。

**核心矛盾**：作者把问题凝练成两个耦合的挑战。**(C1) 无标注本地适应不稳定**——朴素 TTA 没有正则，会把模型推离平坦盆地。**(C2) 持续全局演化不安全**——服务器在无标注顺序场景下没有真值去核验某个客户端更新的质量，若像 FedAvg 那样盲目聚合一个过拟合或基于剧烈域偏移的更新，会「毒化」全局模型、遗忘之前客户端积累的知识；而经典持续学习常用的数据回放在 FL 的隐私约束下又不可行。

**本文目标 / 切入角度**：作者用一个**统一的概率视角**同时解决 C1 和 C2——把全局模型看作一个不断演化的**先验**，把每个客户端的本地适应重新表述为以无标注数据为引导的**近似后验推断**。这样一来，「正则化本地适应」和「安全聚合」就不是两个临时补丁，而是同一个贝叶斯框架的两端。

**核心 idea**：用「全局模型当先验 + MAP 估计稳本地、输出熵当可靠性门控稳全局」替代「无约束 TTA + FedAvg」，把持续联邦适应建模成一个贝叶斯滤波式的递归过程。

## 方法详解

### 整体框架
BPFedCTTA 面向 FedCTTA 设定：系统先用 $N$ 个有标注源客户端经标准 FedAvg 训练出初始全局模型 $\theta_G^{(0)}$；部署后，$K$ 个只有无标注数据流的目标客户端 $\{C_k\}$ 在时间 $t_1,\dots,t_K$ **顺序、异步**到来，且各自分布 $P_k$ 互不相关、严格禁止共享任何特征或 logit。整个目标函数是「本地稳定」与「全局演化」的权衡：

$$\min_{\{\theta'_k,\theta_G^{(k)}\}} \sum_{k=1}^{K}\Big[\underbrace{\tilde{\mathcal{R}}_k(\theta'_k)}_{\text{本地稳定}} + \lambda\underbrace{\big(\textstyle\sum_{s=1}^{N}\mathcal{R}_s(\theta_G^{(k)})\big)}_{\text{全局演化}}\Big]$$

框架是**两级结构**：客户端侧用 **BPA** 把当前全局模型当先验、做 MAP 本地适应，得到个性化模型 $\theta'_k$ 并上报其不确定性；服务器侧用 **UGSA** 按不确定性门控把这个更新融进全局模型 $\theta_G^{(k-1)}\to\theta_G^{(k)}$，再分发给下一个客户端，形成一个不断演化的系统。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["源客户端有标注数据<br/>FedAvg 训练"] --> B["全局模型 θ_G<br/>(平坦损失盆地·先验)"]
    B -->|分发给顺序到来的<br/>无标注客户端 C_k| C["贝叶斯先验引导适应 BPA<br/>全局模型当高斯先验·MAP 估计"]
    C --> D["个性化模型 θ'_k<br/>+ 预测熵不确定性 σ_k"]
    D --> E["不确定性门控单客户端聚合 UGSA<br/>w_k=softmax(-βσ_k) 门控融合"]
    E -->|θ_G←θ_G+γ·w_k·(θ'_k−θ_G)| B
    E --> F["演化后的全局模型<br/>部署到下一客户端"]
```

### 关键设计

**1. 贝叶斯先验引导适应 BPA：用全局模型当先验、MAP 估计把无标注本地适应锁在稳定盆地里**

这一设计直击 C1——只用无标注数据 $X_k$ 适应一个复杂模型本质上是**病态**的，数据少或带噪时极易不稳定或过拟合。作者把本地适应建模成近似后验 $p(\theta|X_k,\theta_G^{(k-1)})$：以当前全局模型为中心定义一个高斯先验 $p(\theta|\theta_G^{(k-1)})=\mathcal{N}(\theta|\theta_G^{(k-1)},\Sigma_0)$，表达「最优参数应当待在那个泛化良好的全局解附近」这一信念；无标注数据的似然 $p(X_k|\theta)\propto\exp(-\mathcal{L}(\theta;X_k))$ 由一个 TTA 损失隐式建模（鼓励高置信、局部一致）。不求完整后验，BPA 只取 **MAP**（后验众数），展开 log 先验后目标变成熵最小化损失加一个二次正则：

$$\theta'_k=\arg\min_\theta\Big(\mathcal{L}_{\text{EM}}(\theta)+\tfrac{1}{2}(\theta-\theta_G^{(k-1)})^\top\Sigma_0^{-1}(\theta-\theta_G^{(k-1)})\Big)$$

这个由贝叶斯推出的二次正则项 $\mathcal{L}_{\text{BPA}}$ 把适应锚定在全局先验的高概率区域，相当于给 TTA 配了一个「数据驱动、有原理依据」的正则化器——这正是朴素 TTA 缺的东西。更巧的是先验精度（协方差逆）会**随模型不确定性自适应缩放**：$\Sigma_0^{-1}=\text{diag}\big(\frac{1}{\sigma_0^2}\mathbb{E}_{x\in X_k}[\mathcal{H}(p(y|x;\theta_G^{(k-1)}))]\big)$，其中 $\mathcal{H}$ 是香农熵。模型置信（熵低）时收紧先验、把参数拉得更牢，模型不确定（熵高）时放松先验、给适应留余地，从而在异构无标注数据流上既防漂移又能学到东西。

**2. 不确定性门控单客户端聚合 UGSA：把聚合当贝叶斯置信更新、用输出熵门控挡住不可靠更新**

这一设计针对 C2——服务器拿不到真值，没法判断某个客户端更新该不该信。BPA 的概率化恰好给了答案：既然本地适应是后验推断，就能用**适应后模型的预测熵**直接量化更新的可靠性。UGSA 把聚合看成对全局参数的一次贝叶斯置信更新：设上一轮全局模型 $p(\theta_G^{(k-1)})=\mathcal{N}(\theta_G^{(k-1)},\Sigma_G^{(k-1)})$、客户端后验 $q_k(\theta)=\mathcal{N}(\theta'_k,\Sigma_k)$，新全局分布通过最小化两个 KL 散度的加权和得到（既贴合客户端后验、又不偏离旧全局先验），其解是两者在密度空间的几何平均 $p(\theta_G^{(k)})\propto p(\theta_G^{(k-1)})^{1-w_k}q_k(\theta)^{w_k}$。门控权重由客户端不确定性给出：

$$\sigma_k=\mathbb{E}_{x\in X_k}[\mathcal{H}(p(y|x;\theta'_k))],\qquad w_k=\frac{\exp(-\beta\sigma_k)}{\sum_j\exp(-\beta\sigma_j)}$$

$\beta$ 控制衰减锐度——置信（低熵）客户端贡献被放大，高不确定客户端被压低，避免噪声漂移。这等价于一个**联邦贝叶斯滤波**：$p(\theta_G^{(k)}|X_{1:k})\propto q_k(\theta|X_k)^{w_k}\,p(\theta_G^{(k-1)}|X_{1:k-1})$，即精度加权的证据累积，让全局模型在无标注异构条件下递归地、安全地吸收新知识。工程上用对角近似 $\Sigma_G$ 提速，落地成一行简洁更新：

$$\theta_G^{(k)}=\theta_G^{(k-1)}+\gamma\,w_k\,(\theta'_k-\theta_G^{(k-1)})$$

$\gamma$ 是服务器学习率。和 FedAvg 的等权平均相比，UGSA 的关键区别是「更新量先经过可靠性门控 $w_k$ 再注入」，这就是它能挡住毒化更新、缓解灾难性遗忘的根本原因。

## 实验关键数据

### 主实验
分类用 ResNet-18（CIFAR10-C / CIFAR100-C，腐蚀严重度 5，10 客户端 / 50 轮），自然图像分割用 SegFormer-B5（Cityscapes→ACDC 四种恶劣天气），医学分割用 U-Net（视网膜眼底 / 前列腺 MRI，按医疗中心天然分域）。对比覆盖 FL（FedAvg）、PFL（FedProx/FedBN/FedGA）、TTA/CTTA（Tent/CoTTA/BeCoTTA/TCA）以及 TTA+FL（ATP/FedTHE+/FedCTTA/TTA-FedDG）。

| 基准 | 指标 | 本文 | 次优基线 | 说明 |
|--------|------|------|----------|------|
| CIFAR10-C | 平均 Acc | **68.44** | 68.09 (TTA-FedDG) | 顺序腐蚀适应，领先但优势小 |
| CIFAR100-C | 平均 Acc | **67.58** | 65.58 (TTA-FedDG) | 类别更多时优势拉大 +2.0 |
| Cityscapes→ACDC | mIoU(Seq1→Seq4) | 62.88→**65.19** | FedCTTA 61.66→63.28 | 序列内**逐步上升**，体现持续学习有效 |
| 医学分割 | 平均 Dice | 75.06~75.52 | 与 TTA-FedDG 互有胜负 | 4 个序列中 3 个进前二，跨序列最稳 |

对照组里，标准 TTA 在顺序场景明显退化（CoTTA 在 Table 2 从 58.60% 掉到 56.99%、Tent 从 57.06% 崩到 52.72%），PFL 因需标注数据表现平平（FedBN 64.34/57.91），而源模型与 FedAvg/本地微调在分割上随时间**持续掉点**（FedAvg 54.30→53.51），反衬出 BPFedCTTA 的 mIoU 随序列上升的稳定性。

### 消融实验
组件消融（Table 4，逐步叠加）与不确定性度量方式对比（Table 5）：

| 配置 | CIFAR10-C | CIFAR100-C | Cityscapes Seq1→Seq4 | 说明 |
|------|---------|----------|------|------|
| FedAvg | 65.49 | 59.50 | 54.30→53.51 | 无适应，分割随时间退化 |
| + Tent | 64.72 | 57.51 | 57.06→52.72 | 朴素 TTA 顺序场景崩溃 |
| + BPA | 67.15 | 62.89 | 59.82→58.30 | 仅本地正则，较 Tent +2.43/+5.38 |
| + Tent + UGSA | 66.83 | 61.45 | 58.53→58.12 | 仅聚合门控，缓解遗忘但有限 |
| **完整 (BPA+UGSA)** | **67.74** | **64.57** | 62.88→**65.19** | 两者互补，序列内逐步上升 |

| UGSA 不确定性度量 | CIFAR10-C | 单客户端耗时(s) | 遗忘 ∆%(越低越好) |
|------|---------|------|------|
| 预测熵（默认） | 68.44 | 4.49 | 4.21 |
| 能量分数 | 68.62 | 8.83 | 3.95 |
| 预测一致性 | 68.79 | 19.46 | 3.82 |
| MC Dropout(T=10) | 68.75 | 64.92 | 3.45 |
| 集成(M=3) | 68.65 | 20.13 | 3.54 |

### 关键发现
- **BPA 与 UGSA 互补且都不可少**：BPA 负责稳本地（相比 Tent 在 CIFAR100-C 提升 +5.38%），UGSA 负责安全全局合并；仅有其一都打不过完整模型，且只有完整版能在分割序列里实现 62.88→65.19 的逐步上升。
- **不确定性度量是速度-精度权衡**：预测熵精度与速度平衡最好（4.49s）；多视角方法（一致性 / MC Dropout / 集成）遗忘更低、精度略高，但 MC Dropout 慢到 64.9s/客户端，实用性差，故默认选熵。
- **超参 $\beta$ 至关重要**：$\beta=0$（关闭 UGSA 门控）时稳定性崩到 51.8%；$\beta=0.2$ 稳定性峰值 62.9%、总体 mIoU 最优 63.5%；服务器学习率 $\gamma=0.5$ 最佳，$\gamma$ 过小压塑性、过大伤稳定性。
- **抗异构**：在极端 Non-IID（Dirichlet $\alpha=0.05$）下达 64.5%/62.5%，比最强基线高 1.4%/1.8%；从 $\alpha=10$ 到 $\alpha=0.05$ 仅退化 5.3%/6.7%，远小于 FedAvg 的 9.0%/9.3%。

## 亮点与洞察
- **一个贝叶斯视角同时解两个耦合难题**：把「本地正则」和「安全聚合」统一成先验—后验推断，BPA 的二次正则项和 UGSA 的熵门控不是两个临时补丁，而是同一框架自然推出的结果——这种「机制自洽」让方法读起来很顺，也好解释为什么有效。
- **复用平坦盆地几何当先验**：作者敏锐地指出 FL 全局模型落在平坦盆地、对噪声更新敏感，于是直接把这个盆地当高斯先验来锚定适应，是把「FL 模型本身的性质」转成「TTA 正则化器」的巧思。
- **不确定性既驱动本地、又驱动全局**：预测熵在 BPA 里自适应缩放先验精度、在 UGSA 里当门控权重，一个量打通两端，工程上也省事。
- **落地形式极简**：UGSA 最终化成 $\theta_G\!\leftarrow\!\theta_G+\gamma w_k(\theta'_k-\theta_G)$ 一行，几乎是带门控权重的 FedAvg，迁移到现有联邦框架几乎零成本，这个「理论重、落地轻」的转化值得借鉴。

## 局限与展望
- **CIFAR10-C 上优势很薄**：68.44 vs TTA-FedDG 68.09 仅 +0.35，且在若干腐蚀类型上反被基线超过（如 Snow/Frost 列），说明简单顺序场景下相对收益有限，亮点主要在更难的 CIFAR100-C 和分割。
- **表格数值存在不一致**：正文称 CIFAR100-C 达 67.58%，但消融 Table 4 与 Table 5 中完整模型只有 64.57%（⚠️ 以原文为准），两处口径疑似不同（如是否含某变体），读者引用时需注意。
- **高斯先验与对角协方差是强假设**：BPA 用各向同性高斯先验、UGSA 用对角 $\Sigma_G$ 近似，参数间相关性被忽略；C1/C2 的理论证明放在附录、正文未展开，严格性需查附录。
- **依赖熵作可靠性代理有风险**：OOD 样本上模型可能「自信地错」（低熵但错），此时门控会放大坏更新；消融也显示多视角不确定性遗忘更低，暗示熵并非最稳的可靠性度量。
- **客户端被假设分布完全无关**：现实中顺序客户端常有时间相关性，方法未利用这种结构，若能建模相邻客户端关系或有进一步收益。

## 相关工作与启发
- **vs 朴素 TTA（Tent/CoTTA）**：它们靠输出级熵最小化或伪标签自训练，在联邦平坦盆地里无正则、顺序场景下崩溃（Tent 57.06→52.72）；本文用全局先验二次正则锚住，把「无约束适应」变成「MAP 受约束适应」。
- **vs 联邦 CTTA 前作 [40]（FedCTTA）**：前作是同步、预定义簇、已知空间偏移，且为相似度计算共享噪声样本 logit（隐私松）、偏个性化；本文针对异步顺序、分布完全无关的极端时空异构，严禁特征/logit 共享，且聚焦全局模型的安全持续演化。
- **vs 个性化 FL（FedBN/FedProx/FedGA）**：PFL 多数假设目标客户端有标注、只管训练阶段，无法处理测试时域偏移与顺序到来；本文全程无标注、面向部署后持续适应。
- **vs FedTHE+/TTA-FedDG 等 TTA+FL**：同属把 TTA 接入 FL，但本文的差异点在「贝叶斯统一 + 不确定性门控聚合」，在类别多/序列长的难场景（CIFAR100-C、Cityscapes 序列末端）拉开更大差距。

## 评分
- 新颖性: ⭐⭐⭐⭐ 贝叶斯统一视角把本地正则与安全聚合一体化，FedCTTA 设定也更贴近真实部署，但单个组件（MAP 正则、熵门控）各有渊源。
- 实验充分度: ⭐⭐⭐⭐ 覆盖分类 + 自然/医学分割四类基准、十余基线、Non-IID 与超参敏感性都做了，但表格存在数值不一致、CIFAR10-C 优势偏薄。
- 写作质量: ⭐⭐⭐⭐ 动机—挑战—方法逻辑清晰、公式推导完整，框架自洽好读；个别正文与表格数据对不上略影响可信度。
- 价值: ⭐⭐⭐⭐ 面向隐私敏感、客户端异步到来的真实联邦部署，落地形式极简（一行门控聚合），对工业界持续适应系统有参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] HiLoRA: Hierarchical Low-Rank Adaptation for Personalized Federated Learning](hilora_hierarchical_low-rank_adaptation_for_personalized_federated_learning.md)
- [\[CVPR 2026\] From Selection to Scheduling: Federated Geometry-Aware Correction Makes Exemplar Replay Work Better under Continual Dynamic Heterogeneity](from_selection_to_scheduling_federated_geometry-aware_correction_makes_exemplar_.md)
- [\[CVPR 2026\] FedAdamom: Adaptive Momentum for Improved Generalization in Federated Optimization](fedadamom_adaptive_momentum_for_improved_generalization_in_federated_optimizatio.md)
- [\[CVPR 2026\] Domain Sensitive Federated Learning with Fisher-Informed Pruning](domain_sensitive_federated_learning_with_fisher-informed_pruning.md)
- [\[CVPR 2026\] Personalized Federated Training of Diffusion Models with Privacy Guarantees](personalized_federated_training_of_diffusion_models_with_privacy_guarantees.md)

</div>

<!-- RELATED:END -->
