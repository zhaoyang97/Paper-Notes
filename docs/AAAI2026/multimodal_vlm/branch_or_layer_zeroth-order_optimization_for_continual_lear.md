# Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models

**会议**: AAAI 2026  
**arXiv**: [2506.12409](https://arxiv.org/abs/2506.12409)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 零阶优化, 持续学习, CLIP, 参数高效微调, 模态感知优化  

## 一句话总结
本文系统探索了零阶（ZO）优化在基于PEFT的视觉-语言持续学习（VLCL）中的应用，发现全ZO替换会导致训练不稳定，提出从分支级（branch-wise）到层级（layer-wise）的渐进式ZO-FO混合策略，并基于视觉模态方差更大的理论发现提出MoZO策略（梯度符号归一化+视觉扰动约束），在四个benchmark上达到SOTA。

## 背景与动机
基于VLM（如CLIP）的持续学习（VLCL）近年发展迅速，PEFT策略（如LoRA、MoE adapter）让这些模型能以较低资源开销实现竞争力性能。然而，现有方法几乎清一色使用一阶（FO）优化（SGD/Adam），其确定性更新路径在PEFT的低维参数子空间中容易陷入尖锐局部极小值，导致对当前任务过拟合，加剧灾难性遗忘。零阶（ZO）优化通过随机扰动估计梯度，天然具有跳出局部极小的能力，且不需要反向传播，显存占用更低。但ZO在VLCL中的适用性从未被系统研究。

## 核心问题
**如何将ZO优化有效地集成到VLCL中以提升性能？** 具体而言：（1）简单地把FO全部替换为ZO会导致什么问题？（2）ZO应该被应用在哪个模态分支（视觉 vs 语言）？（3）在单个分支内部，ZO应该被应用在哪些层（连续层 vs 交错层）？（4）两个模态在ZO优化下的行为差异如何处理？

## 方法详解

### 整体框架
基于冻结的CLIP-ViT-B/16骨架，仅训练附加在每一层上的适配器模块（MoE adapter或LoRA）。核心思想是将FO和ZO优化协同应用于不同模态分支和网络层：一部分可训练单元用ZO优化（随机扰动估计梯度，提供探索能力），另一部分保持FO优化（精确梯度，提供稳定性）。最终通过MoZO策略进一步优化。

### 关键设计

1. **分支级ZO探索（Branch-wise）**：首先研究三种分支级配置——Dual（双分支ZO）、Vision（仅视觉ZO+语言FO）、Language（仅语言ZO+视觉FO）。实验发现：Dual w/ ZO导致严重的loss震荡和性能崩溃（Last./Avg.平均下降8.5%/9.5%）；单分支ZO则表现接近甚至超越baseline，且Language w/ ZO普遍优于Vision w/ ZO，因为语言分支张量维度更低、对随机扰动更鲁棒。

2. **层级ZO探索（Layer-wise）**：在确定单分支ZO有效后，进一步探索四种层级配置：Hop-odd（奇数层ZO）、Hop-even（偶数层ZO）、Prefix(six)（前6层ZO）、Suffix(six)（后6层ZO）。关键发现：交错式（Hop-odd/even）显著优于连续式（Prefix/Suffix），因为浅层关注局部特征、深层捕获抽象语义，统一优化方法忽略了这种多样性，而ZO-FO交错能更好匹配每层对探索和稳定的不同需求。Dual w/ ZO + layer-wise平均比全分支ZO提升9.4%。

3. **MoZO（Modality-aware ZO）策略**：通过分析梯度方差分布，发现视觉分支在ZO下的梯度方差显著大于语言分支，导致优化不稳定。MoZO包含两个组件：
   - **梯度符号归一化**：对ZO估计的梯度取sign，只保留方向信息，丢弃幅度信息，抑制异常大梯度
   - **模态差异化扰动**：为视觉分支设置更小的扰动因子 $\epsilon_v < \epsilon_l$，约束视觉分支的参数探索幅度

### 损失函数 / 训练策略
- ZO梯度估计：$\nabla_{ZO}\mathcal{L}(\theta) \approx \frac{\mathcal{L}(\theta + \varepsilon\Delta) - \mathcal{L}(\theta)}{\varepsilon} \cdot \Delta$，其中 $\Delta$ 为随机方向向量，$\varepsilon=0.001$
- 采用保守ZO策略：评估多个候选更新，选择loss最低的那个应用（而非激进的单次估计直接更新）
- MoZO更新规则：视觉分支用 $\epsilon_v \xi_t$ 扰动，语言分支用 $\epsilon_l \xi_t$ 扰动，$\epsilon_v < \epsilon_l$
- FO/ZO混合比例 $\lambda=1$，在第一个task上验证超参后保持不变

## 实验关键数据

基线为MoE4Adapter（CVPR 2024 SOTA），骨架CLIP-ViT-B/16。

| 数据集 (配置) | 指标 | Baseline (FO) | Dual ZO | Vision ZO | Language ZO | Layer-wise最优 |
|---|---|---|---|---|---|---|
| CIFAR Inc10 | Last. | 80.47 | 69.29 (-11.2) | 76.05 | 80.94 (+0.5) | **82.41** (Vis. Hop-even) |
| CIFAR Inc10 | Avg. | 86.97 | 77.36 (-9.6) | 83.93 | 87.00 | **88.51** (Lan. Hop-odd) |
| TinyImg Inc10 | Last. | 77.52 | 67.64 (-9.9) | 72.98 | 76.74 | **79.39** (Vis. Hop-odd) |
| TinyImg Inc10 | Avg. | 85.21 | 75.88 (-9.3) | 82.08 | 85.03 | **87.05** (Lan. Hop-odd) |
| TinyImg Inc20 | Last. | 52.13 | 42.40 (-9.7) | 49.65 | 49.14 | **52.27** (Vis. Hop-even) |
| TinyImg Inc20 | Avg. | 60.55 | 47.64 (-12.9) | 57.90 | 58.69 | **60.98** (Vis. Hop-even) |
| ImgR Inc20 | Last. | 65.36 | 58.56 (-6.8) | 62.54 | 64.38 | **65.68** (Vis. Hop-odd) |
| ImgR Inc20 | Avg. | 71.53 | 65.92 (-5.6) | 69.84 | 70.38 | **72.16** (Lan. Hop-even) |

**MoZO进一步提升**（Dual w/ Hop-even → MoZO）：

| 数据集 | Last. | Avg. |
|---|---|---|
| CIFAR Inc10 | 79.12 → 79.87 (+0.75) | 86.81 → 87.25 (+0.44) |
| TinyImg Inc20 | 51.95 → 52.46 (+0.51) | 60.53 → 61.23 (+0.70) |
| ImgR Inc20 | 64.99 → 65.80 (+0.81) | 71.29 → 71.82 (+0.53) |

**显存效率**（MoE设置）：

| 配置 | MoE显存 | LoRA显存 |
|---|---|---|
| Baseline (全FO) | 19.96GB | 15.11GB |
| Dual w/ ZO | 2.17GB (-89.1%) | 1.73GB |
| Vision w/ ZO | 6.93GB (-65.3%) | 5.71GB |
| Language w/ ZO | 12.39GB (-37.9%) | 11.09GB |

### 消融实验要点
- **ZO策略选择**：激进ZO*（单次估计直接更新）性能最差（Dual: 66.67 Last.），保守ZO（多候选选最优）中等，加上Sign梯度归一化后最好（Language+Sign: 78.52 Last.），验证了梯度幅度控制的重要性
- **交错 vs 连续层**：交错式（Hop-odd/even）始终优于连续式（Prefix/Suffix），因为交错时梯度方差显著更低，训练更稳定
- **LoRA设置下的一致性**：将MoE替换为LoRA后，所有趋势一致——layer-wise ZO有效、交错优于连续、MoZO进一步提升
- **显著性分析**：5次运行中，Language w/ ZO的方差最小、性能最优，验证了语言分支更适合ZO的结论

## 亮点
- **系统性的实证探索**：从branch-wise到layer-wise的渐进式研究路径清晰、实验设计严谨，逐层揭示ZO在VLCL中的最佳应用方式
- **理论动机与实验验证结合**：视觉模态方差更大这一发现不是凭空假设，而是通过梯度方差分布图实验验证后，再提出针对性解决方案
- **显存优势明显**：ZO消除了反向传播，Dual ZO减少89%显存，为资源受限场景提供可行方案
- **对PEFT优化的新视角**：揭示了FO在低维子空间容易陷入局部极小这一被忽视的问题，ZO-FO协同是一种新颖的解决思路

## 局限性 / 可改进方向
- **仅限CLIP**：只在CLIP-ViT-B/16上验证，未测试更大模型（如ViT-L/14）或其他VLM架构（如BLIP-2、LLaVA）
- **仅限图像-文本模态**：未探索音频、视频等其他模态，而论文自己也指出这是limitation
- **数据集规模偏小**：CIFAR-100、Tiny-ImageNet、ImageNet-R都是相对小规模数据集，未在更大规模CL benchmark上验证
- **MoZO提升幅度有限**：虽然一致提升，但幅度较小（0.4~0.8%），和layer-wise ZO带来的大幅提升相比显得不够显著
- **缺乏与其他CL方法的直接对比**：主要对比的是MoE4Adapter+不同ZO配置，缺少与PROOF、CLAP4CLIP等其他VLCL方法的全面对比
- **超参$\epsilon_v$和$\epsilon_l$的选择**：论文未详细说明视觉/语言扰动因子的具体值和调参过程
- **计算开销**：保守ZO策略需要评估多个候选更新，虽然省显存但可能增加训练时间，论文缺少训练时间对比

## 与相关工作的对比
- **vs MoE4Adapter (CVPR 2024)**：本文的直接baseline，用MoE架构实现PEFT-based VLCL的SOTA。本文在其基础上引入ZO优化，通过layer-wise配置可超越其全FO性能，同时大幅减少显存
- **vs ZeroFlow (2025)**：同为将ZO应用于持续学习的工作，但ZeroFlow未区分模态差异，且主要面向CNN架构。本文首次系统研究ZO在VLM多模态场景下的应用，发现并解决了模态间优化差异问题
- **vs MeZO (NeurIPS 2023)**：MeZO将ZO应用于LLM微调，但未考虑持续学习场景和多模态分支差异。本文发现简单全ZO替换在VLCL中失效，必须进行分支级和层级的精细配置
- **vs BOFA**：同在AAAI 2026的CLIP-based持续学习工作，使用正交低秩融合。两者解决同一问题但路径不同——BOFA从参数空间正交性出发，本文从优化器选择出发

## 启发与关联
- ZO-FO混合优化的思想可以推广到其他PEFT场景（如VLM的instruction tuning），特别是在显存受限时
- 视觉模态方差更大这一发现暗示，在多模态学习中视觉和语言分支可能需要不同的优化策略（不仅是学习率），这对多模态训练策略设计有启发
- 交错层优于连续层的发现与ResNet中skip connection的思想异曲同工，提示层间的优化器多样性有益于特征学习

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统研究ZO在VLCL中的应用，视角新颖；但ZO本身不是新技术，MoZO的设计（sign+小扰动）较为直觉
- 实验充分度: ⭐⭐⭐⭐ 分支级+层级的系统探索非常全面，消融丰富；但数据集偏小，缺少与更多VLCL方法的对比
- 写作质量: ⭐⭐⭐ 逻辑框架清晰，但行文有冗余，部分段落重复表述同一观点；公式符号不够统一
- 价值: ⭐⭐⭐⭐ 为PEFT-based VLCL提供了新的优化视角和实用方案，显存优势明显；但CLIP+小数据集的场景限制了实际影响力
