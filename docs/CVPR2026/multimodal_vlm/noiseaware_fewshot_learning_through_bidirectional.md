# Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment

**会议**: CVPR 2026  
**arXiv**: [2603.11617](https://arxiv.org/abs/2603.11617)  
**代码**: 无 (论文未提供代码链接)  
**领域**: 视觉-语言模型 / 小样本学习 / 噪声标签学习  
**关键词**: 噪声标签, prompt learning, 最优传输, CLIP, 小样本学习  

## 一句话总结
提出NA-MVP框架，通过双向（clean+noise-aware）多视图prompt设计配合非平衡最优传输（UOT）实现细粒度patch-to-prompt对齐，并用经典OT对识别出的噪声样本做选择性标签修正，在噪声小样本学习场景下持续超越SOTA。

## 背景与动机
- CLIP等视觉-语言模型通过prompt learning可高效适配下游任务，但当训练标签存在噪声时，少量错误标签会不成比例地影响梯度更新
- 现有噪声prompt学习方法存在三大局限：
  - **prompt表达力不足**：多数方法仅用1-2个prompt（如正/负对），单视角对齐无法捕捉细粒度语义线索
  - **显式负标签僵硬**：为每张图像分配一个硬负标签，固定的反类信号在噪声环境下常不准确或无信息量
  - **去噪粗糙**：依赖固定置信度阈值或无选择性的伪标签，导致错误传播
- 核心洞察：鲁棒的噪声小样本学习需要从全局匹配转向**区域感知的细粒度对齐**，自适应区分clean和noisy语义

## 核心问题
如何在VLM prompt learning中，在标签噪声严重的少样本设置下，(1) 自适应地区分干净和噪声语义信号；(2) 实现细粒度的图像区域-prompt对齐以抑制噪声区域；(3) 选择性地修正错误标签而不过度纠正。

## 方法详解

### 整体框架
NA-MVP包含两个核心模块协同工作：
1. **噪声感知对齐** (蓝色路径)：为每个类构建多个clean和noise-aware prompt，通过UOT与局部图像patch做细粒度对齐，生成clean/noisy概率
2. **选择性标签修正** (绿色路径)：基于双向对齐信号自适应识别错标样本，用经典OT修正其标签
两个模块迭代地更新训练集并优化prompt，产出去噪数据集用于鲁棒预测。

### 关键设计
1. **双向多视图Prompt构建**:
   - 每个类 $k$ 构建两组可学习prompt：clean-oriented $\{Prompt_{m,k}^c\}_{m=1}^N$ 和 noise-aware $\{Prompt_{m,k}^n\}_{m=1}^N$
   - 每个prompt由 $M$ 个可学习context token + 类别特定token组成
   - Clean prompt捕捉类别相关语义，noise-aware prompt作为自适应过滤器抑制误导信号
   - 非目标类作为隐式负样本，避免显式负标签的僵硬问题

2. **基于UOT的细粒度噪声感知对齐**:
   - 将局部图像特征 $F_i \in \mathbb{R}^{L \times d}$ 和prompt特征 $G_k \in \mathbb{R}^{N \times d}$ 视为离散分布
   - 用余弦相似度计算代价矩阵 $C_k = 1 - F_i G_k^\top$
   - UOT放松严格质量守恒约束（$T\mathbf{1}_N \leq \mu$ 而非等号），允许部分匹配
   - 通过Dykstra算法的快速实现（Sinkhorn + 熵正则化）求解
   - 核心优势：不强制所有特征都对齐，允许噪声/不相关patch被"丢弃"

3. **选择性标签修正**:
   - **噪声识别**：计算样本与clean/noise-aware prompt的UOT距离，得到相似度 $s_{i,k}^c$ 和 $s_{i,k}^n$，通过自适应阈值 $\phi_{i,k} = \frac{\exp(s_{i,k}^n/\tau)}{\exp(s_{i,k}^c/\tau) + \exp(s_{i,k}^n/\tau)}$ 判定样本是否noisy
   - **标签修正**：对识别为noisy的样本，用经典OT（严格质量守恒）计算全局图像特征与类prompt特征之间的最优传输计划 $T^*$，选 $\arg\max_j T^*_{ij}$ 作为伪标签
   - 仅修正 $p_{ik}^c < \phi_{i,k}$ 的样本，保留可信样本不变，避免过度修正

### 损失函数 / 训练策略
- **早期阶段** (前 $T_{sup}$ 个epoch)：$\mathcal{L}_{sup} = \mathcal{L}_{gce} + \lambda_i \cdot \mathcal{L}_{itbp}$
  - GCE (Generalized Cross-Entropy)：对噪声标签鲁棒的损失函数
  - ITBP Loss：辅助双向对比损失，鼓励图像特征与clean prompt对齐、远离noise-aware prompt
- **后期阶段**：激活标签修正，在去噪数据集上继续用GCE训练
- 推理时：$p(y=k|x_i) = (1 - p_{ik}^n) \cdot p_{ik}^c$，同时利用clean和noise-aware概率
- SGD优化 (lr=0.002, momentum=0.9, weight_decay=5e-4)，50 epochs，16 shared context tokens，ResNet-50 image encoder

## 实验关键数据

### 合成噪声 (5数据集均值, 16-shot)
| 数据集 | 指标 | NA-MVP | NLPrompt | 提升 |
|--------|------|--------|----------|------|
| OxfordPets (75%Sym) | Acc | **86.23** | 70.77 | +15.46 |
| OxfordPets (50%Sym) | Acc | **88.13** | 83.17 | +4.96 |
| DTD (75%Sym) | Acc | **48.63** | 39.80 | +8.83 |
| Caltech101 (75%Sym) | Acc | **89.37** | 86.70 | +2.67 |

### 真实世界噪声 (Food101N)
| 方法 | 4-shot | 8-shot | 16-shot | 32-shot |
|------|--------|--------|---------|---------|
| NLPrompt | 70.57 | 73.93 | 76.46 | 76.87 |
| **NA-MVP** | **76.10** | **76.27** | **76.90** | **77.03** |

在极少样本(4-shot)下优势尤为明显（+5.53），因为少样本时噪声标签影响更大。

### 消融实验要点
- **双向prompt**：单prompt(48.08%) → +显式负标签(48.82%) → +隐式双向(49.63%) → +多视图(51.83%)，逐步提升
- **UOT vs OT vs KL**：UOT(54.18%) > OT(53.37%) > KL(52.60%)，UOT的放松约束在噪声环境下优势明显
- **选择性修正**：全局OT修正在低噪声时会误改正确标签（25%噪声: 59.60%），选择性修正更稳定（63.13%）
- **prompt数量N**：N=4达到最佳平衡（N=1太少，N=8冗余）
- **vs DEFT**：在所有噪声水平上持续优于DEFT（75%噪声: 86.23 vs 75.87）

## 亮点
- **概念上的新视角**：将噪声鲁棒性问题重新定义为"区域感知的clean-noisy语义分解"问题，而非简单的全局匹配
- **UOT的巧妙使用**：放松质量约束天然适配噪声场景——噪声patch无需强制对齐，可被安全"丢弃"
- **两种OT的互补设计**：UOT用于局部细粒度对齐（识别噪声），经典OT用于全局标签修正（严格保质量保证分配合理性）
- 在高噪声率(75%)下的提升尤为显著，说明框架的噪声鲁棒性确实优于prior work
- 推理公式 $(1-p^n) \cdot p^c$ 简洁优雅

## 局限性 / 可改进方向
- 仅在分类任务上验证，未扩展到检测、分割等更复杂视觉任务
- 依赖CLIP的ResNet-50 backbone，未验证在更强backbone（ViT-L/14）上的表现
- UOT的Sinkhorn迭代会引入额外计算开销，论文未详细讨论计算成本
- 假设噪声标签是对称/非对称的标准模式，实际噪声分布可能更复杂（如instance-dependent noise）
- 4个多视图prompt的最优数量可能依赖数据集特性

## 与相关工作的对比
- **vs CoOp**: 基础prompt learning，完全不考虑噪声，性能随噪声率急剧下降
- **vs NLPrompt**: 使用OT-Filter做噪声识别 + 全局OT重标注，属于粗粒度全局方法；NA-MVP通过patch级UOT + 自适应阈值实现更精细的噪声处理
- **vs DEFT**: 用固定阈值(0.5)做clean判定，NA-MVP的自适应阈值 $\phi_{i,k}$ 更灵活
- **vs PLOT**: PLOT用OT做多prompt对齐但针对clean数据，NA-MVP扩展为噪声感知的双向设计
- **vs CLIPN**: CLIPN用正/负prompt对做OOD检测，NA-MVP将其思想迁移到噪声标签学习并加入多视图

## 启发与关联
- UOT在噪声环境中的"放松匹配"思想可迁移到其他噪声场景（如目标检测中的噪声标注、医学图像分割中的不精确标注）
- "隐式负样本"优于"显式负标签"的结论对NLP中的对比学习也有启发
- 双向prompt的设计思路可扩展为"多维度prompt"（如clean/noisy/ambiguous三向）
- 已有相关idea引用此论文：[Verifier Pseudo Label for Open World Detection](../../ideas/object_detection/20260316_verifier_pseudo_label_open_world.md)

## 评分
- 新颖性: ⭐⭐⭐⭐ 将UOT引入prompt-based噪声标签学习是新颖的，双向多视图设计有独到之处
- 实验充分度: ⭐⭐⭐⭐ 5个合成噪声数据集 + 1个真实噪声数据集，消融实验非常全面（组件、对齐方式、prompt数量、阈值策略）
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，Figure 1的三大局限总结到位，方法描述系统
- 价值: ⭐⭐⭐⭐ 在噪声小样本学习这一实用场景下提供了有效方案，高噪声下的大幅提升有实际意义

### 消融实验要点
- UOT 对齐比 Sinkhorn 和余弦相似度更鲁棒, 高噪声下优势尤为显著
- 双向 prompt 对齐 (视觉→文本 + 文本→视觉) 优于单向
- 噪声自适应阈值策略比固定阈值更有效

## 亮点与洞察
- 将最优传输理论引入 prompt-based 噪声标签学习是新颖的跨领域迁移
- 双向多视图对齐同时处理 prompt 噪声和标签噪声
- 在 60% 对称噪声下仍保持较高准确率, 实用性强

## 局限性 / 可改进方向
- UOT 计算开销随 prompt 数量和样本量增加
- 仅在分类任务验证, 未扩展到检测/分割等下游任务
- 噪声类型假设较为理想化 (对称/非对称), 真实噪声模式可能更复杂

## 相关工作与启发
- **vs CoOp/CoCoOp**: 标准 prompt learning 方法, 不处理噪声标签. 本文在有噪声时大幅领先
- **vs ProGrad**: 用梯度过滤噪声 prompt, 但不如 UOT 的全局最优传输对齐有效

## 与我的研究方向的关联
- 可能关联: `20260316_adaptive_model_routing.md`
- 可能关联: `20260316_cross_species_framework.md`
- 可能关联: `20260316_concept_bottleneck_world_model.md`

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐
