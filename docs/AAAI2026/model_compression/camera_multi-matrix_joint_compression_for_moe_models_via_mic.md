# CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis

**会议**: AAAI 2026  
**arXiv**: [2508.02322](https://arxiv.org/abs/2508.02322)  
**代码**: [https://github.com/xuyuzhuang11/CAMERA](https://github.com/xuyuzhuang11/CAMERA)  
**领域**: 模型压缩  
**关键词**: MoE压缩, 微专家, 结构化剪枝, 混合精度量化, training-free  

## 一句话总结
提出"micro-expert"概念将MoE层的输出分解为跨矩阵（up/gate/down_proj）的微专家线性组合，基于能量排序进行结构化剪枝(Camera-P)和混合精度量化(Camera-Q)，在Deepseek-MoE-16B/Qwen2-57B/Qwen3-30B上20%-60%剪枝率全面超越NAEE和D²-MoE，且分析Qwen2-57B仅需单卡A100不到5分钟。

## 背景与动机
MoE架构通过稀疏激活实现高效扩展，但参数量增加并未带来等比例的性能提升，存在显著的结构冗余。现有MoE压缩方法有两大局限：(1) 整expert剪枝/合并粒度太粗，信息损失大或假设过强（假设expert间功能相似）；(2) 部分expert剪枝在矩阵级独立操作，忽视了up_proj、gate_proj和down_proj三个矩阵间的功能依赖关系。

## 核心问题
如何在保持功能完整性的前提下，找到更细粒度且跨矩阵协调的压缩单元来高效压缩MoE模型？核心挑战是：精确评估每个压缩单元的重要性是NP-hard问题（Column Subset Selection Problem），且现代MoE模型的微专家数量达$10^5$级别。

## 方法详解

### 整体框架
将MoE层分解为微专家的线性组合：MoE输出$\mathbf{y} = \sum_{i}^{N_e} \phi_i \mathbf{w}_i^{down}$，其中$\phi_i$是标量组合系数，$\mathbf{w}_i^{down}$是basis vector。基于此分解，Camera算法高效估计每个微专家的"解码时能量"进行排序，Camera-P用于剪枝，Camera-Q用于混合精度量化。

### 关键设计
1. **Micro-Expert定义**: 每个expert的第i个微专家由三个向量联合定义：$\mathbf{w}_i^{up}$（up_proj第i行）、$\mathbf{w}_i^{gate}$（gate_proj第i行）、$\mathbf{w}_i^{down}$（down_proj第i列）。MoE输出是所有微专家输出的线性组合，组合系数$\phi_i = A_i \cdot \sigma(\mathbf{w}_i^{gate}\mathbf{x}) \cdot \mathbf{w}_i^{up}\mathbf{x}$为标量。这个分解揭示了MoE层的本质结构。

2. **Camera能量排序算法**: 定义微专家能量$\mathcal{E}_i = [(1-\alpha)\|\mathbf{\Phi}_{:,i}\|_2^2 + \alpha\|\mathbf{\Phi}_{:,i}\|_\infty^2] \cdot \|\mathbf{w}_i\|_2^2$，同时考虑activation系数的L2范数（整体贡献）和L∞范数（最大贡献）以及basis vector的范数。理论保证：基于能量排序的剪枝误差与最优SVD近似之间仅差$O(k)$-delta。

3. **Camera-P结构化剪枝**: 按能量排序后，将低能量微专家的三个关联向量同时置零，保持跨矩阵的功能完整性。逐层处理，每层先收集校准样本、排序微专家、剪枝后重新计算输出传递到下一层。

4. **Camera-Q混合精度量化**: 将微专家按能量分为三组，分别分配不同bit-width（如3/2/1 bit）。关键是保证同一微专家的三个参数使用相同精度，不同于传统方法按单矩阵的input dimension切分。

### 损失函数 / 训练策略
完全training-free和gradient-free，仅需128条2048长度的校准序列（Wikitext2），逐层进行。Camera-P对Qwen2-57B分析仅需<5分钟/单卡A100。

## 实验关键数据

| 模型 | 剪枝率 | Camera-P Avg | NAEE Avg | D²-MoE Avg |
|--------|------|------|----------|------|
| Deepseek-MoE-16B | 20% | **61.03** | 60.51 | 58.97 |
| Deepseek-MoE-16B | 40% | **58.58** | 54.94 | 54.32 |
| Deepseek-MoE-16B | 60% | **51.62** | 45.28 | 46.72 |
| Qwen2-57B-A14B | 20% | **67.28** | 66.11 | 66.38 |
| Qwen2-57B-A14B | 40% | **66.81** | 63.92 | 64.40 |
| Qwen2-57B-A14B | 60% | **65.17** | 51.40 | 56.32 |
| Qwen3-30B-A3B | 20% | **69.94** | 69.64 | 66.35 |

Camera-Q（2.25-bit平均）在Deepseek-MoE-16B上均分56.56 vs GPTQ 53.45 vs MC 54.45。

### 消融实验要点
- 微专家能量分布高度不均匀，证实了按能量剪枝的有效性
- 匹配方式对比：Camera-Q（跨矩阵一致精度）56.56 vs Camera-Q†（单矩阵切分）52.69，证明跨矩阵功能完整性至关重要
- α参数（L2 vs L∞权重）对perplexity和平均准确率影响小，但对特定任务有影响
- 校准数据量和来源不敏感（128~512样本，Wiki2 vs C4），鲁棒性好
- Camera-P直接减少权重数，20%剪枝实现1.03-1.06x解码加速，40%达1.04-1.42x

## 亮点
- **"微专家"概念非常fundamental**——将MoE层分解为basis vector的线性组合，为理解MoE内部工作机制提供了全新视角
- **跨矩阵联合压缩保持功能完整性**——与传统逐矩阵压缩形成鲜明对比，Camera-Q vs Camera-Q†的对比清晰证明了这一点
- **效率惊人**：单卡A100分析57B模型<5分钟，比现有方法快100x+，真正实用可落地
- 理论有保证：pruning误差与最优SVD的差距可控($O(k)$-delta)
- 可扩展到dense模型的FFN剪枝，且与Wanda等单矩阵方法互补

## 局限性 / 可改进方向
- 在expert数较少的older MoE模型（Mixtral-8x7B、Phi3.5-MoE）上优势不明显
- 能量排序是静态的，未考虑不同输入样本可能需要不同的微专家组合
- Camera-P剪枝后不做fine-tuning恢复，高剪枝率下可能有空间
- 未与LoRA等参数高效微调方法结合
- 混合精度量化部分依赖GPTQ，可以探索与更先进量化方法（SpinQuant等）的结合

## 与相关工作的对比
- **vs NAEE**: NAEE做整expert搜索（brute-force组合），不 scale到多expert模型，Camera在micro-expert级别操作更精细且快100x+
- **vs D²-MoE**: D²-MoE先合并expert再低秩分解，假设expert间可合并且存在数值不稳定问题，Camera直接识别重要微专家保留全精度
- **vs MC量化**: MC在expert级别分配bit-width太粗糙，Camera-Q在micro-expert级别分配更精细

## 启发与关联
- 微专家视角可以扩展到分析VLM中的视觉expert冗余——与EM-KD中的token压缩互补
- 能量排序的思路可以用于动态推理：按输入动态选择高能量微专家子集
- 与 `ideas/model_compression/20260316_adaptive_model_routing.md` 相关——微专家粒度的动态路由

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 微专家概念是genuinely original的贡献，为MoE理解提供了新视角
- 实验充分度: ⭐⭐⭐⭐⭐ 3+2个MoE模型、20-60%剪枝率、9个zero-shot任务、量化实验、丰富的消融
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，motivation清晰，实验分析深入
- 价值: ⭐⭐⭐⭐⭐ 对MoE压缩有重大实用价值，方法简洁高效可落地
