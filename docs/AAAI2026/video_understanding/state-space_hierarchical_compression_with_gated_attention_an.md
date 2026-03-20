# MambaMia: State-Space Hierarchical Compression for Hour-Long Video Understanding in Large Multimodal Models

**会议**: AAAI 2026  
**arXiv**: [2506.13564](https://arxiv.org/abs/2506.13564)  
**代码**: [https://github.com/naver-ai/mambamia](https://github.com/naver-ai/mambamia)  
**领域**: 多模态大模型 / 视频理解  
**关键词**: 长视频压缩, 状态空间模型, Mamba, 门控Patch聚合, 自适应帧采样  

## 一句话总结
MambaMia 提出了基于双向 Mamba 的两阶段层次化视频 Token 压缩框架：门控 Patch 聚合（GPA）做空间-时间局部压缩 + 时间轴聚合器（TAA）利用 Mamba 的自适应步长 $\Delta_t$ 做数据驱动的关键帧采样，将小时级视频压缩到仅 4.7K Token，在 LVBench 上达到 44.6 分超越 Qwen2-VL 和 mPLUG-Owl3。

## 研究背景与动机
1. **领域现状**：大型多模态模型（LMM）在图像和短视频理解上表现出色，但处理小时级长视频时面临严重的 Token 爆炸问题——数百帧的视频会产生数十万个 Token，远超标准模型和硬件的容量。
2. **现有痛点**：(1) 逐帧空间池化/Token 剪枝只解决单帧冗余，无法解决帧间时序累积问题；(2) 基于查询的选择方法依赖特定任务，牺牲了通用上下文建模能力；(3) 暴力扩展上下文窗口需要巨大计算资源，不适合学术/生产环境。
3. **核心矛盾**：长视频包含两类冗余——帧内空间冗余（大量相似 patch）和帧间时序冗余（连续帧内容高度相似），但同时也包含需要保留的细粒度关键事件。需要一个既能大幅压缩又不丹失关键信息的通用方案。
4. **本文要解决什么？** 如何在标准硬件上高效压缩小时级视频的视觉 Token，同时保持理解性能？
5. **切入角度**：利用状态空间模型（Mamba）的线性复杂度处理超长序列，并巧妙复用 Mamba 内部的自适应步长 $\Delta_t$ 作为帧重要性信号做自适应采样。
6. **核心 idea 一句话**：用双向 Mamba + 门控聚合做空间压缩，再复用 Mamba 步长做自适应时间帧筛选，实现层次化长视频压缩。

## 方法详解

### 整体框架
两阶段压缩流水线：输入 384 帧视频 → 视觉编码器提取每帧 576 个 patch Token（共约 221K Token）→ **Stage 1: 时空压缩层（GPA）** 将每帧压缩为 24 个 anchor Token（共约 9.2K）→ **Stage 2: 时间轴聚合器（TAA）** 进一步通过 delta 采样压缩到约 4.7K Token → 送入 LLM。

### 关键设计

1. **门控 Patch 聚合（GPA）**:
   - 做什么：在双向 Mamba 处理后的序列中，利用插入的可学习 query anchor 聚合周围 patch 信息
   - 核心思路：每行 24 个 patch 对应一个 query anchor。GPA 用 query-conditioned 加权池化聚合相邻 patch：$\boldsymbol{\alpha} = \text{softmax}(\mathbf{W}_\alpha \mathbf{q} + \mathbf{b}_\alpha)$，$\mathbf{a} = \sum_i \alpha_i \mathbf{x}_i$。然后用门控机制自适应混合：$\mathbf{f} = (1-g)\mathbf{q} + g \cdot \mathbf{a}$，其中 $g = \sigma(\mathbf{W}_g \mathbf{q} + b_g)$
   - 设计动机：$g \approx 0$ 时保留 query 自身信息（状态空间上下文），$g \approx 1$ 时吸收局部 patch 信息。比 BIMBA 使用的 3D 平均池化更灵活，消融显示 GPA 带来约 7% 的平均提升

2. **时间轴聚合器（TAA）+ Delta 采样**:
   - 做什么：沿时间轴建模帧间依赖，并利用 Mamba 自适应步长 $\Delta_t$ 做数据驱动的关键帧筛选
   - 核心思路：单向 Mamba 处理帧级 anchor 序列，其内部 $\Delta_t = \text{softplus}(\mathbf{W}_\Delta \mathbf{f}_t + \mathbf{b}_\Delta)$ 是端到端学习的。将 $\Delta_t$ 解释为帧重要性分数——$\Delta_t$ 大的帧是模型认为更重要的帧。用累积 delta 采样算法：累积 $\Delta_t$，超过阈值 $\delta_{\text{thresh}}$ 时选中该帧并重置累积器。默认保留约 50% 帧（384→192）
   - 设计动机：巧妙复用 SSM 内部的 $\Delta_t$ 而非额外训练选择器——这个步长本来就反映了输入的信息量（大 $\Delta_t$ = 更大状态更新 = 更多新信息）。可视化显示 $\Delta_t$ 峰值与场景切换/关键事件对齐

3. **双向 Mamba 时空压缩器**:
   - 做什么：在 GPA 之前处理整个时空 Token 序列，共享空间和时间信息
   - 核心思路：3 层双向 Mamba2 块处理约 230K Token 的序列。双向设计使每个 Token 能看到前后上下文
   - 设计动机：用单向 Mamba 替换双向会下降约 1.7 分，说明双向建模对时空特征的共享很重要

### 训练策略
三阶段 LLaVA 风格训练：图像理解 → 模块对齐（仅训练压缩层）→ 视频指令微调（解冻 LLM）。训练 128 帧，推理 384 帧。Delta 采样仅在推理时使用。压缩模块约 247M 参数。

## 实验关键数据

### 主实验 — 长视频 Benchmark

| 模型 | LLM | Max Token | LVBench | MLVU | VideoMME | VNBench |
|------|-----|-----------|---------|------|----------|---------|
| Qwen2-VL | Qwen2-7B | - | 42.0 | 64.2 | 55.6 | 33.9 |
| LLaVA-Video | Qwen2-7B | 12.5K | 43.8 | 70.8 | 63.3 | 37.0 |
| mPLUG-Owl3 | Qwen2-7B | - | 43.5 | - | 53.5 | - |
| **MambaMia** | Qwen2-7B | **4.7K** | **44.6** | 68.0 | 58.3 | 41.5 |

### 消融实验 — 压缩模块设计

| 配置 | GPA | TAA | LVBench | MLVU | MME | Avg |
|------|-----|-----|---------|------|-----|-----|
| BIMBA (3D pool) | ✗ | ✗ | 35.3 | 53.8 | 47.3 | 45.4 |
| +GPA | ✓ | ✗ | 41.1 | 62.4 | 53.2 | 52.2 |
| +GPA+TAA (Full) | ✓ | ✓ | 41.1 | 64.0 | 55.7 | **53.6** |

### 关键发现
- 仅 4.7K Token 即可达到与使用 12.5K Token 的 LLaVA-Video 可比的性能，Token 效率提升约 2.6 倍
- GPA 替换 3D 平均池化带来约 7 分的平均提升——可学习的门控聚合远好于固定池化
- Delta 采样比均匀采样在 LVBench 上高 1.2 分（44.6 vs 43.4），统计显著（p=0.047）
- 即使使用 Mamba 作为 LLM 骨干也需要专门的压缩——vanilla Mamba LLM 性能远低于有压缩的版本
- 性能在 384 帧时饱和，更多帧无额外收益
- VNBench（needle-in-a-video-haystack）上表现优异（41.5），说明压缩不会丢失关键细粒度信息

## 亮点与洞察
- **复用 Mamba $\Delta_t$ 作为帧重要性信号**：最巧妙的设计——SSM 步长本身就编码了输入信息量，无需额外的重要性预测器。这个思路可以推广到任何使用 SSM 的序列处理场景
- **LLM 前的模块化压缩**：与 VAMBA（在 LLM 内压缩）不同，MambaMia 在 LLM 之前独立压缩，保持了模块化和轻量性
- **严谨的实验方法论**：强调 from-scratch 训练、控制变量对比、多随机种子验证、统计检验——实验设计值得学习

## 局限性 / 可改进方向
- 384 帧时性能饱和，说明压缩层可能存在信息瓶颈——更多帧但更好的压缩策略值得探索
- 当前 GPA 的 query conditioned pooling 只看 query token，没有 content-aware attention（为了效率），可能丢失 patch 间的关系
- $\delta_{\text{thresh}}$ 是手动设定的，可以改为自适应阈值
- 仅测试了 7B 模型，对更大规模 LLM 的效果待验证
- 训练时用均匀采样、推理用 delta 采样，这种 train-test mismatch 可能限制性能

## 相关工作与启发
- **vs BIMBA**: 同为 Mamba+周期查询的架构，但 BIMBA 用 3D 平均池化、MambaMia 用可学习门控聚合+delta 采样，消融显示 MambaMia 全面优于 BIMBA
- **vs LLaVA-Video**: LLaVA-Video 用 12.5K Token，MambaMia 用 4.7K Token 达到可比性能——效率优势明显
- **vs Video-XL**: Video-XL 在 LLM 内部做聚合 + CLIP 选帧，MambaMia 在 LLM 外独立压缩 + 学习选帧，架构更模块化
- 启发：TTF-VLA 论文中的帧间时序融合思路和 MambaMia 的时间压缩是互补的——可以先 TTF 增强再 MambaMia 压缩

## 评分
- 新颖性: ⭐⭐⭐⭐ 复用 $\Delta_t$ 做帧采样很巧妙，GPA 设计也有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 7 个 benchmark、5 种压缩对比、多随机种子统计检验、成本分析，极其详尽
- 写作质量: ⭐⭐⭐⭐⭐ 方法描述精确，附录 13 节覆盖所有细节，可复现性极强
- 价值: ⭐⭐⭐⭐⭐ 4.7K Token 处理小时级视频对社区有重大实用价值
