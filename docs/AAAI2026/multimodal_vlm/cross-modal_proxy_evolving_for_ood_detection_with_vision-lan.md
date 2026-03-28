# Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models

**会议**: AAAI 2026  
**arXiv**: [2601.08476](https://arxiv.org/abs/2601.08476)  
**代码**: 未公开  
**领域**: 多模态VLM / OOD检测  
**关键词**: OOD检测, VLM, 跨模态代理演化, zero-shot, test-time adaptation, CLIP, 负标签  

## 一句话总结

提出 CoEvo，一个 training-free 和 annotation-free 的 test-time 框架，通过双向 sample-conditioned 的文本/视觉 proxy 协同演化机制动态更新正负代理缓存，在 ImageNet-1K 上比最强负标签基线 AUROC 提升 1.33%、FPR95 降低 45.98%（从 18.92% 降至 10.22%），实现 SOTA 的 zero-shot OOD 检测。

## 背景与动机

VLM（如 CLIP）在开放世界部署时需要可靠的 OOD 检测来拒绝未见过的类别。Zero-shot OOD 检测不需要标注负样本，当前主流的"负标签"方法从 WordNet 等词库中选取与 ID 类语义不相关的文本标签作为 OOD proxy，通过比较测试图像与正/负标签的相似度来判断 ID/OOD。

然而，静态负标签设计存在两个根本缺陷：

1. **未建模的负空间（Unmodeled Negative Space）**：全局固定的负标签集只能稀疏采样 ID 类别之外的广阔语义空间，很多与当前测试样本相关的负语义未被覆盖
2. **跨模态不对齐（Modality Misalignment）**：test-time 分布偏移下视觉特征发生漂移，但文本负标签保持不变，导致跨模态相似度几何结构扭曲，判决阈值不稳定

AdaNeg 部分解决了第一个问题（用视觉 proxy 适配），但文本负标签仍然是静态的——适配是单向的。

## 核心问题

如何在不训练、不标注的前提下，实现文本和视觉双模态 proxy 的双向、sample-conditioned 动态适配，以应对 test-time 分布偏移下的 zero-shot OOD 检测？

## 方法详解

### 整体框架

CoEvo 在 test-time 维护两个模态特定的 proxy 缓存（文本缓存 + 视觉缓存），每个缓存包含正/负队列。核心是 **Proxy-Aligned Co-Evolution 机制**：视觉线索引导文本负标签的动态挖掘，更新后的文本 proxy 反过来精化视觉决策边界，形成闭环。

### 关键设计

1. **文本 Proxy 缓存**：
   - **正 proxy 队列** $\mathbf{T}_p$：ID 类名的 CLIP 文本编码，固定不变以保持 ID 语义锚点
   - **负 proxy 队列** $\mathbf{T}_n$：初始化为从大规模词库（WordNet/CSP）中选取的 M 个负类编码，在推理过程中动态演化

2. **视觉 Proxy 缓存**：
   - **正 proxy 队列** $\mathbf{V}_p \in \mathbb{R}^{K \times L \times D}$：每个 ID 类存储 L 个视觉实例，初始化为对应文本编码，高置信 ID 样本逐步入队
   - **负 proxy 队列** $\mathbf{V}_n$：高置信 OOD 样本入队，优先队列策略淘汰低置信/过时样本

3. **Proxy-Aligned Co-Evolution 机制**（核心贡献）：
   - **文本 proxy 演化**：通过置信度 margin 门控（基于自适应阈值 $\delta$ 和余量 $\gamma$），对高置信样本：
     - 预测为 OOD → 检索语义上接近测试样本的文本负标签（Near Negatives），收紧局部开集边界
     - 预测为 ID → 检索语义上远离测试样本的文本负标签（Far Negatives），扩大负空间覆盖
     - 去重约束避免重复入队
   - **视觉 proxy 演化**：文本 proxy 更新后扩展视觉负 proxy 队列以容纳新暴露的 OOD 语义；基于熵（entropy）的置信度度量决定是否替换现有 proxy，确保队列始终存储最高置信样本

4. **OOD 分数演化**：
   - **Pre-evolution 分数**（用于 proxy 更新）：$\mathcal{S}^{\text{pre}} = \lambda \mathcal{S}_T^{\text{pre}} + (1-\lambda) \mathcal{S}_V^{\text{pre}}$，文本权重更高（$\lambda=0.8$），因为冷启动时视觉 proxy 稀疏不可靠
   - **Post-evolution 分数**（用于最终判决）：$\mathcal{S}^{\text{post}} = (1-\lambda) \mathcal{S}_T^{\text{post}} + \lambda \mathcal{S}_V^{\text{post}}$，权重翻转——演化后视觉 proxy 因累积实例级信息而更具判别力
   - 这种"权重翻转"策略是关键设计：冷启动信任语义先验，充分演化后信任视觉证据

5. **自适应阈值**：采用数据驱动的阈值估计（最小化 ID/OOD 分数的类内方差），比固定阈值更鲁棒。

### 训练策略

完全 Training-free 和 Annotation-free。不修改 CLIP 骨干参数，仅在 test-time 在线演化 proxy 缓存。

## 实验关键数据

### ImageNet-1K 主实验（CLIP ViT-B/16）

| 方法 | iNaturalist FPR95↓ | SUN FPR95↓ | Places FPR95↓ | Textures FPR95↓ | 平均 FPR95↓ | 平均 AUROC↑ |
|------|-------------------|------------|--------------|----------------|------------|------------|
| NegLabel | 1.91 | 20.53 | 35.59 | 43.56 | 25.40 | 94.21 |
| CSP | 1.54 | 13.66 | 29.32 | 25.52 | 17.51 | 95.76 |
| AdaNeg | 0.59 | 9.50 | 34.34 | 31.27 | 18.92 | 96.66 |
| **CoEvo-NegLabel** | **0.53** | **4.42** | **23.51** | **12.42** | **10.22** | **97.95** |
| **CoEvo-CSP** | 0.46 | 4.68 | 25.83 | 12.78 | 10.94 | 97.85 |

- FPR95 从 18.92%（AdaNeg）降至 10.22%，相对降低 45.98%
- 在 Textures 数据集上改进最大（31.27% → 12.42%），该数据集分布偏移最严重

### OpenOOD 基准（Near-OOD + Far-OOD）

| 方法 | Near-OOD FPR95↓ | Far-OOD FPR95↓ | ID ACC↑ |
|------|----------------|----------------|---------|
| AdaNeg | 67.51 | 17.31 | 67.13 |
| CoEvo-NegLabel | 64.64 | 15.24 | 66.83 |
| CoEvo-CSP | 66.88 | **14.47** | **67.36** |

- Far-OOD 场景提升明显，Near-OOD 改善较温和

### 消融实验

| Proxy 演化配置 | 平均 FPR95↓ | 平均 AUROC↑ |
|---------------|------------|------------|
| 无演化（NegLabel） | 24.97 | 94.56 |
| 仅文本演化 | 21.77 | 95.38 |
| 仅视觉演化 | 17.41 | 96.99 |
| **双向协同演化** | **10.22** | **97.95** |

- 文本和视觉演化都有独立贡献，双向组合产生超加性效果
- 视觉演化的单独贡献（-7.56 FPR95）大于文本演化（-3.20 FPR95）

### 数据不平衡鲁棒性（FPR95↓，ImageNet vs SUN）

| ID:OOD 比例 | NegLabel | AdaNeg | CoEvo-NegLabel |
|------------|---------|--------|---------------|
| 1:100 | 23.00 | 28.00 | **17.00** |
| 1:1 | 21.55 | 8.01 | **5.27** |
| 100:1 | 19.69 | 17.40 | **14.77** |

- 在所有比例下均保持最优，极端不平衡（100:1，仅 100 个 OOD）下仍有效

## 亮点

- **双向 co-evolution 是核心创新**：文本引导视觉、视觉反馈文本的闭环设计，解决了现有方法单向适配的根本局限
- **Pre/Post 权重翻转策略巧妙**：冷启动信任文本先验、充分演化后信任视觉证据，自适应调节双模态贡献
- **完全 training/annotation-free**：即插即用任何 CLIP 模型，无需额外训练或标注OOD数据
- **FPR95 降低 45.98% 是非常显著的改进**——对实际部署的安全性有直接影响

## 局限性 / 可改进方向

- **Near-OOD 改善有限**：在 SSB-hard 等 fine-grained OOD 场景下提升不如 Far-OOD 明显
- **Test-time 计算开销**：每个样本需要检索 Top-N 文本候选 + 更新双模态缓存，推理延迟高于静态方法
- **依赖初始负标签质量**：如果 WordNet/CSP 初始负标签覆盖不足，演化起点较差
- **顺序敏感性**：online 演化的结果可能受测试样本顺序影响（文中未充分讨论）
- **仅在 CLIP ViT-B/16 上验证**：更大规模 VLM（如 ViT-L、SigLIP）的效果未知

## 与相关工作的对比

- **vs NegLabel/CSP（静态负标签）**：CoEvo 通过动态演化 proxy 解决静态方法的稀疏性和不对齐问题，FPR95 降低约 60%
- **vs AdaNeg（单向适配）**：AdaNeg 仅适配视觉 proxy 而文本保持不变；CoEvo 实现双向适配，额外收益显著（FPR95 从 18.92% 降至 10.22%）
- **vs MCM（最大 softmax）**：MCM 不使用负标签，仅基于 ID 类相似度判断，性能远低于负标签方法
- **vs 训练型方法（LoCoOp/LAPT）**：CoEvo 作为 training-free 方法超越了需要 prompt tuning 的训练型方法

## 启发与关联

- Co-evolution 的思路可推广到 VLM 的其他 test-time adaptation 场景（域适应、continual learning、开集识别）
- "冷启动信任先验、演化后信任数据"的权重翻转策略是一个通用的 online learning 设计模式
- Proxy 缓存 + 熵筛选的在线更新机制可借鉴到检索增强系统中

## 评分

- 新颖性: ⭐⭐⭐⭐ 双向 proxy co-evolution 机制新颖，权重翻转设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ ImageNet-1K + OpenOOD + 消融 + 超参敏感度 + 不平衡分析，非常完整
- 写作质量: ⭐⭐⭐⭐ 问题分析深入，公式推导清晰，算法伪代码完整
- 价值: ⭐⭐⭐⭐ 对 VLM 安全部署有直接价值，training-free 特性利于实际应用
