# Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models

**会议**: AAAI 2026  
**arXiv**: [2511.15311v2](https://arxiv.org/abs/2511.15311v2)  
**代码**: [有](https://github.com/Mehran-TAM/Uni-Adapter)  
**领域**: 3D视觉 / 测试时适应 / 视觉-语言基础模型  
**关键词**: 测试时适应, 3D点云, 视觉-语言基础模型, 动态原型学习, 无训练适应  

## 一句话总结

提出 Uni-Adapter，一种面向3D视觉-语言基础模型(VLFM)的无训练在线测试时适应框架，通过基于聚类的动态原型缓存和图正则化标签平滑来应对分布偏移，在多个3D损坏基准上取得SOTA。

## 背景与动机

3D视觉-语言基础模型（如Uni3D、ULIP-2、OpenShape）通过对齐点云-图像-文本三模态表征，在开放世界点云识别上展现了强大的零样本能力。但在实际部署中，采集到的点云常常受传感器噪声、稀疏性、分辨率低等因素影响，导致与训练分布产生偏移，模型性能显著下降。

现有测试时适应(TTA)方法分两类：(1) 基于训练的方法（如TPT）需要在推理时反向传播优化prompt/参数，计算开销大；(2) 基于缓存的无训练方法（如TDA、Point-Cache）仅缓存高置信样本作为原型，但这种策略在3D数据中存在固有缺陷——同一语义类别的点云特征往往形成多个不同的聚类簇（如"飞机"类在t-SNE中形成多个清晰的子群），高置信原型只能覆盖部分模式，导致决策边界不准确。

## 核心问题

如何在不训练、不微调的前提下，让3D VLFM在推理阶段自动适应含噪声、不完整或域外的点云数据，同时全面捕获类内分布的多模式特性？

## 方法详解

### 整体框架

Uni-Adapter由三个核心模块串联组成：
1. **在线原型模块(Online Prototyping)**：对每个类别维护多个聚类中心作为原型，通过在线聚类持续更新
2. **原型重分配模块(Prototype Reassignment)**：构建原型间的相似度图，通过图正则化平滑来修正噪声伪标签
3. **基于熵的融合(Entropy-Based Fusion)**：将缓存logit与原始VLFM logit按熵加权融合，得到最终预测

整个流程完全无需反向传播，仅在推理时前向计算+缓存更新。

### 关键设计

**1. 基于聚类的缓存策略**

与Point-Cache仅缓存高置信样本不同，Uni-Adapter对每个类别维护最多 $N$ 个聚类中心。对于新输入点云 $\mathbf{X}_t$，编码为特征 $\mathbf{f}_t$，先通过与文本嵌入的余弦相似度预测伪类别 $k$，然后在该类别的已有原型中找到最相似的一个进行更新：

$$\mathbf{c}_{k,n}^{\text{new}} = \frac{\alpha_t \mathbf{f}_t + b_{k,n} \alpha_{k,n} \mathbf{c}_{k,n}^{\text{old}}}{\alpha_t + b_{k,n} \alpha_{k,n}}$$

其中 $\alpha_t = \exp(-\beta \cdot H_t)$ 是基于预测熵的置信权重，$b_{k,n}$ 是该原型累积的样本计数。若该类别原型未满则直接初始化新原型。这种设计确保每个聚类中心能代表类内的一个分布模式，而非仅反映高置信区域。

**2. 图正则化标签平滑**

在线原型难免受噪声伪标签污染。该模块将所有原型的特征矩阵 $\mathbf{U} \in \mathbb{R}^{M \times d}$ 构建余弦相似度矩阵 $\mathbf{A} = \mathbf{U}\mathbf{U}^\top$，以阈值 $\gamma$ 稀疏化后计算归一化图拉普拉斯 $\mathbf{L}_{\text{norm}}$，然后求解优化问题：

$$\mathbf{Z}^* = \arg\min_{\mathbf{Z}} \|\mathbf{Z} - \mathbf{Z}^{(0)}\|_F^2 + \lambda_{\text{reg}} \cdot \text{Tr}(\mathbf{Z}^\top \mathbf{L}_{\text{norm}} \mathbf{Z})$$

闭式解为 $\mathbf{Z}^* = (\mathbf{I} + \lambda_{\text{reg}} \mathbf{L}_{\text{norm}})^{-1} \mathbf{Z}^{(0)}$，使用共轭梯度法高效求解（复杂度从 $O(M^3)$ 降至 $O(\rho \cdot \text{nnz}(\mathbf{L}_{\text{norm}}))$）。直观上，相似的原型会互相"拉"向一致的标签分配，从而修正错误伪标签。

**3. 缓存logit计算与熵融合**

缓存logit按类别归一化原型数量后计算：$\mathbf{s}_{\text{cache}} = \mathbf{\Lambda} \mathbf{Z}^{*\top} (\mathbf{U} \mathbf{f}_t)$，其中 $\mathbf{\Lambda}$ 是按每类原型数归一化的对角矩阵。最终预测通过交叉熵加权融合：

$$\mathbf{s}_{\text{final}} = \frac{H_{\text{cache}} \cdot \mathbf{s}_{\text{main}} + H_t \cdot \mathbf{s}_{\text{cache}}}{H_{\text{cache}} + H_t}$$

熵高（不确定性大）的一方权重反而给对方更大的话语权，实现自适应的置信度融合。

### 损失函数 / 训练策略

本方法**完全无需训练**，不涉及任何损失函数或梯度计算。所有适应操作都在推理的前向过程中完成，包括：
- 在线聚类更新（移动平均）
- 图正则化求解（共轭梯度法，最多100次迭代）
- 逐样本适应（batch size = 1）

关键超参数：聚类中心数 $N=30$，稀疏阈值 $\gamma=0.5$，置信衰减 $\beta=10$，标签平滑系数 $\lambda_{\text{reg}}=0.3$。

## 实验关键数据

| 数据集 | Source-Only | Point-Cache (CVPR25) | **Uni-Adapter** | 提升 |
|---|---|---|---|---|
| ModelNet-40C | 59.15% | 66.73% | **69.70%** | +10.55% |
| ScanObjectNN-C | 38.07% | 42.13% | **46.33%** | +8.26% |
| ShapeNet-C | 57.92% | 57.70% | **62.41%** | +4.49% |

在Uni3D-Large上评估。在干净数据集上同样有提升：ModelNet40 83.96%、ScanObjectNN 64.03%、ShapeNet 81.23%。

吞吐量对比（Uni3D，test/s）：Zero-shot 39.19 → Uni-Adapter 36.93（仅损失~6%），而Point-Cache仅9.73（损失75%）。

跨模型验证：在ULIP-2和OpenShape上同样有效，ModelNet-40C上分别提升+7.97%和+4.64%。

### 消融实验要点

1. **组件贡献**：Online Prototyping贡献主要提升（59.15→68.48），Prototype Reassignment再加1.22%（→69.70）
2. **聚类 vs 置信缓存**：在ShapeNet-C所有corruption类型上，基于聚类的缓存一致优于基于置信度的缓存
3. **聚类中心数N**：N=30最优，太少无法覆盖类内分布，太多引入噪声
4. **标签平滑$\lambda_{\text{reg}}$**：0.3最优，接近0时平滑效果消失、接近1时过度平滑
5. **共轭梯度 vs 直接求逆**：共轭梯度更快（27.07ms vs 29.20ms），MAE<0.0005%
6. **统计显著性**：所有对比的p值远低于0.05，最强对比Point-Cache在ModelNet-40C上p=8.04×10⁻⁷

## 亮点

1. **真正的无训练适应**：不需要反向传播、不修改模型参数、不需标注数据，batch size=1即可工作
2. **聚类缓存设计精巧**：解决了高置信缓存的模式覆盖不足问题，用在线聚类捕获类内多模式分布
3. **图正则化标签平滑**：利用原型间的拓扑关系修正伪标签，比简单的置信度过滤更优雅
4. **计算效率突出**：吞吐量接近零样本推理（仅降6%），远优于Point-Cache（降75%），且内存开销可忽略
5. **模型无关性**：在Uni3D、ULIP-2、OpenShape三种3D VLFM上均有效
6. **实验全面**：覆盖损坏数据集（15种损坏×5级别）、干净数据集、大规模数据集（1156类），并做了统计显著性检验

## 局限性 / 可改进方向

1. **冷启动不稳定**：缓存初始化阶段（原型尚未充分积累时），在严重噪声输入下性能不稳定——作者也承认了这一点
2. **伪标签累积偏差**：虽然图平滑能修正部分错误，但基于argmax的伪标签生成本身在域偏移极大时可能持续错误
3. **固定聚类数N**：所有类别共享同一个最大聚类数，但不同类可能有不同的分布复杂度
4. **仅评估分类任务**：3D点云的分割、检测等下游任务未涉及
5. **未考虑连续域漂移**：实验中假设corruption类型固定，未测试域随时间连续变化的场景
6. **未来方向**：作者提出可引入轻量自监督训练（对比损失或原型一致性目标）改善早期适应稳定性

## 与相关工作的对比

| 方法 | 类型 | 是否无训练 | 3D专用 | VLFM专用 | ModelNet-40C |
|---|---|---|---|---|---|
| TENT | 训练型TTA | ❌ | ❌ | ❌ | 59.48 |
| T3A | 无训练TTA | ✅ | ❌ | ❌ | 64.12 |
| TPT | 训练型TTA | ❌ | ❌ | ✅ | 61.02 |
| TDA | 无训练TTA | ✅ | ❌ | ✅ | 63.63 |
| CloudFixer | 输入适应 | ✅ | ✅ | ❌ | 56.09 |
| Point-Cache | 无训练TTA | ✅ | ✅ | ✅ | 66.73 |
| **Uni-Adapter** | 无训练TTA | ✅ | ✅ | ✅ | **69.70** |

关键区别：Point-Cache用高置信缓存+k-means局部特征，Uni-Adapter用在线聚类全局原型+图平滑。Uni-Adapter统一了一个cache结构就完成了Point-Cache双cache（全局+局部）的功能，且吞吐量远优。

## 启发与关联

1. **与ideas/model_compression/20260316_foundation_model_tta.md的关联**：该idea探讨视觉基础模型的TTA，Uni-Adapter验证了无训练cache-based TTA在3D领域的有效性。值得注意的是Uni-Adapter不需要微调任何参数（连LoRA都不用），纯靠缓存机制就能取得显著提升——这对我们探索更广泛基础模型TTA有参考意义
2. **聚类原型 + 图正则化的组合**可能迁移到2D VLFM的TTA场景（如CLIP在domain shift下的适应）
3. **在线聚类策略**的思路可以推广到其他streaming inference场景，如持续学习中的类别原型维护
4. **熵交叉加权融合**是一个简洁有效的多来源预测融合策略，可以用于其他多模型集成场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 聚类缓存替代置信缓存+图平滑修正是有意义的创新，但整体框架仍是cache-based TTA的改进
- 实验充分度: ⭐⭐⭐⭐⭐ 3个损坏数据集+3个干净数据集+2个大规模数据集，3种3D VLFM，统计显著性检验，消融全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分，公式推导完整
- 价值: ⭐⭐⭐⭐ 3D VLFM的无训练TTA是实用且及时的课题，方法高效实用

## 实验关键数据
- Without retraining, Uni-Adapter 有效地 mitigates distribution shifts, achieving 最先进 performance on diverse 3D benchmarks over different 3D VLFMs, improving ModelNet-40C by 10.55%, ScanObjectNN-C by 8.26%, and ShapeNet-C by 4.49% over the source 3D VLFMs.

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 被AAAI 2026接收
- 待深读后补充

## 局限性 / 可改进方向
- 待深读后补充

## 与相关工作的对比
待深读后补充。

## 与我的研究方向的关联
- 待分析

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 对我的价值: ⭐⭐⭐
