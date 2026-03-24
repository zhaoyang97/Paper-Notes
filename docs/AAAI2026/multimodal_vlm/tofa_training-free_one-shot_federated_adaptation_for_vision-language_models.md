# TOFA: Training-Free One-Shot Federated Adaptation for Vision-Language Models

**会议**: AAAI 2026  
**arXiv**: [2511.16423](https://arxiv.org/abs/2511.16423)  
**代码**: 待确认  
**领域**: 多模态VLM  
**关键词**: 联邦学习, CLIP适配, One-Shot FL, 层次贝叶斯, 训练无关

## 一句话总结
提出TOFA框架，在联邦学习场景下通过层次贝叶斯模型学习个性化视觉prototype分布 + 全局对齐的LLM文本增强 + 自适应模态融合，实现无需训练、仅一轮通信的CLIP高效适配，在9个数据集上超越one-shot基线甚至部分多轮训练方法。

## 研究背景与动机
1. **领域现状**：CLIP等VLM在联邦学习(FL)中的适配日益受关注。现有方法主要通过prompt learning (PromptFL, pFedPrompt) 或fine-tuning来适配下游任务，但依赖多轮client-server通信。
2. **现有痛点**：
   - **通信开销大**：多轮交互导致高通信成本，且要求系统长期稳定运行
   - **计算资源不足**：大量移动端client和低配server无法进行模型训练
   - **One-shot方法不适配VLM**：已有的one-shot FL方法（如FedLPA、FENS）主要为传统模型设计，不善于利用VLM的多模态信息
   - **数据异质性**：non-IID数据分布导致local和global优化目标不一致
3. **核心矛盾**：如何在零训练、仅一轮通信的严格约束下，充分利用VLM的多模态信息，同时处理数据异质性
4. **本文要解决什么？** 设计一个training-free + one-shot的FL框架，实现VLM的高质量适配
5. **切入角度**：分别从视觉和文本两条pipeline提取互补信息——视觉侧用贝叶斯方法提取个性化prototype，文本侧用LLM增强+全局对齐得到鲁棒文本表示
6. **核心idea一句话**：用层次贝叶斯做个性化视觉分布推断 + LLM文本增强全局对齐 + confidence-based模态融合，在FL中实现无训练一轮VLM适配

## 方法详解

### 整体框架
TOFA由三个模块组成：
- **Visual Pipeline**: 各client计算local visual statistics → 上传server → server用uninformative prior计算全局class prototype分布 → 发回client → client用全局分布作为prior推断个性化local posterior → GDA分类
- **Textual Pipeline**: 各client用本地LLM生成augmented text descriptions → 计算各text prompt的importance score → server全局对齐，筛选robust text prompts → 加权组合用于分类
- **Adaptive Fusion**: 基于confidence的sample-wise融合，动态平衡visual和textual预测

### 关键设计

1. **协同Prompt分布学习（Visual Pipeline）**:
   - 做什么：为每个client学习个性化的class-specific视觉prototype分布
   - 核心思路：假设CLIP视觉特征服从class-specific高斯分布 $\mathcal{N}(\mathbf{w}_c, \mathbf{\Sigma})$。采用**层次贝叶斯模型**：全局后验分布 $q(\theta) = \pi(\theta|D)$ 通过汇聚所有client的local statistics计算；然后以全局后验作为informative prior，结合local data计算个性化后验 $q(\theta^k) \propto L(D^k|\theta^k)[L(D|\theta)]^\alpha \pi(\theta)$，其中power prior参数 $\alpha$ 控制全局信息的影响权重。使用Normal-Inverse-Wishart共轭prior保证后验有closed-form解，无需迭代优化。最终用GDA做分类
   - 设计动机：直接用mean prototype会忽略数据分布的方差信息。贝叶斯框架自然地在全局generalization和local personalization之间做trade-off，$\alpha$控制两者的平衡。共轭prior保证one-shot可行（无需迭代）

2. **全局对齐的文本增强（Textual Pipeline）**:
   - 做什么：从LLM生成的丰富文本描述中筛选robust且generalizable的text prompts
   - 核心思路：各client使用LLM生成dataset-aware的class descriptions $\{t_c^m\}_{m=1}^M$。每个client计算每个text prompt对各class的分类置信度 $p_c^k(t_c^m)$。Server端用类KL散度的importance scoring $r(t_c^m) = \frac{1}{K}\sum_{k=1}^K u^k(t_c^0)\log\frac{u^k(t_c^m)}{u^k(t_c^0)}$ 对text prompts评分，其中 $u^k$ 衡量区分目标class与其他class的confidence。高分prompt在各种异质数据环境下都表现稳定
   - 设计动机：手工模板"A photo of a {class}"太简单，LLM增强可以引入丰富语义但质量参差不齐。全局对齐确保选出的text prompts跨heterogeneous clients都有效

3. **自适应模态融合**:
   - 做什么：sample-wise地融合visual和textual预测
   - 核心思路：融合公式 $f_M^k(\mathbf{z}) = \eta(\mathbf{z})f_V^k(\mathbf{z}) + (1-\eta(\mathbf{z}))f_T(\mathbf{z})$，权重 $\eta(\mathbf{z}) = \sigma(\log\frac{\max_j \text{softmax}(f_V^k(\mathbf{z}))_j}{\max_j \text{softmax}(f_T(\mathbf{z}))_j})$。通过Theorem 1证明：当 $\eta$ 与两种模态的loss差成正比时，融合分类器的泛化误差最小。用校准后的confidence作为accuracy的代理
   - 设计动机：不同样本在不同模态上的可靠性不同——对某些样本visual更准，对另一些textual更准。固定权重无法适应这种变化

### 损失函数 / 训练策略
TOFA完全training-free：
- Visual pipeline只需传递sufficient statistics（均值、散布矩阵、样本数）
- Textual pipeline只需传递importance scores
- **一轮通信**：client→server→client即完成

## 实验关键数据

### 主实验
CLIP Datasets (16-shot, 10 clients, label shift):

| 方法 | Training-free | One-shot | OxfordPets | Flowers102 | Food101 | Caltech101 | DTD |
|------|:---:|:---:|------|----------|--------|----------|-----|
| CoOp | ✗ | ✗ | 89.18 | 69.03 | 82.54 | 90.62 | 63.97 |
| pFedPrompt | ✗ | ✗ | 91.84 | 96.46 | 92.26 | 96.54 | 77.14 |
| Zero-Shot CLIP | ✓ | ✓ | 85.77 | 66.14 | 77.31 | 86.29 | 42.32 |
| CLIP-GDA | ✓ | ✓ | 88.81 | 91.23 | 79.05 | 92.55 | 60.64 |
| FedLPA+PromptFL | ✗ | ✓ | 83.42 | 78.60 | 74.74 | 88.69 | 52.75 |
| **TOFA** | **✓** | **✓** | **91.23** | **95.78** | **85.49** | **94.58** | **71.68** |

CIFAR-10/100 (100 clients, Dir(0.3)):

| 方法 | CIFAR-10 | CIFAR-100 |
|------|----------|-----------|
| FedAvg | 75.10 | 42.52 |
| Zero-Shot CLIP | 87.71 | 64.92 |
| CoOp | 93.11 | 74.83 |
| **TOFA** | **93.18** | **76.63** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| Visual only | 低于full model | 缺少robust textual信息 |
| Textual only | 低于full model | 缺少personalized visual信息 |
| w/o Global Alignment | 性能下降 | LLM text质量不稳定 |
| w/o Adaptive Fusion (fixed weight) | 性能下降 | 无法适应sample-level差异 |
| Full TOFA | 最优 | 三模块互补 |

### 关键发现
- TOFA作为training-free+one-shot方法，在多个数据集上超越了多轮训练的CoOp和PromptFL
- 在extreme heterogeneity（CIFAR-100, 100 clients, Dir(0.3)）下仍然有效，展现强鲁棒性
- 在DomainNet feature shift场景下也有竞争力，说明方法对label shift和feature shift都有效

## 亮点与洞察
- **层次贝叶斯的妙用**：用全局后验作为local的informative prior，elegantly地在一轮通信内实现个性化。共轭prior保证closed-form解，避免了任何迭代优化——这是实现training-free的关键数学infrastructure
- **Text增强的全局对齐**：不是简单averaging各client的text评分，而是用类KL散度选出跨异质环境都robust的text prompts，比直接用LLM输出质量高很多
- **Sample-wise fusion有理论支撑**：Theorem 1将模态融合的generalization error bound与mixing coefficient联系起来，不是拍脑袋设计的confidence weighting

## 局限性 / 可改进方向
- **假设高斯分布**：CLIP特征是否真的服从类高斯分布？复杂场景下（如细粒度分类）可能需要更灵活的分布假设
- **LLM一致性要求**：需要各client使用相同版本LLM生成text augmentation，这在实际FL场景中可能难以保证
- **仅适用于分类任务**：GDA-based视觉pipeline限制了方法只能做分类，无法扩展到detection/segmentation等任务
- **隐私分析不够深入**：虽然只传递statistics而非raw data，但class-specific mean和covariance是否可能泄露隐私值得深入分析

## 相关工作与启发
- **vs PromptFL/pFedPrompt**: 多轮训练方法，需要多次client-server通信 + 梯度计算。TOFA完全无训练、一轮完成，在多数数据集上性能相当甚至更好
- **vs CLIP-GDA**: 同样使用GDA，但CLIP-GDA是纯local方法，不做联邦聚合。TOFA通过层次贝叶斯引入全局信息，在异质环境下更robust
- **vs FedLPA**: one-shot但需要client训练，且只用视觉模态。TOFA无训练+双模态，性能大幅领先

## 评分
- 新颖性: ⭐⭐⭐⭐ 层次贝叶斯+全局文本对齐+自适应融合的组合在FL+VLM领域是首创
- 实验充分度: ⭐⭐⭐⭐ 9个数据集、多种异质性设置、4类baseline对比、消融实验完整
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，但公式密度很高，可读性一般
- 价值: ⭐⭐⭐⭐ 为资源受限的联邦VLM适配提供了实用方案，training-free+one-shot约束下的性能很impressive
