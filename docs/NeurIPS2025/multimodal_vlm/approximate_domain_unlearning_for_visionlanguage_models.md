# Approximate Domain Unlearning for Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.08132](https://arxiv.org/abs/2510.08132)  
**代码**: [https://kodaikawamura.github.io/Domain_Unlearning/](https://kodaikawamura.github.io/Domain_Unlearning/)  
**领域**: 多模态VLM / 机器遗忘  
**关键词**: approximate unlearning, domain unlearning, vision-language model, CLIP, prompt tuning

## 一句话总结

提出 Approximate Domain Unlearning (ADU) 新任务，通过 Domain Disentangling Loss (DDL) 和 Instance-wise Prompt Generator (InstaPG) 两个模块，让预训练 VLM 选择性遗忘指定域（如插画、素描）的识别能力，同时保持其他域（如真实照片）的分类精度，在四个多域数据集上大幅超越所有基线。

## 研究背景与动机

1. **领域现状**：预训练 VLM（如 CLIP）具有极强的域泛化能力，能跨域识别各类物体，但在特定下游任务中，这种全面的泛化能力并非必要，甚至可能带来安全隐患和信息泄露风险。
2. **现有痛点**：现有的近似遗忘方法聚焦于 class-level unlearning（遗忘特定类别），但在许多实际场景中仅遗忘类别远远不够。例如自动驾驶系统需要识别真实的汽车，但不应将路边广告上的插画汽车误认为真实汽车。
3. **核心矛盾**：VLM 的强域泛化能力导致不同域的特征分布在隐空间中高度纠缠（entangled），直接套用类遗忘策略（对 forget 域最大化熵 + 对 memorize 域最小化交叉熵）无法有效区分哪些特征属于哪个域，导致遗忘和保留相互干扰。
4. **本文要解决什么？** 如何让 VLM 在域级别（而非类级别）进行精细的选择性遗忘——降低指定域的识别准确率，同时保持其他域的性能。
5. **切入角度**：既然域纠缠是核心障碍，那就先把域分布在隐空间中拆开（disentangle），再分别对不同域施加遗忘/保留策略。
6. **核心idea一句话**：通过域分布解纠缠 + 实例级自适应 prompt 生成，实现 VLM 的精准域级遗忘。

## 方法详解

### 整体框架

整体 pipeline 基于 CLIP 的 vision prompt tuning 框架：在 ViT 图像编码器的前 9 层插入可学习的 vision prompt token，通过三部分损失联合优化——(1) 对 memorize 域最小化分类交叉熵，(2) 对 forget 域最大化分类熵，(3) DDL 强制域分布分离。InstaPG 嵌入在中间 Transformer 层，根据输入图像动态生成实例级 prompt。文本端固定使用 "a photo of a [class]" 模板，不做额外调整。

### 关键设计

1. **Approximate Domain Unlearning (ADU) 问题定义**:
   - 做什么：给定训练集 $\{(\mathbf{x}, y, d)\}$，其中 $d \in \mathcal{D}$ 为域标签，定义 $\mathcal{D}_{\text{memorize}}$ 为需保留的域，$\mathcal{D}_{\text{forget}} = \mathcal{D} \setminus \mathcal{D}_{\text{memorize}}$ 为需遗忘的域
   - 核心思路：对 memorize 域最小化交叉熵 $\mathcal{L}_{\text{memorize}}$，对 forget 域最小化到均匀分布的交叉熵 $\mathcal{L}_{\text{forget}}$（等价于最大化熵）
   - 设计动机：直接将类遗忘的两个损失搬到域遗忘上是最直接的 baseline，但由于域分布纠缠严重，单靠这两个损失效果有限

2. **Domain Disentangling Loss (DDL)**:
   - 做什么：在隐空间中显式地将不同域的特征分布推开
   - 核心思路：由两个互补的损失函数组成。第一个是辅助域分类器的交叉熵损失 $\mathcal{L}_{\text{CE}}$，要求模型能正确预测样本的域标签；第二个是 Maximum Mean Discrepancy (MMD)，在再生核希尔伯特空间中最大化域间距离：
     $$\mathcal{L}_{\text{domain}} = \gamma \mathcal{L}_{\text{CE}} - \lambda \text{MMD}^2$$
     其中 MMD 前取负号表示最大化域间距离（与传统域适应中最小化 MMD 恰好相反）。$\gamma=30$，$\lambda=10$ 为默认超参。
   - 设计动机：如果不同域的特征分布被良好分离，那么针对某个域施加遗忘损失就不会影响其他域。CE 从判别式角度保证域可分，MMD 从分布距离角度保证域分离，二者互补。

3. **Instance-wise Prompt Generator (InstaPG)**:
   - 做什么：根据每张输入图像的 patch 特征动态生成个性化的 vision prompt
   - 核心思路：嵌入在 ViT 的中间 Transformer block 中，采用 cross-attention 机制——可学习的 vision prompt 作为 query，图像 patch features 作为 key 和 value，生成实例级别的 prompt 送入后续层
   - 设计动机："域"本身是模糊的概念。例如"插画"风格跨度极大，有的接近写实、有的接近卡通。统一的 prompt 无法捕捉这种实例级的域差异，InstaPG 通过注意力机制让 prompt 自适应于每张图像的特征

### 损失函数 / 训练策略

总损失由三部分组成：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{memorize}} + \mathcal{L}_{\text{forget}} + \mathcal{L}_{\text{domain}}$$

- $\mathcal{L}_{\text{memorize}}$：对 memorize 域样本的标准分类交叉熵
- $\mathcal{L}_{\text{forget}}$：对 forget 域样本到均匀分布的交叉熵（最大化预测熵）
- $\mathcal{L}_{\text{domain}} = \gamma \mathcal{L}_{\text{CE}} - \lambda \text{MMD}^2$：DDL 域解纠缠损失

训练细节：使用 ViT-B/16 作为图像编码器，deep prompting 策略，8 个上下文 token，SGD 优化器，学习率 0.0025，训练 50 epoch。每个域仅使用 8 个标注样本（few-shot setting）。

## 实验关键数据

### 主实验

在 ImageNet、Office-Home、Mini DomainNet 三个数据集上对比 LP++、CLIPFit、BBF 和 Baseline（$|\mathcal{D}_{\text{forget}}|=1$ 的结果）：

| 方法 | ImageNet H↑ | Office-Home H↑ | Mini DomainNet H↑ |
|------|------------|----------------|-------------------|
| LP++ | 50.69 | 30.46 | 31.73 |
| CLIPFit | 71.31 | 43.44 | 53.56 |
| BBF | 45.56 | 31.25 | 32.12 |
| Baseline | 74.66 | 52.59 | 62.07 |
| **Ours** | **77.02** | **69.96** | **75.56** |

在 Office-Home 上，本文方法 H 指标比最强基线 Baseline 高 **+17.37**；在 Mini DomainNet 上高 **+13.49**。当遗忘域数量增加到 3 个时，优势更加明显（Office-Home 上 H=75.89 vs Baseline 59.47）。

### 消融实验

DDL 和 InstaPG 的消融（Office-Home，$|\mathcal{D}_{\text{forget}}|=1$）：

| 配置 | H↑ | Mem↑ | For↑ | 说明 |
|------|-----|------|------|------|
| 无 DDL 无 InstaPG (Baseline) | 52.59 | 79.96 | 39.88 | 朴素遗忘策略 |
| 仅 InstaPG | 56.41 | 83.55 | 44.05 | InstaPG 单独贡献 +3.82 |
| 仅 DDL | 60.82 | 74.51 | 51.72 | DDL 单独贡献 +8.23 |
| DDL + InstaPG (完整方法) | **69.96** | 77.93 | **64.34** | 两者组合效果最佳 |

DDL 内部 CE 与 MMD 的消融（Office-Home，$|\mathcal{D}_{\text{forget}}|=1$）：

| CE | MMD | H↑ | Mem↑ | For↑ |
|----|-----|----|------|------|
| ✗ | ✗ | 56.41 | 83.55 | 44.05 |
| ✗ | ✓ | 68.62 | 82.41 | 59.47 |
| ✓ | ✗ | 64.01 | 82.97 | 53.62 |
| ✓ | ✓ | **69.96** | 77.93 | **64.34** |

### 关键发现

- DDL 的贡献大于 InstaPG，但两者组合后效果远超单独使用，说明域解纠缠和实例级自适应是互补的
- 域分类准确率从 zero-shot CLIP 的 25.80% 提升到本文方法的 79.43%，证明 DDL 确实有效分离了域分布
- 超参 $\gamma$ 和 $\lambda$ 在超过一定阈值后性能稳定，方法对超参不敏感
- 随着训练样本数增加，本文方法持续提升，而 Baseline 在 Mini DomainNet 上出现过拟合趋势
- t-SNE 可视化清晰展示了方法前后域分布从纠缠到分离的变化
- 注意力图显示：对 forget 域，模型注意力从目标物体上消散；对 memorize 域，注意力保持甚至增强

## 亮点与洞察

- **首次定义 ADU 问题**：将近似遗忘从类级别扩展到域级别，打开了全新的研究方向，且有清晰的实际应用动机（自动驾驶安全等）
- **反转 MMD 的用法**：传统域适应最小化 MMD 来对齐域分布，本文反其道而行最大化 MMD 来拆分域分布，思路简洁而有效
- **few-shot 设定下即可工作**：每个域仅用 8 个样本就能实现高质量的域遗忘，实用性强
- **InstaPG 的 cross-attention 设计**：巧妙利用 prompt 作为 query、图像 patch 作为 key/value，让 prompt 能感知每张图的域特性
- **注意力图可解释性**：通过 attention heatmap 直观展示了遗忘效果——forget 域的注意力被"分散"，memorize 域的注意力被"保留甚至增强"

## 局限性 / 可改进方向

- **需要域标签**：方法假设所有训练样本都有域标签，而实际中域标注往往不完整。论文在附录中用伪标签做了初步探索，但更鲁棒的域估计方案值得深入研究
- **域的定义依赖先验**：哪些是"域"由人工定义，如果域划分不当可能影响效果；未探索自动发现域的设定
- **仅在 CLIP 上验证**：所有实验基于 CLIP ViT-B/16，未测试其他 VLM（如 BLIP-2、LLaVA 等更大的多模态模型）
- **仅针对图像分类**：ADU 的定义限于分类任务，是否能扩展到生成式 VLM（如 text-to-image）的域遗忘是一个有趣的开放问题
- **可能的隐私遗忘基准**：当前评估侧重分类性能，缺少对信息泄露/隐私保护效果的直接度量

## 相关工作与启发

- **vs BBF (Kuwana et al., 2024)**：BBF 是 VLM 上 SOTA 的类遗忘方法，但直接用于域遗忘时 For 指标比本文低 30% 以上，说明类遗忘和域遗忘是根本不同的问题
- **vs CLIPFit / LP++**：这些是 SOTA 的 CLIP 微调方法，配合标准遗忘损失后在 ADU 上 H 值仍然很低（30-53），说明仅靠微调策略不够，必须解决域纠缠问题
- **vs 域适应/泛化方法 (DAN, JAN)**：这些方法最小化 MMD 追求域不变性，本文反向最大化 MMD 追求域可分性，体现了 ADU 与 DA/DG 在目标上的根本差异
- **启发**：DDL 的 "先分离再遗忘" 思路或许可以推广到其他需要细粒度控制的遗忘场景，如按时间段遗忘、按来源遗忘等

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次提出 ADU 问题，切入角度新颖且有实际意义
- 实验充分度: ⭐⭐⭐⭐ 四个数据集、多种遗忘域数量配置、完整消融和可视化分析，但仅限 CLIP 一个模型
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，动机阐述到位，图表直观
- 价值: ⭐⭐⭐⭐ 开辟了域级遗忘的新方向，但距离实际部署还需解决域标签依赖等问题
