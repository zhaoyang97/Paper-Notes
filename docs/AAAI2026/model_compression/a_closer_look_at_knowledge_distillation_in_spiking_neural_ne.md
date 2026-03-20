# A Closer Look at Knowledge Distillation in Spiking Neural Network Training

**会议**: AAAI 2026  
**arXiv**: [2511.06902](https://arxiv.org/abs/2511.06902)  
**代码**: [https://github.com/SinoLeu/CKDSNN](https://github.com/SinoLeu/CKDSNN)  
**领域**: 模型压缩/神经形态计算  
**关键词**: 知识蒸馏, 脉冲神经网络, 激活图对齐, 噪声平滑, 能效训练  

## 一句话总结

针对ANN→SNN知识蒸馏中教师ANN连续特征/logits与学生SNN离散稀疏spike特征/logits之间分布差异被忽视的问题，提出基于显著性缩放激活图蒸馏（SAMD）和噪声平滑logits蒸馏（NLD）的CKDSNN框架，在CIFAR-10/100、ImageNet-1K和CIFAR10-DVS上均取得SNN训练的新SOTA。

## 背景与动机

脉冲神经网络（SNN）受生物神经元启发，以事件驱动的二进制脉冲传递信息，可将乘法替换为加法运算，具有极高的能效优势，可高效运行于神经形态硬件（如Intel Loihi）。但SNN训练面临两个主要困境：（1）ANN-to-SNN转换方法需要大量时间步才能保证精度；（2）直接训练方法虽减少了时间步，但因替代梯度的估计误差，SNN与ANN之间仍存在明显精度差距。

近期工作引入知识蒸馏（KD），以预训练ANN为教师、SNN为学生来提升SNN训练质量，取得了一定成效。但现有方法（如KDSNN、BKDSNN）在做KD时存在两个被忽视的关键问题：

1. **特征分布差异**：ANN的中间特征是连续浮点值，而SNN的特征是多时间步上的离散二值脉冲（0/1），且SNN的spike主要集中在显著区域，二者在分布上天然不匹配。
2. **logits分布差异**：SNN的分类logits因源自二值特征而呈现更稀疏和尖锐的分布，与ANN连续平滑的logits分布有显著差异。

现有方法简单地做逐元素对齐（element-wise alignment），忽视了上述本质差异，导致蒸馏效果不理想。

## 核心问题

如何在ANN→SNN知识蒸馏中有效弥合教师（连续浮点特征+平滑logits）与学生（离散二值spike特征+稀疏logits）之间在特征和输出层面的分布差异，使蒸馏真正有效？

## 方法详解

### 整体框架

CKDSNN框架接收预训练ANN教师模型和可学习SNN学生模型，从两个互补层面进行知识蒸馏：
- **特征层面**：通过SAMD将教师ANN的类激活图（CAM）蒸馏给学生SNN的脉冲激活图（SAM），而非直接对齐原始特征。
- **logits层面**：通过NLD用高斯噪声平滑学生SNN的稀疏logits，拉近与教师ANN连续logits的分布后再做对齐。

总损失 = 标准交叉熵损失 + β·SAMD损失 + γ·NLD损失。

### 关键设计

1. **Saliency-scaled Activation Map Distillation (SAMD)**：分三步完成。
   - **CAM生成（教师端）**：利用Grad-CAM对预训练ANN提取类激活图 $M^{te} \in \mathbb{R}^{H \times W}$，通过梯度加权的通道求和获得与目标类相关的空间显著区域。
   - **SAM生成（学生端）**：由于SNN中替代梯度存在估计误差，Grad-CAM式的方法在SNN上产生的激活图不准确。因此论文另辟蹊径，直接利用SNN的spike本身——对所有时间步和通道维度上的spike进行求和，得到脉冲激活图 $M^{st} = \sum_{t}\sum_{c} F^{st}_{t,c}$。这一做法利用了SNN的天然特性：spike本身就只在显著区域产生。
   - **显著性缩放对齐**：CAM和SAM虽语义一致（都高亮显著区域），但数值量级差异大（一个来自浮点加权，一个来自二值累加）。论文用softmax函数将两者都转化为概率分布 $P^{te}$ 和 $P^{st}$（带温度参数 $\mathcal{T}$），然后用KL散度损失对齐：$\mathcal{L}_{SAMD} = \mathcal{T}^2 \cdot KL(P^{te} \| P^{st})$。

2. **Noise-smoothed Logits Distillation (NLD)**：SNN的logits因二值特征导致分布稀疏尖锐，直接与ANN平滑logits对齐效果差。NLD的核心是用**自适应高斯噪声**来"软化"SNN的logits。噪声的均值和标准差直接从SNN logits自身的统计量导出（$\epsilon \sim \mathcal{N}(\bar{z}^{st}, \sigma(z^{st})^2)$），既保持原始分布特征又增加连续性。融合后 $z^{soft} = z^{st} + \lambda \epsilon$，再通过softmax+KL散度与教师logits对齐：$\mathcal{L}_{NLD} = \tau^2 \cdot KL(y^{te} \| y^{soft})$。

3. **SAM的架构无关性**：与CATKD等仅适用于CNN架构的激活图蒸馏方法不同，SAM的生成不依赖梯度信息（不受替代梯度误差影响），也不依赖特定网络结构，因此可以同时应用于Spiking CNN（如ResNet）和Spiking Transformer（如Spikformer）架构。

### 损失函数 / 训练策略

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \beta \cdot \mathcal{L}_{SAMD} + \gamma \cdot \mathcal{L}_{NLD}$$

超参数设置：$\mathcal{T}=2.0$, $\tau=2.0$, $\lambda=0.1$, $\beta=1.0$, $\gamma=1.0$。SAMD应用在网络最后一个stage（深层特征包含更精确的语义信息）。使用SGD优化器(momentum 0.9, weight decay 1e-4)，Cosine Annealing学习率调度。

## 实验关键数据

| 数据集 | 架构 | 时间步 | 本文(CKDSNN) | 之前SOTA | 提升 |
|--------|------|--------|------|----------|------|
| CIFAR-10 | ResNet-19 | T=1 | 96.11% | 95.37% (EnOF) | +0.74% |
| CIFAR-100 | ResNet-19 | T=1 | 79.11% | 77.08% (EnOF) | +2.03% |
| CIFAR-10 | ResNet-19 | T=4 | 97.81% | 96.19% (EnOF) | +1.62% |
| CIFAR-100 | ResNet-19 | T=4 | 83.88% | 82.43% (EnOF) | +1.45% |
| CIFAR-100 | Spikformer-4-384 | T=1 | 83.07% | 81.26% (BKDSNN) | +1.81% |
| ImageNet-1K | SEW-R34 | T=4 | 73.05% | 71.24% (BKDSNN) | +1.81% |
| CIFAR10-DVS | ResNet-20 | T=10 | 81.55% | 80.50% (EnOF) | +1.05% |

能效方面：在ImageNet-1K上使用ResNet-34，CKDSNN(T=2)以71.33%精度仅需8.0%发火率和3.61W功耗，而BKDSNN(T=4)需15.0%发火率和3.98W功耗获得71.24%精度——CKDSNN用更少时间步和更低功耗超越了之前SOTA。

### 消融实验要点

- **SAMD和NLD缺一不可**：去掉任一策略在各种设置下都会显著掉点，说明特征级和logits级蒸馏互补有效。
- **显著性缩放方式**：softmax缩放(79.11%)显著优于无缩放(75.56%)、Z-score(76.48%)和L2-norm(74.78%)，约领先2-3个点。softmax能有效归一化并识别最显著区域。
- **CAM-SAM vs 传统激活图KD**：使用CAM-CAM（如e2KD、CATKD等ANN式梯度激活图方法）在SNN上效果显著差于提出的CAM-SAM方案，验证了SNN中替代梯度误差对Grad-CAM的影响。CATKD还受限于CNN架构，而SAM适用于各种架构。
- **自适应噪声 vs 固定噪声**：自适应高斯噪声(NLD)显著优于任何固定标准差的随机噪声。小噪声效果差，大噪声反而有害，而NLD通过从logits自身派生噪声参数，自适应地保持了分布特征。
- **SAMD应用位置**：在最后一个stage效果最好（79.11%），前面stage效果递减（Stage 1: 77.01%）。
- **损失景观**：CKDSNN训练的模型具有更平坦的损失景观，较少鞍点，有利于收敛到更好的局部最优。

## 亮点

- **利用SNN自身特性生成SAM而非强行用Grad-CAM**是核心洞察。SNN中spike天然只在显著区域产生，直接累加spike就能构建有意义的激活图，巧妙绕开了替代梯度误差问题。思路简单优雅且计算高效。
- **自适应噪声平滑logits**的设计很实用：噪声的统计参数从logits自身导出，既软化了分布又保留了原始特征，有最大熵原理支撑的理论依据。
- **两个策略高度互补**——SAMD处理特征级分布差异，NLD处理logits级分布差异，分别从不同侧面解决了ANN-SNN蒸馏的对齐问题。
- **架构无关性**：SAM生成方式不依赖梯度和特定架构，既适用于Spiking CNN也适用于Spiking Transformer，泛化性好。
- 在精度和能效之间取得了更好的平衡：以更少时间步即可超越之前方法的全时间步性能。

## 局限性 / 可改进方向

- **仅验证了图像分类任务**：SNN在NLP、多模态等任务上的KD效果未探索，方法的跨任务泛化性有待验证。
- **CAM的精度依赖教师质量**：当教师ANN的CAM本身有误差时（如在困难样本上），可能误导学生的SAM学习。虽然论文发现NLD可以部分缓解此问题，但根本上仍受限于教师的质量。
- **只蒸馏最后一个stage**：多层级的渐进式蒸馏可能进一步提升效果，但论文实验显示浅层stage效果较差，可能需要针对不同层设计不同的对齐策略。
- **噪声超参数λ仍需手动调节**：虽然噪声分布参数是自适应的，但融合权重是固定的，可以探索训练过程中动态调整λ。
- **时间维度的信息利用**：SAM简单地在时间维度求和，丢失了时序动态信息。不同时间步的spike重要性可能不同，可以引入时间注意力机制。

## 与相关工作的对比

- **vs KDSNN (CVPR'23)**：KDSNN直接逐元素对齐ANN和SNN的特征和logits，完全忽略分布差异。CKDSNN认识到这种差异并针对性设计了SAMD和NLD，在各数据集上均大幅领先（如ImageNet上66.92% vs 63.42%）。
- **vs BKDSNN (ECCV'24)**：BKDSNN通过模糊矩阵处理SNN的spike特征来改善特征匹配，但仍在原始特征层面操作。CKDSNN则提升到语义激活图层面，语义一致性更强。ImageNet上73.05% vs 71.24%。
- **vs EnOF (NeurIPS'24)**：EnOF通过增强输出特征来改善SNN训练。CKDSNN从蒸馏对齐的角度出发，提出了更根本的解决方案，在CIFAR-100上T=1时79.11% vs 77.08%。
- **vs ANN激活图KD方法（e2KD、CATKD）**：这些方法设计于ANN-to-ANN蒸馏场景，直接应用于SNN时因替代梯度误差而效果差。CKDSNN专为SNN特点设计了SAM替代方案。

## 启发与关联

- **SAM的设计思路可迁移**：直接利用模型内在特性（如SNN的spike分布）构建蒸馏信号，而非依赖通用但不适用的方法（如Grad-CAM），这种"因地制宜"的思路可推广到其他异构模型间的蒸馏。
- **噪声平滑logits的思路**值得在其他量化/稀疏模型的蒸馏中尝试——任何产生稀疏分布的模型都可能从类似策略中获益。
- 对SNN直接训练领域有重要参考价值：CAM-SAM框架可能不仅用于蒸馏，也可用于SNN的自监督/对比学习中作为特征表示。
- 结合量化SNN（如二值SNN）或混合精度SNN的蒸馏训练是一个有价值的交叉方向。

## 评分

- 新颖性: ⭐⭐⭐⭐ 利用SNN spike特性设计SAM替代Grad-CAM的洞察很有启发性，但整体框架仍是"特征KD+logits KD"的组合范式
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖三个静态数据集+一个神经形态数据集，多架构(CNN+Transformer)，详尽的消融、能效分析、可视化、理论分析
- 写作质量: ⭐⭐⭐⭐ 问题阐述清晰，方法描述系统，理论分析部分稍显冗长但增强了说服力
- 价值: ⭐⭐⭐⭐ 对SNN-KD领域贡献明确，SOTA提升显著，但影响范围受限于SNN这一相对小众方向