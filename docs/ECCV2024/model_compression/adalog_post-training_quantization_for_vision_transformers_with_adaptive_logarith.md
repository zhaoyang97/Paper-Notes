# AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer

**会议**: ECCV 2024  
**arXiv**: [2407.12951](https://arxiv.org/abs/2407.12951)  
**代码**: [https://github.com/GoatWu/AdaLog](https://github.com/GoatWu/AdaLog) (有)  
**领域**: 模型压缩 / 后训练量化 / Vision Transformer  
**关键词**: PTQ, 非均匀量化, 自适应对数底, 超参搜索, ViT量化  

## 一句话总结
提出自适应对数底量化器AdaLog，通过可搜索的对数底替代固定log₂/log√2量化器来处理ViT中post-Softmax和post-GELU激活的幂律分布，并设计快速渐进组合搜索(FPCS)策略高效确定量化超参，在极低比特(3/4-bit)下显著优于现有ViT PTQ方法。

## 背景与动机
Vision Transformer在分类、检测、分割等视觉任务上取得了优异性能，但其巨大的计算量和内存开销阻碍了在边缘设备上的部署。后训练量化(PTQ)只需少量校准数据即可完成模型量化，是最高效的压缩手段。然而，ViT中有两类特殊的激活分布——post-Softmax和post-GELU——它们都呈现**幂律分布**，大量值集中在接近零的区域，均匀量化器会造成巨大的量化误差。

现有的非均匀量化方案(FQ-ViT的log₂量化器、RepQ-ViT的log√2量化器)采用固定的对数底。但问题在于：log₂在4-bit下对大值的舍入误差大，而log√2在3-bit下又会把大部分值截断为0。不同层、不同bit-width下最优的对数底是不同的，固定底无法适应这种变化。此外，log√2量化器在反量化时需要浮点乘法，不利于硬件部署。

## 核心问题
1. **对数底不灵活**: 固定log₂或log√2无法在不同bit-width和不同层之间自适应调整，极低bit下精度严重下降
2. **超参搜索空间稀疏**: ViT激活分布范围大，传统网格搜索的均匀稀疏划分容易陷入局部最优
3. **硬件不友好**: log√2量化器的反量化过程涉及逐元素浮点乘法，无法用纯整数推理

## 方法详解

### 整体框架
AdaLog作用于标准ViT block中的post-Softmax层(MatMul2)和post-GELU层(FC2)。其他层(QKV, Proj, FC1)仍使用均匀量化器，但统一采用FPCS搜索最优超参。整体流程：对每一层，用32张校准图片获取输入/输出特征，然后通过FPCS搜索确定量化超参（AdaLog层搜索对数底b和缩放因子s，均匀量化层搜索缩放因子和零点），最终得到完整的量化模型。

### 关键设计
1. **自适应对数底量化器(AdaLog)**: 将量化公式从固定底推广到任意底b：$A^{(\mathbb{Z})} = \text{clamp}(\lfloor -\log_b \frac{A}{s} \rceil, 0, 2^{bit}-1)$。关键技巧是用有理数近似$\log_2 b \approx q/r$，将反量化分解为查表+位移操作：$\hat{A} = s \cdot (2^{-\tilde{A}^{(\mathbb{Z})}} \circ 2^{-\tilde{U}})$，其中$\tilde{A}^{(\mathbb{Z})}$和$2^{-\tilde{U}}$都可预计算并存入查找表（表长仅$2^{bit}$），推理时只需两次查表和一次位移，完全避免浮点运算，硬件友好。

2. **偏置重参数化(Bias Reparameterization)**: post-GELU激活包含负值（主要集中在(-0.17, 0]），而对数量化要求非负输入。解决方法是将FC2的线性层重写为$Y = W \cdot (X + 0.17) + (b - 0.17 \cdot \hat{W} \cdot \mathbf{1})$，使AdaLog的输入$X' = X + 0.17$变为非负，同时将偏移量吸收到偏置中，不影响输出等价性。

3. **快速渐进组合搜索(FPCS)**: 受NLP中beam search启发。先在粗粒度网格上评估所有候选组合(复杂度$O(xy)$，$xy=n$)，选出top-k最优组合，再围绕每个最优点做细粒度扩展(每个扩展$z$个候选，$kz=n$)，迭代$p$步。总复杂度为$O(pn)$，与交替搜索相当，但通过渐进细化搜索空间避免了局部最优，精度接近暴力搜索(复杂度$O(nm)$)。

### 损失函数 / 训练策略
- 搜索目标为**逐层MSE损失**：$\text{MSE}(\phi^{(l)}(X_l, a, b), O_l)$，即量化层输出与全精度输出的均方误差
- 校准数据仅需32张ImageNet图片(分类)或32张COCO图片(检测/分割)
- 权重采用通道级量化，激活采用层级量化+scale重参数化技术
- 超参设置：$r=37$（素数以保证互素），$n=128$，搜索步数$p=4$

## 实验关键数据

| 数据集 | 模型 | bit(W/A) | 指标 | AdaLog | RepQ-ViT | 提升 |
|--------|------|----------|------|--------|----------|------|
| ImageNet | ViT-S | W4/A4 | Top-1 | 72.75% | 65.05% | +7.70% |
| ImageNet | ViT-B | W4/A4 | Top-1 | 79.68% | 68.48% | +11.20% |
| ImageNet | DeiT-B | W4/A4 | Top-1 | 78.03% | 75.61% | +2.42% |
| ImageNet | Swin-B | W4/A4 | Top-1 | 82.47% | 78.32% | +4.15% |
| ImageNet | ViT-S | W3/A3 | Top-1 | 13.88% | 0.10% | +13.78% |
| ImageNet | ViT-B | W3/A3 | Top-1 | 37.91% | 0.10% | +37.81% |
| COCO | Mask R-CNN Swin-T | W4/A4 | AP^box | 39.1% | 36.1% | +3.0% |
| COCO | Cascade Mask R-CNN Swin-S | W4/A4 | AP^box | 50.6% | 49.3% | +1.3% |

### 消融实验要点
- **AdaLog贡献最大**: 在ViT-S W4A4上，仅加AdaLog（不用FPCS）即从62.20%提升到72.01%（+9.81%）；仅加FPCS提升不到1%
- **FPCS在3-bit下重要性增大**: ViT-B W3A3上，FPCS从9.68%提升到15.50%（+5.82%），因为低bit下搜索空间更关键
- **AdaLog+FPCS组合效果最佳**: 两者叠加在所有模型/bit-width上均超过单独使用
- **post-Softmax量化器对比**: AdaLog在2-bit仍保持合理精度(如ViT-S 70.36%)，而log₂和log√2均崩溃到0.10%
- **post-GELU量化器对比**: AdaLog在所有7个模型上均达到最高精度，而其他量化器表现不稳定
- **FPCS vs 暴力搜索**: 精度接近（DeiT-T W3A3: 31.56% vs 32.04%），但时间仅为1/45（4.1min vs 183min）
- **可集成到现有框架**: 在BRECQ框架中加入AdaLog后，LSQ训练激活参数的精度从崩溃(0.93%)提升到62.50%
- **效率**: 查找表内存开销极小，4-bit DeiT-T仅需~3KB，不到模型大小的0.2%

## 亮点
- **对数底可学习化**的核心思想简洁有力，一个超参就把固定底量化器的所有局限都解决了
- 查表+位移的硬件友好反量化方案设计精妙：通过有理数近似$\log_2 b$，把任意底的幂运算变成两次查表+一次位移
- 偏置重参数化让AdaLog兼容负值激活(post-GELU)，扩展了适用范围
- FPCS的beam search思想迁移到量化超参搜索，兼顾了搜索精度和效率
- 在3-bit极端低bit下，其他方法完全崩溃而AdaLog仍有合理精度，说明非均匀量化在低bit场景的核心优势

## 局限性 / 可改进方向
- 仅在post-Softmax和post-GELU层使用AdaLog，其他层仍用均匀量化器——是否可以对所有层统一使用自适应非均匀量化？
- 实验仅覆盖分类/检测/分割，未验证在生成类任务(如扩散模型)或更大规模模型(如VLM)上的效果
- 3-bit下绝对精度仍然不高（ViT-S仅13.88%），实用性有限，2-bit几乎不可用
- 对数底的搜索范围和$r$值的选择缺乏理论指导，依赖经验设定
- 未探索与QAT结合的可能性——AdaLog的非均匀量化函数是否可微？
- → 可延伸到 [注意力感知混合精度量化](../../ideas/model_compression/20260316_attention_aware_quant.md)：不同注意力区域用不同对数底/bit-width

## 与相关工作的对比
- **vs RepQ-ViT (ICCV 2023)**: RepQ-ViT用固定log√2底+scale重参数化，但反量化需要逐元素浮点乘法，硬件不友好；AdaLog底可搜索且完全整数推理。W4A4下AdaLog平均高5.13%
- **vs PTQ4ViT (ECCV 2022)**: PTQ4ViT用Twin-Uniform量化器处理幂律分布，本质仍是均匀量化的变体；AdaLog的非均匀对数量化更匹配幂律分布。PTQ4ViT在3-bit下几乎完全崩溃
- **vs FQ-ViT (IJCAI 2022)**: 提出log₂量化器的开创性工作，但固定底的局限性在低bit下暴露无遗

## 启发与关联
- **对注意力感知量化idea的启发**: AdaLog证明了非均匀量化在处理ViT特殊激活分布上的核心优势。我们的[注意力感知混合精度量化](../../ideas/model_compression/20260316_attention_aware_quant.md)idea可以在此基础上更进一步——不同attention head/不同空间区域使用不同的AdaLog底，让量化精度与注意力重要性自适应匹配
- **对SVD-量化联合设计的参考**: [SVD-Quantization Co-Design](../../ideas/multimodal_vlm/20260316_svd_quant_dense_prediction_vlm.md)中，AdaLog可作为密集预测VLM的激活量化方案之一，特别是在attention层的post-Softmax量化上
- **对医学量化的参考**: [无解码器蒸馏量化](../../ideas/model_compression/20260317_decoder_free_quant_medical.md)中，AdaLog可用于编码器中的Transformer层量化，改善编码器的激进量化精度
- FPCS的渐进搜索思想可迁移到任何需要多超参联合搜索的量化场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 自适应对数底的思想自然但此前无人尝试，查表方案设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 分类/检测/分割三个任务，7种模型，3/4/6-bit，消融全面
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，图示直观，但部分符号较多需要仔细跟读
- 价值: ⭐⭐⭐⭐ 对ViT低bit PTQ有实质性推进，FPCS搜索策略可复用性强
