---
title: >-
  [论文解读] ScAle: Attention Head Scaling as a Minimal Adapter for Spatial Reasoning in Vision–Language Models
description: >-
  [ECCV 2026][VLM Reasoning][空间推理] ScAle提出一种极轻量的VLM空间推理适配方法：在完全冻结的backbone中，为每层每个注意力头和MLP输出学习一个bounded scalar（通过tanh约束在(1-s_max, 1+s_max)范围内），仅对last token的激活值做乘法缩放，以约1K可训练参数在SpatialEval上实现最高134.1%的相对精度提升，参数效率比LoRA高2500倍以上。
tags:
  - "ECCV 2026"
  - "VLM Reasoning"
  - "空间推理"
  - "参数高效微调"
  - "激活缩放"
  - "注意力头"
  - "VLM"
---

# ScAle: Attention Head Scaling as a Minimal Adapter for Spatial Reasoning in Vision–Language Models

**会议**: ECCV 2026  
**arXiv**: [2606.29579](https://arxiv.org/abs/2606.29579)  
**代码**: [https://github.com/rchowdhubnor/ScAle.git](https://github.com/rchowdhubnor/ScAle.git)  
**领域**: 多模态VLM / LLM效率  
**关键词**: 空间推理, 参数高效微调, 激活缩放, 注意力头, VLM

## 一句话总结
ScAle提出一种极轻量的VLM空间推理适配方法：在完全冻结的backbone中，为每层每个注意力头和MLP输出学习一个bounded scalar（通过tanh约束在(1-s_max, 1+s_max)范围内），仅对last token的激活值做乘法缩放，以约1K可训练参数在SpatialEval上实现最高134.1%的相对精度提升，参数效率比LoRA高2500倍以上。

## 研究背景与动机
VLMs在通用视觉问答上表现强劲，但在空间推理任务上仍严重不足——LLaVA-7B在SpatialEval三个子任务（网格定位、迷宫导航、地图关系推理）上的冻结基线准确率仅22.7%-31.7%。现有解决方案要么需要全量微调（成本高且损害泛化），要么用LoRA等PEFT方法（仍需百万级可训练参数，训练开销大）。

核心矛盾在于：空间推理能力在VLMs中是存在的（预训练已编码），但被其他竞争模式（competing modes）淹没——2D空间信号在1D token序列中被弱化，被语义、纹理等其他激活模式干扰。本文的关键洞察来自一个初步实验：仅对LLaVA-7B某一层的MLP last-token激活乘以一个合适的标量（不更新任何权重），SpatialEval准确率就能从约20-30%飙升到红色热力区。这一现象暗示空间推理能力的瓶颈不在于"缺少知识"，而在于"信号被压制"——只需要重新校准激活幅度就能释放已有能力。

核心idea：用可学习的bounded scalars对frozen VLM的注意力头和MLP输出做last-token级别的激活重缩放，通过梯度下降让各层各头协同校准，以极端参数效率（约1K参数）实现空间推理能力的定向提升。

## 方法详解

### 整体框架
ScAle不修改任何预训练权重，只在transformer的残差流中插入可学习的标量缩放操作。给定一张图像I、一个问题q和四个候选答案{A,B,C,D}，frozen VLM正常前向传播，但在每个transformer层的两个位置对last token的激活值做乘法缩放：（1）每个注意力头的输出在进入输出投影WO之前；（2）MLP块的输出之后。缩放后的激活继续参与后续计算，最终用last token的logits对四个选项打分，取最高分选项。整个流程中只有缩放标量ϕ可训练，原始参数θ完全冻结。

缩放函数设计为bounded tanh形式：$eff(u) = 1 + s_{max} \cdot \tanh(u)$，其中u是可学习参数，$s_{max}=10$，将缩放范围约束在$(-9, 11)$。这个设计保证数值稳定——不会因为极端缩放值导致激活爆炸或消失。

对注意力头，第l层第h个头的last token输出$z_{l,h} \in \mathbb{R}^{B \times 1 \times d_h}$被缩放为$\tilde{z}_{l,h} = eff(u^{attn}_{l,h}) \cdot z_{l,h}$，然后送入输出投影WO。对MLP，第l层MLP的last token输出$Y_l \in \mathbb{R}^{B \times 1 \times d}$被缩放为$\tilde{Y}_l = eff(u^{MLP}_l) \cdot Y_l$。注意力部分共$L \times H$个标量（L层×H头），MLP部分共L个标量（每层一个），参数量远小于任何主流PEFT方法。

### 关键设计

**1. Bounded tanh缩放公式：保证数值稳定的激活调制**

核心问题是如何在允许足够调制幅度的同时防止数值不稳定。直接学习无界标量会导致训练中激活值发散，而用sigmoid约束到(0,2)又限制了抑制能力（无法将激活缩到接近0或取负值以反相抑制）。

本文用$eff(u) = 1 + s_{max} \cdot \tanh(u)$，$s_{max}=10$。这个公式有三个精妙之处：（1）以1为中心——u=0时eff=1，等价于原始前向，保证初始化时模型行为不变；（2）对称范围(-9, 11)——既能大幅抑制无关模式（eff接近-9时实际上翻转激活方向，主动抵消干扰），也能大幅放大有用信号（eff接近11）；（3）tanh的饱和特性天然提供正则化——极端缩放需要较大的|u|，被L2正则项惩罚，模型只在"值得"时才大幅偏离1。这种「默认不动、有必要才调」的机制使得ScAle即使在极少训练数据下也不会过拟合。

**2. 注意力头级别 + last-token级别：精准定位空间推理信号的调控点**

在何处缩放、以何种粒度缩放，是方法有效性的关键。本文做了两个关键设计决策：

第一，仅对last token缩放。last token是VLM做选择题打分时唯一使用的token——它的残差流状态直接决定最终答案logits。如果对整个序列的所有token缩放，会同时改变上下文编码和查询适配，可能破坏上游视觉和文本的编码过程。只缩放last token则精准作用于决策阶段，不影响视觉特征的提取和文本语义的理解。形式上，虽然$z_{l,h}$的维度写作$\mathbb{R}^{B \times T \times d_h}$，但在ScAle中T=1（仅保留last token），因此每个头每层只引入一个标量。

第二，注意力头粒度的缩放（每头一个标量）远优于MLP层粒度（每层一个标量）。实验表明ScAle(attn)以1024参数达到68.9%/68.6%/45.0%（Maze-Nav/Spatial-Grid/Spatial-Map），而ScAle(mlp)仅32参数只能达到44.0%/30.7%/38.0%。这说明空间推理的失效根因在于token间的注意力交互没有被正确校准（attention mixing出了问题），而非前馈网络的容量不足。每个注意力头捕捉不同的关系模式——有些头可能专注于空间关系，有些头处理语义/颜色/纹理——per-head scaling让模型能够选择性放大空间相关头的输出、抑制干扰头的输出，实现对"competing modes"假设的精准操作。

**3. 全层联合优化与L2正则：让各层协同校准**

一个自然的问题是：既然初步实验中单层缩放就能带来增益，为什么还要在所有层学习标量？答案是单层最优缩放因子是通过暴力搜索找到的（遍历-10到+10），而实际应用中无法对每个任务做这种搜索。通过在所有层放置可学习标量并用梯度下降联合优化，各层可以自动协调：某些层负责抑制干扰模式（eff接近0或负值），某些层负责放大空间信号（eff>1），形成分布式的校准模式。

可视化结果显示缩放因子在中间层最活跃——这与LLaMA式transformer中"中层做特征合成、深层做输出校准"的分工一致。具体来说，第10-20层（LLaVA-7B共32层）的attention head缩放值偏离1最远，早期层（第1-5层）和最后几层的缩放值接近1，说明浅层编码和后层决策都不需要大幅调制，空间推理的"竞合"主要发生在中间特征合成阶段。

### 损失函数 / 训练策略

训练目标：最大化正确答案的对数概率。给定样本$(I, q, c^*)$，模型前向得到logits $\ell_{\phi}(I, q)$，损失为：

$$\mathcal{L}(\phi) = \text{CE}(\ell_{\phi}(I, q), c^*) + \lambda_{attn}\sum_{l,h}\|u^{attn}_{l,h}\|_2^2 + \lambda_{MLP}\sum_l\|u^{MLP}_l\|_2^2$$

其中$u^{attn}_{l,h}$和$u^{MLP}_l$是各层各头/MLP的可学习标量参数，$\lambda_{attn} = \lambda_{MLP} = 10^{-4}$。L2正则鼓励标量接近0（即eff接近1，不做缩放），只在必要处才大幅偏离——这起到了自动特征选择的作用。

三种变体对应不同的ϕ集合：（1）ScAle(attn)：仅含$u^{attn}_{l,h}$，$\lambda_{MLP}=0$；（2）ScAle(mlp)：仅含$u^{MLP}_l$，$\lambda_{attn}=0$；（3）ScAle(both)：两者皆有。

优化器为AdamW，学习率$10^{-3}$，micro batch size=1（每步一个样本），训练5个epoch。每任务1500样本按固定随机种子划分为train/test，分别用5%/10%/20%（即75/150/300样本）训练，剩余做测试，确保所有方法在相同数据划分下比较。

## 实验关键数据

### 主实验

**SpatialEval on LLaVA-7B（表1）**：不同训练数据量下的方法对比。

| 方法 | 可训练参数 | Maze-Nav (20%) | Spatial-Grid (20%) | Spatial-Map (20%) |
|------|-----------|---------------|-------------------|------------------|
| Frozen | 0 | 31.7% | 29.3% | 22.7% |
| LoRA-all (rank=1) | 2.65M | 71.6% | 81.9% | 83.2% |
| (IA)³ | 663.6K | 70.8% | 84.8% | 80.7% |
| LoRA-last (rank=1) | 78.1K | 53.6% | 29.2% | 24.3% |
| ScAle (attn) | **1,024** | 68.9% | 68.6% | 45.0% |
| ScAle (mlp) | 32 | 44.0% | 30.7% | 38.0% |
| ScAle (both) | 1,056 | 69.7% | 75.3% | 45.3% |

ScAle(attn)用1024参数在Maze-Nav和Spatial-Grid上分别达到68.9%和68.6%，相对frozen基线提升117.3%和134.1%，逼近LoRA-all性能但参数少约2500倍。LoRA-last参数量多76倍但性能远不如ScAle(attn)（Maze-Nav仅53.6% vs 68.9%），说明"在哪改"比"改多少参数"更重要——只在最后一层加adapter远不如在全层做轻量激活调制。

**跨模型泛化（表2）**：20%训练数据下四种模型的SpatialEval结果。

| 方法 | LLaVA-7B | LLaVA-13B | Qwen2.5-VL-3B | Qwen2.5-VL-7B |
|------|----------|-----------|---------------|---------------|
| ScAle参数量 | 1.0K | 1.6K | 576 | 784 |
| Frozen (Maze-Nav) | 31.7% | 26.7% | 26.0% | 38.3% |
| ScAle attn (Maze-Nav) | 68.9% | 69.8% | 70.3% | 66.2% |
| Frozen (Spatial-Grid) | 29.3% | 36.7% | 63.7% | 73.7% |
| ScAle attn (Spatial-Grid) | 68.6% | 62.0% | 75.8% | 83.8% |
| Frozen (Spatial-Map) | 22.7% | 28.0% | 51.7% | 66.3% |
| ScAle attn (Spatial-Map) | 45.0% | 51.5% | 66.8% | 76.8% |

ScAle在四种模型、两种架构家族上一致有效，无需任何架构特定修改。Qwen系列基础空间推理能力更强（frozen基线更高），但ScAle仍能带来可观增益。Spatial-Map任务对所有方法都偏难——即使LoRA-all也在LLaVA-7B上仅达83.2%，说明地图内多实体间关系推理确实比位置识别和路径导航更复杂。

**真实空间推理与幻觉鲁棒性（表3）**：在WhatsUp-VLM（VGQA/COCOQA）和POPE上的表现。

| 方法 | LLaVA-7B VGQA | LLaVA-7B COCOQA | LLaVA-7B POPE | Qwen-3B VGQA | Qwen-3B COCOQA | Qwen-3B POPE |
|------|-------------|----------------|--------------|-------------|----------------|-------------|
| Frozen | 56.5% | 56.3% | 82.1% | 84.5% | 83.9% | 87.3% |
| ScAle (attn) | 92.9% | 91.5% | 88.2% | 93.5% | 95.0% | 90.4% |
| (IA)³ | 97.8% | 96.6% | 89.8% | 98.4% | 98.4% | 90.4% |

ScAle在真实空间VQA上带来35-36个绝对百分点的提升，在POPE幻觉基准上也有6个点的增益（LLaVA-7B从82.1%到88.2%），说明激活缩放不仅增强空间推理，还改善视觉grounding——两者可能共享某些被压制的视觉-语言对齐通路。

### 消融实验

| 配置 | 参数量 | Maze-Nav (20%) | Spatial-Grid (20%) | 关键结论 |
|------|--------|---------------|-------------------|---------|
| ScAle (attn) | 1,024 | 68.9% | 68.6% | 注意力缩放是核心贡献者 |
| ScAle (mlp) | 32 | 44.0% | 30.7% | MLP单独缩放效果有限 |
| ScAle (both) | 1,056 | 69.7% | 75.3% | MLP在Spatial-Grid上提供+6.7%增量 |
| 5%数据 (attn) | 1,024 | 66.7% | 51.7% | 75样本即可接近全量性能 |
| 10%数据 (attn) | 1,024 | 69.0% | 67.7% | 150样本基本饱和 |
| LoRA-QKV + ScAle | 935K | 78.9% | 85.8% | 混合方案以36%参数超越LoRA-all |

注意力缩放是关键贡献者——仅注意力缩放就基本达到both的性能。MLP缩放在Spatial-Grid上额外带来6.7%提升，说明部分空间推理信号也需要前馈网络的配合放大。数据量方面，即使在5%数据（仅75样本）下，ScAle(attn)仍能达到66.7%(Maze-Nav)和51.7%(Spatial-Grid)，展示出极强的少样本适应能力。ScAle与LoRA-QKV的混合方案以935K参数（LoRA-all的36%）在Maze-Nav上达到78.9%，超越LoRA-all的71.6%——再次证明注意力头是最值得投入参数预算的地方。

### 关键发现
- **注意力交互失校准是空间推理失败的主因**：注意力缩放远比MLP缩放有效，表明空间推理失败主要源于token间交互权重不当，而非前馈网络容量不足。这与论文的"competing modes"假说一致——问题不是缺少空间知识，而是空间信号在注意力混合中被其他模式淹没。
- **跨任务迁移能力超预期**：在Spatial-Grid上训练的ScAle适配器直接用于VGQA（84.5%→92.2%）和COCOQA（83.9%→90.9%），甚至提升GLUE/MRPC（66.7%→73.8%）和GLUE/SST2（Spatial-Map训练：65.8%→88.3%），说明学到的缩放模式捕捉了可迁移的空间结构而非任务特定捷径。更重要的是，没有观察到灾难性遗忘——空间推理的提升不以牺牲通用语言能力为代价。
- **与LoRA互补而非互斥**：ScAle + LoRA-QKV混合方案展示了"先诊断后投入"的工作流——用ScAle的head级缩放快速定位关键模块（注意力头），再将有限的参数预算集中投入这些模块的LoRA adapter，实现参数效率与绝对性能的帕累托最优。

## 亮点与洞察
- **"问题不在权重，在信号强度"的洞察非常漂亮**：初步实验用单层标量扫描就发现从22%到70%+的性能跃迁（热力图红蓝对比），直接推翻了"需要更新权重才能获得新能力"的默认假设。这本质上是一种"模型能力挖掘"而非"能力注入"——空间推理已经在模型里了，只是被埋没了。这个认知框架可以推广到其他VLM短板（如计数能力、深度估计），先问"信号是否被压制"再决定是否加参数。
- **Tanh bounded scaling是简单优雅的设计**：以1为中心、对称范围、饱和区天然正则化——每个特性都有明确的motivation。$s_{max}=10$这个值让缩放范围足够宽（(-9, 11)），能容纳从"完全翻转抑制"到"11倍放大"的调制需求，同时tanh在远端饱和也防止了梯度爆炸。
- **Last-token only是最被低估的设计决策**：如果不加这个约束，对所有序列token缩放会引入T倍参数膨胀和训练不稳定。只缩last token既精简又精准——last token本来就是选择题打分的唯一入口。这个设计的深层含义是：ScAle本质上是在"操控读出层（readout）的输入"，而非在"改变编码过程（encoding）"。
- **可迁移到任何transformer-based模型的通用Adapter范式**：因为ScAle只插入标量乘法、不改变任何模块的接口，理论上可插到任何有注意力头和MLP的transformer中（纯LLM、VLM、甚至扩散模型中的DiT backbone）。换一个下游任务（如数学推理、代码生成）是否能同样有效，是值得探索的方向。

## 局限与展望
- **Spatial-Map任务提升有限**：ScAle(attn)在Spatial-Map上仅从22.7%提升到45.0%（LLaVA-7B），远低于Maze-Nav的68.9%和Spatial-Grid的68.6%。作者未深入分析地图推理为何更难被激活缩放修复——可能地图关系需要多实体间的相对位置建模和多跳推理，仅靠last-token的幅度调制难以捕捉这种复杂的空间关系链。
- **仅验证了选择题场景**：所有实验都是四选一multiple-choice，模型只需从四个选项中挑最优，不需要生成空间描述或坐标。实际空间推理应用（如机器人导航输出坐标序列、AR场景描述物体空间关系）需要生成式评估，ScAle在这些开放式生成场景下是否有效完全未知。
- **$s_{max}=10$未做消融**：缩放范围的上下界直接影响调制自由度，但论文没有报告不同$s_{max}$值的对比实验。更大的$s_{max}$可能带来更好性能但也可能不稳定，更小的$s_{max}$可能不足以激活被深度抑制的空间模式。
- **可解释性探索可以更深**：论文可视化了学到的缩放因子热力图，观察到中层活跃的pattern，但没有进一步做因果分析——比如哪些具体的头在被缩放后改变了注意力pattern（可以像Hua et al.那样做head-level的功能归因），也没有分析被大幅缩放的头原本在关注什么。
- **具体改进思路**：（1）将ScAle扩展到生成式空间推理任务，评估开放式空间描述的准确性；（2）研究per-token scaling（而非仅last token）是否能进一步提升Spatial-Map类多跳空间推理任务；（3）结合activation steering技术，用对比prompt自动发现"空间promoting"的方向向量，与ScAle的标量缩放互补，形成"方向+幅度"的双重调制。

## 相关工作与启发
- **vs LoRA / (IA)³**：LoRA通过学习低秩增量矩阵$\Delta W = BA$来改变模型行为，(IA)³学习逐维度的缩放向量——两者本质上都在修改表示空间。ScAle的根本区别在于：它不改变表示的"方向"，只改变其"幅度"。这一定位使得ScAle的参数效率达到极致（比LoRA少约2500倍），但也决定了其表达能力上限——无法引入新的表示方向，只能重新加权已有的。这种"方向vs幅度"的二分法是理解ScAle与其他PEFT方法关系的关键框架。
- **vs Activation Steering**：SteerVLM等方法通过对比prompt学习steering vector加到残差流上，需要微调数百万参数（约占模型14%）。ScAle可以看作是一种"软化"的steering——不添加新的方向向量，而是通过标量选择性放大/抑制已有的方向。这更安全（不会引入训练时未见的行为模式），也更轻量（1K vs 数百万参数）。本质上，activation steering是在学"往哪推"，ScAle是在学"放大哪个已有的推"。
- **vs Mechanistic Interpretability的head patching**：Meng et al.和Hua et al.用激活缩放做因果分析——手动调某个头的缩放观察输出变化，以此定位功能头。ScAle把这个"分析工具"变成了"训练方法"——让梯度下降自动找到最优的缩放配置。这种"分析工具→训练方法"的转化思路值得借鉴：未来可能有更多interpretability方法被改造成efficient adaptation方法（如path patching→learnable path weights、logit lens→learnable early-exit classifiers）。
- **vs Prefix-tuning**：Prefix-tuning在输入端插入可学习的虚拟token序列（数千到百万参数），通过改变attention key/value来影响模型行为。ScAle与之相反——不改输入、不改KV、不改权重，只调残差流中已有信号的幅度。两种方法在"在哪干预"上有根本分歧：prefix-tuning认为在输入端加引导信号最有效，ScAle认为在中间层调信号强度最有效。可以推测两者互补——prefix-tuning设定任务上下文，ScAle校准模型内部对上下文的响应强度。

## 评分
- 新颖性: ⭐⭐⭐⭐ 将激活缩放从可解释性分析工具转化为PEFT方法的思路新颖，"不更新权重、只调信号强度"的哲学和主流PEFT形成鲜明对比。核心操作本身简单（标量乘激活），亮点在于问题定位（preliminary experiment热力图）和系统验证的完整度。
- 实验充分度: ⭐⭐⭐⭐ 覆盖4个模型、2种架构家族、3个SpatialEval子任务、2个真实VQA、POPE幻觉基准、GLUE语言理解、交叉任务迁移、LoRA混合方案——实验矩阵相当全面。缺少$s_{max}$消融和对Spatial-Map弱点的深入诊断是两个遗憾。WhatsUp-VLM和POPE上的表现有效缓解了对"只在合成benchmark上有效"的质疑。
- 写作质量: ⭐⭐⭐⭐ 从preliminary experiment引出motivation的叙事逻辑清晰，hotspot热力图非常有说服力。方法描述公式完整、notation一致。实验分析有insight（如"spatial reasoning failures arise from mis-calibrated token interactions"）。第4.5节Advantages略显重复，可精简。
- 价值: ⭐⭐⭐⭐⭐ 在边缘部署、多任务适配、联邦学习等参数/带宽受限场景中，1K参数的适配器有极强的实用价值。即使作为LoRA的补充（混合方案），也能以36%的参数超越全量LoRA。这种"先诊断再微调"的工作流（用ScAle快速找到关键模块→针对性加更大容量adapter）可能成为新的best practice。论文揭示的"模型能力挖掘而非能力注入"范式，对理解VLM的内部机制也有深远启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2026\] On Test-Time Scaling for Vision-Language Models](on_test-time_scaling_for_vision-language_models.md)
- [\[ECCV 2026\] Towards Spatial Trace with Reasoning in Vision-Language Models for Robotics](towards_spatial_trace_with_reasoning_in_vision-language_models_for_robotics.md)
- [\[ICLR 2026\] Pursuing Minimal Sufficiency in Spatial Reasoning](../../ICLR2026/vlm_reasoning/pursuing_minimal_sufficiency_in_spatial_reasoning.md)
- [\[ICLR 2026\] InternSpatial: A Comprehensive Dataset for Spatial Reasoning in Vision-Language Models](../../ICLR2026/vlm_reasoning/internspatial_a_comprehensive_dataset_for_spatial_reasoning_in_vision-language_m.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](../../ICLR2026/vlm_reasoning/omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)

</div>

<!-- RELATED:END -->
