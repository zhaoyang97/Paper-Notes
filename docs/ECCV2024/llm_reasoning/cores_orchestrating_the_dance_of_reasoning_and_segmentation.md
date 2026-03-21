# CoReS: Orchestrating the Dance of Reasoning and Segmentation

**会议**: ECCV 2024  
**arXiv**: [2404.05673](https://arxiv.org/abs/2404.05673)  
**代码**: https://chain-of-reasoning-and-segmentation.github.io (有)  
**领域**: LLM推理  
**关键词**: reasoning segmentation, chain-of-thought, MLLM, multi-modal reasoning, SAM

## 一句话总结
提出 CoReS（Chains of Reasoning and Segmenting），一种双链结构的多模态思维链框架，通过推理链和分割链的层次化协作，结合 in-context 引导策略，实现对复杂推理文本中目标物体的渐进式精确分割，在 ReasonSeg 数据集上超越 LISA 6.5%。

## 研究背景与动机
1. **领域现状**：推理分割（reasoning segmentation）是一类新兴任务，要求模型理解复杂推理查询并精确定位和分割目标区域。现有方法主要分两类：为 MLLM 配备分割解码器（如 LISA 将 SAM 接入 LLaVA）或直接让 LLM 以文本形式输出 mask（如 VistaLLM）。
2. **现有痛点**：现有 MLLM 在对象级别分割效果不错，但在处理推理文本中的精确定位时表现挣扎。例如分割"给狗提供嗅觉的部位"时，LISA 直接搜索"圆形、暗色、感知重要"的区域，结果错误地分割了狗的眼睛而非鼻子——因为两者在特征上有高度语义相似性。
3. **核心矛盾**：推理分割需要同时具备复杂推理能力和精确定位能力。MLLM 有推理能力但直接用 [SEG] token 一步到位地定位容易受语义相似实例干扰。
4. **本文要解决什么？**：如何让 MLLM 模仿人类视觉搜索的认知过程——从粗到细、逐步缩小范围——来完成复杂推理分割任务。
5. **切入角度**：借鉴人类视觉搜索的自顶向下认知过程——先根据先验知识定位大致区域，再逐步聚焦到精确目标。例如找"婚礼信物"，人会先找手的位置，再在手的局部区域找到戒指。
6. **核心idea一句话**：将 LLM 的 chain-of-thought 从纯文本推理扩展为多模态双链结构（推理链 + 分割链），通过层次化的语义逻辑引导逐步精确的分割。

## 方法详解

### 整体框架
输入：图像 $Q_{img}$ + 查询文本 $Q_{text}$ → MLLM（LLaVA）处理图文输入，生成符合层次化逻辑的推理链输出（包含 [LOC] 和 [SEG] 两个特殊 token）→ [LOC] 和 [SEG] 的 token embedding 作为分割链的引导，驱动 SAM 框架逐层生成分割结果 → 输出：最终精确的目标分割 mask。此外，随机采样的 in-context 文本示例作为额外输入，引导 MLLM 生成符合层次逻辑的输出。

### 关键设计

1. **Chain-of-Reasoning（推理链）**:
   - 做什么：约束 MLLM 的输出遵循自顶向下的语义逻辑层次
   - 核心思路：使用响应模板 "It appears on [LOC]. It is [SEG]." 约束 MLLM 输出的句子结构。这迫使 MLLM 在不同 token 位置注入不同层次的信息——[LOC] 位置注入目标最可能出现的场景或物体信息（如"刀或螺旋桨"），[SEG] 位置注入目标本身的特征信息（如"平坦的切割或推进面"）。通过 LoRA 微调 MLLM，使用交叉熵损失 $\mathcal{L}_{CoR} = \mathcal{L}_{CE}(\mathbf{p}(H|Q_{img}, Q_{text}), \mathbf{t})$ 监督推理链输出
   - 设计动机：仅靠单个 [SEG] token 无法捕获层次化的推理逻辑；通过模板化输出结构，强制模型在不同位置编码不同粒度的语义信息，形成从粗到细的推理链

2. **Chain-of-Segmenting（分割链）**:
   - 做什么：利用推理链的逻辑引导，迭代生成从粗到精的分割结果
   - 核心思路：从 MLLM 最后一层提取 [LOC] 和 [SEG] 位置的 embedding（$\mathbf{h}^0, \mathbf{h}^1 \in \mathbb{R}^{1\times256}$），依次作为 SAM mask decoder 的文本提示。分割链是迭代的：前一层级的分割结果 $m^{t-1}$ 通过 SAM prompt encoder 处理后作为下一层级的 mask prompt（$\mathcal{M}^t = \theta(m^{t-1})$），引导分割逐步精化。最终分割过程为 $m^t = \gamma(F_v(Q_{img}), \mathcal{M}^t, \hat{\mathbf{h}}^t)$
   - 设计动机：分割链将推理链的语义层次直接映射到视觉模态——第一层根据 [LOC] 做场景级定位，第二层根据 [SEG] 在定位区域内做精确分割

3. **Token Refiner（token 精化器）**:
   - 做什么：利用前一层的视觉分割结果反向校准下一层的文本 token embedding
   - 核心思路：使用 masked average pooling（MAP）从前一层 mask 对应的视觉特征中创建原型，然后通过 cross-attention 校准下一层的 token embedding：$\hat{\mathbf{h}}^t = \mathcal{R}(\beta(\mathbf{h}^t), F_v(Q_{img}), m^{t-1})$，其中 $\mathcal{R}(h, i, m) = h + CA(h, MAP(i, m))$
   - 设计动机：MLLM 输出的 token 是一次前向传播生成的，缺乏视觉信息的反馈；token 精化器让视觉模态的中间结果反向增强文本引导，实现双模态层次化的真正交互

4. **In-Context Guidance（上下文引导）**:
   - 做什么：在训练和推理时为 MLLM 提供逻辑规则的示范
   - 核心思路：预构建一个纯文本的 context library（由 ChatGPT 根据手动样例生成更多 QA 对）。每次前向传播时随机抽取与当前查询无关的文本 QA 示例，作为 in-context 输入放在用户输入之前。这些示例不包含特殊 token，仅以自然语言描述"物体通常出现在哪里"和"物体的常见特征"，隐式传达了自顶向下的输出逻辑规则
   - 设计动机：仅用模板约束句子结构不足以让 MLLM 主动发现和输出层次化的逻辑关系。通过 in-context 示例作为"规则提供者"，MLLM 从纯文本上下文中提取自顶向下的规则并迁移到多模态任务输出中。这种方式避免了构建额外多模态 CoT 数据集或使用两阶段 MLLM 的高计算代价

### 损失函数 / 训练策略
- **推理链损失**：$\mathcal{L}_{CoR} = \mathcal{L}_{CE}(\mathbf{p}(H|Q_{img}, Q_{text}), \mathbf{t})$，标准交叉熵损失监督文本输出
- **分割链损失**：$\mathcal{L}_{CoS} = \lambda_d \mathcal{L}_{DICE}(m^T, M_{gt}) + \lambda_c \mathcal{L}_{CE}(m^T, M_{gt})$，仅对最终层 mask 做监督（$\lambda_d=0.5, \lambda_c=2.0$）
- **总损失**：$\mathcal{L}_{total} = \lambda_R \mathcal{L}_{CoR} + \lambda_S \mathcal{L}_{CoS}$，$\lambda_R = \lambda_S = 0.5$
- 训练策略：LoRA 微调 MLLM（LLaVA-7B-v0）；SAM Image Encoder 和 Prompt Encoder 冻结以保持泛化能力；投影层和 mask decoder 可训练
- 训练数据：follow LISA，混合使用语义/参考/推理分割数据集，不使用 VQA 数据

## 实验关键数据

### 主实验

| 方法 | MLLM | ReasonSeg val gIoU | ReasonSeg test gIoU |
|------|------|---------------------|---------------------|
| OVSeg | CLIP ViT-L | 28.5 | 26.1 |
| GRES | BERT | 22.4 | 21.3 |
| X-Decoder | UniCL | 22.6 | 21.7 |
| SEEM | UniCL | 25.5 | 24.3 |
| LISA | LLaVA-7B | 44.4 | 36.8 |
| **CoReS** | LLaVA-7B | **54.8** | **48.7** |
| LISA (ft) | LLaVA-7B | 52.9 | 47.3 |
| **CoReS (ft)** | LLaVA-7B | **59.4** | **52.4** |
| LISA (ft) | LLaVA-13B | 56.2 | 51.7 |
| **CoReS (ft)** | LLaVA-13B | **61.8** | **55.9** |
| LISA (ft) | LLaVA-v1.5-13B | 65.0 | 61.3 |
| **CoReS (ft)** | LLaVA-v1.5-13B | **68.1** | **65.5** |

### 消融实验

| InC | CoR | CoS | CoS-R | gIoU | cIoU |
|-----|-----|-----|-------|------|------|
| | | | | 52.9 | 54.0 |
| ✓ | | | | 53.2 | 55.4 |
| | ✓ | | | 54.0 | 54.1 |
| | | ✓ | | 53.9 | 55.6 |
| | ✓ | ✓ | | 55.3 | 56.9 |
| | ✓ | | ✓ | 55.4 | 58.0 |
| | | ✓ | ✓ | 56.9 | 59.8 |
| | ✓ | | ✓ | 57.5 | 60.4 |
| ✓ | ✓ | ✓ | | 58.4 | 59.3 |
| ✓ | ✓ | | ✓ | **59.4** | **62.1** |

In-context input 数量消融：1个示例最优（59.4 gIoU），2个和4个略微下降但仍优于不用（57.5）

逻辑层级消融：2层（[L]+[S]）= 59.4 vs 3层（[L]+[P]+[S]）= 59.7，提升有限但远优于1层（[S]）= 53.2

### 关键发现
1. **未微调的 CoReS 超越微调的 LISA**：未使用 ReasonSeg 微调的 CoReS（54.8 gIoU）已经超过使用 ReasonSeg 微调的 LISA（52.9 gIoU），说明性能提升来自更好地挖掘 MLLM 潜力而非依赖训练数据
2. **双链协作是核心**：CoR 引导下的 CoS-R（带 token 精化器）贡献了 +4.6% 的主要提升，证明双模态层次化交互的有效性
3. **In-context 引导是额外增益**：在双链基础上加 in-context input 再提升 +1.9%，证明逻辑规则引导的效果
4. **在 refCOCOg 上优势更明显**：refCOCOg 的查询文本更长更复杂（平均 8.4 词），CoReS 的提升更显著（+2%），印证了多模态思维链主要增强复杂查询场景
5. **2层逻辑已足够**：3层结构仅微小提升，ReasonSeg 的问题复杂度不需要更深的逻辑链，且更长链条带来反向传播困难

## 亮点与洞察
1. **首创多模态 CoT 用于密集预测**：将 chain-of-thought 从视觉理解任务扩展到分割等密集预测任务，是 CoT 应用范围的重要突破
2. **In-context 规则迁移**：文本域的 in-context 示例能将输出逻辑规则迁移到多模态任务，无需构建多模态 CoT 数据集，非常实用
3. **Token Refiner 实现跨模态反馈**：视觉模态的中间结果反向校准文本 token，构建了真正的双向交互，而非单向的文本→视觉引导
4. **思路的普适性**：自顶向下的视觉层次搜索思路可以推广到其他需要精确定位的多模态任务（如 grounding、检测）

## 局限性 / 可改进方向
1. **逻辑层级深度受限**：3层结构带来的提升有限，可能是因为 ReasonSeg 数据集本身不够复杂。更复杂的推理任务可能需要自适应决定层级深度
2. **In-context library 质量瓶颈**：随着示例数量增加性能不升反降，暗示 context library 的质量和多样性不足。可考虑检索增强（根据查询相似度选择 in-context 示例）
3. **仅在 ReasonSeg 上验证**：推理分割任务的基准目前较少，方法的泛化性需要更多数据集验证
4. **中间层 mask 无监督**：分割链中间层级没有 ground-truth 监督，可能导致中间结果不稳定；可探索弱监督或自监督的中间层training
5. **计算开销**：双链结构 + in-context input 增加了输入长度和计算量，对部署不友好

## 相关工作与启发
- **LISA** [Lai et al., 2023]：CoReS 的直接基线，将 SAM 接入 LLaVA 做推理分割。CoReS 在此基础上加入层次化推理结构
- **V\*** [Wu et al., 2024]：利用 LLM + MLLM 构建视觉搜索算法，超越 GPT-4V。启发了自顶向下视觉搜索的思路
- **KAM-CoT** [Mondal et al., 2024]：用知识图谱辅助多模态 CoT 训练，减少计算成本。启发了低成本 CoT 构建
- **启发**：可以将 CoReS 的双链结构思路迁移到 3D 场景理解（先定位房间再定位物体）或视频理解（先定位关键帧再定位目标）

## 评分
- ⭐⭐⭐⭐ 新颖性：多模态 CoT 从理解到密集预测的扩展有开创性，双链 + in-context 引导设计精巧
- ⭐⭐⭐⭐ 实验充分度：ReasonSeg 主实验详实，消融全面覆盖各组件，指标提升显著
- ⭐⭐⭐⭐ 写作质量：论文结构清晰，motivation 用 "找狗鼻子" 的例子很直观
- ⭐⭐⭐⭐⭐ 价值：为推理分割提供了一种系统化的框架思路，in-context 规则迁移的思想有广泛适用性
