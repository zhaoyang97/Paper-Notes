# Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration

**会议**: AAAI 2026  
**arXiv**: [2411.17686](https://arxiv.org/abs/2411.17686)  
**代码**: [https://github.com/kawhiiiileo/FiCoCo](https://github.com/kawhiiiileo/FiCoCo)  
**领域**: 多模态VLM / 模型压缩  
**关键词**: 视觉token压缩, MLLM加速, training-free, 信息回收, 冗余度量  

## 一句话总结
提出FiCoCo三阶段框架（Filter-Correlate-Compress），通过集成视觉感知+语义感知冗余度量筛选丢弃token，利用token间相关性自适应回收信息，实现training-free的MLLM加速。在LLaVA-NeXT上达14.7×FLOPs压缩同时保留93.6%性能，在5种MLLM架构上全面超越FastV、SparseVLM等SOTA。

## 背景与动机
MLLM中视觉token数量远超文本token，且自然视觉信号固有冗余度高。prefilling阶段占总延迟高达80%（如Qwen2-VL），成为部署瓶颈。现有training-free token reduction方法存在两个问题：(1) 使用单一冗余度量（如attention score）判断token重要性可能不准确——FastV等方法混淆了"可预测性"和"重要性"；(2) 被丢弃的token中可能仍包含对任务有益的信息，简单剪枝造成不可逆信息损失。

## 核心问题
如何在不需要重训练的情况下，精确识别MLLM中的冗余视觉token，并在丢弃过程中最大程度地保留有用信息？核心挑战是：(1) 单一指标无法全面衡量冗余性；(2) 被丢弃token中的有用信息需要重新分配到保留token中，但naive averaging会稀释保留token的核心内容。

## 方法详解

### 整体框架
三阶段流水线：Filter→Correlate→Compress，在每一层逐步压缩视觉token。FiCoCo-V在视觉编码器中操作（task-agnostic），FiCoCo-L在LLM解码器中操作（task-aware，利用文本先验）。两种变体可独立使用，实现不同精度-效率权衡。

### 关键设计
1. **Redundancy-based Token Discard (Filter)**: 集成两种冗余度量：Vision-aware redundancy（一个token从其他patch token接收的平均attention越高→越冗余，因为它的信息可被邻居预测）+ Semantic-aware redundancy（与[CLS] token的attention越低→语义贡献越少→越冗余）。核心insight颠覆了FastV的假设：**高attention-in = 高可预测性 = 高冗余**，而非"高重要性"。还设计了**local penalty策略**：将2D grid分割为窗口，惩罚同一窗口内的高冗余token，避免空间集中丢弃。

2. **Correlation-based Information Recycling (Correlate)**: 不用固定K值选择目标token，而是**token-adaptive K**——对每个被丢弃token，计算其与所有保留token的correlation（attention在vision encoder中/attention+indirect text correlation在LLM decoder中），用quantile阈值自适应确定相关token数量。不同层、不同token的最优K差异极大（Pairwise Token Correlation Gap分析证明）。

3. **Self-preserving Compression (Compress)**: 保留token更新时，保证至少保留50%自身信息（$\mathbf{X}_j^\mathbb{T} \leftarrow \frac{\mathbf{X}_j^\mathbb{T} + \sum_{i \in \mathbb{I}_j} \alpha_{ij} \mathbf{X}_i^\mathbb{S}}{1 + \sum_{i \in \mathbb{I}_j} \alpha_{ij}}$），且从高相关的丢弃token中回收更多信息（$\alpha_{ij}$按correlation归一化）。"dense"信息路径（多对多）优于"convergent"（多对一）。

4. **FiCoCo-L的task-aware增强**: 在LLM decoder中，利用视觉token与文本token的attention作为任务相关性指标（替代语义冗余），同时引入indirect semantic correlation——通过文本token作为桥梁测量两个视觉token间的间接语义关联。

### 损失函数 / 训练策略
完全training-free，仅需不到10行额外代码即可即插即用。FiCoCo-V从ViT第12层开始压缩，FiCoCo-L从LLM第4层开始。超参数$\lambda=0.35$, $\beta=0.6$, $\gamma=0.6$, $\varepsilon=0.998$。

## 实验关键数据

| 模型 | 方法 | TFLOPs | Avg Acc | Avg(%) |
|--------|------|--------|---------|--------|
| LLaVA-1.5-7B | Vanilla | 8.5 | 70.3 | 100% |
| LLaVA-1.5-7B | SparseVLM | 1.5 | 61.0 | 86.8% |
| LLaVA-1.5-7B | FiCoCo-V | 1.5 | 65.2 | **92.7%** |
| LLaVA-1.5-7B | FiCoCo-L | 1.5 | 65.3 | **92.8%** |
| LLaVA-NeXT-7B | Vanilla | 42.7 | 58.6 | 100% |
| LLaVA-NeXT-7B | PDrop(5.0T) | 5.0 | 53.8 | 91.7% |
| LLaVA-NeXT-7B | FiCoCo-L(2.9T) | 2.9 | 54.9 | **93.6%** |
| Video-LLaVA | FiCoCo-V | 2.6 | 50.3 | **92.8%** |

效率：LLaVA-NeXT上throughput提升2.08×(FiCoCo-V)，KV-Cache减少80%，GPU显存减少36%。

### 消融实验要点
- Vision-aware redundancy（+VR）显著优于用attention作为重要性指标（-VR），证实"高attention-in=高冗余"假说
- Semantic-aware redundancy影响更大（去掉后SQA -3.7, TextVQA -6.7）
- token-adaptive K > fixed K=0(pruning) > fixed K=1/2（固定K导致信息过稀释/噪声）
- Self-preserving compression >> naive averaging（TextVQA +2.2%）
- Local penalty在FiCoCo-V中有效（+2.2 TextVQA），但在FiCoCo-L中反而略有负面效果
- 无[CLS]时用Key均值替代，仅损失0.16%性能

## 亮点
- **"attention-in as redundancy"的信息论视角**非常精辟——颠覆了FastV等方法"高attention=高重要性"的假设，理论和实验双重验证
- **token-adaptive K**的设计很有洞察力——Pairwise Token Correlation Gap分析清晰展示了固定K的不足
- **self-preserving compression**保证保留token至少50%自身信息，简洁但有效
- training-free + <10行代码 + 可跨5种MLLM架构直接使用，实用性极强
- 视频理解也适用（Video-LLaVA 92.8%保留率），不仅限于图像

## 局限性 / 可改进方向
- 极端压缩下（90%+ token丢弃）Fine-grained理解任务（如OCR）仍有明显退化
- FiCoCo-V是task-agnostic的，在需要精确文本识别的场景中不如FiCoCo-L
- 超参数虽然不太敏感，但不同MLLM架构仍需微调
- 未探索与training-based方法（如TokenPacker）的组合

## 与相关工作的对比
- **vs FastV**: FastV用attention作为重要性→不准确，FiCoCo用attention-in as redundancy+[CLS] attention双指标→更准确，且FastV只做pruning无信息回收
- **vs SparseVLM**: SparseVLM只用text-visual attention剪枝，FiCoCo集成多维度冗余+自适应信息回收，极端压缩下FiCoCo-V比SparseVLM高5.9%
- **vs ToMe**: ToMe用fixed K=1→信息稀释严重；FiCoCo用token-adaptive K+self-preserving compression→保留核心信息

## 启发与关联
- 与 `ideas/model_compression/20260316_task_aware_token_compression.md` 高度相关——FiCoCo的task-aware FiCoCo-L设计正是这个方向的实现
- "attention-in as redundancy"的视角可以推广到ViT pruning、视频理解等场景
- 信息回收机制可能可以与EM-KD的Hungarian匹配结合——在蒸馏中也做被丢弃token的信息重分配
- **Idea触发**：FiCoCo的冗余指标和回收机制是静态的（逐层独立），能否设计一个全局跨层的token importance预测器？

## 评分
- 新颖性: ⭐⭐⭐⭐ "attention-in as redundancy"和adaptive-K信息回收是新颖的，整体框架设计elegant
- 实验充分度: ⭐⭐⭐⭐⭐ 5种MLLM+图像/视频+6档压缩率+极其详尽的消融
- 写作质量: ⭐⭐⭐⭐⭐ 三阶段命名直观，分析深入，可视化出色
- 价值: ⭐⭐⭐⭐⭐ training-free+即插即用+跨架构泛化，实用价值极高
