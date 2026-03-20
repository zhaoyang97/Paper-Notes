# ApET: Approximation-Error Guided Token Compression for Efficient VLMs

**会议**: CVPR 2026  
**arXiv**: [2602.19870](https://arxiv.org/abs/2602.19870)  
**代码**: [https://github.com/MaQianKun0/ApET](https://github.com/MaQianKun0/ApET)  
**领域**: 多模态VLM / 模型加速 / Token压缩  
**关键词**: 近似误差, 信息论, 线性近似, FlashAttention兼容, 无注意力token压缩  

## 一句话总结
从信息论角度出发，通过线性近似重建每个visual token并用重建误差衡量其信息量（误差大=信息多=应保留），提出完全不依赖注意力权重的ApET框架，在LLaVA-1.5-7B上88.9%压缩保留95.2%精度，视频任务甚至达100.4%超基线，且完全兼容FlashAttention。

## 背景与动机
现有VLM token压缩方法（SparseVLM、PyramidDrop等）依赖注意力权重评估token重要性，存在两大问题：(1) **位置偏差**——LLM的注意力对靠近文本的视觉token(序列后端)系统性给予更高权重，与实际信息量无关；(2) **与FlashAttention不兼容**——FlashAttention不输出注意力矩阵，需要额外计算注意力权重反而增加开销。实验证实：在Qwen2.5-VL上，注意力引导的压缩方法反而比直接用FlashAttention的baseline更慢。

## 核心问题
能否完全抛弃注意力信号，用token的内在信息量（可重建性）作为压缩依据，同时保持与FlashAttention的完全兼容？

## 方法详解

### 整体框架
ApET可在视觉编码器输出和LLM中间层（如第16层）两个位置插入。每个位置执行三步：(1) **Token选择**：用FPS从所有visual token中采样$M$个basis token；(2) **近似误差计算**：用basis token线性近似所有token $v' \approx \sum \alpha_i b_i$，计算重建误差$\xi = \|v - v'\|_2$；(3) **Token合并**：按误差排序保留高误差token，低误差token与最近似的已保留token合并（average merging）。

### 关键设计
1. **信息论基础**：从互信息最大化出发，$\max_S I(V;S) = H(V) - H(V|S)$。由于$H(V)$固定，目标是最小化$H(V|S)$。Shannon定理提供下界：$\frac{1}{2\pi e}\exp(\frac{2H(V|S)}{d}) \leq \xi$，即最小化重建MSE等价于最小化条件熵。因此**重建误差大的token包含更多无法被子集表达的独特信息**，应该被保留。

2. **线性近似替代重建模型**：不需要训练额外的重建网络，直接用$V \approx BA$求解线性系统（$B$是basis token集，$A$是系数矩阵）。计算开销极低：$M=10$个basis token即可，$M \ll N$(576)。FPS采样确保basis token的多样性，DPC也可用但计算量更大。

3. **Token合并策略**：低误差token不是直接丢弃，而是与最相似的高误差token合并（average merging），减少信息损失。basis token自动保留在保留集中，确保近似基底不丢失。

### 损失函数 / 训练策略
完全training-free。在视觉编码器输出和LLM第16层（LLaVA系列）或第14层（Qwen2.5-VL）处进行两次压缩。$M=10$作为默认值（对$M$不敏感，5-20范围内性能变化<2%）。代码开源。

## 实验关键数据
**LLaVA-1.5-7B（9个基准平均）**：

| 保留token | ApET | VisionZip | PDrop | SparseVLM | FastV |
|--------|------|------|----------|------|------|
| 192 (33%) | **98.0%** | 97.8% | 97.2% | 96.1% | 90.4% |
| 128 (22%) | **97.1%** | 96.2% | 96.2% | 93.7% | 85.4% |
| 64 (11%) | **95.2%** | 92.7% | 86.6% | 87.2% | 76.7% |

**Video-LLaVA（256 tokens, 87.5%↓）**：ApET **100.7%**反超基线 vs VisionZip 94.4% vs FastV 83.9%

**Qwen2.5-VL-7B（20%保留率）**：ApET 93.3% vs PDrop 90.3% vs SparseVLM 89.8%

**效率（LLaVA-1.5-7B, 11.1%保留率）**：总推理1.46×加速，prefill 1.38×加速

### 消融实验要点
- **FPS最优采样策略**：FPS ≈ DPC >> Random（Random也能工作说明近似误差本身有效）
- **$M$对结果不敏感**：$M \in [5,20]$范围内POPE变化<2%，$M=10$最优
- **关键优势在极端压缩**：64 tokens(11%)时ApET vs VisionZip差距2.5%，vs PDrop差距8.6%
- **视频优势突出**：因消除了注意力的位置偏差（长视频序列中位置偏差更严重）
- **在Qwen2.5-VL上其他方法变慢**：注意力方法需重新计算权重比baseline更慢，而ApET 1.19×加速

## 亮点
- **信息论视角独特**——从Shannon条件熵→重建MSE→近似误差的推导链清晰优雅
- 完全不依赖注意力=完全兼容FlashAttention=在现代VLM上真正实用
- 极致简洁：仅需10个basis token做线性近似+L2误差+FPS采样，整个方法几行代码
- 视频理解超基线（100.7%）进一步验证了"冗余token有害"的假设——去噪效果
- 与V2Drop（变化量视角）、GACD（梯度视角）形成互补的"token重要性评估三角"

## 局限性 / 可改进方向
- 线性近似可能不足以捕捉非线性特征关系——更强的近似方法可能更准确
- FPS采样引入$O(NM)$额外计算（不过$M$很小所以实际开销极低）
- 压缩后token合并用简单average——weighted average或attention-based merge可能更好
- 未与V2Drop等基于变化量的方法直接对比——两种信号（近似误差 vs 层间变化量）的互补性值得探索
- 未在训练中使用（仅推理时）——类似DUET-VLM的训练+推理统一可能进一步提升

## 与相关工作的对比
- **vs V2Drop (CVPR'26)**：V2Drop用层间变化量（也不需要注意力，也兼容FA），ApET用近似误差。两者思路类似但信号来源不同——V2Drop看"token在层间变了多少"，ApET看"token能被其他token代表多少"
- **vs VisionZip (CVPR'25)**：VisionZip用CLS注意力选dominant token+全局合并，ApET用近似误差+FPS+局部合并，在64 tokens时95.2% vs 92.7%
- **vs DUET-VLM (CVPR'26)**：DUET-VLM双阶段（视觉侧聚类+语言侧注意力剪枝），ApET单一原则（近似误差）但在两处应用。DUET-VLM可训练，ApET纯推理
- **vs GACD (CVPR'26)**：GACD用梯度做token贡献估计缓解幻觉，ApET用近似误差做效率压缩——侧重点不同但都避免了直接注意力依赖

## 启发与关联
- ApET的"可被其他token线性重建=冗余"与V2Drop的"层间变化小=不重要"可能可以**统一成一个框架**：近似误差度量token的内在信息量（静态），变化量度量token被网络利用程度（动态），两者乘积可能是最佳的token重要性信号
- 线性近似的idea可以推广到**KV cache压缩**——用少量basis KV线性近似其余KV，按近似误差选择保留哪些KV

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 信息论+线性近似误差的框架完全原创，与所有注意力方法正交
- 实验充分度: ⭐⭐⭐⭐⭐ 4个模型(LLaVA/LLaVA-NeXT/Video-LLaVA/Qwen2.5-VL)、9+基准、效率分析、多种消融
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，motivation→theory→method→experiment逻辑链完美
- 价值: ⭐⭐⭐⭐⭐ 信息论新视角+FlashAttention兼容+开源=对VLM压缩领域影响大
