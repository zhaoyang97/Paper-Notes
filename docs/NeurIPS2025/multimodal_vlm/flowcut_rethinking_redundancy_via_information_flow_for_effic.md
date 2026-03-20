# FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.19536](https://arxiv.org/abs/2505.19536)  
**代码**: [https://github.com/TungChintao/FlowCut](https://github.com/TungChintao/FlowCut)  
**领域**: 模型压缩 / 多模态VLM  
**关键词**: 视觉token剪枝, 信息流, VLM效率, 注意力分析, Training-free  

## 一句话总结
从信息流（Information Flow）视角重新理解VLM中视觉token的冗余性：发现CLS token是信息中继站、冗余渐进式涌现、单层单标准评分不够可靠，提出FlowCut——基于信息流感知的多标准累积重要性剪枝框架，在LLaVA-1.5-7B上以88.9%的token减少率超越SOTA 1.6%，在LLaVA-NeXT-7B上超越4.3%。

## 背景与动机
VLM因大量视觉token导致计算开销高。已有方法用单层注意力分数做token重要性排序和剪枝（如FastV、SparseVLM），但作者提出一个根本性质疑：token间和层间的交互如此复杂，仅看单层注意力分数是否足以识别冗余？

## 核心问题
从信息流角度理解视觉token冗余的本质是什么？如何设计与模型固有信息传播行为对齐的剪枝策略？

## 方法详解

### 整体框架
三大创新模块：(1) 基于注意力熵的层自适应剪枝比例；(2) 多标准评分策略（注意力强度+信息密度+语义相关性）；(3) 累积流重要性跟踪。

### 关键发现（信息流分析）

1. **Insight 1 — CLS token是信息中继站**: 通过分析ViT各层的信息流入/流出，发现patch token在浅层主要关注邻近token和CLS token，在深层则关注远距离的"hub token"。CLS token充当全局信息广播器——它先从所有patch token收集信息，再传递给每个patch token。因此**CLS token的注意力可以作为全局信息流的代理**。

2. **Insight 2 — 冗余渐进式涌现**: CLS token的注意力分布随层加深而逐步集中（熵递减），冗余不是静态属性而是在编码过程中层层涌现。第11-15层熵急剧下降。

3. **Insight 3 — 单标准不可靠**: CLS高度关注的token可能信息密度很低（Value L1范数小）或语义相关性低（与CLS余弦相似度低）——不同标准会给出矛盾的重要性排序！

### 关键设计

1. **层自适应剪枝比例**: 使用注意力熵指导每层的剪枝强度——熵低（注意力集中→冗余多）的层剪枝更激进，熵高的层保守剪枝。不再用固定的per-layer ratio。

2. **多标准融合评分**: 综合三个维度评估token重要性：
   - **注意力强度**: CLS token对该token的注意力分数
   - **信息密度**: Value向量的L1范数（信号强度）
   - **语义相关性**: 与CLS token的余弦相似度（全局语义相关性）

3. **累积流重要性跟踪**: 不只看当前层，而是跨层累积重要性：`S_cum^(l) = 0.5 × I_cur^(l) + 0.5 × S_cum^(l-1)`。每2层剪枝一次，确保历史和当前信息都被考虑。

### 损失函数 / 训练策略
- 完全无需训练（training-free），推理时即插即用
- 剪枝在ViT的中间层执行

## 实验关键数据

**LLaVA-1.5-7B（保留64个token，原始576个，↓88.9%）**:

| 方法 | 平均准确率 (相对100%) |
|------|---------------------|
| Vanilla (576 tokens) | 100% |
| FastV | 77.5% |
| SparseVLM | 84.6% |
| PDrop | 86.1% |
| VisionZip | 94.4% |
| **FlowCut** | **96.0%** (+1.6%) |

**LLaVA-NeXT-7B（保留32个token，原始2880个，↓94.4%）**: FlowCut 91.9% vs VisionZip 87.6% (+4.3%)

- 保留192个token（↓66.7%）时：FlowCut达到100.2%——**甚至超过原始模型**！
- prefilling阶段3.2×加速
- 在InternVL2和Qwen2-VL上也展示了泛化性

### 消融实验要点
- **多标准 vs 单标准**: 多标准评分比仅用attention显著更好
- **累积 vs 单层**: 累积重要性跟踪提升2-3%
- **自适应比例 vs 固定比例**: 自适应剪枝比固定ratio高1-2%
- **三个标准的贡献**: 注意力强度是基础，信息密度和语义相关性各贡献约0.5-1%提升

## 亮点
- **理论驱动**: 从信息流的基本视角出发，三个insight层层递进，分析→设计水到渠成
- **保留64 token达96%**: 88.9%的token减少率下仍保持96%相对性能——在同等压缩率下远超所有方法
- **超越原模型的可能**: 192 token时100.2%——剪枝后性能甚至微超原模型，说明冗余token可能有害
- **CLS作为信息中继的发现**: 对ViT内部机制的理解有独立学术价值

## 局限性 / 可改进方向
- 仅在ViT架构的视觉编码器上分析信息流，其他架构（如SigLIP without CLS）尚未验证
- 多标准的权重（0.5/0.5）是手动设置的，可以学习自适应权重
- 未与KV-Latent等维度级压缩方法结合
- prefilling加速3.2×，但decoding阶段的加速未报告（KV cache大小也会减少）
- 评估以理解任务为主，生成任务（如图像描述）的影响程度未充分测试

## 与相关工作的对比
- **vs FastV**: FastV用第2层attention做一次性剪枝，FlowCut跨层累积+多标准，大幅领先
- **vs VisionZip**: VisionZip也是strong baseline但只用attention+少量固定保留，FlowCut自适应+多标准更优
- **vs mPLUG-DocOwl2 (cross-attention压缩)**: DocOwl2需要额外的压缩模块和训练，FlowCut完全无需训练

## 启发与关联
- 信息流分析框架可以推广到LLM的token级分析——不仅是视觉token，文本token也可能存在类似的信息集中现象
- 层自适应剪枝比例的思路与TrimLLM（层级压缩）互补——FlowCut在层内做token剪枝，TrimLLM在层间做层删除
- "CLS作为信息中继"的发现可以用于KV-Latent——维度降采之后，CLS token的KV可以保持更高维度

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 信息流视角和三个insight都是该领域的重要理论贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 4个VLM架构、12个benchmark、详细消融
- 写作质量: ⭐⭐⭐⭐⭐ Figure 2-4的信息流可视化极其精美和直观
- 价值: ⭐⭐⭐⭐⭐ 当前VLM token剪枝领域的SOTA方法，理论和实践价值兼具
