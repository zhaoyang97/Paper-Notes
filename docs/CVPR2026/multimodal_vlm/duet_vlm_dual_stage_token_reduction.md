# DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference

**会议**: CVPR 2026  
**arXiv**: [2602.18846](https://arxiv.org/abs/2602.18846)  
**代码**: [https://github.com/AMD-AGI/DUET-VLM](https://github.com/AMD-AGI/DUET-VLM)  
**领域**: 多模态VLM / 模型加速 / Token压缩  
**关键词**: 视觉token压缩, 双阶段压缩, 局部聚类聚合, 文本引导剪枝, 训练+推理加速  

## 一句话总结
提出DUET-VLM双阶段视觉token压缩框架：先在视觉编码器侧通过局部聚类聚合将冗余token合并为信息保持的紧凑表示（V2V），再在语言骨干侧通过文本引导的层级自适应剪枝逐步删减低信息量token（T2V），在LLaVA-1.5-7B上67%压缩保留99%精度，89%压缩保留97%精度。

## 背景与动机
VLM的视觉token数量庞大（LLaVA-1.5 576个，LLaVA-NeXT 2800+），导致注意力计算二次增长。现有压缩方法存在单侧不足：(1) 视觉侧方法（VisionZip/PruMerge）过早合并导致信息丢失，缺乏下游自适应性；(2) 语言侧方法（PyramidDrop/FastV）均匀丢弃缺乏语义自适应性；(3) 所有方法都是单向的——要么在融合前压缩，要么在推理中剪枝——没有联合优化视觉冗余去除和上下文感知保留。

## 核心问题
如何设计一个统一的双阶段token压缩框架，同时在视觉编码器侧去除空间冗余、在语言骨干侧基于文本语境自适应剪枝，实现极端压缩下的精度保持？

## 方法详解

### 整体框架
两阶段：(A) **V2V视觉侧**：从CLIP最后一层自注意力图中选择$k_1$个dominant token（最高注意力得分），剩余token按注意力引导的局部聚类合并为$k_2$个contextual token（每个由宽度$w$的邻域平均得到）。(B) **T2V语言侧**：在语言骨干的$M$个stage之间，用salient text token与visual token的cross-attention得分排序，按比例$\lambda$逐步丢弃低得分visual token。

### 关键设计
1. **局部聚类聚合 (Local Cluster Aggregation)**：替代VisionZip的全局平均（全局平均会稀释语义信息）。从残余token中选$k_2$个中心点（按V2V注意力得分排名），每个中心点的邻居由注意力亲和度确定（宽度$w$），局部平均生成contextual token。关键优势：(a) 局部聚合保留细粒度线索；(b) 因$w \cdot k_2 < |\mathbf{X}_{res}|$，未分配token被早期丢弃，减轻语言骨干负担。消融显示局部聚类在所有预算下都优于VisionZip的全局方案（97.1% vs 96.5% at 192 tokens）。

2. **Salient Text Token引导的层级剪枝**：非使用单一的最后text token（PyramidDrop默认），而是识别注意力最高的salient text token子集$\mathcal{S}$做T2V cross-attention计算。实验对比三种策略：(C)仅最后text token、(C+all)所有text token、(C+S)salient text tokens。在训练场景下(C+S)在低token预算时最优（97.6% at 64 tokens），因为精选的text token提供更尖锐的上下文信号。

3. **训练+推理统一**：DUET-VLM独特之处是可以**在训练时就用压缩token训练**（不只是推理时压缩），使模型适应压缩表示。训练结果：192 tokens→99.7%精度（+训练时间减少26%），128→99.1%（-31%训练时间），64→97.2%（-36%训练时间）。推理直接复用训练好的压缩配置。

### 损失函数 / 训练策略
使用与LLaVA-1.5相同的训练流程和损失。默认语言骨干压缩配置：layer 16丢弃50%，layer 24丢弃全部（深层token已冗余，注意力热图证实信息已转移到隐状态）。集群宽度$w=4$。AMD MI325 GPU×8训练。

## 实验关键数据
**推理（LLaVA-1.5-7B）**：

| Token预算 | DUET Avg | VisionZip | PyramidDrop | FitPrune | SparseVLM |
|--------|------|------|----------|------|------|
| 192 (67%↓) | **99.0%** | 97.7% | 96.4% | 97.8% | 95.7% |
| 128 (78%↓) | **98.1%** | 96.3% | 95.6% | 94.8% | 93.6% |
| 64 (89%↓) | **95.4%** | 92.8% | 86.7% | 85.0% | 86.4% |

**训练（LLaVA-1.5-7B）**：192 tokens训练→99.7%精度（超越基线的VisionZip/PyramidDrop的98.4%/96.4%），同时训练时间减少26%。

**视频（Video-LLaVA-7B）**：53.1%压缩→100.8%反而超越基线！93.4%压缩→97.6%。

**跨架构（Qwen2.5-VL-7B）**：160 tokens→98.4%（VisionZip 96.9%），验证泛化性。

### 消融实验要点
- **局部vs全局聚类**：局部聚类在所有预算下持续优于全局（192: 97.1 vs 96.5，128: 95.7 vs 94.8，64: 92.2 vs 91.3）
- **聚类宽度$w$的影响**：$w=4,6$最优——过小过度碎片化，过大过度平滑
- **$k_1$ vs $k_2$配比**：高预算倾向更多dominant($k_1$)，低预算需要平衡的dominant-contextual分配
- **深层token完全可删除**：第24层丢弃所有visual token性能几乎不变——信息已转移到隐状态
- **Salient text token也改善PyramidDrop**（99.5% vs 99.2%），证明是通用的改进
- **视频优势最显著**：因为视频帧间时序冗余巨大，DUET-VLM在93.4%压缩下仍达97.6%

## 亮点
- **双阶段设计理念清晰**——视觉侧去空间冗余（结构化），语言侧去语义冗余（上下文化），两阶段互补
- 局部聚类聚合是一个简洁但有效的改进——保留邻域结构而非全局平均稀释
- **训练阶段也能压缩**的发现很有价值——不只是推理加速，还能减少训练成本
- 视频场景的超基线表现（>100%）说明适度压缩可能有正则化效果
- 代码开源+在多架构（LLaVA/Qwen2.5-VL）上验证，实用性强

## 局限性 / 可改进方向
- 未报告精确推理/训练时间，仅报告token数和speedup指标
- 语言侧剪枝层的选择（8/16/24三阶段划分）是固定启发式，未做自适应
- 仅在7B模型上验证，更大模型(13B+)的效果未知
- Salient text token的选择在推理时需额外注意力计算，但论文未量化该开销
- 未与V2Drop等基于token变化量的方法对比——V2Drop的变化量信号可能比注意力更鲁棒

## 与相关工作的对比
- **vs VisionZip (CVPR'25)**：VisionZip只做视觉侧全局合并。DUET-VLM加入局部聚类+语言侧层级剪枝，192 tokens时99.0% vs 97.7%
- **vs PyramidDrop (CVPR'25)**：PyramidDrop只做语言侧均匀剪枝。DUET-VLM加入视觉侧聚类+文本引导自适应剪枝，64 tokens时95.4% vs 86.7%
- **vs V2Drop (CVPR'26)**：V2Drop用token变化量做剪枝信号，兼容FlashAttention。DUET-VLM用注意力做信号，但额外有视觉侧聚类阶段。二者思路互补
- **vs FastV (ECCV'24)**：FastV一次性剪枝，性能迅速下降（64 tokens仅70.7%）。DUET-VLM渐进剪枝+预压缩保持95.4%

## 启发与关联
- DUET-VLM的"训练时也用压缩token"与GKD(CVPR2026)的"解耦表示学习和任务学习"思想有共鸣——都是让模型适应压缩/蒸馏后的表示
- 与`ideas/model_compression/20260317_hashgrid_pgi_token_compression.md`相关——该idea探索更精确的token重要性评估方法，DUET-VLM的局部聚类可以作为一种空间先验
- 在强压缩(89%+)时候，与V2Drop的token变化量信号结合可能带来进一步提升——聚类保结构，变化量保语义重要性

## 评分
- 新颖性: ⭐⭐⭐⭐ 双阶段统一框架和局部聚类是清晰的增量改进，但各组件不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 4个模型(LLaVA/LLaVA-NeXT/Video-LLaVA/Qwen2.5-VL)、8个基准、推理+训练、详细消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，消融详尽，但符号较多且图排版稍显拥挤
- 价值: ⭐⭐⭐⭐⭐ 即插即用+训练可用+开源+跨架构验证=实用性极强
