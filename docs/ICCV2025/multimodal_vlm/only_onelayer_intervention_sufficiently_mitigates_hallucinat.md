# ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models

**会议**: ICCV 2025  
**arXiv**: [2507.00898](https://arxiv.org/abs/2507.00898)  
**代码**: [https://github.com/zifuwan/ONLY](https://github.com/zifuwan/ONLY)  
**领域**: 多模态VLM / 幻觉缓解  
**关键词**: hallucination mitigation, contrastive decoding, text-to-visual entropy ratio, training-free, single-layer intervention  

## 一句话总结
提出ONLY，一种training-free的单层干预解码方法——通过Text-to-Visual Entropy Ratio（TVER）选择偏向文本的attention head生成textually-enhanced logits，然后与原始logits做自适应对比/协作解码，仅增加1.07×推理时间就在POPE上比VCD/M3ID高3.14%，在CHAIR上降低CHAIR_S 6.2个点。

## 研究背景与动机
1. **领域现状**：LVLM幻觉的主流缓解方法是contrastive decoding——对比原始输出和扰动版本的输出（如VCD用噪声图像、M3ID去掉图像）。但这些方法需要2次或更多完整推理查询，推理时间翻倍，不适合实时应用。
2. **现有痛点**：VCD需要2.01×推理时间和1.05×GPU内存，M3ID需要2.03×推理时间，OPERA需要7.12×，HALC需要6.52×。虽然准确率有提升但效率-性能权衡不合理——增加100%时间只换来几个点的提升。
3. **核心矛盾**：contrastive decoding需要对比两个分布，但现有方法通过跑两次完整模型来获取这两个分布，极其浪费。能否在单次推理内部直接构造出一个"textually-enhanced"分布？
4. **本文要解决什么**：用单层intervention替代双重推理，在不显著增加计算的情况下获取用于对比解码的增强logits。
5. **切入角度**：观察到当图像输入被扰动时（如加噪声），文本attention的熵上升而视觉attention的熵下降。直接选择TVER高的attention head就能模拟"visual distortion"效果，无需真的扰动图像。
6. **核心idea一句话**：用text-to-visual entropy ratio筛选attention head构造textually-enhanced输出，只需一层额外attention计算就能实现与双重推理等效的对比解码。

## 方法详解

### 整体框架
在LVLM正常解码过程中，额外计算一个Textual-Enhanced MHA输出：选择一层（默认第0层），用TVER筛选attention head（保留TVER>=平均值的head，其余置零），将增强输出通过残差连接到最后一层并过MLP得到textually-enhanced logits。最终用曼哈顿距离自适应决定对比解码还是协作解码。

### 关键设计

1. **Text-to-Visual Entropy Ratio (TVER)**:
   - 做什么：衡量每个attention head对文本vs视觉token的信息分散程度。
   - 核心思路：将attention矩阵按照文本/视觉token indices拆分为$a^{\mathcal{T}}$和$a^{\mathcal{V}}$，分别计算归一化后的entroy：$\text{TVER}_{\ell,i} = \frac{\text{Entropy}(a^{\mathcal{T}}_{\ell,i})}{\text{Entropy}(a^{\mathcal{V}}_{\ell,i})}$。TVER高的head说明它对文本信息的uncertainty高（更依赖语言先验），低的head更关注视觉信息。
   - 关键发现：TVER与幻觉高度相关——response级别的平均TVER越高，CHAIR_I越高（幻觉越多）；token级别上，幻觉token和非幻觉token在textual-enhanced logits与原始logits间的曼哈顿距离有显著分布差异。
   - 设计动机：不需要实际扰动图像，直接从attention head的entropy特征中提取语言偏差信息。

2. **Textual-Enhanced MHA (TE-MHA)**:
   - 做什么：在选定层$\tilde{\ell}$中，保留TVER高于该层平均值的head，屏蔽其余head，生成偏向文本的attention输出。
   - 核心思路：$\tilde{a}_{\ell,i} = a_{\ell,i}$ if $\text{TVER}_{\ell,i} \geq \text{average}(\text{TVER}_\ell)$, 否则置0。用筛选后的attention做MHA计算。
   - 通过两个残差连接连到原始最后一层输出：$\tilde{\bar{\mathcal{H}}}^{L-1}_t = \tilde{\mathcal{H}}^{\tilde{\ell}}_t + \mathcal{H}^{L-1}_t$，再过MLP得到增强logits。
   - 设计动机：只需计算一层额外attention（复用已有的K/V），开销极小。

3. **自适应解码策略**:
   - 做什么：根据原始logits和增强logits之间的曼哈顿距离，动态选择协作或对比解码。
   - 核心思路：$d_t = \sum_{y_t} |p_\theta - \tilde{p}_\theta|$。当$d_t < \gamma$（两者分布接近）时做协作解码（加权合并）；当$d_t \geq \gamma$（差异大，可能有幻觉风险）时做对比解码（减去增强logits）。
   - 公式：${f_\theta^{\text{final}}} = f_\theta + \alpha_1 \tilde{f}_\theta$ (collaborative) 或 $(1+\alpha_2)f_\theta - \alpha_2 \tilde{f}_\theta$ (contrastive)。
   - 设计动机：不是所有token都需要对比解码。当模型有信心时协作可以增强细节，当模型不确定时对比可以压制幻觉。

### 效率分析
只需一层额外attention计算 → 1.07×推理时间，几乎无额外GPU内存（14951MB vs 原始14945MB）。相比VCD (2.01×, 15749MB)、M3ID (2.03×, 15575MB)、OPERA (7.12×, 22706MB)、HALC (6.52×, 23084MB)，效率提升巨大。

## 实验关键数据

### POPE Benchmark（LLaVA-1.5，MS-COCO Random）

| 方法 | Accuracy | F1 | 推理时间 | GPU内存 |
|------|---------|-----|---------|--------|
| Regular | 83.44 | 83.09 | 1.00× | 14945MB |
| VCD | 87.15 | 85.45 | 2.01× | 15749MB |
| M3ID | 87.52 | 84.50 | 2.03× | 15575MB |
| **ONLY** | **89.10** | **87.85** | **1.07×** | **14951MB** |

### CHAIR Benchmark（LLaVA-1.5，Max token=128）

| 方法 | CHAIR_S↓ | CHAIR_I↓ | Len |
|------|---------|---------|-----|
| Regular | 26.2 | 9.4 | 55.0 |
| VCD | 24.4 | 7.9 | 54.4 |
| M3ID | 21.4 | 6.3 | 56.6 |
| HALC | 21.7 | 7.1 | 51.0 |
| **ONLY** | **20.0** | **6.2** | 49.8 |

MME-Hallucination（LLaVA-1.5）：ONLY 635.55 vs 次优M3ID 598.11 (+37.44)

### 消融实验

| 配置 | POPE F1 | CHAIR_S↓ | 说明 |
|------|---------|---------|------|
| Regular | 81.27 | 26.2 | 基线 |
| 零Visual Attention | 84.82 | 21.2 | 直接清零视觉attention |
| 加噪Visual Attention | 84.79 | 22.1 | 给视觉attention加噪 |
| 2×Textual Attention | 84.96 | 21.6 | 文本attention翻倍 |
| Sum Ratio选择 | 84.46 | 23.1 | 用attention weight sum ratio选head |
| **TVER选择（Ours）** | **85.37** | **20.0** | entropy ratio选head |

### 关键发现
- **ONLY在1.07×时间内超过所有2×+方法**：在POPE上89.10% vs VCD 87.15% (+1.95%)；CHAIR_S 20.0 vs VCD 24.4 (降4.4)。用1/30的额外时间获得了更好结果。
- **层选择对结果影响很小**：32层中任意一层做intervention，F1都在84.62~85.37之间，而VCD仅83.38、M3ID仅84.05。说明方法非常鲁棒。
- **TVER比直接操纵attention更好**：直接清零视觉attention或加噪都不如TVER head选择，因为TVER从信息论角度精确定位了语言偏差最强的head。
- 自适应解码中$\gamma=0.2$最优（LLaVA-1.5），$\alpha_1=3, \alpha_2=1$最优。
- 在更强模型LLaVA-NeXT-7B/13B上同样有效且一致优于VCD/M3ID。

## 亮点与洞察
- **信息论驱动的head选择**：TVER源于条件熵的直觉——$H(\mathcal{T}|\mathcal{V})$高意味着文本attention对视觉information不敏感（更像纯语言推理），选择这些head就是在增强语言先验来做对比解码。这比VCD的"加噪声"和M3ID的"去图像"在理论上更优雅。
- **自适应对比/协作切换**：不是简单地总做对比，而是根据距离判断——模型有信心时协作增强细节，不确定时对比压制幻觉。这捕捉了幻觉发生具有token-level动态性的本质。
- **极致效率**：1.07×时间、0额外内存，是目前已知最高效的幻觉缓解方法。核心insight是：获取对比信号不需要跑完整模型两遍，只需要在一层中选择性地激活某些attention head。

## 局限性 / 可改进方向
- 仅在LLaVA-1.5、InstructBLIP、Qwen-VL上验证，未测试更新的大模型（InternVL、Qwen2-VL等）。
- 超参数$\alpha_1, \alpha_2, \gamma$需要对不同模型做调整（LLaVA用$\gamma=0.2$，QwenVL用$\gamma=0.4$）。
- 生成长度略有缩短（49.8 vs 55.0 token），可能遗漏了部分正确信息。
- 未在视频理解任务上验证。
- 只选择一层做intervention，多层组合可能进一步提升但增加开销。

## 相关工作与启发
- **vs VCD**: VCD用扩散噪声扰动图像做对比解码，需要2×推理。ONLY直接从attention entropy中提取语言偏差信号，1.07×推理就超过VCD。
- **vs M3ID**: M3ID去掉图像做无条件预测对比，也需要2×推理。ONLY的TVER head选择在本质上达到了类似"抑制视觉、增强文本"的效果但只需一层计算。
- **vs MRGD（同批前文）**: MRGD用独立的奖励模型引导搜索，ONLY直接在解码过程中修改attention。两者都是推理时方法但切入点不同：MRGD重搜索策略，ONLY重logits修正。
- **vs ShortV（同批前文）**: ShortV发现很多层对视觉token无效而跳过。ONLY从互补角度利用这种层间差异——在一层中选择性增强文本attention来提取语言偏差。两者侧面印证了"MLLM的层对视觉和文本的贡献不均"这一现象。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ TVER驱动的head选择+单层intervention是非常优雅的设计，信息论动机清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 6个benchmark，3个LVLM，效率对比，大量消融（层选择、策略对比、超参数），GPT-4V评估
- 写作质量: ⭐⭐⭐⭐⭐ 问题分析精准（Figure 1的效率-性能trade-off图），TVER与幻觉的关联验证（Figure 4）
- 价值: ⭐⭐⭐⭐⭐ 1.07×时间超过2×+的方法，实用价值极高，很可能成为default hallucination mitigation方法
