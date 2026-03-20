# SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs

**会议**: ICCV 2025  
**arXiv**: [2506.05344](https://arxiv.org/abs/2506.05344)  
**代码**: [https://github.com/CR400AF-A/SparseMM](https://github.com/CR400AF-A/SparseMM)  
**领域**: 多模态VLM / 高效推理 / Attention分析  
**关键词**: visual heads, attention sparsity, KV-Cache compression, training-free, OCR-based identification  

## 一句话总结
发现MLLM中仅约5%的attention head主动参与视觉理解（称为"visual heads"），提出基于OCR任务的training-free识别方法量化每个head的视觉相关性，并设计SparseMM——按visual score非对称分配KV-Cache预算的策略，在DocVQA上仅用5.3%的cache（256/4830）即可维持Qwen2-VL的性能，实现1.87×加速和50%内存减少。

## 研究背景与动机
1. **领域现状**：MLLM通过视觉编码器+adapter+LLM的架构实现多模态理解。但LLM是如何"学会看图"的——哪些内部组件负责处理视觉信息——还不清楚。现有KV-Cache压缩方法（SnapKV、PyramidKV、AdaKV）均匀对待所有attention head，忽视了视觉处理的特殊性。
2. **现有痛点**：均匀压缩KV-Cache在多模态场景下不合理——视觉token占输入的90-99%，如果压缩时不区分"视觉重要head"和"文本专用head"，会不均匀地损失视觉信息。
3. **核心矛盾**：高效推理需要压缩KV-Cache，但不知道哪些head承载关键视觉信息，盲目压缩可能导致视觉理解崩溃。
4. **本文要解决什么**：(1) 系统研究MLLM中的视觉head分布；(2) 提出识别方法；(3) 基于发现设计更智能的KV-Cache压缩策略。
5. **切入角度**：用OCR任务作为"锚"——OCR输出的每个字符都能精确对应到图像的特定patch区域，因此可以直接追踪哪些attention head在关注正确的视觉区域。
6. **核心idea一句话**：MLLM中<5%的attention head负责视觉理解，按这些head的视觉分数非对称分配KV-Cache预算可以实现更好的效率-精度权衡。

## 方法详解

### 整体框架
两个阶段：(1) Visual Head识别：用OCR图像让模型生成文字，追踪每个输出token对应的图像patch区域，检查每个head最高attention是否命中该区域，聚合1000个样本的hit率得到visual score矩阵。(2) SparseMM推理加速：按visual score给每个head分配不同的KV-Cache预算。

### 关键设计

1. **Visual Head识别（OCR-based）**:
   - 做什么：为每个attention head计算一个visual score，衡量它在多大程度上关注视觉内容。
   - 核心思路：对每个OCR输出token $y_i$，找到它对应的图像区域的patch indices $I_{y_i}$。检查head $h$的attention矩阵$A_h$中最大attention值是否落在这些patches内。Visual Score = 命中次数 / 图像token数，在1000个Synthdog数据上聚合归一化。
   - 关键发现：(1) **极度稀疏**——仅~5%的head有显著visual score；(2) **跨架构通用**——在MHA (Vicuna)和GQA (Mistral, Qwen2)中都观察到；(3) **任务无关**——用OCR识别的visual heads在物体识别、场景理解等其他任务中同样重要。
   - 设计动机：OCR提供了最精确的text-to-visual对应关系（每个字符→特定图像区域），这比一般VQA中模糊的答案-图像关联更精确可靠。

2. **SparseMM: 三部分KV-Cache分配**:
   - 做什么：根据visual score非对称地给不同head分配不同的KV-Cache预算。
   - 三部分组成：
     - **Local Window Cache**：每个head固定保留最近32个token的KV（局部窗口）。
     - **Uniform-Based Cache**：剩余预算的$\rho=10\%$均匀分配给所有head（保底最低额度）。
     - **Score-Preferred Cache**：剩余$90\%$按visual score比例分配——visual head获得更多cache。
   - 公式：head $(i,j)$的总预算 = $w + r + b_{ij}^{score}$，其中$b_{ij}^{score} = B_{remain2} \cdot \frac{s_{ij}}{\sum s_{ij}}$。
   - 在每个head的预算内，用observation window（末尾32个token的query）计算attention score选择最重要的KV对保留。
   - 设计动机：visual head需要保留更多视觉token的KV来维护视觉理解能力，而non-visual head可以激进压缩。$\rho=0$（完全按score分配）会导致某些head完全没有cache，性能崩溃，所以需要uniform保底。

### 验证实验：Masking Visual Heads
屏蔽top 5% visual heads → OCRBench性能显著下降（~20%↓）；随机屏蔽同比例head → 影响很小。进一步屏蔽top 10%的额外5%影响明显小于前5%，证明visual heads的分布确实非常稀疏且关键。

## 实验关键数据

### 主实验（256 KV-Cache budget，16K input tokens）

| 方法 | DocVQA | OCRBench | TextVQA | ChartQA | TextCaps | 延迟(ms) |
|------|--------|---------|---------|---------|---------|--------|
| FullKV | 0.68 | 0.52 | 0.65 | 0.55 | 0.73 | 52.9 |
| **SparseMM** | **0.68** (-0.00) | **0.52** (-0.00) | **0.65** (-0.00) | **0.54** (-0.01) | **0.73** (-0.00) | **37.1** (-30%) |
| SnapKV | 0.64 (-0.04) | 0.46 (-0.06) | 0.62 (-0.03) | 0.50 (-0.05) | 0.65 (-0.08) | 35.3 |
| PyramidKV | 0.65 (-0.03) | 0.48 (-0.04) | 0.62 (-0.03) | 0.53 (-0.02) | 0.65 (-0.08) | 34.9 |
| AdaKV | 0.65 (-0.03) | 0.48 (-0.04) | 0.62 (-0.03) | 0.53 (-0.02) | 0.65 (-0.08) | 35.1 |

### 效率评估（LLaVA-NeXT-Vicuna-7B，32K input tokens）

| 指标 | FullKV | SparseMM (256 budget) |
|------|--------|-----------------------|
| Decode延迟 | 52.9ms/token | ~28ms/token (1.87× 加速) |
| 峰值内存 | 32.87GB | 17.38GB (**-47%**) |

Qwen2-VL-7B在DocVQA上：256 budget = 仅5.3%的cache → **性能不降**（0.9345 vs FullKV 0.9394）。

### 消融实验

| 配置 | MMBench结果 |
|------|-----------|
| Local+Uniform+Score (完整) | 81.5 @256 budget |
| Local only | 80.5 @256 budget |
| Local+Uniform (无Score) | 81.4 @256 budget |

$\rho$超参数消融：$\rho=0.1$最优；$\rho=0$在Mistral模型上性能崩溃（0.145），因为部分head完全没有cache。

### 关键发现
- **SparseMM在5个文本密集型benchmark上几乎无损**（FullKV vs SparseMM差距 ≤ 0.01），但SnapKV/AdaKV在同预算下损失3-8%。
- **在多选benchmark（MMBench、VQAv2）上也有效**：96 token budget下SparseMM保持full performance，而其他方法已开始下降。
- OCR识别的visual head在不同数据集（MLT、CTW、COCO）上分布一致，验证了方法的鲁棒性。但检测任务（COCO larger bbox）识别的visual heads噪声更大。
- **与ShortV（同系列论文）的互补关系**：ShortV发现层级视觉冗余，SparseMM发现head级视觉稀疏。两者都指出MLLM中视觉处理高度集中在少数"组件"中。

## 亮点与洞察
- **"Visual Heads"是一个重要的分析性发现**：仅5%的attention head负责视觉理解——这意味着MLLM在SFT阶段只"调整"了极少数head来适应视觉，绝大多数head仍然是纯文本处理。这挑战了我们对multimodal fine-tuning的理解。
- **OCR作为探针工具的巧妙设计**：OCR任务提供了精确的字符-到-patch对应关系，使得attention追踪成为可能。这种"用精确对应任务探测模型内部机制"的方法论可以推广到其他分析场景。
- **三部分cache分配机制简洁有效**：Local Window（保局部性）+ Uniform（保底线）+ Score-Preferred（利用稀疏性）的组合比任何单一策略都好，且逻辑清晰。

## 局限性 / 可改进方向
- Visual Head识别依赖OCR数据（1000张Synthdog），对非文本密集型场景的泛化性虽然验证了但仍有待进一步确认。
- 仅在7B模型上验证，更大模型的visual head分布可能不同。
- 三部分分配的超参数（window size=32, $\rho$=0.1）在不同模型间可能需要调整。
- 未与token pruning方法（FastV、SparseVILA）联合使用——两者是正交的（head-level vs token-level）。

## 相关工作与启发
- **vs ShortV（同批前文）**: ShortV从层级角度发现视觉token在60%的层中可以跳过处理。SparseMM从head级角度发现仅5%的head主动处理视觉。两者共同说明：MLLM的视觉处理高度集中在少数层的少数head中。
- **vs SparseVILA**: SparseVILA在prefill/decode阶段解耦视觉token稀疏化（属于token-level），SparseMM在head级别分配cache（属于head-level）。两者可以组合使用。
- **vs SnapKV/AdaKV**: 这些方法是通用KV-Cache压缩，不区分多模态。SparseMM的核心优势就是利用了"visual head稀疏"这一多模态特有的性质。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "Visual Heads"的发现本身就是重要的分析贡献，是首次系统研究MLLM中attention head的视觉特异性
- 实验充分度: ⭐⭐⭐⭐⭐ 3个模型(MHA+GQA)、7个benchmark、效率评估、masking验证、跨数据集鲁棒性、可视化，极其全面
- 写作质量: ⭐⭐⭐⭐ 清晰，visual head identification的描述详细
- 价值: ⭐⭐⭐⭐⭐ 发现级工作——"MLLM中仅5%的head负责视觉"这一结论对理解多模态模型机制有深远意义，实际加速效果也很强
