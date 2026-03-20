# Visual Symbolic Mechanisms: Emergent Symbol Processing in Vision Language Models

**会议**: ICLR 2026 Oral  
**arXiv**: [2506.15871](https://arxiv.org/abs/2506.15871)  
**代码**: 有（将开源数据集、分析和干预代码）  
**领域**: 多模态VLM / 可解释性  
**关键词**: visual binding, position IDs, mechanistic interpretability, causal mediation, VLM  

## 一句话总结
发现 VLM 内部涌现了一套三阶段符号处理机制（ID retrieval → ID selection → feature retrieval），利用内容无关的空间位置索引（position IDs）来解决视觉绑定问题，并证明绑定错误可直接追溯到这些机制的失败。

## 研究背景与动机

1. **领域现状**：VLM 使用组合性表示（如"红色"+"方形"）来高效编码视觉场景。语言模型中已发现涌现的 binding IDs——内容无关的符号索引——用于追踪实体与属性的绑定关系。

2. **现有痛点**：VLM 在需要精确绑定的任务上表现极差（计数、视觉搜索、视觉类比），例如无法区分"红方+蓝圆"与"蓝方+红圆"。这就是经典的 **绑定问题（binding problem）**。但 VLM 内部是否像纯文本 LM 一样存在符号处理机制，尚一无所知。

3. **核心矛盾**：组合性表示的代价是必须解决绑定问题——将正确的特征绑定到正确的物体。VLM 的很多"谜题"式失败（如计数错误）本质上都是绑定失败，但我们不知道这些失败的内部机制是什么。

4. **本文要解决什么？** (a) VLM 是否使用类似符号的机制来处理视觉绑定？ (b) 这些机制具体是什么？ (c) 绑定错误是否可以追溯到这些机制的失败？

5. **切入角度**：借鉴纯文本 LM 中 binding IDs 的发现和认知科学中的视觉索引理论（Pylyshyn, 2001），假设 VLM 可能利用 **空间位置** 作为内容无关的索引来绑定物体特征。

6. **核心idea一句话**：VLM 涌现出三种注意力头（ID retrieval、ID selection、feature retrieval），利用空间位置作为符号变量来索引和检索视觉对象特征。

## 方法详解

### 整体框架
通过场景描述任务（给定多物体图片和部分描述，补全缺失物体）作为探测工具，结合表征分析（PCA、RSA）、因果中介分析（CMA）和干预实验，在 7 个 VLM（Qwen2-VL, Qwen2.5-VL-3B/7B/32B, Llava1.5-7B/13B, Llava-OneVision-7B）上系统研究视觉绑定的内部机制。

### 关键设计

1. **三阶段 Position ID 架构**
   - 做什么：描述 VLM 解决绑定问题的完整处理流程
   - 核心思路：
     - **Stage 1 - ID Retrieval**（~Layer 12-16）：给定 prompt 中描述的物体（如"红色方形"），ID retrieval heads 从对应的图像 token 中检索该物体的空间位置索引。输出是抽象的空间指针，不是物体特征
     - **Stage 2 - ID Selection**（~Layer 18-21）：基于已检索到的 position IDs，计算目标物体（需要被描述的缺失物体）的 position ID。通过排除已知物体的位置来推断目标位置
     - **Stage 3 - Feature Retrieval**（~Layer 23-27）：使用 Stage 2 确定的 position ID 作为查询，从图像 token 中检索目标物体的语义特征（颜色、形状）
   - 设计动机：认知科学中的视觉索引理论预测了类似的内容无关空间索引机制。PCA 分析直观验证：Layer 19 的表示按空间位置聚类（特征重叠），Layer 27 按物体特征聚类（位置重叠），完美对应 Stage 2→3 的转变

2. **因果中介分析（CMA）定位注意力头**
   - 做什么：因果性地识别执行每个阶段的具体注意力头
   - 核心思路：设计三个 CMA 条件，每个针对一种头：
     - ID retrieval 条件：clean 和 modified 图片中物体位置交换，patch prompt token 处的注意力头输出
     - ID selection 条件：同样交换位置，但 patch 最后一个 token 处的输出
     - Feature retrieval 条件：clean 和 modified 图片中目标物体特征不同，patch 最后 token 处的输出
     - CMA score: $s = (M(c_1^*)[a_1^*] - M(c_1^*)[a_1]) - (M(c_1)[a_1^*] - M(c_1)[a_1])$
   - 设计动机：纯表征分析只能说明相关性，CMA 提供因果证据。每个条件精确针对一个阶段，通过预测 patching 的下游效应来验证机制

3. **Position ID 干预实验**
   - 做什么：验证 position IDs 的功能性、泛化性和可迁移性
   - 核心思路：通过加性干预 $\tilde{o}_h(x) = o_h(x) + \alpha \cdot (d_t - d_o)$ 编辑 position ID，其中 $d_t$ 和 $d_o$ 是目标和原始 ID 的估计。在逼真图像（PUG 环境）、真实图像（COCO）、跨任务（场景描述→空间推理）等设置下测试
   - 设计动机：如果 position IDs 真的是通用的索引机制，那么编辑它们就应该系统性地改变模型输出——在合成图像和真实图像上都有效，在不同任务间可迁移

### 损失函数 / 训练策略
本文是分析工作，不涉及模型训练。所有实验基于预训练好的开源 VLM。

## 实验关键数据

### 主实验

| 实验 | 模型 | 干预有效率 | 说明 |
|------|------|----------|------|
| PUG 图像干预 (ID retrieval) | 7 模型平均 | >79% | 编辑 ID retrieval heads 可控制模型输出 |
| PUG 图像干预 (ID selection) | 7 模型平均 | >79% | 编辑 ID selection heads 同样有效 |
| PUG 图像干预 (feature retrieval) | 7 模型平均 | >79% | 编辑 feature retrieval heads 有效 |
| 颜色检索干预 | 所有模型 | 高 | Position IDs 存储在物体对应的图像 patch keys 中 |
| 跨任务迁移 | 5/7 模型 | 显著提升 | 场景描述任务的 IDs 迁移到空间推理任务有效 |

### 消融/分析实验

| 分析 | 关键发现 | 说明 |
|------|---------|------|
| 相对 vs 绝对编码 | Qwen 系列使用**相对**位置编码 | Llava 系列差异不明显 |
| 高/低熵绑定错误 | 低熵场景（物体共享特征）ID retrieval 与 selection 精度下降 | 直接解释了绑定错误的原因 |
| 高熵 ID → 低熵修复 | Qwen2.5-VL-3B 提升 11.1%，Llava1.5-13B 提升 10.4% | 因果性证明 ID 机制失败导致绑定错误 |
| 计数任务消融 | 消除 top-250 heads → 性能降至 0%，bottom-500 → 仍 ~70% | Position ID heads 对计数任务至关重要 |

### 关键发现
- **三阶段架构跨模型/跨尺度一致**：在 Qwen2-VL, Qwen2.5-VL (3B/7B/32B), Llava1.5 (7B/13B), Llava-OneVision 7 个模型上都观察到相同模式
- **Position IDs 使用相对编码**：Qwen 系列明确使用相对于物体组的相对位置，而非图像中的绝对位置
- **IDs 存储在 patch keys 中**：Position IDs 存在于物体对应的图像 patch 的 key 向量中，独立于 RoPE 位置编码
- **机制可迁移**：从场景描述任务估计的 IDs 可直接用于改善空间推理任务的表现
- **绑定错误可因果性修复**：将高熵（特征区分度高）条件下的精确 IDs 注入低熵（特征相似）条件，可显著减少绑定错误

## 亮点与洞察
- **揭示了 VLM "看世界"的方式**：VLM 不是直接处理像素特征，而是先在中间层建立空间索引系统，再用索引去取特征。这与认知科学中的视觉索引理论（Pylyshyn 2001）高度一致，暗示深度神经网络可能独立发现了类似生物视觉的解决方案
- **从诊断到修复的路径**：论文不仅发现了机制，还展示了通过干预 position IDs 可以部分修复绑定错误（低熵场景提升 4.6-11.1%）。这为架构改进提供了具体方向——可以设计显式支持空间索引的架构，或通过 spatial pointing 任务训练来增强这些机制
- **跨模态的统一符号处理**：纯文本 LM 用 binding IDs，VLM 用 position IDs，本质上都是涌现的内容无关符号变量。这支持了"大规模 Transformer 会自发发展符号处理能力"的假说

## 局限性 / 可改进方向
- **实验场景偏简单**：主要用 2-3 个形状/颜色的合成图片，PUG 环境虽然逼真但物体仍简单。真实世界的多物体场景（如室内场景有几十个物体）是否仍适用？
- **仅分析开源模型**：GPT-4V、Gemini 等闭源模型无法分析，且这些模型在绑定任务上表现更好，可能有不同的机制
- **未研究文本-视觉交互**：Position IDs 如何与 prompt 中的语言表示交互？当 prompt 更复杂时（多跳推理问句）机制是否改变？
- **修复效果有限**：高熵→低熵干预最多提升 11.1%，说明绑定错误还有其他来源未被捕获
- **改进方向**：可考虑在训练中加入 spatial pointing/grounding 任务来增强 position ID 机制；或设计 object-centric attention 架构（如 Slot Attention）来显式支持空间索引

## 相关工作与启发
- **vs Binding IDs (Feng & Steinhardt 2023)**：纯文本 LM 的 binding IDs 是在序列维度上分配符号变量，VLM 的 position IDs 则利用二维空间作为索引维度，更自然地对应视觉场景的空间结构
- **vs Yang et al. 2025 (Emergent symbolic mechanisms in LMs)**：该工作发现 LM 在抽象推理中用符号变量，本文将此扩展到视觉域，且发现的电路功能不同（场景解析 vs 关系归纳），不是简单的跨模态迁移
- **与 VHD-Guided Adaptive Visual Re-injection idea 相关**：Position ID 机制的发现暗示长链多模态推理中，视觉信息可能通过空间索引来持续追踪，长推理链条中 position ID 的退化可能是视觉信息遗忘的原因之一

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次揭示 VLM 内部的视觉符号处理机制，理论意义重大
- 实验充分度: ⭐⭐⭐⭐⭐ 7 个模型×3 种分析方法（PCA/RSA/CMA）×多种干预实验，证据极其充分
- 写作质量: ⭐⭐⭐⭐⭐ Yoshua Bengio 参与，写作极为清晰，图表精美，逻辑链条完整
- 价值: ⭐⭐⭐⭐⭐ 对理解 VLM 失败模式和未来架构设计有直接启发，连接 AI 和认知科学
