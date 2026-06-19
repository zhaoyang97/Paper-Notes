---
title: >-
  [论文解读] B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens
description: >-
  [ICCV 2025][模型压缩][视觉大语言模型] 本文提出B-VLLM框架，通过文本条件自适应帧选择、时序帧Token合并和空间Token采样三个模块，在VLLM的上下文窗口限制内动态平衡视频的时空线索，在MVBench上带来10%的性能提升。 当前VLLM在处理视频特别是长视频时面临视觉Token过载问题——视频帧数增…
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "视觉大语言模型"
  - "视频理解"
  - "Token平衡"
  - "帧选择"
  - "Token合并"
---

# B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens

**会议**: ICCV 2025  
**arXiv**: [2412.09919](https://arxiv.org/abs/2412.09919)  
**代码**: [https://github.com/zhuqiangLu/B-VLLM.git](https://github.com/zhuqiangLu/B-VLLM.git)  
**领域**: Model Compression / Video Understanding  
**关键词**: 视觉大语言模型, 视频理解, Token平衡, 帧选择, Token合并

## 一句话总结

本文提出B-VLLM框架，通过文本条件自适应帧选择、时序帧Token合并和空间Token采样三个模块，在VLLM的上下文窗口限制内动态平衡视频的时空线索，在MVBench上带来10%的性能提升。

## 研究背景与动机

当前VLLM在处理视频特别是长视频时面临视觉Token过载问题——视频帧数增加导致Token数量迅速增长，可能超出LLM上下文窗口限制并大幅增加计算成本。现有解决方案存在时空线索失衡的问题：

- **均匀下采样帧数**（如VideoLLaMA2采样固定8帧）：忽略了视频的时序动态，可能错过与任务相关的关键帧
- **压缩每帧Token数**（如LLaMA-VID将每帧压缩为2个Token）：无法保留帧内的空间细节，导致在需要空间理解的任务上表现不佳

作者指出这是一种"时空Token失衡"问题：减少帧级Token使时序线索占主导，而均匀帧采样使空间线索被淹没。需要一种能根据任务自适应平衡时空Token分配的方法。

## 方法详解

### 整体框架

B-VLLM的流程：（1）视觉编码器将所有帧编码为初始视觉Token（包含[CLS] Token）；（2）文本条件帧选择模块利用[CLS] Token和问题文本选择最相关的$L^*$帧；（3）时序帧Token合并去除重复帧；（4）空间Token采样从选中帧中提取最相关的$R$个Token；（5）可选的空间Token合并进一步控制Token上限；（6）投影到LLM特征空间与文本Token拼接输入LLM。

### 关键设计

1. **文本条件自适应帧选择**:

    - 利用每帧的[CLS] Token（携带高层语义信息）而非所有Token来定位相关帧，平衡计算效率
    - 采用Q-Former作为帧选择网络，联合编码[CLS] Token序列$V_{[CLS]}$和文本上下文$T$，生成$L^*$个query
    - 通过Gumbel-Softmax实现可微的离散帧选择：$V^* = \Phi(Q \cdot V_{[CLS]}^\top, \tau) \cdot V$
    - 温度参数$\tau$控制离散程度，$\tau \to 0$时Gumbel-Softmax逼近one-hot分布（实际设$\tau=0.1$）
    - 选择Gumbel-Softmax而非普通Softmax的原因：Softmax会平滑掉丰富的空间视觉线索，特别是聚合多帧时
    - 选择后恢复帧的时序顺序

2. **时序帧Token合并**:

    - 问题：帧选择可能产生重复（如帧数$L < L^*$时）
    - 利用选择矩阵$S_\tau$的行向量余弦相似度检测重复帧
    - 相似度超过阈值$\gamma$的帧视为重复，通过均值合并：$V_\alpha^* = \frac{1}{|D_\alpha|} \sum_{\beta \in D_\alpha} V_\beta^*$
    - 迭代执行去重合并过程

3. **空间视觉Token采样**:

    - 使用空间Q-Former从选中帧的$M$个Token中采样$R$个最相关Token（$R \ll M$），输入为帧Token和文本上下文
    - 可选的渐进式空间Token合并策略：当Token数超过预算$\theta$时，通过Bipartite Merging反复合并最相似Token直到满足预算
    - 首次提出迭代式Token合并策略实现精细的Token数量控制

4. **与骨干LLM的集成**:

    - 通过可训练MLP投影视觉Token到LLM特征空间
    - 框架设计灵活：可独立使用（以Qwen2为骨干），也可集成到现有VLLM（如LLaMA-VID、VideoLLaMA2）

### 损失函数 / 训练策略

- 仅在LLaMA-VID-Dataset和Valley数据集上训练，确保公平对比
- 使用标准的语言建模自回归损失

## 实验关键数据

### 主实验——视频基准

| 方法 | #帧 | MVBench | VideoMME-s | VideoMME-m | VideoMME-l |
|------|-----|---------|------------|------------|------------|
| LLaMA-VID | 1fps | 39.0 | 34.2 | 34.7 | 27.1 |
| LLaMA-VID + Ours | 1fps | 43.5(+4.5) | 44.7(+10.5) | 38.8(+4.1) | 35.2(+8.1) |
| VideoLLaMA2 | 8 | 45.5 | 48.9 | 42.7 | 37.7 |
| VideoLLaMA2 + Ours | 1fps | 46.5(+1.0) | 47.2 | 44.4(+1.7) | 41.5(+3.8) |
| **B-VLLM** | 1fps | **50.8** | **60.8** | **51.8** | **47.9** |

长视频场景提升最为显著：LLaMA-VID在VideoMME-Long上提升8.1%。

### 消融实验——模块贡献

| 帧选择 | 时序合并 | 空间采样 | MVBench | VideoMME | MMBench | POPE |
|--------|----------|----------|---------|----------|---------|------|
| ✗ | ✗ | ✗ | 39.0 | 33.6 | 49.8 | 75.3 |
| ✓ | ✗ | ✗ | 39.4 | 36.5 | 58.0 | 81.1 |
| ✓ | ✓ | ✗ | 42.1 | 38.1 | 54.6 | 67.3 |
| ✓ | ✓ | ✓ | **43.5** | **39.6** | **59.3** | **83.8** |

### 关键发现

- **帧选择的有效性**：Q-Former优于Resampler（MVBench: 46.5 vs 44.3），因Q-Former的交叉注意力更适合处理跨模态Token
- **[CLS] Token的价值**：虽然信息不如Mean Pooling丰富，但训练效率高出一倍（10hrs vs 19hrs），且性能持续优于随机猜测
- **Token数量-性能权衡**：512个Token时B-VLLM性能即趋于饱和，说明高效预算控制的可行性
- **超参数敏感性**：低$\tau$（0.1）+ 高$\gamma$（0.75-1.0）效果最佳；$\tau$越低Gumbel-Softmax越接近离散选择，$\gamma$越高越避免合并不相似帧
- 8-32关键帧对大多数视频理解任务已足够
- 视频理解任务更依赖时序动态而非空间细节（从32降到16帧仅降0.9%，但空间Token从32降到16降0.6%）

## 亮点与洞察

- 将视频VLLM的Token管理问题清晰地分解为"选帧—去重—选空间Token"三个解耦阶段，每个阶段有明确的技术方案
- [CLS] Token在VLLM中的重新利用是一个简洁有效的设计——大多数VLLM直接丢弃[CLS] Token
- Gumbel-Softmax实现可微离散帧选择的技巧优雅，既保证了端到端可训练性又实现了真正的帧选择
- 框架的即插即用特性使其具有广泛的实用价值

## 局限与展望

- [CLS] Token作为帧级表示信息量有限，可能导致对需要细粒度空间信息才能判断相关性的任务选帧不准
- 在VizWiz、TextVQA等需要空间推理和OCR的图像基准上因Token数量限制表现不佳
- 帧选择模块本身（Q-Former）也引入额外计算开销和参数
- 空间Token采样固定数量$R$，未能根据帧内容自适应调整

## 相关工作与启发

- 与LLaMA-VID（极端空间压缩）和VideoLLaMA2（均匀帧采样）形成互补
- Token Merging技术（ToMe）从图像生成领域引入到VLLM的Token管理中
- 文本条件的自适应处理思路可扩展到更多多模态场景（如多图推理、文档理解）

## 评分

- 新颖性: ⭐⭐⭐⭐ 文本条件帧选择+迭代Token合并的组合设计新颖实用
- 实验充分度: ⭐⭐⭐⭐ 多基准评测、集成验证、详细消融、可视化分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，模块设计逐层推进
- 价值: ⭐⭐⭐⭐ 通用视频VLLM框架，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Vision-centric Token Compression in Large Language Model](../../NeurIPS2025/model_compression/vision-centric_token_compression_in_large_language_model.md)
- [\[NeurIPS 2025\] Learning to Factorize and Adapt: A Versatile Approach Toward Universal Spatio-Temporal Foundation Models](../../NeurIPS2025/model_compression/learning_to_factorize_and_adapt_a_versatile_approach_toward_universal_spatio-tem.md)
- [\[CVPR 2025\] DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models](../../CVPR2025/model_compression/dycoke_dynamic_compression_of_tokens_for_fast_video_large_language_models.md)
- [\[ACL 2025\] Quantification of Large Language Model Distillation](../../ACL2025/model_compression/quantification_of_large_language_model_distillation.md)
- [\[CVPR 2026\] TimeRipples: Accelerating vDiTs by Understanding the Spatio-Temporal Correlations in Latent Space](../../CVPR2026/model_compression/timeripples_accelerating_vdits_by_understanding_the_spatio-temporal_correlations.md)

</div>

<!-- RELATED:END -->
