---
title: >-
  [论文解读] Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models
description: >-
  [ICLR 2026][多模态VLM][KV Cache压缩] 发现LVLM中KV Cache存在模态特异和注意力头特异的语义冗余，仅靠重要性选择会丢失语义覆盖，提出MixKV按头自适应混合重要性与多样性分数进行KV Cache压缩，在极端压缩下平均提升5.1%。 领域现状：LVLMs处理高分辨率图像和长视频时生成大量KV对…
tags:
  - "ICLR 2026"
  - "多模态VLM"
  - "KV Cache压缩"
  - "语义冗余"
  - "多样性"
  - "注意力头"
  - "视觉语言模型"
---

# Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models

**会议**: ICLR 2026  
**arXiv**: [2510.20707](https://arxiv.org/abs/2510.20707)  
**代码**: [GitHub](https://github.com/xuyang-liu16/MixKV)  
**领域**: 多模态VLM/推理效率  
**关键词**: KV Cache压缩, 语义冗余, 多样性, 注意力头, 视觉语言模型

## 一句话总结
发现LVLM中KV Cache存在模态特异和注意力头特异的语义冗余，仅靠重要性选择会丢失语义覆盖，提出MixKV按头自适应混合重要性与多样性分数进行KV Cache压缩，在极端压缩下平均提升5.1%。

## 研究背景与动机

**领域现状**：LVLMs处理高分辨率图像和长视频时生成大量KV对，KV Cache成为内存瓶颈。现有方法（SnapKV、AdaKV等）基于注意力重要性保留关键KV对、丢弃次要的。

**现有痛点**：(1) 视觉信息比文本有更多语义冗余——图像中相似纹理/重复模式导致KV对间余弦相似度高达0.6-0.8（文本仅0.2-0.4）；(2) 不同注意力头的冗余度差异巨大——有些头平均相似度>0.9，有些<0.3；(3) 仅按重要性选择→保留的KV对方高度相似→丢失了全局语义覆盖。

**核心矛盾**：t-SNE可视化清晰显示：SnapKV（仅重要性）选中的KV对只覆盖了完整分布的一个小子集，大量信息丧失。

**切入角度**：在重要性基础上引入多样性——高冗余头（KV对相似度高）更强调多样性以避免冗余，低冗余头保持重要性优先。

**核心 idea**：按头自适应地将冗余度作为重要性和多样性分数的混合权重。

## 方法详解

### 整体框架
MixKV 要解决的是：现有 KV Cache 压缩方法只按注意力重要性挑 KV 对，结果在视觉这种高冗余场景里保留下来的全是彼此相似的对，丢掉了全局语义覆盖。它的做法是给每个 KV 对同时算两类分数——衡量"该不该留"的重要性分数和衡量"留了会不会重复"的多样性分数——再按每个注意力头各自的冗余程度，把两者混成一个综合分数，最后取综合分数最高的 B 个 KV 对保留。整套流程不改原方法的重要性计算，只在打分阶段插入多样性，因此能即插即用地套在 SnapKV、AdaKV 等任意基于重要性的压缩方法上。

### 关键设计

**1. 冗余度量化：用一个 O(T) 的标量刻画每个头有多冗余**

不同注意力头的冗余差异巨大——有的头里 KV 对平均相似度超过 0.9，有的不到 0.3，所以混合权重必须按头来定，先得有个便宜的冗余度量。MixKV 对每个头取归一化后的 Key 向量，算它们之间离对角线的平均余弦相似度 $\bar{r}_h^l$ 作为该头的冗余度。直接两两比较是 $O(T^2)$，论文用代数恒等式 $\sum_{i,j} R_{i,j} = T^2 \|\hat{\bar{K}}_h^l\|_2^2$ 把求和折叠成"平均 Key 的模长"，于是只要 $O(T)$ 就能算出 $\bar{r}_h^l = \frac{T^2\|\hat{\bar{K}}_h^l\|_2^2 - T}{T(T-1)}$。$\bar{r} \to 1$ 说明这个头里的 Key 高度雷同、该强调多样性；$\bar{r} \to 0$ 说明已经够散、维持重要性优先即可。

**2. 多样性分数：用与平均 Key 的偏离程度近似"独特性"**

要避免两两比较的 $O(T^2)$ 开销，MixKV 不去算每个 Key 和其它所有 Key 的关系，而是只算它跟全局平均 Key 的关系：取负余弦相似度 $s_i^{\text{div}} = -\hat{K}_{h,i}^l \cdot \hat{\bar{K}}_h^l$ 作为多样性分数。越不像平均值的 Key，说明它越偏离主流、携带的信息越独特，分数就越高；扎堆在平均值附近的 Key 互相冗余，分数就低。这样多样性也压到了 $O(T)$，和重要性分数同一个量级。

**3. 自适应混合：让冗余越高的头越听多样性的**

有了按头的冗余度和每个 Key 的两类分数，最后把它们融成综合分数 $s_i^{\text{comp}} = (1-\bar{r}_h^l) \cdot s_{\text{imp},i} + \bar{r}_h^l \cdot s_{\text{scaled},i}^{\text{div}}$。权重直接由冗余度 $\bar{r}_h^l$ 决定：头越冗余，多样性占的比重越大，从而主动跳过那些"重要但和已选项高度相似"的 KV 对；头本身就散，则几乎退回纯重要性选择。这正是 MixKV 区别于固定加权的关键——混合比例不是全局超参，而是逐头随冗余度自适应。

**4. 重要性分数增强：把两类重要性信号都纳进来**

为了让重要性这一侧的信号更全，$s_{\text{imp}}$ 同时整合外在和内在两类来源：外在重要性 $s_{\text{imp}}^{\text{ex}}$ 来自注意力窗口（KV 对被后续 query 关注的程度），内在重要性 $s_{\text{imp}}^{\text{in}}$ 来自 VNorm（Value 向量本身的范数），两者相加得 $s_{\text{imp}} = s_{\text{imp}}^{\text{ex}} + s_{\text{imp}}^{\text{in}}$，再送进上面的混合公式。

## 实验关键数据

### 主实验
极端压缩(budget=64)下多模态理解：

| 方法 | DocVQA | OCRBench | TextVQA | ChartQA | 平均提升 |
|------|--------|----------|---------|---------|---------|
| SnapKV | 47.3 | 31.9 | 57.1 | 42.7 | — |
| SnapKV+MixKV | **48.8** | **36.1** | **59.0+** | **45+** | +5.1% |
| AdaKV | 基线 | 基线 | 基线 | 基线 | — |
| AdaKV+MixKV | **+** | **+** | **+** | **+** | +5.1% |

### GUI Grounding任务（ScreenSpot-v2）

| 方法 | 准确率 | 说明 |
|------|--------|------|
| SnapKV | 基线 | budget=64 |
| SnapKV+MixKV | **+8.0%** | 多样性在UI元素定位中很重要 |
| AdaKV+MixKV | **+9.0%** | 更大提升 |

### 关键发现
- t-SNE可视化证实MixKV让SnapKV的选择覆盖了更广的KV分布
- GUI Grounding任务提升最大(+8-9%)——因为UI元素分散在图像各处，多样性选择覆盖更多位置信息
- 推理效率与基线方法相当——冗余度和多样性分数都是 $O(T)$ 计算
- 在纯文本LLM(Qwen2.5、Llama-3.1)上也有一致提升

## 亮点与洞察
- **视觉KV冗余的量化分析**：首次系统量化LVLM中KV对的模态特异和头特异冗余。余弦相似度从LLM的0.2-0.4飙升到LVLM的0.6-0.8，这个数据有说服力。
- **t-SNE可视化的直觉**：一张图说明了仅靠重要性为什么不够——SnapKV选中的点只覆盖分布的一角，MixKV覆盖更广。
- **$O(T)$ 冗余度计算**：利用代数恒等式避免了 $O(T^2)$ 的两两比较，保证了实际可用性。

## 局限与展望
- 多样性分数仅考虑Key（不考虑Value），Value的冗余模式可能不同
- 负余弦相似度作为多样性代理是否是最优选择？其他距离度量未探索
- 全局平均Key作为锚点可能受异常值影响
- 仅在7-8B模型上验证，更大模型（70B+）效果未知

## 相关工作与启发
- **vs SnapKV**: SnapKV仅用注意力重要性，MixKV在其基础上+多样性，即插即用+5.1%
- **vs AdaKV**: AdaKV自适应分配各头的淘汰预算，MixKV自适应分配各头的重要性vs多样性权重，正交且可叠加
- **vs SparseMM**: SparseMM用头重要性分配不对称预算，MixKV关注头内部的冗余特性

## 评分
- 新颖性: ⭐⭐⭐⭐ 重要性+多样性的混合思路清晰有效，冗余度分析有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型、多任务、多budget、即插即用验证充分
- 写作质量: ⭐⭐⭐⭐ 分析可视化丰富，方法描述清晰
- 价值: ⭐⭐⭐⭐ 对LVLM部署优化有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression](../../CVPR2026/multimodal_vlm/flashcache_frequency_kv_cache_compression.md)
- [\[ICCV 2025\] AirCache: Activating Inter-Modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference](../../ICCV2025/multimodal_vlm/aircache_activating_inter-modal_relevancy_kv_cache_compression_for_efficient_lar.md)
- [\[ICLR 2026\] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)
- [\[ICLR 2026\] A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models](a-tpt_angular_diversity_calibration_properties_for_test-time_prompt_tuning_of_vi.md)
- [\[NeurIPS 2025\] CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization](../../NeurIPS2025/multimodal_vlm/coido_efficient_data_selection_for_visual_instruction_tuning_via_coupled_importa.md)

</div>

<!-- RELATED:END -->
