---
title: >-
  [论文解读] GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity
description: >-
  [NeurIPS 2025][幻觉检测][object hallucination] 提出GLSim，一种无训练的LVLM物体幻觉检测方法，通过融合全局场景相似度（物体token与最后instruction token的余弦相似度）和局部视觉定位相似度（物体token与Visual Logit Lens定位的Top-K图像patch的余弦相似度），在MSCOCO上以83.7% AUROC超越SVAR 9%、Internal Confidence 10.8%。
tags:
  - "NeurIPS 2025"
  - "幻觉检测"
  - "object hallucination"
  - "hallucination detection"
  - "global-local similarity"
  - "visual logit lens"
  - "training-free"
---

# GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity

**会议**: NeurIPS 2025  
**arXiv**: [2508.19972](https://arxiv.org/abs/2508.19972)  
**代码**: [https://github.com/deeplearning-wisc/glsim](https://github.com/deeplearning-wisc/glsim)  
**领域**: 幻觉检测  
**关键词**: object hallucination, hallucination detection, global-local similarity, visual logit lens, training-free

## 一句话总结
提出GLSim，一种无训练的LVLM物体幻觉检测方法，通过融合全局场景相似度（物体token与最后instruction token的余弦相似度）和局部视觉定位相似度（物体token与Visual Logit Lens定位的Top-K图像patch的余弦相似度），在MSCOCO上以83.7% AUROC超越SVAR 9%、Internal Confidence 10.8%。

## 研究背景与动机

**领域现状**：大型视觉语言模型(LVLM)会产生物体幻觉——生成图像中不存在的物体的描述。这严重影响了模型在医疗影像、自动驾驶等高风险领域的可靠部署。**现有痛点**：现有幻觉检测方法要么依赖外部标注数据（CHAIR等），要么需要外部LLM判断（FaithScore等），要么只使用单一视角的信号。基于token概率的方法(NLL)因LLM偏好语言流畅性而失效；基于注意力的方法(SVAR)受attention sink影响；Internal Confidence直接用Visual Logit Lens的最大概率可能过度自信。**核心矛盾**：单一全局或局部信号各有盲区——全局方法会因场景语义关联误判上下文合理但视觉不存在的物体（如生日场景中的"dining table"）；局部方法会因视觉相似物体误判（如摩托车皮座与"handbag"）。**切入角度**：首次将全局和局部embedding相似度信号统一到一个框架中，利用两者互补的优势。

## 方法详解

### 整体框架
GLSim是一个无训练的物体级幻觉检测框架。对于LVLM生成文本中提及的每个物体o，计算两个分数：(1) 全局相似度——物体embedding与场景embedding的余弦相似度；(2) 局部相似度——物体embedding与通过Visual Logit Lens定位的Top-K图像patch embedding的平均余弦相似度。最终GLSim分数是两者的加权组合。

### 关键设计

1. **基于Visual Logit Lens的无监督物体定位**:

    - 功能：在不依赖外部标注或检测器的情况下，定位图像中与某物体最相关的区域
    - 核心思路：将每个视觉token v_i在decoder层l的隐藏表示h_l(v_i)通过unembedding矩阵W_U映射到词汇空间，得到每个视觉patch预测物体词o的概率softmax(VLL_l(v_i))[o]，选择概率最高的Top-K个patch作为物体o的定位区域I(o)
    - 设计动机：Visual Logit Lens比注意力权重更准确地定位物体（相关实验表明AUROC提升12.5%），同时不需要任何外部检测器

2. **局部相似度分数**:

    - 功能：检验物体是否在图像的特定区域有真实的视觉证据
    - 核心思路：计算物体token embedding h_{l'}(o)与Top-K个定位patch的隐藏表示h_l(v_i)之间的平均余弦相似度：s_local = (1/K)·Σ_{v_i∈I(o)} sim(h_l(v_i), h_{l'}(o))。真实物体的对应区域embedding会与其有高相似度，幻觉物体则对应到不相关区域导致低相似度
    - 设计动机：直接使用embedding similarity比使用Logit Lens的概率值更稳定可靠——概率可能过度自信（如Internal Confidence的问题），而embedding空间的相似度提供更fine-grained的信号

3. **全局相似度分数**:

    - 功能：判断物体是否与整体场景语义一致
    - 核心思路：计算物体token embedding与instruction prompt最后一个token的隐藏表示之间的余弦相似度：s_global = sim(h_l(v,t), h_{l'}(o))。最后一个instruction token编码了模型对图像和文本上下文的综合理解
    - 设计动机：最后instruction token比"最后image token"或"所有image token平均"更能捕获场景语义（消融实验中AUROC高8%），它提供了一个物体在场景中"合不合理"的高层判断

### 损失函数 / 训练策略
完全无训练，直接利用LVLM的内部表示。最终GLSim分数 = w·s_global + (1-w)·s_local，w=0.6在多个场景下一致表现最优。层索引l和l'通过消融实验选择（LLaVA: l=32, l'=31; Shikra: l=30, l'=27）。

## 实验关键数据

### 主实验

| 数据集/模型 | 指标 | GLSim | SVAR | Internal Conf. | Contextual Lens | NLL |
|------------|------|-------|------|----------------|-----------------|-----|
| MSCOCO/LLaVA-7B | AUROC | **83.7** | 74.7 | 72.9 | 75.4 | 63.7 |
| MSCOCO/LLaVA-13B | AUROC | **84.8** | 75.2 | 71.0 | 78.7 | 63.1 |
| MSCOCO/MiniGPT-4 | AUROC | **87.0** | 83.6 | 75.7 | 84.9 | 59.4 |
| MSCOCO/Shikra | AUROC | **83.0** | 70.7 | 69.1 | 69.5 | 60.4 |
| Objects365/LLaVA-7B | AUROC | **72.6** | 64.9 | 68.7 | 63.2 | 62.9 |
| Objects365/MiniGPT-4 | AUROC | **74.8** | 71.0 | 68.5 | 70.2 | 56.7 |

### 消融实验

| 配置 | LLaVA AUROC | Shikra AUROC | 说明 |
|------|-------------|-------------|------|
| 仅全局(s_global) | 79.3 | 78.9 | 单独使用已超越所有基线 |
| 仅局部Top-K(s_local) | 78.8 | 76.8 | 与全局互补 |
| GLSim(全局+局部Top-K) | **83.7** | **83.0** | 融合提升+4.9/+6.2 |
| 定位方法:Attention | 66.3(局部) | 65.0 | 注意力权重不可靠 |
| 定位方法:Cosine Sim | 76.2(局部) | 70.1 | 次优 |
| 定位方法:Logit Lens | **78.8**(局部) | **76.8** | 最优定位 |
| w=0.4 | 82.5 | - | 偏向局部 |
| w=0.6 | **83.7** | - | 最优平衡 |
| w=0.8 | 82.0 | - | 偏向全局 |
| K=8/16/32/64 | 82→83→**83.7**→82 | - | K=32最优，约6%图像token |

### 关键发现
- GLSim在所有LVLM和数据集组合上一致超越所有基线，在Shikra上提升尤为显著(+12.7% AUROC vs SVAR)
- 全局和局部信号确实互补：两者单独使用均已超越所有现有方法，融合后进一步提升
- Internal Confidence可能对幻觉物体过度自信——因为Visual Logit Lens直接给出的概率值可能指向错误区域但概率很高
- Visual Logit Lens作为定位方法比注意力权重好12.5%，比余弦相似度好2.6%
- 最优层选择在后期中间层（而非最后一层），支持"最优任务层不一定是最终层"的观察

## 亮点与洞察
- 全局-局部互补的思路简洁直觉且极为有效，首次在幻觉检测中证明两者的互补价值
- 无训练即插即用——不需要额外训练或外部模型，直接利用LVLM内部表示
- 全面的基准测试——系统性地首次比较了5种物体级幻觉检测方法，填补了该领域的benchmarking空白
- 定性分析非常直观：Figure 2清晰展示了全局失败/局部成功和局部失败/全局成功的互补案例

## 局限与展望
- 仅处理物体存在性幻觉，不处理属性(颜色/大小)和关系(空间位置)级幻觉
- K和w的选择虽然在实验中鲁棒，但最优值可能随输入分辨率变化
- 需要选择合适的层索引l和l'，不同模型的最优层不同
- 检测后如何利用GLSim分数来修正或缓解幻觉是待探索的方向

## 相关工作与启发
- **CHAIR**：基于ground-truth匹配的幻觉评估指标，而GLSim不需要标注
- **Internal Confidence**：使用Visual Logit Lens概率，GLSim改进为用embedding相似度+Top-K聚合
- **SVAR**：注意力权重方法，受attention sink和文本token偏移影响
- 启发：全局-局部互补思路可迁移到其他检测任务；GLSim可与训练端的幻觉缓解方法(如Causal-LLaVA)串联使用

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次在幻觉检测中系统性地融合全局和局部信号，Visual Logit Lens的创新应用
- 实验充分度: ⭐⭐⭐⭐⭐ 5种LVLM × 2个数据集 × 5种基线 × 大量消融（K/w/层/定位方法/全局设计），非常充分
- 写作质量: ⭐⭐⭐⭐ 动机和方法描述清晰，定性分析直观有效
- 价值: ⭐⭐⭐⭐ 无训练即插即用的特性使其实用性极强，代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] PAS: Prelim Attention Score for Detecting Object Hallucinations in Large Vision-Language Models](../../CVPR2026/hallucination/pas_prelim_attention_score_for_detecting_object_hallucinations_in_large_vision-l.md)
- [\[CVPR 2026\] Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations](../../CVPR2026/hallucination/beyond_global_scores_fine_grained_token_grounding_as_robust_detector_of_lvlm_hallucinations.md)
- [\[AAAI 2026\] Causally-Grounded Dual-Path Attention Intervention for Object Hallucination Mitigation in LVLMs](../../AAAI2026/hallucination/causally-grounded_dual-path_attention_intervention_for_objec.md)
- [\[ACL 2025\] Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models](../../ACL2025/hallucination/retrieval_visual_contrastive_decoding_to_mitigate_object_hallucinations_in_large.md)
- [\[ICCV 2025\] Why LVLMs Are More Prone to Hallucinations in Longer Responses: The Role of Context](../../ICCV2025/hallucination/why_lvlms_are_more_prone_to_hallucinations_in_longer_responses_the_role_of_conte.md)

</div>

<!-- RELATED:END -->
