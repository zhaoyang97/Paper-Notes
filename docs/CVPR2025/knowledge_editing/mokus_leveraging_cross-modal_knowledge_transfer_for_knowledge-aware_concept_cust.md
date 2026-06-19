---
title: >-
  [论文解读] MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization
description: >-
  [CVPR2025][知识编辑][概念定制] 提出 MoKus 框架，发现并利用"跨模态知识迁移"现象——在 LLM 文本编码器中更新知识会自动传递到视觉生成端——实现知识感知的概念定制，两阶段设计：先学视觉锚点表示，再秒级更新文本知识绑定。 领域现状：概念定制（Concept Customization）旨在根据用户提供的…
tags:
  - "CVPR2025"
  - "知识编辑"
  - "概念定制"
  - "跨模态知识迁移"
  - "LLM文本编码器"
  - "DiT"
---

# MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization

**会议**: CVPR2025  
**arXiv**: [2603.12743](https://arxiv.org/abs/2603.12743)  
**代码**: [GitHub](https://chenyangzhu1.github.io/MoKus)  
**领域**: 知识编辑  
**关键词**: 概念定制, 知识编辑, 跨模态知识迁移, LLM文本编码器, DiT

## 一句话总结
提出 MoKus 框架，发现并利用"跨模态知识迁移"现象——在 LLM 文本编码器中更新知识会自动传递到视觉生成端——实现知识感知的概念定制，两阶段设计：先学视觉锚点表示，再秒级更新文本知识绑定。

## 研究背景与动机
**领域现状**：概念定制（Concept Customization）旨在根据用户提供的概念图像生成新的定制图像。现有方法（DreamBooth、Textual Inversion 等）用 rare token（如 \<sks\>）表示目标概念。

**Rare token 的两大缺陷**：
   - **不稳定性**：rare token 在预训练数据中几乎不出现，与其他文本组合时生成质量不稳定
   - **知识无感知**：rare token 仅绑定视觉外观，无法承载概念的固有知识（如"小美人鱼雕像在丹麦"）

**核心矛盾**：如何将多条自然语言知识高效绑定到目标视觉概念，实现知识感知的定制生成？

**关键观察——跨模态知识迁移**：在 LLM 文本编码器中用知识编辑技术更新某个问题的答案后，生成的图像会对应更新后的答案。例如把"贝多芬最喜欢的乐器"的答案更新为"吉他"，生成图像就会出现吉他。

## 方法详解

### 整体框架（两阶段）

**阶段一：视觉概念学习（Visual Concept Learning）**
- 将目标概念与 rare token 关联，作为"锚点表示"（anchor representation）
- 用 Rectified Flow 训练 DiT + LoRA，最小化预测速度场与真实速度场的 MSE
- LoRA 参数仅加在 MMDiT 自注意力层，高效微调
- 输出：锚点表示 h = ϕ(P)，存储目标概念的视觉信息

**阶段二：文本知识更新（Textual Knowledge Updating）**
- 将每条知识 $k_i$ 转换为问题格式 $q_i$，期望输出为锚点表示 y
- 输入问题到 LLM 编码器，提取隐藏状态 $h_i$ 和梯度
- 计算更新方向 $v_i = -\eta \cdot \|h_i\|^2 \cdot \nabla y_i$
- 求解正则化最小二乘得到参数偏移的闭式解 $\Delta\theta_t^* = (H^\top H + I)^{-1} H^\top V$
- 直接加到 LLM 编码器可更新层的参数上：$\hat{\theta}_t = \theta_t + \Delta\theta_t^*$
- 仅修改 MLP 层（Gate/Up Projection，第 18-26 层），单条知识更新只需几秒

### KnowCusBench 基准
- 35 个概念（来自 DreamBench、CustomConcept101、Unsplash）
- 每个概念 5 条知识（6 个角度：个人关系、物理属性、功能、价值、来源、情感）
- 199 条生成 prompt（4 个角度）
- 共 5,975 张评估图像

## 实验关键数据

### 定量对比（表1）

| 方法 | 重建 CLIP-I↑ | 重建 CLIP-I-Seg↑ | 生成 CLIP-I↑ | 生成 CLIP-I-Seg↑ | 生成 CLIP-T↑ | Pick Score↑ | 训练时间↓ |
|------|-----------|----------------|-----------|----------------|-----------|-----------|----------|
| Naive-DB | **0.874** | 0.758 | 0.789 | 0.717 | 0.291 | 20.80 | ~27min |
| Enc-FT | 0.582 | 0.553 | 0.591 | 0.562 | 0.197 | 18.34 | ~10min |
| **MoKus** | 0.867 | **0.764** | 0.761 | **0.718** | **0.305** | **21.30** | **~6min** |

- MoKus 在更关键的 CLIP-I-Seg 指标上最优（过滤背景后评估概念保真度更准确）
- Enc-FT 直接微调 LLM 编码器严重破坏输出分布，各指标全面崩溃

### 知识数量消融（表2）

| 知识数量 | 重建 CLIP-I-Seg | 生成 CLIP-T | 训练时间(s) |
|---------|----------------|-----------|------------|
| 1 | 0.761 | 0.304 | 331.3 |
| 3 | 0.761 | 0.305 | 345.1 |
| 5 | 0.764 | 0.305 | 360.0 |

- 知识数量从 1 增到 5，性能稳定甚至略升，每增一条仅多约 7 秒

### 缩放因子消融
- η=1e-4 时性能严重崩溃（CLIP-I 降至 0.557），η=1e-6 为最优点
- η 在 1e-5~1e-8 范围内性能稳定，对超参不敏感

### 关键发现
- 可扩展到虚拟概念创建（纯文字描述即可创建新概念）和概念擦除（修改外观描述实现概念删除）
- 在世界知识基准 WISE 上也能提升模型表现

### 实现细节
- 基于 Qwen-Image 模型，8 张 H800 GPU
- 视觉概念学习：lr=2e-4，AdamW 优化器，Diffusers 默认 LoRA 配置
- 知识更新：使用 UltraEdit 方法，修改 LLM 编码器第 18-26 层的 Gate/Up Projection 矩阵（共 16 个参数矩阵），scaling factor η=1e-6
- 评估：重建+生成两部分，每部分 5 个不同随机种子，共 5,975 张图像

### 与并行工作的区别
- GapEval 和 UniSandbox 通过直接微调 LLM 文本编码器探索跨模态知识迁移，但均未发现显著证据
- MoKus 使用知识编辑技术（非直接微调）实现精确更新，成功观察到跨模态迁移

## 亮点
- **跨模态知识迁移的发现与利用**：首次系统性地证明 LLM 文本编码器中的知识更新可以迁移到视觉生成，区别于并行工作（GapEval、UniSandbox）的失败尝试
- **秒级知识绑定**：闭式解使得每条知识更新在秒级完成，极大优于需重训的方案
- **新任务+新基准**：定义了 Knowledge-Aware Concept Customization 任务并构建 KnowCusBench
- **锚点表示的桥梁作用**：rare token 从"最终表示"降级为"中间桥梁"，自然语言知识成为真正的概念载体

## 局限性
- 依赖 LLM 文本编码器的知识编辑能力，编辑干扰（locality）可能影响其他知识
- 知识以文本问答格式表达，非所有概念知识都适合这种形式化
- 仅在 Qwen-Image 上测试，对其他 T2I 架构的泛化性待验证
- KnowCusBench 概念数量有限（35 个），更大规模评估

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 跨模态知识迁移观察新颖，知识感知定制是全新任务
- 实验充分度: ⭐⭐⭐⭐ 自建基准、多 baseline 对比、多应用扩展
- 写作质量: ⭐⭐⭐⭐ 动机清晰，观察-方法-验证的叙事流畅
- 价值: ⭐⭐⭐⭐ 为概念定制技术开辟了知识感知的新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Do Text Edits Generalize to Visual Generation? Benchmarking Cross-Modal Knowledge Editing in UMMs](../../ICML2026/knowledge_editing/do_text_edits_generalize_to_visual_generation_benchmarking_cross-modal_knowledge.md)
- [\[ACL 2025\] BMIKE-53: Investigating Cross-Lingual Knowledge Editing with In-Context Learning](../../ACL2025/knowledge_editing/bmike-53_investigating_cross-lingual_knowledge_editing_with_in-context_learning.md)
- [\[ACL 2025\] Structure-aware Domain Knowledge Injection for Large Language Models](../../ACL2025/knowledge_editing/structure-aware_domain_knowledge_injection_for_large_language_models.md)
- [\[ACL 2025\] ToxEdit: Adaptive Detoxification Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing](../../ACL2025/knowledge_editing/adaptive_detoxification_safeguarding_general_capabilities_of_llms_through_toxici.md)
- [\[ACL 2025\] Towards a Principled Evaluation of Knowledge Editors](../../ACL2025/knowledge_editing/towards_a_principled_evaluation_of_knowledge_editors.md)

</div>

<!-- RELATED:END -->
