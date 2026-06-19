---
title: >-
  [论文解读] Hyperbolic Safety-Aware Vision-Language Models
description: >-
  [CVPR 2025][LLM安全][双曲空间] HySAC 提出在双曲空间中构建安全感知的视觉语言模型，通过蕴含锥（entailment cone）将安全/不安全内容映射到双曲空间的不同区域（安全内容靠近原点、不安全内容远离原点），使模型具备安全内容分类和动态重定向能力，在检索安全性和NSFW检测上显著超越现有遗忘方法。
tags:
  - "CVPR 2025"
  - "LLM安全"
  - "双曲空间"
  - "内容安全"
  - "CLIP"
  - "蕴含学习"
  - "NSFW检测"
---

# Hyperbolic Safety-Aware Vision-Language Models

**会议**: CVPR 2025  
**arXiv**: [2503.12127](https://arxiv.org/abs/2503.12127)  
**代码**: [https://github.com/aimagelab/HySAC](https://github.com/aimagelab/HySAC)  
**领域**: 多模态VLM  
**关键词**: 双曲空间, 内容安全, CLIP, 蕴含学习, NSFW检测

## 一句话总结
HySAC 提出在双曲空间中构建安全感知的视觉语言模型，通过蕴含锥（entailment cone）将安全/不安全内容映射到双曲空间的不同区域（安全内容靠近原点、不安全内容远离原点），使模型具备安全内容分类和动态重定向能力，在检索安全性和NSFW检测上显著超越现有遗忘方法。

## 研究背景与动机
- **领域现状**: CLIP等VLM在大规模网络数据上训练，不可避免地包含暴力、色情等不安全内容。现有方法（如Safe-CLIP）主要通过"遗忘"（unlearning）来消除模型对不安全概念的知识
- **现有痛点**: 遗忘方法虽然有效减少不安全输出，但限制了模型区分安全/不安全内容的能力——模型丧失了判别力。这对内容审核、用户自主选择等场景不利
- **核心矛盾**: 遗忘 vs 感知——遗忘掉NSFW知识意味着模型无法检测不安全内容，也无法提供安全替代
- **本文解决什么**: 从"遗忘"范式转向"感知"范式，让模型同时知道什么是安全的和什么是不安全的
- **切入角度**: 利用双曲空间天然的层级表征能力，将安全与不安全内容组织为蕴含层级结构
- **核心idea**: 安全内容是一般概念（靠近原点），不安全内容是其具体化（远离原点），利用蕴含锥建模两者的非对称关系

## 方法详解

### 整体框架
HySAC 基于 CLIP 架构，将视觉和文本编码器输出通过指数映射投射到 Lorentz 双曲空间。使用安全/不安全图文四元组数据集 $(I_k, T_k, I_k^\star, T_k^\star)$ 进行微调。训练目标包含两部分：双曲安全对比学习（对齐图文对）和双曲安全蕴含学习（建立安全层级）。推理时通过安全遍历（traversal）机制在双曲空间中移动查询嵌入，实现安全/不安全内容的动态切换检索。

### 关键设计
1. **双曲空间嵌入与蕴含层级**:
    - 功能：在嵌入空间中建立安全内容的显式层级结构
    - 核心思路：使用 Lorentz 模型作为双曲空间实现。定义四层蕴含不等式链：$g_T(T_k) \ll g_I(I_k) \ll g_T(T_k^\star) \ll g_I(I_k^\star)$，即安全文本最靠近原点，安全图像次之，不安全文本再次之，不安全图像最远。通过可学习投影标量 $\alpha_{img}$、$\alpha_{txt}$ 和指数映射将欧氏编码器输出投射到双曲面上
    - 设计动机：双曲空间天然适合表征层级结构（树状结构可低失真嵌入），蕴含关系可以同时保持模态内（文本-图像）和安全性（安全-不安全）两个维度的层级

2. **双曲安全对比 + 蕴含联合损失**:
    - 功能：在双曲空间中同时对齐图文对并分离安全/不安全区域
    - 核心思路：对比损失 $L_{\text{hSC}}$ 基于负 Lorentz 距离对安全对 $(I,T)$、不安全对 $(I^\star,T^\star)$ 及跨安全性对分别计算。蕴含损失 $L_{\text{hSE}}$ 利用蕴含锥的半孔径 $\omega(\mathbf{q}) = \sin^{-1}(\frac{2K}{\sqrt{\kappa}\|\tilde{\mathbf{q}}\|})$ 和外角 $\phi$，约束图像位于对应文本的锥内，并约束不安全文本位于安全图像的锥内。总损失 $L = L_{\text{hSC}} + L_{\text{hSE}}$
    - 设计动机：仅对比学习无法建立安全层级（在欧氏空间中蕴含关系无效），蕴含损失是实现安全感知的关键；联合训练确保检索性能和安全感知同时优化

3. **安全遍历机制 (Safety Traversals)**:
    - 功能：推理时动态调整查询嵌入在安全/不安全区域之间的位置
    - 核心思路：计算每类内容 $X \in \{T, I, T^\star, I^\star\}$ 到根特征 $\mathbf{r}$ 的平均距离 $\mu_X$，定义边界 $\tau_X = \mu_X + \tanh(\frac{\mu_X - \alpha}{\kappa}) + 1$。沿方向向量 $\mathbf{v}_{\text{dir}} = \mathbf{q} - \mathbf{r}$ 移动查询到目标边界：$\mathbf{q}^* = \mathbf{r} + \tau_X \cdot \frac{\mathbf{v}_{\text{dir}}}{\|\mathbf{v}_{\text{dir}}\|}$
    - 设计动机：基于距离的区域分离使得简单的方向移动就可以实现安全/不安全内容的切换，为用户提供灵活的内容审核控制

### 损失函数 / 训练策略
- 使用 AdamW 优化器，weight decay=0.2，batch size=256，训练 20 epochs
- 视觉和文本编码器使用 LoRA (r=16) 微调，减少参数量
- 关键超参数：温度 $\tau=0.07$，投影标量初始化 $\alpha_{img}=\alpha_{txt}=1/\sqrt{512}$，曲率 $c=1.0$（可学习）
- 所有标量在对数空间中学习，混合精度训练（指数映射和损失用FP32保证数值稳定）

## 实验关键数据

### 主实验 — 安全检索（ViSU 测试集）

| 模型 | Safe T→I R@1 | Safe I→T R@1 | Unsafe→Safe T→I R@1 | Unsafe→Safe I→T R@1 |
|------|-------------|-------------|-------------------|-------------------|
| CLIP | 36.8 | 39.8 | 2.0 | 4.6 |
| Safe-CLIP | 45.9 | 45.3 | 8.0 | 19.1 |
| MERU⋆ | 50.0 | 51.2 | 2.3 | 5.7 |
| **HySAC** | **49.8** | **48.2** | **30.5** | **42.1** |

### 消融实验

| 配置 | Safe T→I R@1 | Unsafe→Safe T→I R@1 | Unsafe→Safe I→T R@1 | 说明 |
|------|-------------|-------------------|------|------|
| w/o Ent（仅对比） | 52.3 | 4.1 | 5.5 | 无蕴含损失，安全重定向失败 |
| w/o S-Ent（无安全蕴含） | 51.0 | 1.4 | 7.4 | 无安全层级，重定向几乎无效 |
| HySAC (完整) | 49.8 | 30.5 | 42.1 | 安全重定向大幅提升 |

### 关键发现
- HySAC 在不安全→安全检索上远超 Safe-CLIP（30.5 vs 8.0 R@1），同时在纯安全检索上保持竞争力
- 不安全内容检索方面，HySAC 同样表现最佳（R@1: 81.4 vs CLIP 73.1），证明感知范式比遗忘范式更全面
- 嵌入距离分布可视化清晰展示了四层分离的层级结构（safe text → safe image → unsafe text → unsafe image）
- 在真实NSFW数据集（NudeNet、NSFW URLs、SMID）上，HySAC实现96.2%的安全检索率
- HySAC 还可作为NSFW分类器，在NudeNet和Mixed NSFW上达到竞争性性能，尽管并非为分类设计

## 亮点与洞察
- "感知优于遗忘"的范式转变是核心贡献：保留模型对不安全内容的知识，但赋予其区分能力，比简单删除更灵活可控
- 双曲空间的蕴含锥天然适合建模"安全→不安全"的非对称包含关系，这种层级结构在欧氏空间中无法有效表达
- 安全遍历机制优雅地实现了推理时的灵活控制，同一模型可同时服务于安全检索、内容审核和NSFW分类三种用途
- 四层不等式链 $g_T(T) \ll g_I(I) \ll g_T(T^\star) \ll g_I(I^\star)$ 的设计简洁而完备地覆盖了模态和安全性两个维度的层级
- 嵌入空间可视化（Figure 2）直观展示了四种内容类型的清晰分离，验证了方法的有效性
- LoRA微调策略使得计算开销可控，便于在已有CLIP权重上快速适配
=0.8$ 需要经验设定，在不同数据集分布下可能需要重新校准
- 仅在 CLIP (ViT-L/14) 级别的检索模型上验证，是否能推广到生成式VLM（如扩散模型）或更大模型有待探索
- ViSU 数据集中安全/不安全对是结构化构造的，与真实世界不安全内容的分布可能存在差异
- 安全遍历在语义上可能引入偏移——重定向后的检索虽然安全但相关性可能降低（缺少safe alternative数据集无法评估此方面）
- 当前安全分类基于固定的20个NSFW类别，新型有害内容（如deepfake、AI生成内容）未被纳入
- FP32精度要求使训练成本高于标准CLIP微调散模型）有待探索
- ViSU 数据集中安全/不安全对是人工构造的，与真实世界不安全内容的分布可能存在差异
- 安全遍历在语义上可能引入偏移——重定向后的检索虽然安全但可能不够相关

## 相关工作与启发
- **vs Safe-CLIP**: Safe-CLIP 通过遗忘擦除不安全知识，HySAC 保留知识但建立层级分离，更灵活可控，且检索性能全面超越
- **vs MERU**: MERU 在双曲空间建模模态层级（文本→图像）但不考虑安全性，HySAC 在此基础上扩展为安全性+模态的双层级
- **vs HyCoCLIP**: HyCoCLIP 利用物体级组合增强双曲CLIP的视觉理解，HySAC 引入安全蕴含作为全新维度
- **vs Schramowsky et al.**: 其方法用NSFW概念做负引导，属于推理时干预；HySAC通过训练时重构嵌入空间实现根本性安全感知
- **vs NudeNet/Q16**: 这些是专用分类器，HySAC在NSFW分类上达到可比性能，同时还支持检索和重定向

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将双曲蕴含锥应用于VLM安全性，范式从遗忘到感知的转变具有开创性
- 实验充分度: ⭐⭐⭐⭐ ViSU评估全面，真实NSFW数据集验证有说服力，但缺少在生成模型上的验证
- 写作质量: ⭐⭐⭐⭐ 数学表述严谨，但公式密集使阅读门槛较高
- 价值: ⭐⭐⭐⭐ 为VLM安全性提供了新思路，但实际部署中安全遍历的效果需进一步验证

---

> 本笔记基于论文全文阅读生成，覆盖了 Preliminaries、Method、Experiments 和 Analysis 全部内容。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting](../../NeurIPS2025/llm_safety/zero-shot_robustness_of_vision_language_models_via_confidence-aware_weighting.md)
- [\[NeurIPS 2025\] Geo-Sign: Hyperbolic Contrastive Regularisation for Geometrically Aware Sign Language Translation](../../NeurIPS2025/llm_safety/geo-sign_hyperbolic_contrastive_regularisation_for_geometrically_aware_sign_lang.md)
- [\[NeurIPS 2025\] Approximate Domain Unlearning for Vision-Language Models](../../NeurIPS2025/llm_safety/approximate_domain_unlearning_for_visionlanguage_models.md)
- [\[CVPR 2025\] CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[CVPR 2025\] ForensicZip: More Tokens are Better but Not Necessary in Forensic Vision-Language Models](forensiczip_more_tokens_are_better_but_not_necessary_in_forensic_vision-language.md)

</div>

<!-- RELATED:END -->
