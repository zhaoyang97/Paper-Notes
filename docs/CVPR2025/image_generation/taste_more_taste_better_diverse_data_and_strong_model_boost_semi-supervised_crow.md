---
title: >-
  [论文解读] Taste More, Taste Better: Diverse Data and Strong Model Boost Semi-Supervised Crowd Counting
description: >-
  [CVPR 2025][图像生成][半监督学习] 提出 TMTB 框架，通过扩散模型 inpainting 增强背景多样性、引入 VMamba 骨干网络和抗噪分类分支，在半监督人群计数任务中以仅 5% 标签数据将 JHU-Crowd++ MAE 降至 67.0，大幅刷新 SOTA。 1. 领域现状：人群计数是计算机视觉的基础…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "半监督学习"
  - "人群计数"
  - "数据增强"
  - "状态空间模型"
  - "伪标签"
---

# Taste More, Taste Better: Diverse Data and Strong Model Boost Semi-Supervised Crowd Counting

**会议**: CVPR 2025  
**arXiv**: [2503.17984](https://arxiv.org/abs/2503.17984)  
**代码**: [https://github.com/syhien/taste_more_taste_better](https://github.com/syhien/taste_more_taste_better)  
**领域**: 图像生成  
**关键词**: 半监督学习, 人群计数, 数据增强, 状态空间模型, 伪标签

## 一句话总结
提出 TMTB 框架，通过扩散模型 inpainting 增强背景多样性、引入 VMamba 骨干网络和抗噪分类分支，在半监督人群计数任务中以仅 5% 标签数据将 JHU-Crowd++ MAE 降至 67.0，大幅刷新 SOTA。

## 研究背景与动机
1. **领域现状**：人群计数是计算机视觉的基础任务，广泛应用于公共安全、交通管理和灾害预防。全监督方法依赖大量点标注，标注成本极高（如 UCF-QNRF 约 2000 人时）。
2. **现有痛点**：半监督人群计数方法（基于 Mean Teacher 框架 + 伪标签）虽减少标注需求，但存在两个瓶颈：(a) 现有数据增强方法（Mixup、CutMix）会破坏人群的空间结构，导致密度图失真；(b) CNN 骨干网络过度关注局部细节，缺乏全局上下文建模能力。
3. **核心矛盾**：人群计数的密度图回归高度依赖空间结构完整性，而标准增强方法天然地破坏这种结构；同时密度图标注本身存在噪声，回归头对此十分敏感。
4. **本文目标**：设计适配人群计数的数据增强方法 + 能建模长距离依赖的骨干网络 + 抗噪训练策略。
5. **切入角度**：只修改背景区域（inpainting），保持前景人群完整；用分类任务提供"不精确但准确"的监督信号。
6. **核心 idea**：通过 Stable Diffusion inpainting 背景实现多样性增强，VMamba 替代 CNN 获取全局感受野，Anti-Noise 分类分支抵抗标注噪声。

## 方法详解

### 整体框架
TMTB 基于 Mean Teacher 半监督框架：学生模型和教师模型共享架构（VMamba 骨干 + 回归头 + 分类头），教师模型通过 EMA 更新权重。输入图像经 VMamba 提取特征后送入两个分支：密度图回归分支和计数区间分类分支。Inpainter 模块周期性对训练集进行背景 inpainting 增强。有标签数据直接监督，无标签数据通过一致性正则化学习。

### 关键设计

1. **Inpainting 数据增强**

    - 功能：在不破坏前景人群的前提下增强训练数据多样性
    - 核心思路：利用分类分支预测的前景/背景分割掩码 $M^{\text{inp}}$，仅对背景区域调用 Stable Diffusion 进行 inpainting，随机生成正向文本 prompt 以获得多样化背景。inpainting 后的图像全部作为无标签数据处理。为过滤低质量 inpainting 区域，教师模型计算弱增强和强增强版本之间分类预测的不一致度，生成加权掩码 $M^{\text{incon}}$，以较低权重学习不可靠区域。不一致度权重随训练推进逐渐衰减：$\omega_l^t = \text{softmax}(e^{-l \cdot t / T^{\text{inpw}}})$。
    - 设计动机：Mixup/CutMix 会破坏密度图的空间结构（人群被切割或混合），而 inpainting 仅修改背景、保持前景完整，天然适配密度回归任务

2. **VMamba (Visual State Space Model) 骨干网络**

    - 功能：替代 CNN/Transformer 作为特征提取骨干，建模全局长距离依赖
    - 核心思路：采用 VMamba 的 2D-Selective-Scan (SS2D) 模块，通过互补的 1D 遍历路径让每个像素整合来自不同方向的全局信息，在保持线性时间复杂度 $O(n)$ 的同时获得全局感受野。提取的特征送入回归和分类两个分支。
    - 设计动机：CNN 在极密集、低光照、恶劣天气场景下容易过拟合局部细节；Transformer 全局建模强但计算复杂度为 $O(n^2)$；VSSM 兼顾两者优势

3. **Anti-Noise 分类分支**

    - 功能：提供"不精确但准确"的辅助监督，抵抗点标注噪声
    - 核心思路：将像素级密度值量化为预定义的计数区间 bin，分类头预测每个位置属于哪个区间。用交叉熵损失 $\mathcal{L}_{\text{cls}} = \frac{1}{N}\sum_i^N \mathcal{H}(p_i^{gt}, \hat{p}_i)$ 监督。分类头的输出同时用于生成 inpainting 掩码和不可靠区域过滤。
    - 设计动机：密度图标注存在位置偏差（标注者对人头中心位置判断不一致），导致回归目标本身含噪；分类为"某个区间"比精确回归更鲁棒

### 损失函数 / 训练策略
总损失 $\mathcal{L} = \mathcal{L}^s + \lambda_w \cdot \mathcal{L}^u + \lambda_w \cdot \mathcal{L}^{inp}$：
- **监督损失** $\mathcal{L}^s = \mathcal{L}_{reg}^s + \mathcal{L}_{cls}^s$：回归损失采用多尺度 SSIM + TV loss（CUT loss），分类损失采用交叉熵
- **一致性损失** $\mathcal{L}^u$：MAE(学生密度图, 教师密度图) + MAE(学生分类, 教师分类)，辅以 patch-aligned random masking 强增强
- **Inpainting 损失** $\mathcal{L}^{inp}$：同一致性损失结构，但乘以不一致度加权掩码 $M^{\text{incon}}$ 过滤低质量区域
- $\lambda_w$ 在前 20 个 epoch 从 0 线性 warmup 到 1.0

## 实验关键数据

### 主实验

| 数据集 | 标签比例 | 指标(MAE) | TMTB | MRC-Crowd (前SOTA) | 提升 |
|--------|---------|-----------|------|-------------------|------|
| JHU-Crowd++ | 5% | MAE↓ | **67.0** | 76.5 | -12.4% |
| UCF-QNRF | 5% | MAE↓ | **96.3** | 101.4 | -5.0% |
| ShanghaiTech A | 5% | MAE↓ | **72.4** | 74.8 | -3.2% |
| ShanghaiTech B | 5% | MAE↓ | **10.6** | 11.7 | -9.4% |
| JHU-Crowd++ | 10% | MAE↓ | **66.3** | 70.7 | -6.2% |
| UCF-QNRF | 10% | MAE↓ | **91.7** | 93.4 | -1.8% |
| JHU-Crowd++ | 40% | MAE↓ | **60.0** | 60.0 | 持平 |
| ShanghaiTech B | 40% | MAE↓ | **7.5** | 7.8 | -3.8% |

### 消融实验

| 配置 | JHU 5% MAE | 说明 |
|------|-----------|------|
| Full TMTB | 67.0 | 完整模型 |
| w/o Inpainting Aug | ~73 | 去掉 inpainting 增强后显著退化 |
| w/o VMamba (用CNN) | ~75 | CNN 骨干缺乏全局感受野 |
| w/o Anti-Noise cls | ~71 | 去掉分类分支，回归对噪声敏感 |
| Mixup 替换 Inpainting | ~80+ | Mixup 破坏密度图空间结构 |

### 关键发现
- Inpainting 增强对低标签比例设置（5%）贡献最大，因为此时数据多样性更加关键
- JHU-Crowd++ 5% 场景中 MAE 首次降到 70 以下（67.0），标志性突破
- 不一致度过滤机制 $M^{\text{incon}}$ 对 inpainting 质量把控至关重要，无过滤的 inpainting 反而引入噪声
- 跨数据集泛化实验中，TMTB 甚至超过部分全监督方法

## 亮点与洞察
- **Inpainting 作为密度图任务的增强方案**非常巧妙：传统增强方法都会破坏标注对应关系，而只改背景完美规避了这一问题。这一思路可迁移到任何"标注与空间位置强绑定"的任务（如关键点检测、实例分割）
- **分类分支同时服务三个目的**：抗噪监督、inpainting 掩码生成、不可靠区域检测，一个模块多重复用，设计高效
- **VSSM 在密度估计中的首次应用**验证了线性复杂度全局建模在密集预测任务中的价值

## 局限与展望
- Inpainting 过程需要额外的 Stable Diffusion 推理开销，训练时间显著增加
- 对极端密度场景（几千人以上）的效果未单独分析
- 分类区间 bin 的设定是超参数，不同数据集可能需要调整
- 未来可探索更高效的生成模型替代 SD 做 inpainting，或用可学习的增强策略替代固定规则

## 相关工作与启发
- **vs MRC-Crowd**: 同样采用分类辅助任务，但 MRC-Crowd 用 CNN 骨干且无数据增强创新，本文在两个方面同时推进
- **vs DACount**: DACount 用 Transformer 精炼前景特征，本文用 VMamba 更高效地实现全局建模
- **vs DiffusionMix**: DiffusionMix 也用扩散模型做数据增强，但需要预定义二值掩码，不适用于人群计数中前景位置不可预测的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ inpainting 增强思路在密度估计中是首创，但各模块都有前人工作基础
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集三种标签比例全面覆盖，消融详尽
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机推导自然
- 价值: ⭐⭐⭐⭐ 在半监督人群计数上取得里程碑式突破，JHU 5% MAE<70 首次实现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Boost Your Human Image Generation Model via Direct Preference Optimization](boost_your_human_image_generation_model_via_direct_preference_optimization.md)
- [\[CVPR 2026\] StableMaterials: Enhancing Diversity in Material Generation via Semi-Supervised Learning](../../CVPR2026/image_generation/stablematerials_enhancing_diversity_in_material_generation_via_semi-supervised_l.md)
- [\[CVPR 2025\] Training Data Provenance Verification: Did Your Model Use Synthetic Data from My Generative Model for Training?](training_data_provenance_verification_did_your_model_use_synthetic_data_from_my_.md)
- [\[CVPR 2025\] InsightEdit: Towards Better Instruction Following for Image Editing](insightedit_towards_better_instruction_following_for_image_editing.md)
- [\[CVPR 2025\] DiverseFlow: Sample-Efficient Diverse Mode Coverage in Flows](diverseflow_sample-efficient_diverse_mode_coverage_in_flows.md)

</div>

<!-- RELATED:END -->
