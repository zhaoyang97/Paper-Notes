---
title: >-
  [论文解读] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing
description: >-
  [ICCV 2025][图像生成][图像编辑] 本文提出 ELECT（Early-timestep Latent Evaluation for Candidate selecTion），一个零样本框架，通过在扩散去噪的早期时间步估计背景不一致性来选择最优种子，将计算开销降低 41%（最高 61%），同时提升背景一致性和编辑指令遵循度，且无需外部监督或额外训练。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "图像编辑"
  - "种子选择"
  - "扩散模型"
  - "背景一致性"
  - "零样本"
---

# Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing

**会议**: ICCV 2025  
**arXiv**: [2504.13490](https://arxiv.org/abs/2504.13490)  
**代码**: [https://github.com/Joow0n-Kim/ELECT](https://github.com/Joow0n-Kim/ELECT)  
**领域**: 图像生成  
**关键词**: 图像编辑, 种子选择, 扩散模型, 背景一致性, 零样本

## 一句话总结

本文提出 ELECT（Early-timestep Latent Evaluation for Candidate selecTion），一个零样本框架，通过在扩散去噪的早期时间步估计背景不一致性来选择最优种子，将计算开销降低 41%（最高 61%），同时提升背景一致性和编辑指令遵循度，且无需外部监督或额外训练。

## 研究背景与动机

### 指令引导图像编辑的可靠性问题

基于扩散模型的指令引导图像编辑（如 InstructPix2Pix）允许用户通过自然语言指令修改图像，应用广泛。然而，扩散模型的固有随机性导致同一编辑指令在不同随机种子下产生差异巨大的结果——有的种子能精准编辑前景，有的则严重扭曲背景或完全不编辑。

### 现有方法的不足

用户通常通过"试错法"手动更换种子来找到满意结果，效率极低。已有的候选选择方法存在以下问题：

**仅适用于 T2I 生成**：现有种子选择方法（rejection sampling、seed optimization 等）针对文生图任务设计，关注图像质量和提示吻合度，不考虑与源图像的背景一致性。

**依赖外部验证器**：如美学评分、CLIPScore 等需要完整推理后才能评估，无法实现早期筛选。

**计算成本线性增长**：Best-of-N 策略需要对每个种子完整推理 N 次，计算量正比于 N。

### 核心观察与动机

作者通过三个关键观察推动了方法设计：

**背景 MSE 选择有效**：即使不加额外模型，选择背景 MSE 最小的种子就能有效减少背景扭曲并提升编辑质量。

**聚合相关性图可替代 GT Mask**：从多个种子的编辑相关性图（relevance map）取平均，得到的掩码效果逼近 GT mask。

**早期时间步已定位编辑区域**：分析发现扩散去噪的早期步骤（前 20-30 步）已经确定了主要编辑区域，后续步骤只做细节优化。这意味着可以在早期就对候选进行评估和筛选。

## 方法详解

### 整体框架

ELECT 的流程：(1) 对所有候选种子并行去噪到停止时间步 $t_{\text{stop}}$；(2) 在 $t_{\text{stop}}$ 处通过 Tweedie 公式估计各候选的最终编辑结果；(3) 计算背景不一致性分数（BIS）选择最优种子；(4) 仅对最优种子完成剩余去噪步骤。这样大部分种子只需部分推理，大幅节省计算。

### 关键设计

1. **背景不一致性分数（Background Inconsistency Score, BIS）**:

    - 功能：量化编辑结果中不期望的背景变化
    - 核心思路：BIS 结合软化的编辑掩码和估计的编辑结果来衡量背景偏差：
    $S^{\text{BIS}}(i, t) = (1 - (M_t^{\text{mean}})^2) \odot |\hat{z}_0^i - \mathcal{E}(I)|$
      其中 $M_t^{\text{mean}} = \frac{1}{|\mathbb{S}|}\sum_{i \in \mathbb{S}} M_t^i$ 是聚合的编辑相关性图，$\hat{z}_0^i$ 是通过 Tweedie 公式在时间步 $t$ 估计的第 $i$ 个种子的去噪结果：
    $\hat{z}_0^i = \frac{z_t^i - \sqrt{1-\alpha_t}\epsilon_\theta(z_t^i, t, I, C_T)}{\sqrt{\alpha_t}}$
    - 设计动机：用 $(M_t^{\text{mean}})^2$ 而非阈值化来软化掩码，避免了阈值依赖并防止将保护不好的背景区域误分类为前景。连续权重对过度编辑的背景区域给予高关注，减少正确编辑前景的权重

2. **聚合相关性图替代 GT Mask**:

    - 功能：在没有 GT 前景掩码的推理时，用多种子的相关性图平均来估计编辑区域
    - 核心思路：基于 Watch Your Steps (WYS) 的相关性图，计算条件和无条件噪声预测的差异 $M_t = |\epsilon_\theta(z_t, t, I, C_T) - \epsilon_\theta(z_t, t, I, \emptyset)|$。对所有候选种子取平均后，跨种子一致被编辑的区域在 $M_t^{\text{mean}}$ 中获得高值
    - 设计动机：单个种子的相关性图可能有噪声，聚合多种子结果更稳定可靠

3. **早期停止评估策略**:

    - 功能：在去噪的第 $t_{\text{stop}}$ 步就停下来评估各候选，只保留最优种子继续完整推理
    - 核心思路：通过 Tweedie 公式在早期时间步对最终结果做近似估计。由于信噪比（SNR）在约 20 步后达到 1，此后可以做可靠的比较。经验上大多数扩散模型在 $t_{\text{stop}}=70$（即去噪 30 步后）就稳定
    - 设计动机：传统 Best-of-N 需要完整推理所有 N 个种子（N×100 步），ELECT 只需 $N \times t_{\text{stop}} + 1 \times (100-t_{\text{stop}})$ 步

4. **掩码时序平均**:

    - 功能：在 $t \in [80, 100]$（去噪前 20 步）范围内平均相关性图
    - 核心思路：不同样本捕获编辑区域的最佳时间步不同，取平均可以消除单一时间步的不稳定性
    - 设计动机：消除对固定时间步的依赖，提升掩码提取的鲁棒性

5. **扩展：联合种子+提示词选择**:

    - 功能：当种子选择效果饱和或指令本身有问题时，引入 MLLM 生成替代提示词
    - 核心思路：用 MLLM 评估编辑结果的指令跟随和背景一致性（各打 0/0.5/1 分），如果某项为 0 则让 MLLM 改写指令，再结合 ELECT 选择最优提示词
    - 设计动机：种子选择提升有上限，某些失败案例的根源在于指令不在模型分布内

### 损失函数 / 训练策略

ELECT 是一个**零样本、无需训练**的推理时框架。不涉及任何训练过程，只在推理时利用扩散模型自身的去噪过程来评估候选。同时适用于基于 DDIM 的扩散模型和 Rectified Flow 模型。

## 实验关键数据

### 主实验

PIE-Bench 上各方法对比（N=11 种子，$t_{\text{stop}}=60$，NFE=500）：

| 模型 | 选择方法 | MSE×10⁴↓ | LPIPS×10³↓ | PSNR↑ | CLIP-T↑ | VIEScore↑ |
|------|---------|-----------|-----------|-------|---------|-----------|
| IP2P | Vanilla (1种子) | 248.5 | 162.4 | 20.73 | 24.38 | 3.43 |
| IP2P | Best of 5 (BIS) | 146.2 | 113.8 | 22.95 | 24.68 | 3.57 |
| IP2P | **ELECT** | **127.5** | **103.3** | **23.33** | **24.97** | **3.67** |
| InsDiff | Vanilla | 372.5 | 154.0 | 20.25 | 24.09 | 3.53 |
| InsDiff | **ELECT** | **180.5** | **104.5** | **22.85** | **24.75** | **3.82** |
| MGIE | Vanilla | 341.4 | 145.5 | 21.16 | 24.44 | 3.68 |
| MGIE | **ELECT** | **185.1** | **102.5** | **23.61** | **24.73** | **3.95** |
| UltraEdit | Vanilla | 87.5 | 115.4 | 22.93 | 25.20 | 4.47 |
| UltraEdit | **ELECT** | **63.8** | **92.3** | **24.49** | **25.36** | **4.70** |

### 消融实验

$t_{\text{stop}}$ 对性能的影响（IP2P，PIE-Bench）：

| 配置 | 说明 | 性能趋势 |
|------|------|---------|
| $t_{\text{stop}}=90$ | 仅去噪10步就选择 | 性能较差，噪声过大 |
| $t_{\text{stop}}=80$ | 去噪20步，SNR≈1 | 开始变得可靠 |
| $t_{\text{stop}}=70$ | 去噪30步 | 大多数模型收敛 |
| $t_{\text{stop}}=60$ | 去噪40步 | 稳定，UltraEdit需此点 |
| $t_{\text{stop}}=0$ | 完整推理 | =Best of N by BIS |

计算效率（对比 Best of N 达到相同性能所需的 NFE）：

| 模型 | Best of N NFE | ELECT NFE | 节省率 |
|------|-------------|-----------|--------|
| IP2P | 约500 | 约250 | ~50% |
| MagicBrush | 约500 | 约300 | ~40% |
| InsDiff | 约500 | 约200 | ~60% |
| UltraEdit | 约500 | 约320 | ~36% |
| **平均** | - | - | **41%** |

### 关键发现

- ELECT 在所有测试模型上一致超越 Best of N by BIS，因为在相同 NFE 预算内可以评估更多种子
- 背景一致性的改善同时带来了编辑质量提升（CLIP-T 和 VIEScore 均提升），说明背景保持好的种子通常编辑也更精准
- 聚合相关性图替代 GT mask 在所有指标上达到可比性能，验证了不需要额外标注
- 联合提示词选择可以额外提升 VIEScore +0.56，解决了种子选择无法处理的分布外指令问题
- 在已经失败的案例中约 40% 可以通过 ELECT 成功恢复

## 亮点与洞察

- 问题定义精准：将图像编辑中的种子敏感性问题形式化为可计算的 BIS 指标，使"选好种子"从模糊的人工操作变成可量化的优化问题
- 利用扩散模型自身的中间状态做评估，完全零样本，不需要外部模型或训练
- 软化掩码的设计避免了阈值调整，比 WYS 的硬阈值方法更鲁棒
- 与多种编辑模型兼容（IP2P、MagicBrush、InsDiff、MGIE、UltraEdit），且同时支持扩散和 Rectified Flow 架构
- VIEScore 等 MLLM 评估指标与人类判断高度一致，增强了结果的可信度

## 局限与展望

- BIS 是相对比较指标，在所有候选都很差的情况下无法保证绝对质量
- 可能存在过度优化背景保持而牺牲编辑幅度的风险，但作者指出这种情况罕见且不显著影响性能
- 提示词选择需要调用 MLLM（GPT-4V），引入了外部依赖和额外计算
- 默认 $t_{\text{stop}}$ 需要根据模型类型选择（扩散模型约 70，Rectified Flow 约 60），缺乏完全自适应机制

## 相关工作与启发

- Best-of-N 策略在 LLM 的推理时缩放中已经被广泛使用，本文将其扩展到扩散图像编辑领域
- 编辑相关性图来源于 Watch Your Steps (WYS)，但本文改进了其使用方式（聚合替代单种子、软化替代阈值）
- 与 Focus on Your Instruction、ZONE 等掩码引导方法不同，ELECT 不固定掩码而是通过多种子评估来筛选最优结果
- 早期停止的思想可推广到其他需要评估多候选的生成任务

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] AnyPortal: Zero-Shot Consistent Video Background Replacement](anyportal_zero-shot_consistent_video_background_replacement.md)
- [\[ICCV 2025\] ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation](adiee_automatic_dataset_creation_and_scorer_for_instruction-guided_image_editing.md)
- [\[ICCV 2025\] AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction](aid_adapting_image2video_diffusion_models_for_instruction-guided_video_predictio.md)
- [\[CVPR 2025\] Z-Magic: Zero-shot Multiple Attributes Guided Image Creator](../../CVPR2025/image_generation/z-magic_zero-shot_multiple_attributes_guided_image_creator.md)
- [\[ICCV 2025\] SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing](superedit_rectifying_and_facilitating_supervision_for_instruction-based_image_ed.md)

</div>

<!-- RELATED:END -->
