---
title: >-
  [论文解读] Boosting Adversarial Transferability via Ensemble Non-Attention
description: >-
  [AAAI 2026][对抗迁移性] 提出 NAMEA（Non-Attention Meta Ensemble Attack），首次利用集成模型的非注意力区域（non-attention areas）融合 CNN 和 ViT 的可迁移信息，结合元学习梯度优化，在跨架构对抗迁移性上平均超越 SOTA 方法 AdaEA 和 SMER 分别 15.0% 和 9.6%。
tags:
  - "AAAI 2026"
  - "对抗迁移性"
  - "集成攻击"
  - "非注意力区域"
  - "元学习"
  - "跨架构攻击"
---

# Boosting Adversarial Transferability via Ensemble Non-Attention

**会议**: AAAI 2026  
**arXiv**: [2511.08937](https://arxiv.org/abs/2511.08937)  
**代码**: 无  
**领域**: 其他  
**关键词**: 对抗迁移性, 集成攻击, 非注意力区域, 元学习, 跨架构攻击

## 一句话总结

提出 NAMEA（Non-Attention Meta Ensemble Attack），首次利用集成模型的非注意力区域（non-attention areas）融合 CNN 和 ViT 的可迁移信息，结合元学习梯度优化，在跨架构对抗迁移性上平均超越 SOTA 方法 AdaEA 和 SMER 分别 15.0% 和 9.6%。

## 研究背景与动机

### 对抗迁移性的挑战

深度神经网络对对抗样本高度脆弱，从代理模型生成的对抗样本可以迁移到未知目标模型，使黑盒攻击成为可能。集成攻击通过整合多个代理模型的预测/损失/logits 来提升迁移性。然而，**跨异构架构迁移**（如 CNN→ViT 或混合集成→多种目标）效果严重不足。

### 核心困难——异构梯度方向差异

CNN 和 ViT 的梯度更新方向差异很大，现有方法难以在"减小集成模型梯度方差"和"充分利用单个模型信息"之间取得平衡：
- **AdaEA** 通过差异减少滤波器确保稳定更新方向，但牺牲了模型多样性
- **SMER** 独立优化各代理模型以保持多样性，但更新方向可能不够准确

### 关键观察——注意力区域的互补性

作者发现一个关键现象：**同构模型共享大量注意力区域，但异构模型关注的区域有较大差异**。具体来说：
- 用 ResNet-18 的注意力区域遮盖后，CNN 分类准确率下降高达 30%，但 ViT 仅下降 10% 以内
- 遮盖后的图像反而在异构模型间产生了更高的注意力重叠

这说明 **CNN 的非注意力区域很可能是 ViT 的关注区域，反之亦然**——非注意力区域包含跨架构的可迁移信息。

## 方法详解

### 整体框架

NAMEA 将迭代集成攻击建模为随机梯度下降优化过程，包含 $T$ 个外循环和 $K$ 个内循环。每个外循环执行三步：
1. **注意力元训练**：基于代理模型注意力区域更新梯度
2. **非注意力元测试**：基于代理模型非注意力区域更新梯度
3. **最终更新**：融合两步梯度得到最终更新

### 关键设计

#### 1. **注意力提取与非注意力提取模块（NAE）**

**功能**：利用 Grad-CAM 提取每个代理模型的注意力区域，进而获取非注意力区域。

**核心思路**：对代理模型 $f_n$ 和输入 $x$，按 Grad-CAM 计算注意力图 $\mathbf{H}_n(x)$：

$$\alpha_l^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial f_n(x)[y]}{\partial A_l^c[i,j]}$$

$$\mathbf{H}_n^l(x) = \text{ReLU}\left(\sum_c \alpha_l^c \cdot A_l^c\right)$$

然后生成注意力掩码 $\mathbb{M}_k$（值 ≥ 阈值 $\eta$ 的位置为 1），非注意力掩码 $\bar{\mathbb{M}}_k = \mathbf{1} - \mathbb{M}_k$。

**设计动机**：通过将注意力区域替换为随机高斯噪声来提取非注意力信息：$x_{te}^k = \bar{\mathbb{M}}_k \odot x_{te}^k + \mathbb{M}_k \odot \xi, \quad \xi \sim \mathcal{N}(0,1)$。填充随机噪声比全 0 或全 1 效果更好（约 2.7% 提升），因为随机噪声更强地干扰模型注意力。

#### 2. **元梯度优化（Meta-Gradient Optimization）**

**功能**：通过元学习框架融合注意力和非注意力区域的梯度，平衡更新稳定性和模型多样性。

**核心思路**：

**① 注意力元训练**：初始化 $x_{tr}^0 = x_{adv}^t$，迭代 $K$ 次：

$$g_{tr}^{k+1} = \nabla_{x_{tr}^k} \mathcal{L}(x_{tr}^k, y)$$

$$x_{tr}^{k+1} = \text{Clip}_\epsilon^x(x_{tr}^k + \alpha \cdot \text{sign}(g_{tr}^{k+1}))$$

**② 非注意力元测试**：初始化 $x_{te}^0 = x_{adv}^t$，先用 NAE 模块遮盖注意力区域，再计算梯度：

$$g_{te}^{k+1} = \nabla_{x_{te}^k} \mathcal{L}(x_{te}^k, y)$$

**③ 最终融合**：

$$g^{t+1} = g_{tr}^K + g_{te}^K \odot \bar{\mathbb{M}}_K$$

元测试梯度在融合前被非注意力掩码遮盖，确保注意力区域的可迁移梯度信息不被干扰。

#### 3. **梯度缩放优化模块（GSO）**

**功能**：针对 CNN 和 ViT 的不同特性，分别优化元测试梯度。

- **CNN 层级梯度缩放**：中间层特征更具可迁移性，用缩放因子 $\lambda(l) = \lambda_1 + \lambda_2 \cdot (L/l)$ 增强浅层梯度贡献
- **ViT 通道梯度缩放**：ViT 反向传播梯度较小影响迁移性，对低于平均梯度幅度的通道进行收缩：$g_{te}[c] = g_{te}[c] \cdot \tanh(|{(g_{te}[c] - \phi)}/{\sigma}|)$

### 损失函数 / 训练策略

NAMEA 是即插即用方法，可与 I-FGSM、MI-FGSM、DI-MI-FGSM 等基础攻击结合。超参设置：外循环 $T=10$，内循环 $K=16$，步长 $\alpha=0.8/255$，扰动预算 $\epsilon=8/255$，注意力阈值 $\eta=0.6$。每 $N$ 个连续内循环确保每个代理模型被选择至少一次。

## 实验关键数据

### 主实验

**ImageNet 跨架构迁移性（ASR%，DI-MI-FGSM 基础攻击）**

| 方法 | ViT 平均↑ | CNN 平均↑ | 总平均↑ |
|------|-----------|-----------|---------|
| Ens | 50.0 | 63.5 | 56.8 |
| SVRE | 55.8 | 70.4 | 63.1 |
| AdaEA | 54.8 | 63.9 | 59.4 |
| CWA | 61.4 | 70.3 | 65.9 |
| SMER | 72.1 | 78.5 | 75.3 |
| CSA | 60.3 | 69.0 | 64.7 |
| **NAMEA** | **77.5** | **83.7** | **80.6** |

NAMEA 在 DI-MI-FGSM 基础攻击下平均 ASR 达 80.6%，超越最强基线 SMER 5.3%。

**对防御模型和防御方法的鲁棒性（DI-MI-FGSM）**

| 方法 | 防御模型平均↑ | 防御方法平均↑ |
|------|---|---|
| AdaEA | 56.6 | 40.6 |
| SMER | 69.1 | 56.8 |
| **NAMEA** | **74.2** | **63.3** |

即使面对基于扩散模型的强防御 DiffPure，NAMEA 仍超越基线 10%（50.3% vs 39.9%）。

**真实世界 API 攻击（DI-MI-FGSM）**

| API | SMER | **NAMEA** | 提升 |
|-----|------|-----------|------|
| Google | 52 | **55** | +3 |
| Alibaba | 48 | **53** | +5 |
| Baidu | 61 | **64** | +3 |

### 消融实验

| 配置 | 平均 ASR↑ | 说明 |
|------|-----------|------|
| NAMEA 完整 | ~46.7 (I-FGSM) | 基准 |
| 移除元测试步 (-Mtest) | -9.2% | 非注意力梯度关键 |
| 移除元训练步 (-Mtrain) | -7.4% | 注意力梯度也重要 |
| 移除 GSO 模块 | -2.0% | GSO 有正向贡献 |
| 填充全 0 | -2.7% | 随机噪声优于固定值 |
| 填充全 1 | -2.7% | 同上 |
| 提取注意力区域（非非注意力） | 显著下降 | 验证非注意力的关键作用 |
| 提取随机 patch | 中度下降 | 随机位置不如语义驱动的非注意力 |
| 阈值 η=0.6 | 最优 | CNN 和 ViT 的 ASR 对 η 敏感 |

### 关键发现

1. **非注意力区域的价值被低估**：首次证明集成非注意力区域包含互补的跨架构可迁移信息
2. **元学习是关键**：移除元训练或元测试都导致显著性能下降，二者融合优于任何单一策略
3. **不同于输入多样性**：通过对比实验（NAMEA vs NAMEA_RT），确认性能增益来源于非注意力区域的语义信息，而非简单的输入多样化
4. **对代理模型选择策略不敏感**：同架构/异架构/随机选择的元测试模型效果相近，因为内循环的随机选择已保证足够的非注意力区域探索

## 亮点与洞察

1. **观察驱动的方法设计**：从"异构模型注意力区域差异"的经验观察出发，提出"非注意力区域包含互补迁移信息"的假设，并从理论和实验两个维度验证
2. **即插即用的实用性**：NAMEA 可与各种基础攻击（I-FGSM、MI-FGSM、DI-MI-FGSM 等）无缝组合，泛化性强
3. **注意力解耦的新视角**：不同于以往关注如何利用或干扰注意力区域，本文首次系统探索非注意力区域的价值
4. **真实世界验证**：不仅在标准 benchmark 上测试，还在 Google、阿里云、百度云等商用 API 上验证了攻击有效性

## 局限与展望

1. **计算开销**：NAMEA 在每个内循环维护两个对抗样本（元训练和元测试），计算量约是基线的 2 倍
2. **Grad-CAM 的局限**：注意力提取依赖 Grad-CAM 的质量，对某些架构（如极深网络）可能不够精确
3. **阈值敏感性**：实验显示 ASR 对阈值 $\eta$ 敏感，最优值可能因模型组合而变化
4. **防御适应性**：若防御方自知此攻击策略，可能设计针对非注意力区域加固的防御
5. **仅限图像分类**：未验证在目标检测、语义分割等更复杂视觉任务上的效果

## 相关工作与启发

- **集成攻击**：AdaEA (自适应融合) 和 SMER (强化学习重加权) 是最相关的基线，NAMEA 从梯度解耦+融合的新角度超越它们
- **注意力机制在对抗攻击中的应用**：ATA、AoA 等方法关注干扰注意力区域，本文反其道而行关注非注意力区域
- **元学习在对抗攻击中的应用**：MGAA 用元学习模拟白盒/黑盒攻击，NAMEA 将元学习用于注意力/非注意力梯度融合
- **对防御研究的启示**：非注意力区域的可迁移性暗示当前防御可能过于关注保护注意力区域，需要更全面的鲁棒性设计

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（"集成非注意力"概念新颖，首次系统验证非注意力区域的对抗迁移价值）
- 实验充分度: ⭐⭐⭐⭐⭐（17 种目标模型 + 6 种防御模型 + 9 种防御方法 + 3 个 API + 多轮消融）
- 写作质量: ⭐⭐⭐⭐（结构清晰，动机-观察-方法的逻辑链完整）
- 价值: ⭐⭐⭐⭐（为跨架构对抗迁移性开辟新方向，实用性强）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] ASAG: Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance](toward_the_frontiers_of_reliable_diffusion_sampling_via_adversarial_sinkhorn_att.md)
- [\[ICCV 2025\] NAPPure: Adversarial Purification for Robust Image Classification under Non-Additive Perturbations](../../ICCV2025/others/nappure_adversarial_purification_for_robust_image_classification_under_non-addit.md)
- [\[ACL 2025\] Battling against Tough Resister: Strategy Planning with Adversarial Game for Non-collaborative Dialogues](../../ACL2025/others/battling_against_tough_resister_strategy_planning_with_adversarial_game_for_non-.md)
- [\[AAAI 2026\] On the Edge of Core (Non-)Emptiness: An Automated Reasoning Approach to Approval-Based Multi-Winner Voting](on_the_edge_of_core_non-emptiness_an_automated_reasoning_approach_to_approval-ba.md)
- [\[AAAI 2026\] Guided Perturbation Sensitivity (GPS): Detecting Adversarial Text via Embedding Stability and Word Importance](guided_perturbation_sensitivity_gps_detecting_adversarial_text_via_embedding_sta.md)

</div>

<!-- RELATED:END -->
