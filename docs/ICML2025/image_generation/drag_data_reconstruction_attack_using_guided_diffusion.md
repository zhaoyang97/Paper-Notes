---
title: >-
  [论文解读] DRAG: Data Reconstruction Attack using Guided Diffusion
description: >-
  [ICML 2025][图像生成][数据重建攻击] 提出 DRAG，利用预训练潜在扩散模型（LDM）的图像先验知识，通过引导扩散过程从分割推理（Split Inference）的深层中间表示中高保真地重建原始输入图像，揭示视觉基础模型（CLIP、DINOv2）在 SI 场景下的严重隐私漏洞。 分割推理（Split Infer…
tags:
  - "ICML 2025"
  - "图像生成"
  - "数据重建攻击"
  - "扩散模型引导"
  - "分割推理隐私"
  - "视觉基础模型"
  - "潜在扩散模型"
---

# DRAG: Data Reconstruction Attack using Guided Diffusion

**会议**: ICML 2025  
**arXiv**: [2509.11724](https://arxiv.org/abs/2509.11724)  
**代码**: [ntuaislab/DRAG](https://github.com/ntuaislab/DRAG)  
**领域**: 图像生成  
**关键词**: 数据重建攻击, 扩散模型引导, 分割推理隐私, 视觉基础模型, 潜在扩散模型

## 一句话总结

提出 DRAG，利用预训练潜在扩散模型（LDM）的图像先验知识，通过引导扩散过程从分割推理（Split Inference）的深层中间表示中高保真地重建原始输入图像，揭示视觉基础模型（CLIP、DINOv2）在 SI 场景下的严重隐私漏洞。

## 研究背景与动机

**分割推理（Split Inference, SI）** 是将神经网络拆分为客户端模型 $f_c$ 和服务端模型 $f_s$ 的推理范式——客户端在边缘设备上处理原始数据 $\mathbf{x}^*$ 得到中间表示（IR）$\mathbf{h}^* = f_c(\mathbf{x}^*)$，再发送到云端完成计算。SI 被认为能兼顾隐私与计算效率。

**现有问题**：
1. 已有数据重建攻击（DRA）主要针对较小的 CNN 分类模型（如 ResNet18），对大型视觉基础模型的隐私风险研究不足
2. Vision Transformer（ViT）的 patch tokenization 和 attention 机制与 CNN 有本质区别，攻击有效性未被充分探索
3. ViT 具有 **token 顺序不变性**（token order invariance），这是 CNN 不具备的特性，显著影响攻击效果
4. 从深层 IR 重建原始数据更加困难，因为深层 IR 已高度抽象化

**核心洞察**：预训练 LDM（如 Stable Diffusion）在大规模数据集上学到了丰富的图像先验，可以作为强大的正则化约束，将重建限制在自然图像流形上。

## 方法详解

### 整体框架

DRAG 的核心思路是将扩散模型作为图像先验 $R_\mathcal{I}$，约束优化问题的解空间。具体地，攻击者面对的是一个白盒威胁模型：已知客户端模型 $f_c$ 的架构和参数（在基础模型时代这是合理假设，因为 CLIP、DINOv2 等是公开发布的冻结模型）。

**两种攻击范式**：

1. **优化方法**：直接最小化重建图像与目标 IR 的距离
$$\mathbf{x}' = \arg\min_{\mathbf{x} \in \mathcal{X}} d_\mathcal{H}(f_c(\mathbf{x}), \mathbf{h}^*) + \lambda R_\mathcal{I}(\mathbf{x})$$

2. **学习方法**：训练逆向网络 $f_c^{-1}: \mathcal{H} \to \mathcal{X}$ 从公开数据集学习映射

DRAG 选择优化路线，用扩散模型的迭代采样过程替代传统的梯度下降优化。

### 关键设计

**设计一：引导扩散采样**

基于 DDIM 采样框架，将无条件采样转为条件采样。核心是定义重建目标函数：

$$L(\hat{\mathbf{x}}_0, \mathbf{c}) = d_\mathcal{H}(f_c(\hat{\mathbf{x}}_0), \mathbf{h}^*)$$

其中 $\hat{\mathbf{x}}_0$ 是通过 Tweedie 公式从当前噪声时间步单步估计的干净图像：

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1-\alpha_t}\,\epsilon_\theta(\mathbf{x}_t)}{\sqrt{\alpha_t}}$$

注意这里不能直接用含噪图像 $\mathbf{x}_t$ 计算引导，因为 $f_c$ 只在干净图像上训练，对含噪输入会产生不可靠的梯度。

**设计二：球面高斯约束（DSG）**

采用 DSG（Diffusion with Spherical Gaussian constraint）将引导梯度 $\mathbf{g}_t$ 与噪声 $\epsilon_t$ 融合：

$$\epsilon_t \leftarrow r \cdot \text{Unit}((1-w)\sigma_t\epsilon_t + wr \cdot \text{Unit}(\mathbf{g}_t))$$

其中 $r = \sqrt{n}\sigma_t$，$n$ 是 $\mathbf{x}_t$ 的维度。这一设计减少所需的去噪步数并提升生成质量。

**设计三：自回归（Self-Recurrence）**

由于 $f_c$ 通常是非凸的，单次引导步不足以获得高质量重建。DRAG 采用自回归策略：对每个时间步执行 $k$ 次去噪-加噪循环：

$$\mathbf{x}_t = \sqrt{\alpha_t/\alpha_{t-1}} \cdot \mathbf{x}_{t-1} + \sqrt{1 - \alpha_t/\alpha_{t-1}} \cdot \epsilon$$

这使得模型在每个时间步能获得多次梯度引导，显著提升重建精度。

**设计四：梯度优化技巧**

- 梯度裁剪（gradient clipping）防止梯度爆炸
- 使用 Adam 优化器维护历史引导向量，提升收敛稳定性

### DRAG++ 变体

DRAG++ 结合了学习方法和优化方法的优势：
1. 先用逆向网络 $f_c^{-1}$ 从 IR 获得粗略估计 $\mathbf{x}_{\text{coarse}} = f_c^{-1}(\mathbf{h}^*)$
2. 再通过扩散-去噪过程精化 $\mathbf{x}_{\text{coarse}}$

这种两阶段策略为扩散采样提供了更好的初始化，加速收敛并提升重建质量。

### 损失函数 / 训练策略

**距离度量 $d_\mathcal{H}$**：衡量重建图像经 $f_c$ 后的 IR 与目标 $\mathbf{h}^*$ 之间的距离（如 MSE 或余弦距离）。

**正则化 $R_\mathcal{I}$**：由扩散模型隐式提供——LDM 的去噪过程本身就将重建约束在自然图像分布上，无需额外正则化项（如 Total Variation 或 Deep Image Prior）。

**与已有方法的三个关键区别**：
1. $f_c$ 通常非凸，需要更强的优化策略
2. 可能存在防御机制，攻击者需在对抗设置下工作
3. 客户端可在 $f_c$ 中嵌入随机性（如 token 打乱），进一步增加难度

## 实验关键数据

### 主实验

在 CLIP 和 DINOv2 视觉基础模型上评测，从深层 IR 重建自然图像：

| 方法 | 目标模型 | 图像质量 | 感知相似度 | 优势 |
|------|---------|---------|-----------|------|
| rMLE (He et al.) | ResNet/ViT | 低 | 差 | 仅用 TV 正则 |
| LM (Singh et al.) | ResNet/ViT | 中 | 一般 | 加入深度图像先验 |
| GLASS (Li et al.) | ResNet/ViT | 较高 | 较好 | StyleGAN2 约束 |
| **DRAG (本文)** | **CLIP/DINOv2** | **显著更高** | **最优** | **LDM 图像先验** |
| **DRAG++ (本文)** | **CLIP/DINOv2** | **最高** | **最优** | **逆向网络 + LDM** |

### 消融实验

| 配置 | 关键效果 | 说明 |
|------|---------|------|
| 无 DSG 约束 | 质量下降 | 球面高斯约束是提升质量的关键 |
| 无自回归 ($k=1$) | 重建模糊 | 多次去噪-加噪循环对非凸优化至关重要 |
| 无梯度裁剪 | 训练不稳定 | 梯度爆炸导致生成崩塌 |
| 无 Adam 历史引导 | 收敛慢 | 历史梯度信息加速优化 |
| DRAG vs DRAG++ | DRAG++ 更优 | 逆向网络提供更好初始化 |
| 浅层 IR vs 深层 IR | 浅层更容易 | 深层 IR 信息更抽象，重建难度更大 |

### 关键发现

1. **ViT 的 token 顺序不变性**：ViT 中 token 的排列顺序不影响输出，这意味着从 IR 重建时需要额外处理 token 对应关系，这是 CNN 模型中不存在的挑战
2. **防御有效性有限**：已有防御方法（如 NoPeek、DisP）在 DRAG 面前仍然不堪一击，输入数据仍可被成功重建
3. **基础模型的隐私风险**：CLIP 和 DINOv2 等广泛使用的公开视觉编码器在 SI 场景下存在严重的隐私泄漏风险
4. **LDM 先验的强大作用**：预训练 LDM 提供的图像先验远优于传统的 TV 正则或 GAN 约束，尤其在深层 IR 重建时效果显著

## 亮点与洞察

1. **巧妙利用扩散模型双重角色**：LDM 既提供强大的图像先验（正则化），又通过迭代去噪实现优化——将传统的梯度优化转化为引导扩散采样，elegant 且有效
2. **Tweedie 公式的妙用**：通过单步去噪估计干净图像 $\hat{\mathbf{x}}_0$ 来计算引导梯度，巧妙解决了 $f_c$ 无法处理含噪输入的问题
3. **DRAG++ 的混合策略**：结合逆向网络的快速粗估计和扩散模型的精细优化，兼顾效率与质量
4. **对安全社区的警示**：揭示了基础模型时代 SI 隐私保护的紧迫性——公开的模型权重使白盒攻击成为现实威胁

## 局限与展望

1. **计算开销大**：引导扩散的迭代采样 + 自回归循环导致重建速度较慢，难以实时攻击
2. **依赖预训练 LDM**：当目标图像分布与 LDM 训练集差距较大时（如医学影像、遥感图像），攻击效果可能下降
3. **白盒假设的局限**：虽然基础模型公开使得白盒合理，但客户端可能进行微调或适配，导致模型参数不完全已知
4. **防御探索不足**：论文主要展示攻击能力，对如何有效防御（如差分隐私、对抗训练）的讨论较浅
5. **可扩展性**：能否扩展到更大分辨率图像、视频数据、3D 数据等模态有待验证

## 相关工作与启发

- **GLASS (Li et al., 2023)**：用 StyleGAN2 做图像先验的 DRA，但 GAN 先验的多样性不如 LDM
- **UGD (Bansal et al., 2024)**：通用引导扩散框架，DRAG 将其适配到 DRA 场景
- **DSG (Yang et al., 2024)**：球面高斯约束加速引导扩散，被 DRAG 直接采用
- **DPS (Chung et al., 2023)**：扩散后验采样，用 Tweedie 公式估计干净图像的思路被 DRAG 借鉴
- **启发**：该攻击框架有潜力扩展到联邦学习中的梯度重建攻击（将 IR 替换为梯度信息）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将引导扩散应用于 DRA 是自然但有效的组合，DRAG++ 混合策略有创意
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖多个基础模型和防御方法，消融较全面
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，公式推导完整，图示直观
- **价值**: ⭐⭐⭐⭐⭐ — 对 AI 安全社区有重要警示意义，揭示基础模型 SI 的隐私风险

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion](../../ICCV2025/image_generation/scorehoi_physically_plausible_reconstruction_of_human-object_interaction_via_sco.md)
- [\[AAAI 2026\] Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection](../../AAAI2026/image_generation/diffusion_reconstruction-based_data_likelihood_estimation_for_core-set_selection.md)
- [\[CVPR 2025\] UIBDiffusion: Universal Imperceptible Backdoor Attack for Diffusion Models](../../CVPR2025/image_generation/uibdiffusion_universal_imperceptible_backdoor_attack_for_diffusion_models.md)
- [\[NeurIPS 2025\] GeneMAN: Generalizable Single-Image 3D Human Reconstruction from Multi-Source Human Data](../../NeurIPS2025/image_generation/geneman_generalizable_single-image_3d_human_reconstruction_from_multi-source_hum.md)
- [\[ICML 2025\] Origin Identification for Text-Guided Image-to-Image Diffusion Models](origin_identification_for_text-guided_image-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
