---
title: >-
  [论文解读] Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation
description: >-
  [NeurIPS 2025][3D视觉][自监督深度估计] 首次将Stable Diffusion的视觉先验引入自监督单目深度估计框架，提出Mix-Batch Image Reconstruction（MIR）代理任务保护SD先验不被重投影噪声损坏，并设计Scale-Shift GRU（SSG）桥接SD的尺度-偏移不变性（SSI）与自监督的尺度不变性（SI）深度分布，在KITTI上AbsRel=0.090达到所有SSMDE方法SOTA，且零样本泛化全面超越Marigold、E2E FT、Lotus等有监督SD方法。
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "自监督深度估计"
  - "扩散模型"
  - "扩散先验"
  - "混合批次图像重建"
  - "Scale-Shift GRU"
---

# Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation

**会议**: NeurIPS 2025  
**arXiv**: [2503.15905](https://arxiv.org/abs/2503.15905)  
**代码**: 有（Project Page）  
**领域**: 3D视觉 / 深度估计  
**关键词**: 自监督深度估计, Stable Diffusion, 扩散先验, 混合批次图像重建, Scale-Shift GRU

## 一句话总结

首次将Stable Diffusion的视觉先验引入自监督单目深度估计框架，提出Mix-Batch Image Reconstruction（MIR）代理任务保护SD先验不被重投影噪声损坏，并设计Scale-Shift GRU（SSG）桥接SD的尺度-偏移不变性（SSI）与自监督的尺度不变性（SI）深度分布，在KITTI上AbsRel=0.090达到所有SSMDE方法SOTA，且零样本泛化全面超越Marigold、E2E FT、Lotus等有监督SD方法。

## 研究背景与动机

**领域现状**：自监督单目深度估计（SSMDE）仅从视频序列学习3D信息，无需昂贵深度标注。SD在有监督深度估计中展现出强大视觉先验（锐利边界+强泛化），但此前仅用于有监督设定——用合成数据集的高精度深度标注微调SD。

**现有痛点**：SSMDE的重投影损失天然含噪——遮挡、无纹理区域、光照变化导致伪监督信号，使预测模糊。直接用这种含噪信号微调SD会迅速损坏其VAE潜空间先验。

**核心矛盾**：SD先验需要"高精度监督"来保护latent space质量，但自监督范式天然无法提供。同时，SD输出SSI深度（尺度和偏移均不确定），而自监督需要SI深度（仅尺度不确定、偏移严格为零），两者分布不匹配导致训练不稳定。

**切入角度**：RGB图像本身就是最"高精度"的自监督信号——天然包含完整视觉细节、不依赖外部深度、完美匹配SD原始训练目标。可以用图像重建作为代理任务保护SD先验。

**核心 idea**：用"图像重建"代理任务保护SD先验 + 用GRU迭代对齐SSI→SI深度分布，首次实现SD+自监督深度估计的成功结合。

## 方法详解

### 整体框架

输入图像 $\mathbf{I}_t$ + 噪声 → VAE编码器 → U-Net单步去噪（task switcher控制深度/图像重建任务）→ VAE解码器输出SSI深度 $D_{\text{SSI}}$ 或重建图像 → SSG模块将 $D_{\text{SSI}}$ 转换为 $D_{\text{SI}}$ → 与相邻帧配合pose network做重投影 → photometric loss。Stable Diffusion v2作为backbone，单步去噪。

### 关键设计

1. **Mix-Batch Image Reconstruction (MIR)**:

    - 功能：在不引入任何额外深度监督的情况下保护SD潜空间先验
    - 核心思路：通过task switcher $s \in \{s_x, s_y\}$ 让同一U-Net交替执行深度预测和图像重建任务。图像重建使用photometric loss（SSIM+L1）而非latent space MSE——后者在VAE的1/8分辨率上会产生 $8 \times 8$ 块状伪影。混合批次中随机选择KITTI真实图像或Hypersim合成图像进行重建：$L_s = L_{ph}(\mathbf{I}_h, \mathcal{D}(f_\theta^z(s_y, \mathbf{z}_\tau^y, \mathbf{z}^I)))$
    - 设计动机：三个关键insight——（1）纯KITTI重建因VAE编码不匹配产生块伪影；（2）纯合成图像重建能力无法迁移到减少深度模糊；（3）混合使用让合成数据锚定SD先验、真实数据强制几何一致性。photometric loss替代latent loss使MIR对混合比例不敏感

2. **Scale-Shift GRU (SSG)**:

    - 功能：迭代将SSI深度对齐到SI深度
    - 核心思路：修改GRU迭代公式为 $D_{k+1} = D_\delta + s_c \cdot D_k + s_h$。引入Scale-Shift Transformer（SST）：可学习的scale/shift查询（$Q_{\text{SC}}/Q_{\text{SH}}$）与SD隐层状态做交叉注意力，输出 $s_c$ 和 $s_h$。迭代两次：$D_0$（SSI）→ $D_1$ → $D_2$（SI）。作者通过严格数学推导证明：自监督几何约束下shift必须为零（否则任意深度图退化为平面），而SD的VAE输出范围[-1,1]天然引入非零shift
    - 设计动机：GRU的reset gate可选择性阻断含噪梯度反传——重投影损失的异常梯度被过滤，使 $D_{\text{SSI}}$ 保留SD的精细纹理，$D_{\text{SI}}$ 保持几何一致性

3. **Steady SD Finetuning（稳定微调策略）**:

    - 功能：解决SD大模型+多模块联合训练+间接自监督的训练不稳定
    - 核心思路：引入预训练自监督教师（MonoViT）生成伪深度标签，提供直接监督。伪标签损失权重随训练递减：$\eta_{\text{step}} = \max(1, 30 \cdot (\text{step}_{\text{now}}/\text{step}_{\text{max}}))$，早期强约束、后期释放以突破教师上限
    - 设计动机：SD参数庞大，自监督信号间接含噪，无直接监督时早期训练极易崩溃

### 损失函数 / 训练策略

总损失 $L = L_s + L_{ph} + L_{tc} + L_a \cdot \eta_a$。$L_s$ 是MIR代理任务损失，$L_{ph}$ 是photometric重投影损失，$L_{tc}$ 是递减权重伪标签蒸馏损失，$L_a$ 是辅助损失（GDS loss、edge loss等）。AdamW（lr=3e-5），8张A800 GPU，batch=32，25K步约1天。

## 实验关键数据

### 主实验（KITTI Eigen split）

| 方法 | 类型 | 数据 | AbsRel↓ | RMSE↓ | $a_1$↑ |
|------|------|------|---------|-------|--------|
| Marigold (CVPR24) | 零样本 | Syn(74K+74K) | 0.120 | 4.033 | 0.874 |
| E2E FT (WACV25) | 零样本 | Syn(74K+74K) | 0.112 | 4.099 | 0.890 |
| Lotus (ICLR25) | 零样本 | Syn(59K+59K) | 0.110 | 3.807 | 0.892 |
| MonoViT (3DV22) | 自监督 | K(40K) | 0.096 | 4.292 | 0.908 |
| RPrDepth (ECCV24) | 自监督 | K(40K) | 0.091 | 4.098 | 0.910 |
| **Jasmine** | **自监督** | **KH(68K)** | **0.090** | **3.944** | **0.919** |

### 零样本泛化

| 方法 | DrivingStereo AbsRel | CityScape AbsRel | Foggy AbsRel |
|------|---------------------|------------------|-------------|
| Marigold | 0.178 | 0.164 | 0.146 |
| E2E FT | 0.160 | 0.160 | 0.141 |
| Lotus | 0.173 | 0.147 | 0.150 |
| MonoViT | 0.150 | 0.140 | 0.107 |
| **Jasmine** | **0.136** | **0.123** | **0.098** |

Jasmine在零样本泛化上全面超越有监督SD方法+传统SSMDE方法。

### 消融实验

| ID | 配置 | AbsRel | $a_1$ | 说明 |
|----|------|--------|-------|------|
| 0 | Jasmine完整 | 0.090 | 0.919 | - |
| 1 | w/o SD Prior | 0.516 | 0.258 | 从头训练崩溃，AbsRel↑473% |
| 2 | w/o MIR+SSG | 0.175 | 0.790 | 有SD但无保护，先验被损坏 |
| 3 | w/o SSG | 0.129 | 0.872 | 无分布对齐，AbsRel高43% |
| 4 | w/o MIR | 0.132 | 0.852 | 无先验保护，AbsRel高47% |
| 10 | Latent loss替代ph loss | 0.095 | 0.909 | photometric loss更优 |
| 12 | 辅助图像=KITTI | 0.095 | 0.912 | 合成图像非必须 |
| 13 | KITTI+ETH3D | 0.090 | 0.916 | 域差异大的辅助数据更好 |

### 关键发现

- SD先验是绝对关键：去掉后AbsRel从0.090飙至0.516（5.7倍退化）
- MIR和SSG缺一不可：单独去掉任一导致40%+退化，"先验保护"和"分布对齐"是两个独立且必要的问题
- 辅助数据的域差异比质量更重要：KITTI+ETH3D优于纯KITTI，因为域差异增强SD先验保护
- photometric loss优于latent loss：强调结构一致性而非颜色精度，与深度估计目标更一致
- Jasmine零样本全面超越有监督SD方法，说明自监督几何约束比合成深度标注提供更好的泛化归纳偏置

## 亮点与洞察

- **RGB图像作为"高精度自监督信号"**：SD微调需要高质量监督保护先验，RGB图像恰好满足——天然高质量、无外部依赖、与SD原始训练目标完全对齐。打破了"SD必须有高精度深度标注才能微调"的固有认知。
- **SSI vs SI深度不对齐的理论分析**：通过严格数学推导证明自监督下shift必须为零（否则任意深度图退化为平面），首次清晰揭示这一基础性问题并给出系统解决方案。
- **GRU的reset gate作为梯度过滤器**：利用GRU天然的选择性遗忘特性过滤异常梯度，使SSI深度保留细节而SI深度保持几何一致性。可迁移到任何含噪监督下的精细特征保护场景。
- **首次桥接零样本与自监督深度估计**：深入分析了median alignment和LSQ alignment的差异及适用场景。

## 局限与展望

- 依赖预训练教师（MonoViT）的伪标签稳定训练，教师质量仍影响上限
- 仅在KITTI上训练（驾驶场景），室内/无人机/水下等场景未验证
- Hypersim合成图像虽用量不大但仍算额外数据源
- SD v2非最新，SDXL/SD3上的效果值得探索
- 训练需8张A800 GPU，自监督的低标注成本优势被计算成本部分抵消

## 相关工作与启发

- **vs Marigold/E2E FT/Lotus**: 有监督SD方法依赖合成深度标注。Jasmine不用任何深度标注却在零样本泛化上全面超越，说明自监督几何约束提供了更好的泛化归纳偏置
- **vs MonoViT/MonoDepth2**: 传统SSMDE缺乏SD先验，边缘锐利度和跨域泛化明显弱于Jasmine
- **vs DepthAnything v1/v2**: 依赖大规模图像-深度对数据，Jasmine认为纯自监督+视频数据有更大扩展空间

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将SD引入自监督深度估计，MIR和SSG设计有深刻动机分析和理论支撑
- 实验充分度: ⭐⭐⭐⭐⭐ KITTI+4个零样本数据集，详尽消融覆盖每个组件和设计选择
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义精准，理论推导严谨，图表质量高
- 价值: ⭐⭐⭐⭐⭐ 开创SD+自监督深度估计新范式，组件设计有强可迁移性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] High-Precision Self-Supervised Monocular Depth Estimation with Rich-Resource Prior](../../ECCV2024/3d_vision/high-precision_self-supervised_monocular_depth_estimation_with_rich-resource_pri.md)
- [\[CVPR 2025\] Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron CT Data](../../CVPR2025/3d_vision/regularizing_inr_with_diffusion_prior_self-supervised_3d_reconstruction_of_neutr.md)
- [\[CVPR 2026\] RoSAMDepth: Robust Self-supervised Depth Estimation Leveraging Segment Anything Model](../../CVPR2026/3d_vision/rosamdepth_robust_self-supervised_depth_estimation_leveraging_segment_anything_m.md)
- [\[NeurIPS 2025\] 3D Visual Illusion Depth Estimation](3d_visual_illusion_depth_estimation.md)
- [\[NeurIPS 2025\] Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations](concerto_joint_2d-3d_self-supervised_learning_emerges_spatial_representations.md)

</div>

<!-- RELATED:END -->
