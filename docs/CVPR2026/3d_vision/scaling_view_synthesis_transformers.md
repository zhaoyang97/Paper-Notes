# Scaling View Synthesis Transformers (SVSM)

**会议**: CVPR 2026  
**arXiv**: [2602.21341](https://arxiv.org/abs/2602.21341)  
**代码**: [https://www.evn.kim/research/svsm](https://www.evn.kim/research/svsm)  
**领域**: 3D视觉 / 新视角合成 / 缩放定律  
**关键词**: 新视角合成, 缩放定律, Transformer, encoder-decoder, 计算效率, PRoPE  

## 一句话总结

首次为无几何先验的 NVS Transformer 建立缩放定律：提出有效批量大小假设（B_eff = B·V_T）揭示 encoder-decoder 被低估的根因，设计单向 encoder-decoder 架构 SVSM，在 RealEstate10K 上以不到一半训练 FLOPs 达到新 SOTA（30.01 PSNR），Pareto 前沿比 LVSM decoder-only 左移 3×。

## 研究背景与动机

1. **NVS 缺少缩放分析**：NLP（Chinchilla、Kaplan）和 2D 视觉（DiT）已有系统缩放定律，但 3D 视觉/NVS 领域完全空白——模型设计、训练配置缺乏计算最优的原则性指导
2. **Decoder-only 架构冗余严重**：LVSM decoder-only 渲染每张目标视图都要重新走完全部上下文 token，FLOPs 的 MLP 部分 ∝ V_T×(V_C+1)，注意力部分 ∝ V_T×(V_C+1)²，随目标视图数线性增长
3. **Encoder-decoder 被不公平否定**：LVSM 原文中 encoder-decoder 变体显著弱于 decoder-only，但本文发现根因是：(a) 使用了固定大小场景潜表示引入瓶颈，(b) 在不等计算预算下对比，并非架构本身劣势
4. **目标视图与批量大小的交互效应未知**：NVS 训练标准做法是每个场景重建多个目标视图，但增加 V_T vs 增加 B 对训练动态的影响从未被形式化分析
5. **多视图（V_C>2）缩放是否保持**：将 encoder-decoder 扩展到多视图时，场景表示瓶颈是否会导致缩放退化是开放问题

## 方法详解

### 整体流程

上下文图像 C = {(I_i, g_i, K_i)} → **Transformer Encoder**（双向自注意力）→ 场景表示 z = E[C]（所有 patch token，无固定瓶颈）→ **Cross-Attention Decoder**（单向）→ 并行渲染 V_T 个目标视图 Ĩ = D[z, g_T, K_T]。核心：编码一次、解码多次，目标视图间无交互但可并行。

### 1. SVSM 架构（Section 3）

- **Encoder**：标准 ViT，对所有上下文图像做双向自注意力，输出 patch token 集合作为场景表示。关键区别于 LVSM enc-dec：不压缩为固定数量 learnable token，而是保留全部 patch token，避免信息瓶颈
- **Decoder**：通过 cross-attention 从场景表示 z 中提取信息，自回归地渲染目标视图。各目标视图独立解码但共享 z，可并行执行
- **计算复杂度**：χ_MLP(SVSM) ∝ V_T + V_C，χ_Attn(SVSM) ∝ V_C×(V_T + V_C)。当 V_T ≫ V_C 时降至 O(V_T)，对比 LVSM 的 O(V_T·V_C + V_T) 有显著优势
- **代价**：encoder 无法主动丢弃与目标无关的信息；在参数量和训练步数相同时 SVSM 弱于 LVSM，但通过摊销渲染节省的计算量可加大模型和训练步数，使得**在等计算预算下 SVSM 显著更优**

### 2. 有效批量大小假设（Section 4）

- **定义**：B_eff ≡ B · V_T，其中 B 为场景数、V_T 为每场景目标视图数
- **实验验证**：在 DL3DV（V_C=8）和 RE10K（V_C=2）上固定 B_eff 变换 (B, V_T) 组合进行消融。结果：同 B_eff 下最终 PSNR 差异仅 ±0.1~0.2，训练损失曲线几乎完全重合
- **对 LVSM 的含义**：χ(LVSM) ∝ B·V_T·(V_C+1) = B_eff·(V_C+1)，不依赖 (B, V_T) 拆分方式——调节 V_T 无法省计算
- **对 SVSM 的含义**：χ(SVSM) ∝ B·(V_C + V_T) = B_eff + B·V_C。减少 B、增大 V_T 可以保持 B_eff（保持性能）同时减少总 FLOPs——这就是 enc-dec 效率优势的来源
- **洞察**：LVSM 原文 enc-dec 表现差的根因是在等迭代次数（而非等 FLOPs）下对比，掩盖了 enc-dec 的计算效率

### 3. 立体 Stereo 缩放定律（Section 5，V_C=2）

- **实验设置**：在 RE10K 上，V_T=6，batch size=256，patch size=16，扫描 7M~300M 参数 × 3-4 种训练样本数，总计算跨 10³ 量级（100 petaflops 到 100 exaflops）
- **缩放结果**：log-log 图上两模型族 Pareto 前沿斜率相同，但 SVSM 向左偏移 3×——同性能只需 1/3 FLOPs
- **Chinchilla 分析**：对每个计算预算 χ 确定最优 (N_opt, D_opt)，拟合 N_opt ∝ χ^a、D_opt ∝ χ^b。SVSM：a=0.52, b=0.47（a≈b，与 Chinchilla 一致——增加 k 倍预算应 √k 分给模型、√k 分给数据）；LVSM：a=0.65, b=0.33（更偏模型侧）
- **稳定训练**：应用 1/√L 残差缩放（depth-μP），确保不同深度模型公平对比
- **最终模型**：SVSM-416M（Pareto 最优）和 SVSM-740M（迭代匹配），在约 0.77 zflops（LVSM 的一半）下均超越 LVSM-171M

### 4. 多视图缩放定律（Section 6，V_C>2）

- **问题**：直接扩展 SVSM 到 V_C=4，Pareto 前沿快速饱和，缩放行为消失
- **原因分析**：encoder-decoder 中场景表示是信息瓶颈，位姿信息在深层丢失
- **解决方案 PRoPE**：投影旋转位置编码——每层注意力前将 Q/K/V 通过相机位姿变换到公共参考坐标系执行注意力，再逆变换回各自坐标系。位姿信息直接嵌入每一层而非仅初始嵌入
- **效果**：加 PRoPE 后 SVSM 重新恢复理想缩放趋势，Pareto 前沿仍优于 LVSM+PRoPE

### 5. 固定潜表示缩放实验（Section 7）

- **设置**：Objaverse 数据集，V_C=8，对比 SVSM-fixed（固定潜表示+单向解码）vs LVSM enc-dec（固定潜表示+双向解码）
- **结论**：两者缩放行为类似，SVSM-fixed 仍有 5× 计算优势（Pareto 前沿左移 5×）；但**两者都显著差于无瓶颈设计**——固定潜表示是缩放的主要限制因素

## 实验结果

### 表1：Stereo NVS (V_C=2) 最大模型

| 模型 | 参数量 | 训练FLOPs | PSNR↑ | SSIM↑ | LPIPS↓ | FPS(V_C=4) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| LVSM Enc-Dec | 173M | 2.53 zflops | 28.58 | 0.893 | 0.114 | 52.9 |
| LVSM Dec-Only | 171M | 1.60 zflops | 29.67 | 0.906 | 0.098 | 19.5 |
| SVSM (Iter-matched) | 740M | 0.74 zflops | 29.80 | 0.907 | 0.098 | 42.7 |
| **SVSM (Pareto)** | **416M** | **0.77 zflops** | **30.01** | **0.910** | **0.096** | **61.8** |

### 表2：与显式几何方法对比（RealEstate10K）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:---:|:---:|:---:|
| pixelNeRF | 20.43 | 0.589 | 0.550 |
| pixelSplat | 26.09 | 0.863 | 0.136 |
| MVSplat | 26.39 | 0.869 | 0.128 |
| GS-LRM | 28.10 | 0.892 | 0.114 |
| **SVSM** | **30.01** | **0.910** | **0.096** |

### 表3：多视图 NVS (V_C>2)

| 模型 | 参数量 | 训练FLOPs | PSNR↑ | LPIPS↓ | FPS(V_C=4) | FPS(V_C=16) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| LVSM+PRoPE | 171M | 43 eflops | 26.19 | 0.145 | 104.7 | 23.8 |
| SVSM (Iter) | 711M | 32 eflops | 26.29 | 0.141 | 280.4 | 230.4 |
| **SVSM (Pareto)** | **400M** | **44 eflops** | **26.87** | **0.129** | **411.1** | **333** |

## 核心发现

1. **3× 计算效率**：SVSM Pareto 前沿与 LVSM 斜率相同但左移 3×——同性能只需 1/3 训练计算
2. **Chinchilla 规律跨模态复现**：SVSM 的 a≈0.52, b≈0.47 (a≈b) 与 NLP 发现一致——计算预算加倍应等分给模型和数据
3. **B_eff 决定一切**：有效批量大小 B·V_T 是决定最终性能的唯一因素，(B, V_T) 的具体拆分方式差异 ≤0.2 PSNR
4. **PRoPE 解锁多视图缩放**：无 PRoPE 时 SVSM 在 V_C>2 快速饱和；加 PRoPE 后恢复缩放且前沿仍优于 LVSM
5. **固定潜表示是缩放瓶颈**：无论解码器方向性如何，固定大小场景表示都严重限制缩放能力
6. **推理速度**：SVSM 在 V_C=4 时渲染速度达 LVSM 的 4×，外推到 V_C=16 达 14×

## 亮点与局限

**亮点**：
- 有效批量大小假设概念简洁洞察深刻，一举解释了 enc-dec 被低估的根因并提供利用方法
- 首次在 3D 视觉领域建立 Chinchilla 式计算最优训练配方
- 10³ 量级 FLOPs 的系统扫描、3 个数据集、多种 V_C 设置，实验设计极其严谨

**局限**：
- 训练数据受限：仅使用 RE10K、DL3DV 等小型带位姿数据集并重复采样，与标准 <1 epoch 缩放实践不同
- V_C 大时 encoder 二次复杂度使渲染速度低于 LVSM enc-dec（V_C=8 时）
- 仅覆盖稀疏到中等视图场景，V_C≫16 时线性注意力模型可能更有优势
- 限于确定性渲染，未研究缩放定律对扩散模型式 NVS 的适用性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 有效批量大小假设 + NVS 缩放定律填补 3D 视觉空白
- 实验充分度: ⭐⭐⭐⭐⭐ 10³ FLOPs 系统分析、stereo+multiview+fixed latent 三场景全覆盖
- 写作质量: ⭐⭐⭐⭐⭐ Chinchilla 式严谨呈现，图表专业清晰
- 价值: ⭐⭐⭐⭐⭐ 计算最优训练配方 + 架构指导原则可直接迁移到其他 3D 视觉任务
