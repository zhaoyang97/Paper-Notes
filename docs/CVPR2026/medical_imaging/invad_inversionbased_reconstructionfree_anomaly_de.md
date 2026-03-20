# InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models

**会议**: CVPR 2026  
**arXiv**: [2504.05662](https://arxiv.org/abs/2504.05662)  
**代码**: [https://github.com/SkyShunsuke/InversionAD](https://github.com/SkyShunsuke/InversionAD)  
**领域**: 异常检测 / 扩散模型  
**关键词**: anomaly detection, diffusion model, DDIM inversion, reconstruction-free, industrial inspection  

## 一句话总结
提出"检测即加噪"范式取代传统"检测即去噪"——通过DDIM反转将图像映射到潜在噪声空间，仅用3步推理判断偏离先验分布的程度作为异常分数，无需重建，实现SOTA精度的同时推理速度达88 FPS（比OmiAD快2倍+）。

## 背景与动机
扩散模型做异常检测（AD）的主流范式是"detection via denoising in RGB space"——对输入图像加噪再用正常数据训练的扩散模型去噪重建，用重建误差度量异常。但这个范式有两个根本性限制：(1) **噪声强度敏感**：太强的噪声破坏正常区域（假阳性↑），太弱的噪声让异常区域被完美恢复（漏检↑）；(2) **多步去噪计算昂贵**：大多数方法仅1-2 FPS，无法满足实时需求。

## 核心问题
如何摆脱扩散模型AD中重建的必要性，实现既不需要调噪声强度超参、又不需要多步迭代的高效异常检测？

## 方法详解

### 整体框架
"Detection via noising in latent space"：(1) 用预训练backbone（EfficientNet-B4）提取特征 $z = g_\phi(x)$；(2) 在特征空间做DDIM反转（仅3步，$\tau_3 = [333, 666, 999]$），将特征映射到噪声空间 $z_T$；(3) 通过先验分布偏离度计算异常分数。

### 关键设计
1. **DDIM反转替代重建**：传统方法走 $x_0 \to x_t \to \hat{x}_0$；InvAD走 $x_0 \to x_T$。因为扩散模型仅在正常数据上训练，正常图像沿PF-ODE轨迹映射到高斯先验的高密度区域，异常图像映射到低密度区域。这构建了数据空间到噪声先验的确定性一一映射——不需要重建，直接在噪声端判别。

2. **极少步反转即可**：关键insight是AD不需要精确重建，因此不需要精确的ODE求解。仅用Euler法3步积分（$S=3$），虽然积分精度低，但异常像素仍能被可靠地映射到低密度区域。这使得推理效率从1 FPS提升到88 FPS。

3. **特征空间扩散**：在预训练backbone特征空间而非RGB空间做扩散，利用高级语义信息提升检测性能，同时进一步降低推理开销（特征分辨率16×16 vs 原图256×256）。

4. **NLL+Diff异常评分**：解决reverse-scoring问题——高维空间中正常数据可能获得更低NLL。用两个互补分数：(a) NLL衡量在噪声先验下的典型性；(b) Diff计算$z_T^{normed}$的max-min差异（利用异常局部稀疏的特性）。两者结合对步数$S$鲁棒。

### 训练策略
- 仅训练无条件DiT扩散模型（无类别条件、无伪异常），标准DDPM $\epsilon$-prediction损失
- 300 epochs, AdamW, warmup+cosine, batch=8
- 推理时plug-and-play：可直接替换任何扩散AD方法的推理阶段

## 实验关键数据

| MVTecAD (multi-class) | Image AU-ROC↑ | Image AP↑ | mAD↑ | FPS↑ |
|---|---|---|---|---|
| DiAD (AAAI'24) | 97.2 | 99.0 | 84.0 | 0.1 |
| HVQ-Trans (NeurIPS'23) | 98.0 | 99.5 | 83.6 | 5.6 |
| OmiAD (ICML'25) | 98.8 | 99.7 | 85.3 | 39.4 |
| **InvAD** | **99.0** | 99.6 | 83.7 | **88.1** |

| VisA | Image AU-ROC↑ | mAD↑ | FPS↑ |
|---|---|---|---|
| OmiAD | 95.3 | 79.3 | 35.3 |
| **InvAD** | **96.9** | **80.3** | **74.1** |

| MPDD | Image AU-ROC↑ | mAD↑ | FPS↑ |
|---|---|---|---|
| OmiAD | 93.7 | 78.9 | 49.8 |
| **InvAD** | **96.5** | **80.1** | **120** |

- Plug-and-play: DiAD+InvAD → AU-ROC 97.2→98.2, FPS 0.1→88.1; MDM+InvAD → 91.9→98.2, FPS 2.2→63
- BMAD医学数据集：mAD 87.2，FPS 88，全面超越PatchCore/RD4AD等

### 消融实验要点
- 反转 vs 重建对比（Table 4）：重建方法在S=3/r=40%时AU-ROC仅89.4，InvAD在S=3时达99.0——完全免除噪声强度调参
- 特征空间扩散（FDM）：像素空间单步反转仅44.9 AU-ROC → 特征空间多步反转83.7
- 评分方案（Table 9）：NLL单独在大S时性能下降（reverse-scoring问题），Diff单独也不稳定，NLL+Diff在所有S下保持稳定（99.0→95.4）
- Backbone选择：EfficientNet-B4 > DINO-B > ViT-B；但扩散架构影响小（MLP/UNet/DiT差异不大）
- 反转步数：S=3最优，均匀时间表优于二次/三次/指数

## 亮点
- 范式性创新："加噪检测"替代"去噪检测"——优雅地绕过了噪声强度调参和多步去噪两个根本性痛点
- 3步推理88 FPS——比之前最快的扩散AD方法OmiAD快2倍，且不需要对抗蒸馏
- Plug-and-play设计：只改推理阶段，可直接嵌入任何扩散AD框架
- 理论洞察深刻：分析了为什么低精度反转仍能有效检测异常，以及reverse-scoring问题的缓解

## 局限性 / 可改进方向
- 仍需3次NFE（函数估计），通过蒸馏压缩到1步是有前景的方向
- 像素级定位性能略低于SOTA（pixel AP/F1不如OmiAD），因为特征空间16×16分辨率限制了精细定位
- 依赖预训练backbone的质量——backbone对异常信息的压缩可能丢失小异常
- 评分方案中NLL+Diff的max-min差异对噪声比较敏感

## 与相关工作的对比
- **vs DiAD/GLAD/TransFusion (重建范式)**：它们需要多步去噪+噪声强度调参；InvAD无重建、无调参、速度快几十到几百倍
- **vs OmiAD (蒸馏方法)**：OmiAD用对抗蒸馏压缩到1步NFE但需要重backbone，39 FPS；InvAD 3步NFE轻量DiT，88 FPS
- **vs DeCo-Diff (CVPR'25)**：17 FPS且需要更复杂训练；InvAD标准无条件训练+88 FPS

## 启发与关联
- "反转代替重建"的思想可能推广到其他使用扩散模型做条件生成/对比的任务
- 异常评分的NLL+Diff组合策略对任何高维密度估计任务有参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "加噪检测"范式是扩散AD领域的范式变革，思路优雅简洁
- 实验充分度: ⭐⭐⭐⭐⭐ 4个基准、7个消融维度、plug-and-play验证、可视化分析、详尽补充材料
- 写作质量: ⭐⭐⭐⭐⭐ 动机阐述清晰、对比表(Table 1)直观、理论推导完整
- 价值: ⭐⭐⭐⭐⭐ 速度+精度双SOTA，plug-and-play设计使其可直接融入现有系统
