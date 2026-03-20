# Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction

**会议**: ICCV 2025  
**arXiv**: [2411.14384](https://arxiv.org/abs/2411.14384)  
**代码**: [项目页面](https://caiyuanhao1998.github.io/project/DiffusionGS/) (有)  
**领域**: 3D视觉 / 3D生成与重建  
**关键词**: 3D Gaussian Splatting, 扩散模型, 单视图3D生成, 场景重建, 单阶段  

## 一句话总结
提出DiffusionGS，将3D高斯点云直接嵌入扩散模型的去噪器中，通过单阶段3D扩散实现从单张图片到3D物体生成和场景重建，在ABO/GSO上PSNR超越SOTA 2.20/1.25 dB，RealEstate10K上超1.34 dB，推理速度约6秒（A100）。

## 背景与动机
现有前馈式image-to-3D方法主要分两大类，均存在明显缺陷：
1. **两阶段方法**（LGM、CRM、12345++等）：先用2D多视角扩散生成多视角图像，再用3D重建模型拼合。由于扩散过程缺乏3D模型，无法保证视角一致性，当输入视角方向变化时容易崩溃。
2. **基于triplane-NeRF的3D扩散**（DMV3D、RenderDiffusion等）：虽然引入3D模型，但triplane分辨率有限、体渲染(volume rendering)速度慢，难以扩展到大规模场景。
3. **单视图场景重建方法**（Flash3D、VistaDream等）：依赖单目深度估计器，在严重遮挡或大视角变化下容易失败。

此外，现有方法主要使用以物体为中心(object-centric)的数据集训练，泛化能力有限，对真实场景级别的应用缺乏支持。

## 核心问题
如何在一个单阶段框架中同时实现高保真的3D物体生成和场景重建，保证多视角一致性，且不依赖深度估计器？如何利用有限的3D数据（场景数据尤其稀缺）训练出泛化能力更强的模型？

## 方法详解
### 整体框架
DiffusionGS是一个单阶段3D扩散模型。输入一张干净条件视图和N张加噪视图，通过Transformer去噪器直接预测像素对齐的3D高斯点云(pixel-aligned 3D Gaussians)，然后通过可微光栅化渲染多视角图像进行2D监督。整个框架采用$x_0$-prediction而非$\epsilon$-prediction策略，以获得干净完整的3D高斯表示。

### 关键设计
1. **将3DGS嵌入去噪器（Baking GS into Denoiser）**: 每个去噪时间步都直接输出3D高斯点云$\mathcal{G}_\theta$，每个高斯包含中心位置$\boldsymbol{\mu}$、协方差$\boldsymbol{\Sigma}$、不透明度$\alpha$和RGB颜色$\boldsymbol{c}$。高斯数量固定为$(N+1) \times H \times W$（条件视图+N个噪声视图的像素对齐高斯）。深度通过近远距离的加权插值参数化：$u_t^{(k)} = w_t^{(k)} u_{near} + (1-w_t^{(k)}) u_{far}$。

2. **场景-物体混合训练策略（Scene-Object Mixed Training）**: 
   - **视角选择约束**：对相机位置和朝向施加两个角度约束（位置角度$\theta_{cd} \leq \theta_1$、$\theta_{dn} \leq \theta_2$；朝向约束保证视角重叠），确保训练收敛。
   - **双高斯解码器（Dual Gaussian Decoder）**：物体和场景使用不同的MLP解码器处理不同的深度范围（物体$[u_{near}, u_{far}]=[0.1, 4.2]$，场景$[0, 500]$），混合训练后微调时只保留对应的单个解码器。
   - **点分布损失$\mathcal{L}_{pd}$**：在训练早期warm-up阶段鼓励物体级高斯点云分布更集中。

3. **Reference-Point Plücker Coordinate (RPPC)**:
   传统Plücker坐标使用力矩向量(moment vector)作为相机条件，但力矩向量随相机位移而变化，无法有效捕捉深度和3D几何。RPPC用光线上离世界坐标系原点最近的点(reference point)替代力矩向量：$\boldsymbol{r} = (\boldsymbol{o} - (\boldsymbol{o} \cdot \boldsymbol{d})\boldsymbol{d}, \boldsymbol{d})$。RPPC满足4D光场的平移不变性假设，同时提供更丰富的射线位置和相对深度信息。

### 损失函数 / 训练策略
- **去噪损失$\mathcal{L}_{de}$**: $\mathcal{L}_2$损失 + $\lambda \cdot \mathcal{L}_{VGG}$感知损失，作用于去噪渲染的多视角图像与GT之间
- **新视角损失$\mathcal{L}_{nv}$**: 同样的$\mathcal{L}_2$ + 感知损失，作用于额外的新视角渲染
- **点分布损失$\mathcal{L}_{pd}$**: 仅在物体数据的训练warm-up阶段使用
- **总目标**: $\mathcal{L} = (\mathcal{L}_{de} + \mathcal{L}_{nv}) \cdot \mathbf{1}_{iter > iter_0} + \mathcal{L}_{pd} \cdot \mathbf{1}_{iter \leq iter_0} \cdot \mathbf{1}_{object}$
- **训练流程**：(1) 32×A100混合训练40K iter → (2) 64×A100物体/场景分别微调80K/54K iter → (3) 512×512分辨率再微调20K iter
- **推理**：30步DDIM采样，约6秒（A100）

## 实验关键数据
| 数据集 | 指标 | DiffusionGS | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| ABO (物体) | PSNR/FID | 25.89/9.03 | DMV3D: 23.69/32.28 | +2.20 dB / -23.25 |
| GSO (物体) | PSNR/FID | 22.07/11.52 | DMV3D: 20.82/33.48 | +1.25 dB / -21.96 |
| RealEstate10K (场景) | PSNR/FID | 21.63/15.87 | Flash3D: 20.29/35.03 | +1.34 dB / -19.16 |
| vs PhotoNVS (场景) | PSNR/FID | 21.63/15.87 | 15.31/28.30 | +6.32 dB / -12.43 |
| 用户研究 (1-6分) | 主观质量 | 4.88 | 12345++: 3.81 | +1.07 |
| 推理速度 | 时间 | 5.8s | DMV3D: 31.4s | ~5.4× |

### 消融实验要点
- 在GSO上的逐步消融（从baseline 17.63 dB开始）：
  - +扩散框架: +2.94 dB PSNR, -70.45 FID
  - +$\mathcal{L}_{pd}$: +0.37 dB, -19.45 FID
  - +混合训练: +0.79 dB, -10.62 FID
  - +RPPC: +0.34 dB, -6.27 FID → 最终22.07 dB / 11.52 FID
- 混合训练对场景重建的提升：+0.61 dB PSNR, -10.53 FID（RealEstate10K）
- RPPC对场景重建的提升：+0.28 dB PSNR, -7.09 FID
- 改变随机种子可生成不同形状和纹理的3D资产，具备生成多样性

## 亮点
- **单阶段3D扩散**：将3DGS直接嵌入去噪过程，每步都输出3D高斯点云，从根本上保证视角一致性，摆脱两阶段pipeline的割裂问题
- **场景+物体统一框架**：通过精心设计的混合训练策略和双解码器，首次在一个模型中同时处理物体生成和场景重建
- **不依赖深度估计器**：通过沿相机轨迹生成多视角来预测更精细的高斯点云，无需额外的单目深度估计
- **RPPC设计精巧**：用参考点替代力矩向量，直觉清晰且满足光场理论，有效提升深度感知
- **速度极快**：推理仅~6秒，比DMV3D快5倍以上

## 局限性 / 可改进方向
- 训练成本较高（32-64×A100），不易复现
- 像素对齐高斯的数量固定为$(N+1)HW$，对于复杂场景可能不够灵活（无自适应密度控制）
- 当前分辨率256×256（微调到512），对高分辨率应用仍有提升空间
- 场景数据仍然相对稀缺（~90K样本），更大规模的场景数据可能进一步提升泛化能力
- 未探索与更强的2D基础模型（如Stable Diffusion XL）的结合

## 与相关工作的对比
- **vs DMV3D**（ICLR 2024，triplane-NeRF 3D扩散）：DiffusionGS用3DGS替代NeRF，推理速度快5×，且不受triplane分辨率限制。在ABO上PSNR高2.20 dB。
- **vs LGM**（ECCV 2024，2D多视角扩散+3DGS重建）：LGM是两阶段做法，2D扩散无法保证3D一致性。DiffusionGS在GSO上PSNR高7.80 dB（22.07 vs 14.27）。
- **vs Flash3D**（3DV 2025，单视图场景重建）：Flash3D依赖单目深度估计器，在遮挡区域容易产生伪影和黑斑。DiffusionGS在RealEstate10K上PSNR高1.34 dB且无需深度估计。

## 启发与关联
- 与 [频域安全防御3DGS](../../../ideas/3d_vision/20260316_spectral_defense_3dgs.md) 的关联：DiffusionGS生成的高斯点云可以作为该框架的测试对象，研究生成式3DGS的安全特性
- 与 [多尺度光照场3DGS](../../../ideas/3d_vision/20260317_multiscale_illumination_field_3dgs.md) 的关联：DiffusionGS当前不处理视角依赖光照效果，可考虑将多尺度光照场嵌入扩散过程提升渲染质量
- 与 [语言条件化高斯剪枝](../../../ideas/3d_vision/20260317_language_conditioned_gaussian_pruning_nav.md) 的关联：DiffusionGS输出的大量像素对齐高斯可能受益于任务驱动的剪枝策略
- **新想法方向**：将DiffusionGS的单阶段3D扩散思路扩展到视频→4D生成（时空高斯扩散），或结合更强的图像编码器实现controllable 3D generation

## 评分
- 新颖性: ⭐⭐⭐⭐ 将3DGS嵌入扩散去噪器的思路自然且有效，RPPC和混合训练策略有创意，但pixel-aligned GS和$x_0$-prediction的单独组件已有前人工作
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖物体生成(ABO/GSO)和场景重建(RealEstate10K)，有用户研究、定量对比、消融实验、视觉分析、多样性分析，非常全面
- 写作质量: ⭐⭐⭐⭐ 整体清晰，pipeline图直观，公式推导完整，但场景-物体混合训练的细节较分散
- 价值: ⭐⭐⭐⭐ 在单视图3D生成/重建领域达到新SOTA，速度快且支持场景+物体统一处理，实用价值高
