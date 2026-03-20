# BecomingLit: Relightable Gaussian Avatars with Hybrid Neural Shading

**会议**: NeurIPS 2025  
**arXiv**: [2506.06271](https://arxiv.org/abs/2506.06271)  
**代码**: [jonathsch.github.io/becominglit](https://jonathsch.github.io/becominglit)  
**领域**: 3d_vision  
**关键词**: relightable avatar, 3D Gaussian Splatting, hybrid neural shading, light stage, BRDF, face reconstruction  

## 一句话总结

提出 BecomingLit，基于 3D Gaussian 原语和混合神经着色（neural diffuse BRDF + 解析 Cook-Torrance specular）从低成本 light stage 多视角序列重建可重光照、实时渲染的高保真头部 avatar，并发布了新的公开 OLAT 人脸数据集。

## 研究背景与动机

1. **工业需求强烈**：VR/元宇宙对可重光照的逼真头部 avatar 需求日益增长，但现有方案大多将训练环境的光照"烤入"外观，无法换光。
2. **传统 light stage 成本极高**：已有方法（如 RGCA、Deep Appearance Models）依赖数百个灯/相机的房间级设备，仅少数机构可负担。
3. **公开数据集匮乏**：用于面部外观建模的受控光照多视角数据集极少（Goliath 仅 4 人、分辨率偏低），阻碍了学术研究的广泛展开。
4. **解析 BRDF 不足以描述皮肤**：皮肤存在显著的次表面散射等全局光照效应，纯解析模型（如 Lambertian + Cook-Torrance）无法准确还原漫反射部分。
5. **已有神经方案泛化弱**：RGCA 学习预计算辐射传输（PRT），在未见光照下泛化较差；且使用 per-identity VAE 表情空间，不支持跨身份驱动。
6. **本文动机**：用 16 台相机 + 40 个 LED 的低成本 light stage，结合 FLAME 参数化模型和混合神经 BRDF，以约 1/10 成本达到 SOTA 且支持单目视频驱动。

## 方法详解

### 整体框架

输入 FLAME 表情/姿态参数 → 几何模块 $\mathcal{F}_g$ 预测 3D Gaussian 属性 → 混合神经着色（漫反射 $\mathcal{F}_d$ + 解析高光）→ Gaussian Splatting 渲染 → L1 + SSIM 光度损失 + 正则化联合优化。

### 三大核心设计

**1. 表情依赖的几何模块 $\mathcal{F}_g$（UV 空间 CNN）**

- 在 FLAME UV 贴图上定义固定数量的各向异性 Gaussian 原语（512² UV → 约 202k 原语）。
- $\mathcal{F}_g$ 是转置卷积网络，输入 FLAME 表情系数（109 维），输出每 texel 的位置偏移 $\delta\mu$、旋转 $q$、缩放 $s$、不透明度 $\sigma$ 和表情特征 $f^{expr}$（32 维）。
- Gaussian 中心 = FLAME 网格插值位置 + TBN 旋转后的静态局部偏移 + 动态偏移 $\delta\mu$；静态偏移承载大部分偏移量，动态偏移被正则化为小值，提升新表情泛化。

**2. 混合神经着色（Hybrid Neural Shading）**

- **漫反射 $\mathcal{F}_d$**：3 层 MLP（隐层 64），输入为 6 阶球谐系数（编码入射光）和 $f^{expr}$，输出标量反射率，乘以静态学习的 albedo $a_k$ 得到漫反射颜色。以单色 BRDF 参数化，训练时见白光、推理时按 RGB 通道分别求值以支持彩色光照。隐式学习次表面散射和自遮挡。
- **高光 $f_s$**：基于 Cook-Torrance 模型。NDF 使用 2-Blinn-Phong-lobe 混合（roughness $r$ 线性）；Fresnel 用 Schlick 近似；遮蔽项由 NDF 导出。小型 CNN $\mathcal{F}_v$ 以 $(f^{expr}, \omega_o)$ 预测高光强度 $k_s$ 和法线偏移 $\delta n$，着色法线 = 网格法线 + $\delta n$ 归一化。
- 环境光贴图重光照使用 split-sum 近似（预积分 mipmap + 2D BRDF LUT）。

**3. 低成本 OLAT 数据集**

- 16 台工业相机（2200×3208, 72fps, PTP 亚微秒同步）+ 40 个高 CRI LED，覆盖前半球。
- 10 位受试者，每人约 150s 多序列（预定义表情 + 朗读 + 自由表情），OLAT 帧与全亮跟踪帧 2:1 交替。
- 成本约为 Goliath（144 视角/460 灯）setup 的 1/10。

### 损失函数

$$\mathcal{L} = \underbrace{\lambda_{l1}\mathcal{L}_{l1} + \lambda_{SSIM}\mathcal{L}_{SSIM}}_{\text{photometric}} + \lambda_{normal}\|\delta n\| + \lambda_{alpha}\mathcal{L}_{alpha} + \lambda_{scale}\mathcal{L}_{scale} + \lambda_{pos}\|\delta\mu\|$$

- $\lambda_{l1}=1.0, \lambda_{SSIM}=0.2$；$\lambda_{alpha}=\lambda_{scale}=2\text{e-}2, \lambda_{pos}=1\text{e-}5$。
- Alpha 损失（渲染 alpha 与前景 mask 的 L2）是防止环境光贴图重光照时出现透明伪影的关键正则项——使得仅需前半球灯光即可避免背面伪影。

## 实验

### 主要对比（Table 2，4 位受试者，15 训练视角 / 1 测试视角，4 个 hold-out 光照）

| 方法 | Relighting PSNR↑ | SSIM↑ | LPIPS↓ | Relight+Reenact PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|---|
| RGCA | 29.21 | 0.8462 | 0.1659 | 26.31 | 0.8206 | 0.1917 |
| RGCA_FLAME | 29.78 | 0.8464 | 0.1444 | 26.91 | 0.8282 | 0.1667 |
| **BecomingLit** | **31.38** | **0.8956** | **0.1040** | **28.08** | **0.8730** | **0.1317** |

- 在重光照 PSNR 上超过 RGCA 约 **+2.2 dB**，LPIPS 降低 **37%**。
- 用 FLAME 表情空间替代 RGCA 原始 per-identity VAE 后（RGCA_FLAME），性能一致提升，说明共享表情空间对 reenactment 更优。

### 消融实验（Table 3）

| 变体 | Relight PSNR | Reenact PSNR | 关键发现 |
|---|---|---|---|
| w/ PBR shading | 29.42 | 26.31 | 纯解析 BRDF 无法建模次表面散射，外观塑料感 |
| w/ PRT diffuse | 29.23 | 25.47 | PRT 在未见光照上泛化最差（-2.6 dB vs full） |
| w/ SG specular | 31.55 | 28.09 | 点光源下略优但缺少毛孔级高光细节 |
| w/o alpha loss | 31.34 | 28.07 | 环境光贴图重光照出现透明伪影 |
| w/o expr features | 31.23 | 28.13 | 毛孔细节和高光还原变差 |
| **Full model** | **31.38** | **28.08** | 综合最优 |

### 运行时（Table 4，RTX A6000，202k 原语，1100×1604）

| 方法 | CNN | Diffuse | Specular | Splatting | 总计 |
|---|---|---|---|---|---|
| RGCA | 9ms | 1ms | 1ms | 9ms | 20ms |
| **Ours** | 4ms | 3ms | 1ms | 9ms | **17ms**（~59 FPS） |

### 关键实验发现

1. 混合神经着色的核心优势在漫反射端：隐式学习次表面散射 vs PRT/PBR 分析模型，泛化能力大幅领先。
2. 解析 Cook-Torrance 高光在环境光贴图下产生更自然的毛孔反射，优于 Spherical Gaussian。
3. Alpha 正则使仅前半球灯光 setup 也能支持全方位环境光重光照，将 setup 复杂度/成本/训练计算减半。
4. 表情依赖特征 $f^{expr}$ 同时影响漫反射和高光质量，是连接几何与外观的桥梁。

## 亮点

- **混合着色范式新颖**：漫反射用神经网络隐式学，高光用经典解析模型，物理可解释性与表达力兼得。
- **低成本 setup 实用性强**：16 相机 + 40 LED 即可达到 SOTA，比 Goliath（144 相机 + 460 灯）便宜一个数量级。
- **首个公开高分辨率 OLAT 人脸数据集**：10 人、72fps、2200×3208，填补社区数据空白。
- **全链路端到端**：FLAME 驱动 → 3DGS 几何 → hybrid shading → splatting 渲染，17ms 实时推理。
- **单目视频可驱动**：训练后仅需 FLAME 参数即可动画，天然兼容 VHAP 等单目追踪器。

## 局限性

1. 训练仍需数千帧 light stage 数据和多样表情，无法从随意手机拍摄重建——距消费级应用仍有距离。
2. 几何受限于 FLAME：不建模口腔内部、对视线方向跟踪敏感、细节表达力有限。
3. 漫反射网络 $\mathcal{F}_d$ 从零训练，未利用人脸外观先验——在少量数据下可能表现不稳定。
4. 数据集仅 10 位受试者，种族/年龄多样性有限。
5. 伦理风险：可从单目视频驱动的逼真 avatar 存在深度伪造隐患（虽然需先扫描）。

## 相关工作

| 方向 | 代表方法 | 与本文关系 |
|---|---|---|
| 3DGS 头部建模 | GaussianAvatars, NPGA, RGCA | 本文在 3DGS 基础上增加可重光照能力 |
| Light Stage 面部捕获 | Debevec 2000, Guo 2019, Goliath | 本文提出更低成本的 setup 和新数据集 |
| 预计算辐射传输 | RGCA (PRT SH 系数) | 本文用 neural diffuse 替代 PRT，泛化更好 |
| 神经着色 | LitNeRF, ReNeRF, Neural BRDF | 本文首次在动态 3DGS avatar 上组合 neural diffuse + 解析 specular |
| 可驱动 avatar | FLAME, VHAP, Codec Avatars | 本文直接复用 FLAME 共享表情空间，免去 per-identity encoder |

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 混合神经着色范式（neural diffuse + analytic specular）、低成本 light stage 及公开数据集均为新贡献
- **实验充分度**: ⭐⭐⭐⭐ — 4 人定量对比 + 5 项消融 + 运行时分析 + 环境光/单目驱动应用展示，但受试者数量偏少
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰、方法图示完整、公式推导规范，附录详尽
- **价值**: ⭐⭐⭐⭐ — 公开数据集 + 低成本方案有望推动社区跟进，实时性和驱动灵活性适合 VR/AR 落地
