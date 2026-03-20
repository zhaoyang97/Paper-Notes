# Mobile-VTON: High-Fidelity On-Device Virtual Try-On

**会议**: CVPR 2026  
**arXiv**: [2603.00947]([https://arxiv.org/abs/2603.00947](https://arxiv.org/abs/2603.00947))  
**代码**: 有 ([https://zhenchenwan.github.io/Mobile-VTON/](https://zhenchenwan.github.io/Mobile-VTON/))  
**领域**: 图像生成 / 模型压缩 / 虚拟试穿  
**关键词**: 虚拟试穿, 移动端部署, 知识蒸馏, 扩散模型, 隐私保护  

## 一句话总结
首个全离线移动端扩散式虚拟试穿框架，基于TeacherNet-GarmentNet-TryonNet (TGT)架构，通过特征引导对抗蒸馏(FGA)将SD3.5 Large的能力迁移到415M参数的轻量学生网络，在VITON-HD和DressCode上以1024×768分辨率匹配甚至超越服务器端基线，端到端推理时间约80秒（小米17 Pro Max）。

## 背景与动机
虚拟试穿(VTON)技术在时尚电商领域非常实用，但现有高质量方法几乎全部依赖云端GPU：用户必须上传个人照片到服务器做推理，不仅有延迟和能耗问题，更带来严重的隐私风险（尤其在严格数据保护法规下）。将扩散式VTON部署到移动端面临三大挑战：(1) 模型参数量大、内存和延迟远超移动NPU/GPU能力；(2) 衣物表征在扩散时间步间会发生语义漂移，导致纹理扭曲和细节丢失；(3) 现有方法严重依赖大规模预训练（如ImageNet或大规模文生图），轻量架构无法直接从任务数据学到足够好的生成能力。

## 核心问题
如何在不上传用户数据、仅用一张人像和一张衣物图作为输入的条件下，在普通手机上实现高保真虚拟试穿？核心矛盾是：模型要足够小以在移动端运行，同时生成质量要追平参数量大5-17倍的服务器端方法。

## 方法详解

### 整体框架
Mobile-VTON采用模块化TGT架构：TeacherNet（冻结的SD 3.5 Large，作为知识源）+ GarmentNet（轻量学生，提取一致的衣物特征）+ TryonNet（轻量学生，融合人体和衣物信息生成试穿图像）。Light-Adapter用DINOv2-base替换大型CLIP视觉编码器，通过IP-Adapter机制注入衣物语义。整个系统从任务数据直接训练，不依赖外部预训练。

### 关键设计
1. **FGA蒸馏（Feature-Guided Adversarial Distillation）**：结合两个互补目标训练学生网络。(i) 特征级蒸馏：对齐TeacherNet和学生网络在每个扩散时间步上的score function，ℒ_feature = E_t[‖s_true - s_fake‖²]，采用DMD2式的score matching而非逐像素回归，让学生学到教师的分布行为。(ii) 对抗增强：引入轻量判别器D区分真实图像和TryonNet生成图像，通过标准GAN损失ℒ_GAN提升真实感和细节清晰度。
2. **TCG（Trajectory-Consistent GarmentNet）**：解决衣物特征在不同扩散时间步间的语义漂移问题。直接在每个时间步t对GarmentNet加重建约束 ℒ_cons = E_t[‖X̂_g(t) - X_g‖²]，要求网络在整个扩散轨迹上一致地重建原始衣物图。这种时序正则化使得衣物的颜色、纹理、logo在不同时间步下保持稳定。
3. **Garment-Aware TryonNet**：(i) Latent Concatenation (LC)：将人像和衣物图在高度维度拼接后编码到latent space，同时引入参考条件输入（目标人像+衣物的拼接编码），让TryonNet在无预训练的情况下也能学到衣物-身体对齐。(ii) 特征融合：在TryonNet每一层self-attention中拼接GarmentNet对应层的多尺度特征，cross-attention中同时接受文本和Light-Adapter提供的视觉K-V对，实现多层次的衣物语义注入。
4. **Light-Adapter**：用DINOv2-base替代CLIP大型视觉编码器，将衣物图像特征投影为K、V张量，通过解耦cross-attention注入TryonNet，兼顾语义丰富性和移动端效率。

### 损失函数 / 训练策略
- GarmentNet总损失：ℒ_GarmentNet = λ₁·ℒ_featureG + λ₂·ℒ_cons
- TryonNet总损失：ℒ_TryonNet = ℒ_Diff + λ₁·ℒ_featureT + λ₃·ℒ_GAN（其中ℒ_Diff为衣物感知重建损失）
- 超参数：λ₁=1e-2, λ₂=0.5, λ₃=5e-3
- 两阶段训练：Stage 1在DressCode+VITON-HD合并集上训练140 epochs（lr=1e-4），Stage 2在DressCode上微调100 epochs（lr=5e-5）
- 8×A100 80GB，batch size=256，AdamW优化器

## 实验关键数据
| 数据集 | 指标 | 本文(Mobile-VTON) | 之前SOTA | 对比说明 |
|--------|------|------|----------|------|
| VITON-HD | LPIPS↓ | 0.088 | 0.102 (IDM-VTON) | 超越服务器端最优(mask-based) |
| VITON-HD | SSIM↑ | 0.893 | 0.890 (SD-VITON) | 最佳 |
| DressCode | LPIPS↓ | 0.053 | 0.0513 (BooW-VTON) | 接近最优 |
| DressCode | SSIM↑ | 0.935 | 0.928 (BooW-VTON) | 最佳 |
| VITON-HD In-Wild | LPIPS↓ | 0.133 | 0.137 (IDM-VTON) | 最佳 |
| 内存占用 | GPU Memory | 2.84 GB | 5.80-18.47 GB | 减少51%-85% |
| 部署 | 移动端 | ✓ (小米17 Pro Max, ~80s) | 全部✗ | 唯一可移动端运行的方法 |

### 消融实验要点
- **TCG的贡献**：加入TCG后LPIPS从0.119降到0.111，SSIM从0.874升到0.879，CLIP-I从0.798升到0.805。视觉上可见logo和条纹更清晰、颜色定位更准确
- **LC的贡献**：在TCG基础上加LC，LPIPS进一步从0.111降到0.088，SSIM升到0.893，CLIP-I升到0.833。LC提供了显式衣物几何和外观线索，弥补无预训练的劣势
- **蒸馏的关键性**：去除蒸馏后FID从10.2暴涨到113.6，完全崩溃——说明轻量模型在无教师引导下从头训练根本无法收敛
- **数据集质量影响**：DressCode微调优于VITON-HD微调（轻量模型对数据质量更敏感，DressCode分辨率更统一、视觉更清晰）

## 亮点
- 技术上最大的亮点是FGA蒸馏：score-based distillation + GAN的组合让415M参数学生网络达到了2B+参数教师级别的生成质量
- TCG的设计非常简洁高效——只是一个跨时间步的重建一致性约束，但有效解决了扩散模型中衣物语义漂移的核心问题
- 全系统从任务数据直接训练、不依赖大规模预训练，对资源有限的场景很有参考价值
- DINOv2-base替代CLIP做视觉编码器的选择值得关注——在移动端场景下是一个好的效率-质量trade-off
- 在真实手机上跑通了完整pipeline并给出了实际推理时间（80s），不是纸上谈兵

## 局限性 / 可改进方向
- 80秒的端到端推理时间对用户体验来说仍然偏长，未使用步数缩减、剪枝或系统级加速
- 无法准确生成带文字的衣物（logo、品牌名、口号），因为缺乏文字感知预训练且训练数据中文字衣物较少
- 仅支持上半身试穿，未扩展到全身、裙装等类别
- 作为mask-free方法，需要合成整张图（含背景和身体），FID/KID指标上天然不如mask-based方法公平
- INT8量化在Android NPU上执行，但未报告量化带来的精度损失具体数据

## 与相关工作的对比
- **vs IDM-VTON (18.47GB)**：IDM-VTON是mask-based方法中最强的服务器端基线，在VITON-HD上CLIP-I达0.875。Mobile-VTON在LPIPS和SSIM上超越它，CLIP-I略低(0.833 vs 0.875)，但内存仅需2.84GB且可移动端运行——本质是不同维度的方法。
- **vs CatVTON**：同为mask-free方法，CatVTON也用latent拼接策略。Mobile-VTON在LPIPS/SSIM全面超越CatVTON(0.088 vs 0.161, 0.893 vs 0.872)，说明TGT架构+FGA蒸馏的组合远优于单纯使用CatVTON的拼接策略。
- **vs BooW-VTON**：BooW-VTON是mask-free方法中最强的服务器端基线，在FID/KID上最优。Mobile-VTON在DressCode上SSIM超越它(0.935 vs 0.928)，LPIPS接近(0.053 vs 0.051)，但内存仅需2.84GB vs 18.47GB。

## 启发与关联
- FGA蒸馏策略（score-based + 对抗）可迁移到其他需要部署在边缘设备的扩散模型任务
- TCG的时序一致性约束思路可借鉴到视频生成、3D一致生成等时序/多视角任务
- "数据质量对轻量模型比大模型更重要"这一发现值得在其他蒸馏研究中验证
- 关联idea: `20260316_convnet_dit_hybrid_distill.md`（扩散模型蒸馏相关）

## 评分
- 新颖性: ⭐⭐⭐⭐ [TGT架构设计和FGA蒸馏策略具有系统性创新，首个移动端扩散VTON有实际工程价值]
- 实验充分度: ⭐⭐⭐⭐⭐ [三个数据集、多个基线、详细消融、真实手机部署、数据集质量分析，非常全面]
- 写作质量: ⭐⭐⭐⭐ [结构清晰，图表丰富，方法描述详尽]
- 价值: ⭐⭐⭐⭐ [移动端部署扩散模型是重要工程方向，FGA蒸馏策略具有较好的通用性]
