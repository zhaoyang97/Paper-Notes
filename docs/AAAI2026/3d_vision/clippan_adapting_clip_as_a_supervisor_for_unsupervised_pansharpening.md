---
title: >-
  [论文解读] CLIPPan: Adapting CLIP as A Supervisor for Unsupervised Pansharpening
description: >-
  [AAAI 2026][3D视觉][全色锐化] 提出 CLIPPan，通过轻量微调 CLIP 使其理解多光谱/全色/高分辨率多光谱图像类型及全色锐化过程，然后利用 Wald 协议等文本提示作为语义监督信号，实现无需地面真值的全分辨率无监督全色锐化，可作为即插即用模块兼容任意全色锐化骨干网络。 全色锐化（Pansharpeni…
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "全色锐化"
  - "CLIP"
  - "无监督"
  - "视觉-语言模型"
  - "遥感"
---

# CLIPPan: Adapting CLIP as A Supervisor for Unsupervised Pansharpening

**会议**: AAAI 2026  
**arXiv**: [2511.10896](https://arxiv.org/abs/2511.10896)  
**代码**: [Jiabo-Liu/CLIPPan](https://github.com/Jiabo-Liu/CLIPPan)  
**领域**: 模型压缩  
**关键词**: 全色锐化, CLIP, 无监督, 视觉-语言模型, 遥感

## 一句话总结

提出 CLIPPan，通过轻量微调 CLIP 使其理解多光谱/全色/高分辨率多光谱图像类型及全色锐化过程，然后利用 Wald 协议等文本提示作为语义监督信号，实现无需地面真值的全分辨率无监督全色锐化，可作为即插即用模块兼容任意全色锐化骨干网络。

## 研究背景与动机

全色锐化（Pansharpening）将富含光谱信息的多光谱（MS）图像与具有高空间分辨率的全色（PAN）图像融合，生成高分辨率多光谱（HRMS）图像，在城市规划、环境监测等遥感应用中至关重要。

**核心问题**：现有深度学习方法严重依赖地面真值（GT）进行监督训练，但实际全分辨率场景中 GT 不可获取。通常的做法是在降分辨率模拟数据上训练，导致在真实全分辨率图像上存在严重的域差距（domain gap）。无监督方法虽然避免了 GT 依赖，但仅利用融合输出与输入之间的低级像素关系作为约束，缺乏对融合目标的高级语义指导。

**核心 insight**：如果能告诉模型"融合的目标规则是什么"（如 Wald 协议），就能利用高级语义监督约束融合输出处于 HRMS 域中。CLIP 的图像-文本对齐能力恰好可以将文本描述的融合规则转化为监督信号。

## 方法详解

### 整体框架

CLIPPan 分为两个阶段：

**Stage I — 视觉-语言对齐**：微调 CLIP 使其 (i) 识别 LRMS/PAN/HRMS 图像类型，(ii) 理解遥感图像内容，(iii) 理解全色锐化过程（MS+PAN→HRMS 的映射）。

**Stage II — 语言引导的无监督全色锐化**：用微调后的 CLIP 作为固定语义监督器，结合低级视觉约束训练全色锐化网络。

### 关键设计

**1. 参数高效微调策略**

为保持 CLIP 的强泛化能力，仅插入 6 个轻量 adapter 模块（3 个视觉端 + 3 个文本端）。由于 CLIP 视觉编码器不兼容多光谱图像的多波段输入，用可学习卷积层替换原始 RGB 输入层。

**2. 模间对比学习（InterMCL）**

将每种图像类型绑定到对应的语义空间。使用固定文本描述（而非内容相关描述）：
- MS："a multispectral image"
- PAN："a panchromatic image"  
- HRMS："High-quality reference image adhering to Wald's protocol: spectrally consistent with original data and spatially sharp"

通过对比损失将图像-文本正样本对拉近、负样本对推远：

$$\mathcal{L}_{\text{inter}} = \frac{1}{3}\sum_{M1,M2}\mathcal{L}_{\text{align}}(F^I_{M1}, F^T_{M2})$$

其中 HRMS 图像用传统 BDSD 算法在线生成（因为全分辨率下不可获取）。

**3. 模内对比学习（IntraMCL）**

防止使用固定描述导致的特征坍缩问题。将同一场景的 LRMS/PAN/HRMS 作为正样本，不同场景作为负样本：

$$\mathcal{L}_{\text{intra}} = -\frac{1}{3N}\sum_{i=1}^{3N}\log\frac{\exp(\langle F^{I(i)}_{M1}, F^{I(i)}_{M2}\rangle/\tau_i)}{\sum_{k=1}^{3N}\exp(\langle F^{I(i)}_{M1}, F^{I(j)}_{M1}\rangle/\tau_i)}$$

保证特征多样性，同时促进从自然图像到遥感图像的域迁移。

**4. 融合感知对齐**

引入图像融合适配器（IFA）和文本融合适配器（TFA），学习从 MS+PAN 特征生成融合特征，与 HRMS/Wald 协议特征对齐：

$$\mathcal{L}_{\text{fusion}} = \|F^T_{\text{fuse}} - F^T_{\text{wald}}\|_1 + \|F^I_{\text{fuse}} - F^I_{\text{HRMS}}\|_1$$

**5. 方向向量语义监督（Stage II 核心）**

不能直接用 Wald 协议文本特征做逐元素损失（因为对所有图像固定不变），而是利用**特征位移方向的一致性**：

$$\mathcal{L}_d = 1 - \frac{1}{2}(\langle \Delta\mathbf{V}^I_{\text{MS}}, \Delta\mathbf{V}^T_{\text{MS}}\rangle + \langle \Delta\mathbf{V}^I_{\text{PAN}}, \Delta\mathbf{V}^T_{\text{PAN}}\rangle)$$

其中 $\Delta\mathbf{V}^I_{\text{MS}} = F^I_{\text{out}} - F^I_{\text{MS}}$ 为图像空间的融合变化方向，$\Delta\mathbf{V}^T_{\text{MS}} = F^T_{\text{wald}} - F^T_{\text{MS}}$ 为文本空间的融合目标方向。通过惩罚两个空间中方向的角度偏差，引导输出语义上对齐 HRMS 域。

### 损失函数 / 训练策略

**Stage I**：$\mathcal{L}_{s1} = \mathcal{L}_{\text{inter}} + \mathcal{L}_{\text{intra}} + \mathcal{L}_{\text{fusion}}$

**Stage II 低级视觉约束**：
- 光谱保真：$\mathcal{L}_{\text{spec}} = \|\downarrow(\mathbf{I}_{\text{out}}) - \mathbf{I}_{\text{MS}}\|_2^2 + 1 - \text{SSIM}$
- 空间锐度：$\mathcal{L}_{\text{spat}} = \|\phi(\mathbf{I}_{\text{out}}) - \mathbf{I}_{\text{PAN}}\|_2^2 + 1 - \text{SSIM}$
- QNR 权衡：$\mathcal{L}_{\text{QNR}} = (1-D_\lambda)(1-D_s)$
- 伪监督：$\mathcal{L}_{\text{ship}}$（以降分辨率训练的 SHIP 网络输出为参考）

**总损失**：$\mathcal{L}_{s2} = \mathcal{L}_{\text{spec}} + \mathcal{L}_{\text{spat}} + \mathcal{L}_{\text{QNR}} + \mathcal{L}_{\text{ship}} + \mathcal{L}_d$

GTX-4090，Adam（lr=0.003），batch size=32，1000 iterations。

## 实验关键数据

### 主实验

**Table 1: 全分辨率 + 降分辨率定量结果（QB 与 WV3 数据集）**

| 方法 | QB $D_\lambda$↓ | QB QNR↑ | WV3 $D_\lambda$↓ | WV3 QNR↑ |
|---|---|---|---|---|
| ArbRPN | 0.0140 | 0.9582 | 0.0271 | 0.9383 |
| ArbRPN-C | **0.0030** | **0.9691** | **0.0042** | **0.9582** |
| LFormer | 0.0124 | 0.9602 | 0.0253 | 0.9227 |
| LFormer-C | **0.0053** | **0.9676** | **0.0049** | **0.9572** |
| PanMamba | 0.0134 | 0.9592 | 0.0152 | 0.9426 |
| PanMamba-C | **0.0050** | **0.9672** | **0.0051** | **0.9578** |

CLIPPan（-C 后缀）在所有 5 个骨干网络上均一致提升。ArbRPN-C 在 QB 上光谱失真 $D_\lambda$ 减少 79%，LFormer-C 在 WV3 上空间失真 $D_s$ 减少约 30%。

### 消融实验

**Table 2: 无监督融合损失消融（WV3 降分辨率，ArbRPN 骨干）**

| 损失组合 | MPSNR↑ | ERGAS↓ | SAM↓ | Q2n↑ |
|---|---|---|---|---|
| $\mathcal{L}_{\text{spec}} + \mathcal{L}_{\text{spat}}$ | 29.27 | 8.96 | 9.17 | 0.61 |
| $\mathcal{L}_{\text{unsup}}$ | 32.19 | 5.88 | 6.66 | 0.71 |
| $\mathcal{L}_{\text{unsup}} + \mathcal{L}_d$ | 32.37 | 5.75 | 6.55 | 0.74 |
| $\mathcal{L}_{\text{unsup}} + \mathcal{L}_{\text{ship}} + \mathcal{L}_d$ | **34.72** | **4.49** | **5.54** | **0.80** |

语义损失 $\mathcal{L}_d$ 和伪监督 $\mathcal{L}_{\text{ship}}$ 联合使用效果最佳，MPSNR 提升 5.4 dB。

**Table 3: CLIP 微调损失消融** — IntraMCL、InterMCL、$\mathcal{L}_1$ 逐步添加均带来提升。

**Table 5: 文本描述消融** — Wald 协议文本在所有指标上取得最佳平衡，证实精确的协议文本对语义监督至关重要。

### 关键发现

- 即使没有 GT，CLIPPan 仍能获得与有监督方法接近的效果
- 降分辨率实验也获得一致提升，说明框架对监督/无监督均有效
- 可学习残差卷积处理多光谱输入优于 PCA/RGB/GBNIR 等手动策略

## 亮点与洞察

1. **语言即监督**：首次将 Wald 协议等融合规则作为文本提示转化为 CLIP 语义监督，思路优雅
2. **方向向量损失**：不直接比较固定文本特征，而是比较"融合前后特征位移方向"的一致性，巧妙解决文本特征不变的问题
3. **即插即用通用性**：与 5 种不同骨干网络兼容，均获提升，实用价值高
4. **双向启示**：框架不仅能用协议指导融合，反过来也可以评估协议有效性乃至发现新协议

## 局限与展望

1. CLIP 微调阶段仍需 BDSD 算法生成 HRMS 近似标签，引入了额外先验
2. 文本描述目前为手工设计的固定模板，可探索可学习的 prompt tuning
3. 仅在 WorldView-3 和 QuickBird 两类传感器上验证，更多传感器的泛化性待验证
4. 伪监督 $\mathcal{L}_{\text{ship}}$ 依赖于预训练的 SHIP 网络质量

## 相关工作与启发

- **CLIP-Adapter / CoOp / LoRA-CLIP**：参数高效微调策略，本文采用 adapter 路线
- **RS-CLIP / GeoCLIP**：遥感领域的 CLIP 适配，本文将其扩展到全色锐化任务
- **启发**：该范式可推广到其他遥感图像融合任务（如高光谱-多光谱融合）；"用协议文本做监督"的思想也可应用于其他有明确规则约束但缺乏 GT 的任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 将视觉-语言模型引入全色锐化监督，范式创新性强
- 技术深度: ⭐⭐⭐⭐ — 两阶段设计完整，方向向量损失设计精巧
- 实验充分度: ⭐⭐⭐⭐ — 5 种骨干 × 2 数据集，消融全面（5 组消融实验）
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机论证充分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DINO Eats CLIP: Adapting Beyond Knowns for Open-set 3D Object Retrieval](../../CVPR2026/3d_vision/dino_eats_clip_adapting_beyond_knowns_for_open-set_3d_object_retrieval.md)
- [\[AAAI 2026\] Multi-Modal Assistance for Unsupervised Domain Adaptation on Point Cloud 3D Object Detection](multi-modal_assistance_for_unsupervised_domain_adaptation_on_point_cloud_3d_obje.md)
- [\[CVPR 2026\] Adapting Point Cloud Analysis via Multimodal Bayesian Distribution Learning](../../CVPR2026/3d_vision/adapting_point_cloud_analysis_via_multimodal_bayesian_distribution_learning.md)
- [\[CVPR 2026\] CLIPoint3D: Language-Grounded Few-Shot Unsupervised 3D Point Cloud Domain Adaptation](../../CVPR2026/3d_vision/clipoint3d_language-grounded_few-shot_unsupervised_3d_point_cloud_domain_adaptat.md)
- [\[NeurIPS 2025\] Flux4D: Flow-based Unsupervised 4D Reconstruction](../../NeurIPS2025/3d_vision/flux4d_flow-based_unsupervised_4d_reconstruction.md)

</div>

<!-- RELATED:END -->
