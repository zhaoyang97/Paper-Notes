# DynaQuant: Dynamic Mixed-Precision Quantization for Learned Image Compression

**会议**: AAAI 2026  
**arXiv**: [2511.07903](https://arxiv.org/abs/2511.07903)  
**代码**: [https://github.com/baoyu2020/DynaQuant](https://github.com/baoyu2020/DynaQuant)  
**领域**: 模型压缩  
**关键词**: 图像压缩量化, 混合精度, 动态比特宽度分配, 量化感知训练, 学习图像压缩  

## 一句话总结
针对学习图像压缩（LIC）模型部署效率低的痛点，提出DynaQuant框架，在参数层面通过可学习scale/zero-point + Distance-Aware Gradient Modulator实现内容自适应量化，在架构层面通过轻量Bit-Width Selector动态为每层分配最优比特宽度，在Cheng2020/ELIC/Ballé三个基线上实现接近FP32的R-D性能，同时获得最高5.17×加速和模型大小降至原来的~1/4。

## 背景与动机
LIC模型（如ELIC、Cheng2020）的R-D性能已超越VVC等传统编码器，但计算复杂度和内存需求巨大，难以在手机/无人机等边缘设备部署。现有量化方法存在两个关键缺陷：(1) 采用**全局统一比特宽度**（如全8-bit），忽视了LIC模型中不同层对量化噪声的敏感度差异巨大；(2) 量化参数（scale、zero-point）**静态固定**，无法适应LIC中随输入图像内容剧烈变化的latent特征分布。这导致要么对鲁棒层过度保守（浪费算力），要么对敏感层过于激进（损害R-D性能）。

## 核心问题
如何为LIC模型设计一种**双层动态**量化策略：(1) 参数层面——量化参数能根据输入内容自适应调整；(2) 架构层面——每层的比特宽度能根据层敏感度和数据特性动态分配？同时，如何解决量化中round操作不可微导致的训练困难？

## 方法详解
### 整体框架
DynaQuant包含两个互补的模块：**Dynamic Parameter Adaptation (DPA)** 解决参数级自适应，**Dynamic Bit-Width Selector (DBWS)** 解决层级比特宽度分配。两者都嵌入标准QAT流程中端到端联合优化。Hyperencoder固定使用8-bit量化（参数少且对量化敏感，动态分配收益不大反而增加开销）。

### 关键设计
1. **Content-Aware Quantization Mapping**: 将传统QAT中静态计算的scale $s$ 和zero-point $z$ 改为**可学习的per-channel参数**，通过R-D损失反向传播端到端优化。这使量化映射能适应每张输入图像的latent特征分布变化。

2. **Distance-Aware Gradient Modulator (DGM)**: 针对STE（直通估计器）将round梯度粗暴近似为常数1的局限性，提出新的梯度代理函数 $g(x) = \frac{1}{2} \cdot \frac{\tanh(\beta(x - \lfloor x \rfloor) - 0.5)}{\tanh(0.5)} + 0.5$。其梯度随输入到量化边界（半整数如0.5处）的距离变化：**边界附近的值梯度大**（强调需要继续优化），**量化中心附近的值梯度小**（已经稳定），提供比STE更精确的优化信号。

3. **Dynamic Bit-Width Selector (DBWS)**: 轻量网络模块，取输入激活tensor $A \in \mathbb{R}^{C \times H \times W}$，经AdaptivePool（输出5×5）→ Flatten → 两层MLP（含Dropout p=0.2）→ Reshape → **Gumbel-Softmax**（训练时soft采样、推理时argmax硬选择），输出各层在候选比特宽度集合 $\mathcal{B} = \{b_1, b_2, ..., b_M\}$ 上的概率分布。编码器和解码器各有独立的DBWS，但架构对称——确保编解码器生成一致的比特宽度策略，**无需传输额外的比特宽度配置信息**。

### 损失函数 / 训练策略
联合优化损失：$\mathcal{L} = R + \lambda D + \gamma \mathcal{L}_{\text{bits}}$

- $R$：量化latent的熵模型估计码率
- $D$：重建失真（MSE或MS-SSIM）
- $\mathcal{L}_{\text{bits}} = \frac{1}{L} \sum_{l=1}^{L} \sum_{k=1}^{M} (p_l)_k \cdot b_k$：所有动态量化层的期望平均比特宽度，通过 $\gamma$ 控制R-D性能与计算效率的权衡

DBWS输入策略：编解码器各自第一个module固定8-bit量化，其输出作为对应DBWS的输入；后续所有module（第2到第$BL$个）使用DBWS输出的自适应比特宽度。

## 实验关键数据

**Table 1 主要结果**（BD-Rate loss % / 加速倍数 / 模型大小）：

| 模型 | 方法 | Kodak BD-Rate | 平均BD-Rate | 加速 | 模型大小 |
|------|------|:---:|:---:|:---:|:---:|
| Cheng | FP32基线 | 0.00% | 0.00% | 1.00× | 45.08 MB |
| Cheng | FMPQ | 0.89% | 1.30% | 4.00× | ~11.27 MB |
| Cheng | RDO-PTQ | 4.88% | 4.88% | 4.00× | ~11.27 MB |
| Cheng | **Q-Cheng (DPA)** | **1.02%** | **1.60%** | 4.00× | 11.27 MB |
| Cheng | **DQ-Cheng (DPA+DBWS)** | 7.15% | 12.18% | **5.17×** | **8.72 MB** |
| ELIC | FP32基线 | 0.00% | 0.00% | 1.00× | 137.11 MB |
| ELIC | **Q-ELIC** | 5.97% | 4.92% | 4.00× | 34.28 MB |
| ELIC | **DQ-ELIC** | 7.62% | 6.39% | **4.61×** | **29.78 MB** |
| Ballé | FP32基线 | 0.00% | 0.00% | 1.00× | 19.37 MB |
| Ballé | FMPQ | 6.48% | 7.50% | ~3.98× | ~4.87 MB |
| Ballé | **Q-Ballé** | 5.85% | **5.01%** | 4.00× | 4.84 MB |
| Ballé | **DQ-Ballé** | 7.63% | 6.84% | **4.55×** | **4.26 MB** |

关键结论：Q-（固定8-bit DPA）在Cheng上BD-Rate仅1.60%，优于RDO-PTQ（4.88%），接近FMPQ（1.30%）；DQ-版以略大BD-Rate换取额外~1.2×加速提升。

### 消融实验要点

**Table 2 通用消融**（Cheng2020 q=6, Kodak）:
- DPA INT8: bpp=0.828, PSNR=36.649, R-D loss=1.56（优于PAMS的36.185/1.64）
- DPA-DQ: 平均6.42-bit, PSNR=36.636, R-D loss=1.57（几乎无损地降低25%平均比特宽度）
- PAMS-DQ: 平均6.85-bit, PSNR仅30.262, R-D loss=4.28 → DPA与DBWS具有**协同增效**效应（非简单加法）

**Table 3 DPA组件消融**（去掉任一组件性能下降）:
- 去掉可学习$s$: PSNR 36.649→36.185（-0.464 dB）
- 去掉可学习$z$: PSNR →36.323（-0.326 dB）
- 去掉DGM梯度调节$g(x)$: PSNR →36.288（-0.361 dB）
- 三者都很重要，其中scale $s$ 影响最大

**Table 4 DBWS候选比特集消融**:
- {4,6,8}: 平均5.47-bit, PSNR=36.432
- {6,8,10}: 平均6.42-bit, PSNR=36.636（更好的效率-保真度权衡）

**Bit-Width分配可视化**（Fig.6）: 纹理复杂的Kodim14在gs-1层分配10-bit（超过其他图像的8-bit）；边缘层（ga-0, ga-6, gs-1）倾向更高精度，中间层精度更低——证实了bottleneck层确实需要更多比特。

## 亮点
- **双层动态设计思路清晰**: 参数级和架构级两个层面相互独立又互补，形成了完整的自适应量化pipeline
- **DGM梯度调制器有理论动机**: 不是简单替换STE，而是基于"距决策边界远近"的直觉设计，让量化参数的优化更加targeted
- **编解码器对称DBWS**: 无需传输额外的比特配置元数据，实用性强
- **协同增效实验说服力强**: Table 2清楚展示DPA+DBWS > DPA + DBWS单独效果之和
- **跨架构泛化**: 在Cheng2020、ELIC、Ballé三种不同结构的LIC模型上均有效

## 局限性 / 可改进方向
- **DQ模式下BD-Rate损失显著增大**: DQ-Cheng在JPEG-AI上BD-Rate高达16.52%，说明动态比特宽度在某些数据集/内容类型上仍有明显性能损失，加速换来了不可忽视的质量下降
- **仅在三个相对老的LIC基线上验证**: Cheng2020（2020）、Ballé（2018）、ELIC（2022）都不是最新SOTA，如MambaIC等未涉及
- **Hyperencoder固定8-bit的决策缺乏充分论证**: 仅表述"实验观察到"敏感，缺少量化的敏感度分析
- **候选比特宽度集合需要手动设定**: {4,6,8}或{6,8,10}是预设的，未探索自适应确定候选集
- **未与最新PTQ/QAT通用方法（如GPTQ、AWQ等思想的LIC适配版）对比**
- **DBWS模块引入的额外延时未明确报告**: 虽然声称"轻量"，但具体overhead没有清晰量化

## 与相关工作的对比
- vs **FMPQ**: DynaQuant的固定精度模式（Q-Cheng）与FMPQ相当（1.60% vs 1.30%），但DynaQuant额外提供了动态比特宽度能力，灵活性更强
- vs **RDO-PTQ**: RDO-PTQ作为PTQ方法不需要重训练但BD-Rate损失更大（4.88% vs 1.60%），DynaQuant的QAT方案显然更优
- vs **RAQ**: RAQ在Kodak上BD-Rate高达27.84%，完全不可用级别
- vs **通用混合精度量化（HAQ/HAWQ等）**: 这些方法用强化学习或Hessian信息搜索比特分配，DynaQuant改为轻量MLP+Gumbel-Softmax端到端学习，避免了昂贵的搜索过程
- vs **Instance-aware quantization（InstAQ等）**: DynaQuant同样是内容自适应的，但增加了层级比特宽度动态分配这一维度

## 启发与关联
- DGM梯度调制器的思想可推广到其他需要round操作可微化的场景（如VQ-VAE的codebook学习、neural codec中的量化模块）
- DBWS的"用Gumbel-Softmax进行可微离散选择"范式在NAS/动态网络中常见，但应用于LIC的比特宽度分配是新颖的组合
- **与模型压缩中的token pruning类似**: 都是"不同部分分配不同计算资源"的思想，只是这里是精度维度而非数量维度
- 若结合图像内容的语义区域信息（如ROI），可能实现更细粒度的空间自适应量化

## 评分
- 新颖性: ⭐⭐⭐☆☆ DGM和DBWS单独看都不算全新概念（分别源于QuantSR和NAS中的Gumbel-Softmax），但将两者面向LIC场景整合为unified framework并展示协同效应有一定贡献
- 实验充分度: ⭐⭐⭐⭐☆ 消融实验设计全面（通用/DPA/DBWS各自消融），跨三个基线验证泛化性，但基线模型偏老，缺少与最新量化方法的对比
- 写作质量: ⭐⭐⭐⭐☆ 结构清晰，方法描述完整，图表质量高（特别是Fig.6的bit-width可视化直观有效），但Conclusion中的future work过于简略
- 价值: ⭐⭐⭐☆☆ 对LIC部署有实际价值（5×加速确实显著），但DQ模式下R-D损失较大限制了实用性；核心技术组件的迁移价值一般
