# Cross-Scale Pansharpening via ScaleFormer and the PanScale Benchmark

**会议**: CVPR 2026
**arXiv**: [2603.00543](https://arxiv.org/abs/2603.00543)
**代码**: [GitHub](https://github.com/caoke-963/ScaleFormer)
**领域**: Remote Sensing / Image Fusion
**关键词**: 遥感图像融合, 跨尺度泛化, Transformer, 旋转位置编码, Pansharpening

## 一句话总结
提出首个跨尺度全色锐化数据集PanScale和评测基准PanScale-Bench，以及ScaleFormer框架——将分辨率变化重新解释为序列长度变化，通过Scale-Aware Patchify分桶采样+解耦空间-序列建模+RoPE实现跨尺度泛化。

## 研究背景与动机
1. **领域现状**：全色锐化（Pansharpening）利用高分辨率全色（PAN）图像和低分辨率多光谱（LRMS）图像融合生成高分辨率多光谱图像（HRMS），是遥感图像处理的核心任务。CNN/Transformer方法（MSDCNN、HFIN、ARConv等）已取得长足进步。
2. **现有痛点**：(i) **计算与内存瓶颈**——从训练裁剪尺寸（200-256px）推理到800/1600/2000px时，Transformer显存剧增，常规GPU常在800px就OOM；(ii) **分块推理伪影**——被迫分块推理引入边界不连续和明显块状伪影；(iii) **跨尺度泛化弱**——单一低分辨率训练导致尺度诱导的分布偏移，亮度分布随分辨率增大显著偏移。
3. **核心矛盾**：现有数据集（PanCollection、NBU、PAirMax）仅提供有限尺度多样性和分辨率，缺乏标准化多尺度+高分辨率评估协议。
4. **本文要解决什么？** 在数据、算法、计算三个维度系统解决跨尺度全色锐化的挑战。
5. **切入角度**：将分辨率变化重新表述为序列长度变化——固定空间大小的patch作为token，仅序列长度随图像尺度线性增长。
6. **核心idea一句话**：用Scale-Aware Patchify引入序列轴，将空间建模与尺度建模解耦，配合RoPE实现对未见尺度的外推泛化。

## 方法详解
### 整体框架
ScaleFormer包含三个核心组件：
1. **Scale-Aware Patchify (SAP)**：分桶窗口采样策略
2. **Single Transformer模块**：Spatial Transformer（空间域建模）+ Sequence Transformer（序列/尺度域建模）
3. **Cross Transformer模块**：Spatial-Cross + Sequence-Cross Transformer实现跨模态特征融合

输入PAN图像 $\mathbf{P} \in \mathbb{R}^{H \times W \times 1}$ 和上采样后MS图像 $\mathbf{L} \in \mathbb{R}^{H \times W \times C}$，SAP将其转换为5D张量 $\mathbf{P}_{5d} \in \mathbb{R}^{B \times T \times C \times h \times w}$，其中 $T$ 为序列长度。

### 关键设计
1. **Scale-Aware Patchify (SAP)**：训练时随机采样分桶索引 $t$ 确定窗口大小 $w(t)$，用Patch-to-Sequence Tokenizer将输入划分为不同长度的token序列，暴露模型于多种有效序列长度。推理时使用固定窗口大小，高分辨率仅通过延长序列处理。核心效果：防止均值和方差漂移，使每个token的统计量稳定。

2. **解耦空间-序列建模**：Spatial Transformer在每个patch内建模空间关系：
$$\mathbf{f}_{i,1} = \mathbf{f}_i + SA_{spa}(LN(\mathbf{f}_i))$$
Sequence Transformer在序列维度建模跨patch相关性：
$$\mathbf{f}_{i+1,1} = \mathbf{f}_{i+1} + SA_{seq}(LN(\mathbf{f}_{i+1}))$$
其中 $SA_{seq}$ 操作时将batch和空间维度合并，并注入**RoPE**编码连续相对位置信息以增强尺度外推能力。

3. **Cross Transformer模块**：类似结构但使用交叉注意力实现PAN-MS跨模态交互：
$$\mathbf{f}_{i,1}^{ms} = \mathbf{f}_i^{ms} + CA_{spa}(LN(\mathbf{f}_i^{ms}), LN(\mathbf{f}^{pan}))$$

### 损失函数 / 训练策略
使用L1损失 $\mathbf{L} = \|\mathbf{H}_{out} - \mathbf{G}\|_1$。Adam优化器，初始学习率 $5 \times 10^{-4}$，余弦退火衰减到 $5 \times 10^{-8}$，500 epochs，NVIDIA 3090，32通道。

## 实验关键数据
### 主实验：PanScale数据集跨三个子集的平均结果

| 方法 | Jilin PSNR/SSIM | Landsat PSNR/SSIM | Skysat PSNR/SSIM |
|------|-----------------|-------------------|-------------------|
| HFIN | 38.00/0.9698 | 40.21/0.9666 | 43.96/0.9658 |
| ARConv | 38.23/0.9697 | 39.66/0.9638 | 43.40/0.9797 |
| Pan-mamba | 35.55/0.9480 | 36.73/0.9206 | 41.39/0.9493 |
| **ScaleFormer** | **39.29/0.9761** | **41.04/0.9711** | **44.65/0.9827** |

ScaleFormer在所有数据集上全面领先SOTA，且在分辨率增大时性能保持稳定。

### 消融实验：Landsat数据集

| 消融配置 | 200px PSNR | 400px PSNR | 800px PSNR | 1600px PSNR |
|---------|-----------|-----------|-----------|------------|
| w/o RoPE | 40.46 | 40.95 | 40.76 | 40.69 |
| SeqT→SpaT | 40.91 | 41.30 | 40.72 | 40.51 |
| w/o SAP | 40.53 | 40.93 | 40.62 | 40.39 |
| **Full Model** | **40.61** | **41.37** | **41.13** | **41.03** |

各消融变体均在大分辨率上出现明显性能下降，证实每个组件对跨尺度泛化不可或缺。

### 关键发现
- 模型参数量仅0.52M（HFIN的1/4，ARConv的1/9），计算效率显著优势
- 随分辨率增大，ScaleFormer的GFLOPs和显存增长远慢于HFIN/ARConv
- ARConv在分块推理时出现严重块伪影（DDC-IoU显著下降）
- 全分辨实世界场景评估（无GT）中ScaleFormer同样保持竞争力

## 亮点与洞察
- **问题重构巧妙**：将分辨率泛化转化为序列长度泛化，借用NLP/视频模型中序列建模的思想
- **计算效率突出**：在参数量和GFLOPs上大幅领先SOTA，且优势随分辨率增大而扩大
- **数据集贡献**：PanScale是首个覆盖3种卫星平台（0.5~15m分辨率）的跨尺度全色锐化数据集
- **RoPE应用创新**：将文本/视频领域的RoPE引入遥感融合任务实现尺度外推

## 局限性 / 可改进方向
- 仅关注全色锐化任务，对其他遥感融合任务（超光谱融合、SAR-光学融合）的泛化未验证
- SAP的分桶策略是预定义的固定窗口大小集合，自适应策略可能更优
- 仅使用L1损失，感知损失或GAN损失可能进一步提升视觉质量
- Sequence Transformer的自注意力仍为 $O(T^2)$，超大规模输入时仍有瓶颈

## 相关工作与启发
- 传统方法（GS、IHS、GFPCA）在跨尺度场景表现差（PSNR低10+dB）
- CNN方法（MSDCNN、SFINet、MSDDN）跨尺度泛化有限
- HFIN/ARConv是当前SOTA但显存和计算瓶颈严重
- Pan-mamba用Mamba架构但性能不及Transformer方案
- FlexViT的多分辨率训练和视频生成的分桶训练策略是SAP的灵感来源

## PanScale数据集详情
- **三个子数据集**：Jilin（吉林一号，0.5~1m分辨率）、Landsat（Landsat-8，15m分辨率）、Skysat（Planet SkySat，~1m分辨率）
- **测试集设计**：每个子数据集包含reduced-resolution（200×200到2000×2000）和full-resolution多尺度测试集
- **数据来源**：通过Google Earth Engine (GEE)系统获取和预处理
- **评估指标**：PanScale-Bench整合参考指标（PSNR/SSIM/ERGAS/Q）和无参考指标（$D_\lambda$/$D_S$/QNR）

## 效率优势
| 方法 | 参数量(M) | GFLOPs(G) |
|------|-----------|----------|
| ARConv | 4.4147 | 38.32 |
| HFIN | 1.9836 | 46.21 |
| **ScaleFormer** | **0.5151** | **20.57** |

## 评分 ⭐
- 新颖性: ⭐⭐⭐⭐ — 分辨率→序列长度的重构视角新颖，SAP+RoPE组合有效
- 实验充分度: ⭐⭐⭐⭐⭐ — 三数据集+多尺度+全分辨率+消融+效率分析+可视化全覆盖
- 写作质量: ⭐⭐⭐⭐ — 图表设计优秀，Fig 1/2清晰展示问题和方案对比
- 价值: ⭐⭐⭐⭐⭐ — 数据集+基准+方法三位一体贡献，推动遥感融合领域发展
