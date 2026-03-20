# TrajTok: 学习轨迹Token实现更好的视频理解

**会议**: CVPR 2026  
**arXiv**: [2602.22779](https://arxiv.org/abs/2602.22779)  
**代码**: 待确认  
**领域**: 视频理解 / 模型压缩  
**关键词**: 视频token化, 轨迹token, 视频效率, token压缩, 视频LLM  

## 一句话总结
提出TrajTok——首个端到端可微的轨迹视频tokenizer，通过隐式时空聚类将视频编码为物体轨迹token，无需外部分割/跟踪管线，在分类、检索和长视频QA上全面超越patch-based方法。

## 背景与动机
视频Transformer的标准做法是将视频切成时空patch来生成token，但这种方式随视频长度线性甚至二次增长token数，冗余严重。先前的TrajViT证明了"轨迹token"（按物体轨迹分组）比patch token更优，但它依赖外部SAM+SAM2分割跟踪管线，既慢又不可微，分割粒度与下游任务脱节。

## 核心问题
如何设计一个**端到端可微、轻量高效**的轨迹tokenizer，让视频token的数量与视频时长解耦、与场景复杂度自适应？核心挑战有三：（1）外部分割模型追求像素级精度，但下游理解任务其实只需要语义级分组——如何在不牺牲理解性能的前提下大幅简化分割？（2）如何让分割粒度由下游任务目标反向驱动（比如舞蹈理解需要身体部位级，队形理解只需人物级）？（3）轨迹token的表达力如何自适应物体的运动复杂度？

## 方法详解

### 整体框架
TrajTok = 通用分割器（Universal Segmenter）+ 轨迹编码器（Trajectory Encoder），两者联合训练。输入视频 → 分割器生成软/硬分割掩码 → 编码器将掩码区域聚合为轨迹token → 送入下游Transformer。

### 关键设计

**1. 通用分割器（Universal Segmenter）**
- 用ConvNeXt逐帧提取多尺度特征（1/4分辨率），先不追求像素精度
- 128个可学习query作为聚类原型，通过Perceiver层对特征做cross-attention
- 对patch特征施加1D RoPE编码时空位置
- 输出softmax软分割图，空query自动丢弃 → token数随场景复杂度动态变化
- 训练损失：Dice + Focal（不用交叉熵），强调"发现所有物体区域"而非像素级精确
- **关键trick**：patch特征梯度detach后再进Perceiver，防止patch特征和query之间不稳定的共适应

**2. 轨迹编码器（Trajectory Encoder）**
- 初始嵌入：用软掩码加权聚合patch特征，保持可微
- 精细化：第二个Perceiver模块，用硬掩码做masked cross-attention，每个轨迹query只关注自己对应区域的特征 → 保证解耦
- **自适应Matryoshka token**：每个轨迹可输出1/2/4个sub-token（用Fourier位置嵌入初始化保证多样性），训练时随机采样token数，推理时按计算预算调整

**3. 三种应用场景**
- **TrajViT2**：从头训练视频编码器（CLIP目标），轨迹token直接作为Transformer输入
- **TrajAdapter**：冻结预训练ViT，TrajTok作为特征适配器重组dense feature → 轻量probing
- **TrajVLM**：LLaVA架构中TrajTok替代patch pooling做视觉-语言连接器

### 损失函数/训练策略
- 分割损失：Dice + Focal loss（伪标签来自TrajViT管线注释的8M视频+15M图像）
- 下游目标：CLIP对比损失（TrajViT2）/ 分类损失（TrajAdapter）/ LM损失（TrajVLM）
- 分割器和下游任务联合训练或分割器预训练后冻结复用

## 实验关键数据

| 模型 | K400 Acc↑ | SSv2 Acc↑ | ActivityNet vid2txt R@5↑ | 推理FLOPs |
|------|-----------|-----------|--------------------------|-----------|
| ViT3D | 54.2 | 46.3 | 35.6 | 二次增长 |
| TrajViT | 55.3 | 45.7 | 38.1 | 高斜率线性 |
| **TrajViT2** | **59.1** (+4.8) | **48.7** (+4.1) | **42.2** (+4.1) | ≈ViViT效率 |

| VLM连接器 | LongVideoBench | LVBench |
|-----------|----------------|---------|
| PatchVLM (pool=3) | 基线 | 基线 |
| **TrajVLM** | **+8.8%** | **+5.4%** |

### 消融实验要点
- 分割器：移除Dice loss危害最大（VEQ -3.0, R@5 -2.9）；梯度detach至关重要（不detach则VEQ -8.2）；低分辨率（1/4原图）即够用
- 轨迹编码器：硬掩码masked attention是核心（去掉则R@5暴跌4.7）；Fourier query初始化比随机初始化重要（随机初始化下多token无增益）
- 端到端训练使分割粒度自适应下游任务：CLIP目标驱动更细的前景分割 + 更粗的背景合并

## 亮点 / 我学到了什么
- "不需要像素完美的分割"这个insight很关键——为理解任务做分割时，语义分组能力 > 边界精度
- Matryoshka思路用在轨迹token上很巧妙：运动复杂的轨迹用4个token，简单的用1个
- 端到端训练让分割粒度被下游任务"反向塑造"，比固定管线灵活得多
- TrajVLM在长视频QA上+8.8%说明轨迹token天然适合长视频场景（因为token数不随帧数增长）

## 局限性 / 可改进方向
- ImageNet上略低于ViT3D——单物体简单场景下分割器产生的token太少
- TrajVLM目前是小规模验证，扩展到更大LLM和更多训练数据是未来方向
- 分割器预训练依赖TrajViT管线生成伪标签，能否完全自监督？
- 场景切换时的轨迹连续性处理（temporal chunking）可能丢失跨段信息

## 与相关工作的对比
- vs TrajViT：端到端可微 + 效率提升一个量级 + 更好的数据扩展性
- vs TokenLearner/ToMe/RLT等token压缩方法：效率相当但精度显著更高
- vs ViViT等高效视频Transformer：FLOPs相当但准确率高得多
- vs patch pooling VLM连接器（Molmo/LLaVA）：长视频优势巨大

## 与我的研究方向的关联
- 轨迹token的"语义分组 + 自适应数量"思路可直接迁移到VLM视觉token压缩
- 与EVATok形成互补：EVATok做自适应长度的视频生成tokenizer，TrajTok做理解端
- 分割器设计（轻量ConvNeXt + Perceiver + 可学习query）是可复用的模块

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 首个端到端可微轨迹tokenizer，范式级贡献
- 实验充分度: ⭐⭐⭐⭐⭐ — 三种场景验证+全面消融+扩展性实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，但篇幅较长
- 对我的价值: ⭐⭐⭐⭐⭐ — 轨迹token思路对视觉token压缩研究有直接启发
