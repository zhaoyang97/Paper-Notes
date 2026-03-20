# DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling

**会议**: NeurIPS 2025 (Spotlight)  
**arXiv**: [2505.11196](https://arxiv.org/abs/2505.11196)  
**代码**: [https://github.com/shallowdream204/DiCo](https://github.com/shallowdream204/DiCo)  
**领域**: 图像生成 / 扩散模型 / 高效架构  
**关键词**: ConvNet, diffusion model, 卷积替代注意力, 通道冗余, compact channel attention, 高效生成  

## 一句话总结
重新发掘卷积网络在扩散模型中的潜力——发现预训练DiT的全局自注意力主要捕获局部模式（冗余），提出用标准ConvNet模块+紧凑通道注意力构建纯卷积扩散模型DiCo，在ImageNet-256上以2.05 FID超越DiT-XL/2且速度快2.7倍。

## 背景与动机
Diffusion Transformer (DiT)展现了强大的生成能力但计算开销大。一个反直觉的发现是：分析预训练DiT的注意力模式，发现全局self-attention在大多数层实际上主要捕获的是局部pattern而非全局依赖——这意味着注意力机制在此场景中有很大冗余，可以用更高效的局部操作（如卷积）替代。但简单替换会导致性能下降，需要找到原因并解决。

## 核心问题
能否用纯卷积网络替代DiT中的Transformer实现高效且高质量的扩散建模？为什么简单替换会掉点？如何修复？

## 方法详解

### 整体框架
DiCo是一族完全由标准ConvNet模块构建的扩散模型，不使用任何self-attention操作。每个DiCo block用深度可分离卷积替代self-attention，用紧凑通道注意力解决卷积网络的通道冗余问题。

### 关键设计
1. **DiT注意力冗余的发现**：通过分析预训练DiT各层的注意力图，发现大部分层的注意力集中在局部邻域——全局self-attention的有效感受野实际上很小（类似卷积）。这为用卷积替代提供了理论基础。

2. **通道冗余问题的定位**：简单将attention替换为convolution导致性能下降的根本原因是ConvNet的通道冗余高于Transformer——ConvNet倾向于让很多通道学到相似的特征（通道间相关性高），而Transformer的attention天然促进通道多样性。

3. **紧凑通道注意力（Compact Channel Attention）**：为解决通道冗余，引入一个轻量级的通道注意力模块，动态调整每个通道的激活权重，促进更多样化的通道特征学习。这使得ConvNet能学到与Transformer同样丰富的特征表示。

### 损失函数 / 训练策略
标准扩散训练loss（epsilon/velocity prediction），与DiT相同的训练配置。

## 实验关键数据
| 模型 | 分辨率 | FID↓ | 速度提升 |
|------|--------|------|----------|
| DiT-XL/2 | 256 | 2.27 | 1x |
| **DiCo-XL** | **256** | **2.05** | **2.7x** |
| DiT-XL/2 | 512 | - | 1x |
| **DiCo-XL** | **512** | **2.53** | **3.1x** |

- ImageNet-256上FID 2.05，超越DiT-XL/2 (2.27)
- 256分辨率2.7倍加速，512分辨率3.1倍加速
- 分辨率越高加速比越大（卷积复杂度O(n)对比attention的O(n²)）
- 在MS-COCO上也展示了T2I生成潜力

### 消融实验要点
- 简单Conv替代attention：FID下降
- +紧凑通道注意力：FID恢复并超越DiT
- 通道冗余分析：ConvNet的通道相关性显著高于Transformer
- 通道注意力有效降低通道冗余，提升特征多样性

## 亮点
- **NeurIPS Spotlight**：挑战了"Transformer是扩散模型最优架构"的共识
- **ConvNet的文艺复兴**：在Transformer主导的时代证明了卷积在扩散模型中的独特优势——更高效且可以更好
- **通道冗余是关键洞察**：精准定位了Conv替代Attention掉点的原因，并用轻量通道注意力解决
- **高分辨率优势更大**：511分辨率3.1x加速——对高分辨率T2I应用特别有价值
- **代码开源**，方便社区复现和扩展

## 局限性 / 可改进方向
- 卷积的局部性可能在需要全局建模的场景（如复杂空间关系、长距离一致性）受限
- 仅在class-conditional ImageNet和MS-COCO上验证，大规模T2I（如Laion数据）未测试
- 与MM-DiT（如FLUX/SD3的cross-modal attention）的兼容性未探索

## 与相关工作的对比
- **vs. DiT**：DiT用Transformer做扩散backbone；DiCo用ConvNet——同等质量下快2.7x
- **vs. SANA**：SANA用Linear Attention替代标准Attention加速；DiCo完全去掉Attention用Conv——从不同角度追求效率
- **vs. Dense2MoE**：Dense2MoE通过稀疏化减少每步计算；DiCo通过Conv替代Attention减少每步计算——正交策略可叠加

## 启发与关联
- DiCo + SANA-Sprint（步骤蒸馏）= 每步更快×步数更少 = 可能实现超实时T2I
- 通道冗余分析方法可以迁移到视频扩散模型——视频DiT的冗余可能更严重
- "ConvNet在扩散模型中的复活"可能引发新一轮架构创新

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 在Transformer主导的时代重新证明ConvNet的价值，通道冗余洞察精准
- 实验充分度: ⭐⭐⭐⭐ ImageNet-256/512+MS-COCO，效率和质量全面对比
- 写作质量: ⭐⭐⭐⭐ 从发现问题（注意力冗余）→定位原因（通道冗余）→解决（通道注意力）逻辑清晰
- 价值: ⭐⭐⭐⭐⭐ Spotlight当选，为高效扩散模型架构开辟了ConvNet新路径
