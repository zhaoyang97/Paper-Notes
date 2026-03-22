# Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport

**会议**: ICLR 2026 Oral  
**OpenReview**: [mUIGdUTtk2](https://openreview.net/forum?id=mUIGdUTtk2)  
**代码**: 有  
**领域**: 信息论 / 模型压缩  
**关键词**: optimal transport, rate-distortion, lossy compression, cross-domain, DRC tradeoff  

## 一句话总结
将跨域有损压缩（编码器看退化源、解码器重建不同目标分布）形式化为带压缩率和分类损失双重约束的最优传输问题，推导出 Bernoulli/Gaussian 源的闭式 DRC（失真-率-分类）和 DRPC（失真-率-感知-分类）权衡曲线，在 KODAK 去噪上实现 PSNR 27.90 / SSIM 0.80 的竞争性能，审稿人给出 10/10 评分。

## 研究背景与动机
1. **领域现状**：经典率失真理论假设编码器和解码器在同一域，但实际场景（去噪、超分辨、修复）中编码器处理退化输入、解码器重建干净目标。
2. **现有痛点**：跨域设置下的率失真理论缺乏系统的理论框架和闭式解。
3. **核心idea一句话**：将最优传输与率失真理论和分类约束统一，首次推导出跨域设置下的 DRC/DRPC 闭式权衡曲线。

## 方法详解

### 关键设计
1. **约束最优传输**：同时约束压缩率（信息瓶颈）和分类损失（语义保持）
2. **闭式解**：Bernoulli (Hamming) 和 Gaussian (MSE) 两类源的解析解
3. **One-shot 和渐近 regime**：确定性传输 + 共享随机性下的单次和渐近公式
4. **DRPC 扩展**：加入感知散度（KL、Wasserstein）约束

## 实验关键数据

| 数据集 | 任务 | PSNR | SSIM | LPIPS | PI |
|--------|------|------|------|-------|-----|
| KODAK ($\sigma$=25) | 去噪 | 27.90 | 0.80 | 0.20 | 2.17 |
| Mouse Nuclei ($\sigma$=10) | 去噪 | 33.03 | 0.81 | — | — |
| SIDD | 真实去噪 | 33.61 | 0.90 | — | — |

- 在超分辨（MNIST）、去噪（SVHN/CIFAR-10/ImageNet/KODAK）、修复（SVHN）上验证

## 亮点与洞察
- **信息论 + 最优传输 + 分类的优美统一**：三个不同领域的理论在一个框架下融合
- **闭式解的价值**：不仅指导算法设计，还提供基本性能极限
- **审稿人 10/10**：极高评价反映了理论贡献的深度

## 局限性 / 可改进方向
- 闭式解限于 Bernoulli/Gaussian，更一般分布需要数值方法
- 实际深度压缩模型与理论极限之间仍有差距

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 跨域率失真理论的首个系统化闭式框架
- 实验充分度: ⭐⭐⭐⭐ 理论+多任务验证
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨
- 价值: ⭐⭐⭐⭐⭐ 信息论领域的重要理论贡献
