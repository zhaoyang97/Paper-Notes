# REPA-E: Unlocking VAE for End-to-End Tuning of Latent Diffusion Transformers

**会议**: ICCV 2025  
**arXiv**: [2504.10483](https://arxiv.org/abs/2504.10483)  
**代码**: [https://end2end-diffusion.github.io/](https://end2end-diffusion.github.io/)  
**领域**: 图像生成 / 扩散模型 / 表示学习  
**关键词**: end-to-end training, VAE, latent diffusion, representation alignment, REPA, 训练效率  

## 一句话总结
回答了"潜空间扩散模型能否与VAE端到端联合训练"的基础问题——发现标准扩散loss无法端到端训练但表示对齐（REPA）loss可以，提出REPA-E实现VAE+DiT联合训练，训练速度比REPA快17倍、比vanilla快45倍，在ImageNet 256×256上达到1.12 FID（w/ CFG）的新SOTA。

## 背景与动机
传统深度学习中端到端训练通常是最优选择，但在潜空间扩散模型（LDM）中，VAE和扩散模型一直是分开训练的——先训练好VAE，再在冻结的VAE潜空间上训练扩散模型。之前尝试端到端训练（用diffusion loss反传梯度到VAE）反而导致性能下降。REPA方法通过在DiT内部对齐DINOv2特征加速了扩散训练，但VAE仍然是冻结的。核心问题：能否通过某种loss让VAE和DiT共同进化？

## 核心问题
为什么标准diffusion loss无法有效端到端训练VAE+扩散模型？什么样的loss可以解锁端到端训练？端到端训练对VAE本身有什么影响？

## 方法详解

### 整体框架
REPA-E非常简洁——在REPA的basis上把VAE解冻，让REPA loss的梯度同时流向DiT和VAE encoder/decoder。训练时DiT的中间特征与DINOv2特征做cosine相似度对齐（REPA loss），这个loss因同时作用于潜空间输入（VAE编码的）和DiT内部特征，自然地将梯度传给VAE。

### 关键设计
1. **Diffusion Loss无法端到端的原因**：diffusion loss（去噪重建）在不同时间步的梯度方向对VAE来说是矛盾的——某些时间步想让潜空间更平滑（利于去噪），另一些想保留更多细节（利于重建）。这种梯度冲突导致VAE收到的信号混乱，端到端训练时VAE退化。

2. **REPA Loss解锁端到端训练**：REPA loss让DiT的hidden state与预训练VFM（如DINOv2）的特征对齐。这个loss对VAE提供了一个一致的优化方向——让潜空间编码保留更多语义信息以利于特征对齐。关键洞察：REPA loss通过DiT间接地对VAE施加"让潜空间更结构化"的压力，而非像diffusion loss那样给出矛盾信号。

3. **VAE在端到端训练中的自我改进**：令人惊喜的发现——端到端训练不仅加速DiT学习，还改善了VAE本身。训练后的VAE产生的潜空间更加结构化（特征更可分），即使脱离DiT单独用也有更好的重建质量。这意味着REPA-E实际上是一种"通过扩散模型改善VAE"的方法。

### 损失函数 / 训练策略
- 总损失 = Diffusion loss + λ × REPA loss（cosine similarity alignment with DINOv2）
- VAE和DiT同时可训练，REPA loss的梯度流经DiT反传到VAE
- 基于SiT/REPA架构，ImageNet 256×256训练

## 实验关键数据
| 方法 | 训练步数 | FID↓ (w/o CFG) | FID↓ (w/ CFG) |
|------|----------|---------------|---------------|
| SiT-XL (vanilla) | 7M | ~15 | ~6 |
| REPA | 400K | 1.80 | ~1.4 |
| **REPA-E** | **200K** | **1.69** | **1.12** |

- **FID 1.12** (w/ CFG)：ImageNet 256×256新SOTA
- **FID 1.69** (w/o CFG)：无CFG也是SOTA
- 比REPA快**17倍**（从400K步降到~24K步达到同等FID）
- 比vanilla训练快**45倍**
- 端到端训练后VAE自身改善：潜空间更结构化，下游生成质量更好

### 消融实验要点
- Diffusion loss alone端到端→VAE退化、FID更差
- REPA loss解冻VAE→FID持续改善且收敛更快
- VAE改善与DiT改善是协同的——更好的潜空间让DiT训练更容易
- DINOv2对齐提供了一致的语义梯度方向给VAE

## 亮点
- **回答了一个fundamental question**：端到端训练LDM为什么不work以及如何work——答案优雅且令人意外
- **FID 1.12是ImageNet-256的新SOTA**——且训练效率极高
- **"VAE自我改善"的发现**非常有趣：反直觉地，通过扩散训练改善了tokenizer
- **方法极简**：只需在REPA基础上解冻VAE，几乎零额外工程成本
- **45倍训练加速**对社区的实际价值巨大

## 局限性 / 可改进方向
- 仅在ImageNet 256×256 class-conditional生成上验证
- 未测试text-to-image场景
- 依赖DINOv2作为对齐目标，其他VFM是否同样有效未探索
- VAE改善的程度受限于端到端训练的规模和迭代次数

## 与相关工作的对比
- **vs. REPA**：REPA冻结VAE仅训练DiT；REPA-E解冻VAE实现联合训练，速度更快17倍
- **vs. DC-AE**：DC-AE是先训练更好的VAE再训练DiT（两阶段）；REPA-E通过端到端一步到位
- **vs. MAR/HART**：这些用discrete tokenizer的方法需要token重建+扩散两阶段，REPA-E统一为一个训练过程
- **vs. Scaling Language-Free Visual Repr**：Web-SSL证明SSL encoder可以scale up；REPA-E证明SSL特征（DINOv2）可以通过对齐loss指导VAE+DiT的端到端学习

## 启发与关联
- **重要idea启发**：如果REPA loss能改善VAE，那么同样的思路能否用于改善视频VAE（如CogVideoX的3D VAE）？视频生成中VAE质量是关键瓶颈
- 端到端训练的思路可以扩展到text-to-image——让text encoder也参与端到端优化
- 与SANA-Sprint结合：端到端训练出更好的VAE→再进行步骤蒸馏→可能得到更好的1步生成模型

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 回答了fundamental question，"REPA loss解锁端到端"的发现是paradigm shift
- 实验充分度: ⭐⭐⭐⭐ ImageNet-256 SOTA，但仅限class-conditional场景
- 写作质量: ⭐⭐⭐⭐ 问题定义精准，"为什么不work"的分析有深度
- 价值: ⭐⭐⭐⭐⭐ 1.12 FID + 45x加速，对扩散模型训练范式有深远影响
