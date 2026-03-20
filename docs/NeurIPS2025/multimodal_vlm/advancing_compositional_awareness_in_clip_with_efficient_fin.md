# Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning

**会议**: NeurIPS 2025  
**arXiv**: [2505.24424](https://arxiv.org/abs/2505.24424)  
**代码**: [https://clic-compositional-clip.github.io/](https://clic-compositional-clip.github.io/)  
**领域**: 多模态VLM  
**关键词**: CLIP, compositional reasoning, fine-tuning, hard negatives, image concatenation  

## 一句话总结
提出 CLIC（Compositionally-aware Learning in CLIP），通过拼接图像对 + 跨图词汇交换生成 hard negatives + 多正样本训练的策略，在仅微调文本编码器的情况下同时提升 CLIP 的组合推理能力和检索性能，在 SugarCrepe++ 上取得 CLIP 类模型 SOTA。

## 背景与动机
CLIP 等视觉-语言模型在零样本分类和检索上表现出色，但组合推理能力严重不足——它们倾向于学习"bag of words"表示，无法理解概念之间的关系（如"红衣男人旁边灰马" vs "灰衣男人旁边红马"）。现有改进方法（NegCLIP、DAC、TripletCLIP 等）虽然在 SugarCrepe 上表现好，但在更严格的 SugarCrepe++（引入语义等价但词汇不同的正样本 P2）上反而不如预训练模型，说明它们只学会了词汇敏感性而非真正的语义理解。更糟的是，这些方法通常还会损害下游检索性能。

## 核心问题
如何在不依赖 LLM 生成 hard negatives、不需要合成图像的条件下，**以低成本高效微调**提升 CLIP 的组合推理能力（词汇 + 语义两个维度），同时保持甚至提升检索和分类性能？

## 方法详解

### 整体框架
CLIC 冻结视觉编码器，仅微调文本编码器。训练时每次采样一对图像，拼接成一张新图，从两张图的详细描述中构造 4 个正样本和 1 个 hard negative，然后用三个损失函数联合训练。每隔一步用标准单图 CLIP loss 训练以防止过拟合到拼接图像模式。

### 关键设计
1. **图像拼接 + 跨图词汇交换**: 随机配对两张图像并拼接。从各自 caption 的第一句取出拼接得到 p1, 再交换顺序得到 p2（正样本）。用 spaCy 找到两句中相同词类的词并交换，得到 hard negative n。由于两张图毫无关联，交换后的描述几乎不可能仍然正确。这使 negative 的多样性随数据集大小**二次增长**。

2. **多正样本策略**: 除了 p1（拼接首句）和 p2（首句交换顺序，语义不变但词序变），还从各自 caption 的其他句子中随机取出构造 p3、p4。多个正样本迫使模型学习语义不变性——不同词汇形式描述同一图像应有相似嵌入。

3. **三重损失函数**: 
   - $\mathcal{L}_{Cont}$：扩展到 4 个正样本的标准对比损失
   - $\mathcal{L}_{S\text{-}Neg}$：每个正样本与 hard negative 的单独损失项，确保 hard negative 始终影响训练
   - $\mathcal{L}_{Uni}$：p1 和 p2 文本嵌入的 L2 距离，强制模型对语义等价但词序不同的句子产生相似编码
   
   总损失 $\mathcal{L} = \frac{1}{2}\mathcal{L}_{Cont} + \frac{1}{2}\mathcal{L}_{S\text{-}Neg} + \mathcal{L}_{Uni}$

4. **仅微调文本编码器**: 消融实验证明冻结视觉编码器、仅调文本编码器效果最好——组合推理的提升主要来自文本端。

### 损失函数 / 训练策略
- 每隔一步用标准单图 CLIP loss 训练以保持下游性能
- 仅需约 1M 样本训练 1 epoch（ViT-B/32 在 4×A100 上 <30 分钟）
- 使用高质量详细 caption 数据集（PixelProse 或 CogVLM 重标注的 Laion）

## 实验关键数据

| 方法 | SugarCrepe++ Rep ITT | SugarCrepe++ Swap ITT | SugarCrepe Rep ITT | COCO Text R@5 | COCO Img R@5 |
|------|---------------------|----------------------|-------------------|---------------|--------------|
| CLIP ViT-B/32 | 69.5 | 45.7 | 80.0 | 74.1 | 54.6 |
| NegCLIP | 70.5 | 56.4 | 85.4 | 83.6* | 72.2* |
| DAC-LLM | 53.7 | 32.2 | 89.4 | 63.3 | 58.1 |
| TripletCLIP | 73.5 | 43.4 | 88.0 | 73.3 | 53.5 |
| **CLIC-RedCaps** | **76.0** | **61.5** | 84.8 | **76.4** | **59.4** |
| CLIPS + CLIC (ViT-L) | **84.9** | **75.1** | - | +1.3% | +2.2% |

CLIPS + CLIC 在 SugarCrepe++ 平均 ITT 达 81.0%，为 CLIP 类模型 SOTA。

### 消融实验要点
- **图像拼接 vs 单图**: 拼接版在 SugarCrepe++ 和检索上均优于单图版，关键在于拼接使 hard negative 更多样
- **多正样本**: 添加 p3/p4 显著提升 SugarCrepe++ ITT 但略降 TOT，2 个额外正样本后收益饱和
- **Uni-Modal Loss**: 对 SugarCrepe++ swap 改善不大但帮助模型学习语义不变表示
- **CLIP iterate**: 加入单图标准训练步恢复下游分类性能（ImageNet 从 60.9 回升到 61.7）
- **冻结策略**: 冻结视觉编码器效果最好，冻结文本编码器几乎无提升

## 亮点
- 极其简单高效：不需要 LLM 生成 hard negatives，不需要合成图像，训练 <30 分钟
- 图像拼接 + 跨图词汇交换是非常巧妙的数据增强思路，使 hard negative 多样性二次增长
- **唯一**在 SugarCrepe++ 和检索性能上同时优于预训练模型的方法
- 可泛化到不同架构（ViT-B/32、B/16、L/14）和不同预训练方式（CLIP、CLIPA、CLIPS）
- 不微调视觉编码器就能提升组合推理，说明文本编码器是组合理解的瓶颈

## 局限性 / 可改进方向
- 仅测试了 CLIP 类对比学习模型，未探索 SigLIP 等其他架构
- 仅在 LLaVA 上验证了作为视觉编码器的效果，未测试更多大型 VLM
- 拼接图像的 224×224 低分辨率限制了细节保留
- hard negative 中约 72% 是无意义/不合语法的，但文章论证这反而有助于泛化

## 与相关工作的对比
- **vs NegCLIP**: NegCLIP 在单图内换词，容易生成语义未变的 negative（如"左上角"→"上左角"）；CLIC 跨图换词保证语义确实改变
- **vs DAC**: DAC 依赖 BLIP-2/GPT-Neo 生成 dense caption 和 hard negatives，计算成本高且在 SugarCrepe++ 上大幅退化（53.7 vs 69.5 baseline）
- **vs TripletCLIP**: 需要 LLM 生成 hard negative + 扩散模型生成 hard negative 图像，计算昂贵；在 SugarCrepe++ Swap 上仅 43.4 远低于 CLIC 的 61.5

## 启发与关联
- 拼接图像构造组合训练样本的思路可以迁移到视频理解（跨帧拼接）
- "只调文本编码器就能大幅提升组合推理"这一发现对多模态模型设计有重要启示
- 可以和 Balanced Token Pruning（同批笔记）结合：先用 CLIC 提升组合推理，再用 BTP 加速推理

## 评分
- 新颖性: ⭐⭐⭐⭐ 图像拼接 + 跨图换词的思路简单但出人意料地有效
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型、多数据集、多基线、大量消融、误差分析全面
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰、动机分析透彻、行文流畅
- 价值: ⭐⭐⭐⭐ 实用性强，低成本提升组合推理的通用方案
