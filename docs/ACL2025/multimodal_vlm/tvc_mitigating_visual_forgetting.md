# Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal Long CoT Reasoning

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2503.13360](https://arxiv.org/abs/2503.13360)  
**代码**: [https://sun-hailong.github.io/projects/TVC](https://sun-hailong.github.io/projects/TVC)  
**领域**: 多模态VLM / LLM推理  
**关键词**: 视觉遗忘, 长链推理, CoT, 视觉重注入, 多模态推理  

## 一句话总结
发现MLLM长链推理中存在严重的"视觉遗忘"现象——推理进行到一半时移除图片仅导致2%精度下降，说明模型过度依赖文本输出而非视觉输入。提出Take-along Visual Conditioning (TVC)，在推理过程中周期性重新注入压缩后的图像特征，在5个数学推理基准上平均超越之前SOTA 3.4个点。

## 背景与动机
OpenAI o1等模型展示了长链CoT推理的威力，但多模态推理面临独特挑战。作者进行了一个关键诊断实验：在QVQ-72B的推理过程中途截断，移除图像输入后让模型继续推理——结果MathVista-Hard上精度仅从43.1%降到40.9%（差距仅2%）！这意味着推理链的后半段几乎不依赖图像，模型只是在"续写"之前生成的文本。更进一步：越早移除图像掉点越多（~20%），说明视觉依赖随推理进程呈指数衰减：ΔVisual(k) ∝ e^{-k}。

## 核心问题
如何在长链多模态推理中维持模型对视觉信息的持续关注，避免推理链后半段完全依赖文本而遗忘视觉证据？

## 方法详解

### 整体框架
两阶段方案：(1) 训练阶段——Dynamic Visual Reaffirmation (DVR)在训练数据中人工重新注入图像；(2) 推理阶段——Periodic Visual Calibration (PVC)在推理过程中周期性重新激活视觉信息。两阶段都用adaptive pooling压缩重注入的图像token（4×4池化）。

### 关键设计

1. **视觉遗忘现象的量化验证**: 
   - 在推理链的8个等间隔位置移除图像，测量精度变化
   - 发现：1/8位置移除掉20%，4/8位置移除仅掉2%——近指数衰减
   - 注意力分析：浅层对图像token的attention权重从0.539降到0.005
   - 这不是个别现象——30.9%的样本在完全没有图像时也能答对（靠文本prompt中的隐式信息）

2. **Dynamic Visual Reaffirmation (DVR) 训练**: 
   - 在高质量长链推理数据（从QVQ生成）中，在自省点（self-reflection intervals）手动重新插入图像嵌入+桥接提示（如"Let me see the image again"）
   - 使模型学会"回看图片"的行为模式
   
3. **Periodic Visual Calibration (PVC) 推理**: 
   - 推理时在每个自省间隔周期性重新引入视觉输入
   - 图像经过adaptive pooling压缩到4×4=16个token（原始可能数百个）
   - 类比人类解题时反复看题目图片的行为

### 损失函数 / 训练策略
- 基于Qwen2-VL-7B和72B做SFT
- 训练数据从MathV360K, Geo170K, LLaVA-OneVision策展
- adaptive pooling将重注入图像从原始token数压缩到16个

## 实验关键数据

| 模型 | Size | MathVista | MathVision | MathVerse | DynaMath | OlympiadBench | Avg |
|------|------|-----------|-------------|-----------|----------|---------------|-----|
| Qwen2-VL | 7B | 60.9 | 16.3 | 24.6 | 11.0 | 3.2 | 23.2 |
| QVQ-72B | 72B | 71.4 | 35.9 | 41.5 | 30.7 | 20.4 | 40.0 |
| **TVC** | **7B** | **68.1** | **22.7** | **38.9** | 15.1 | 9.8 | 30.9 |
| **TVC** | **72B** | **72.2** | **41.9** | **48.8** | 30.0 | **24.3** | **43.4** |

- TVC-72B平均超越QVQ-72B 3.4个点
- TVC-7B在MathVerse(38.9%)甚至超越许多27B-72B模型
- MathVision: +6.0(72B), +6.4(7B) vs Qwen2-VL基线

### 消融实验要点
- **DVR + PVC**: 两者都有贡献，联合使用最优（33.9→43.2, +9.3）
- **PVC单独**: 66.7 vs Full TVC 68.1，说明推理时视觉重注入有价值
- **DVR单独**: 66.2 vs Full TVC 68.1，说明训练阶段的视觉重注入习惯也重要
- **压缩率**: 4×4池化是最优平衡——2×2太粗丢信息，8×8太多增开销
- **重注入频率**: 每阶段注入一次最优，过于频繁反而分散推理注意力

## 亮点
- **关键发现推动设计**: "视觉遗忘"现象的量化验证（2%精度差距）是非常有说服力的动机
- **类比人类行为**: 人类解题时反复看图——TVC模拟了这一认知过程
- **7B模型打败72B**: TVC-7B在MathVerse上超越Qwen2-VL-72B和LLaVA-OneVision-72B
- **简洁有效**: 只需要在关键位置重新注入压缩图像，不改变模型架构

## 局限性 / 可改进方向
- 重注入位置是基于固定间隔的，未做自适应（根据推理难度决定何时回看图片）
- 仅在数学推理任务验证，通用VQA、自然图像问答可能效果不同
- adaptive pooling可能丢失关键细节（如小字、公式中的关键符号）
- 训练数据策展过程依赖QVQ生成，受限于QVQ的质量
- 未与VReST（树搜索）结合——如果在MCTS每步都重注入视觉信息可能更强

## 与相关工作的对比
- **vs VReST**: VReST用MCTS探索推理空间，TVC解决推理过程中的视觉遗忘——两者互补
- **vs Improve VLM CoT (SFT+DPO)**: 那篇用DPO校准推理质量，TVC解决推理过程中视觉信息流失——不同维度
- **vs FastV**: FastV做视觉token剪枝加速，TVC反其道——在需要时重新补充视觉token

## 启发与关联
- "视觉遗忘"现象可能在所有长序列多模态任务中存在——不只是数学推理，长视频理解也可能有类似问题
- TVC + VReST组合：在MCTS的每个扩展节点重注入压缩图像，可能显著提升
- 视觉token的自适应重注入可以用VHD（Vision-aware Head Divergence）来判断"何时需要回看图片"

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "视觉遗忘"现象的发现和量化是该领域的重要贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 5个benchmark、诊断实验+注意力分析+消融极其详尽
- 写作质量: ⭐⭐⭐⭐⭐ Figure 1的诊断实验图和Figure 2的注意力可视化直接说明问题
- 价值: ⭐⭐⭐⭐⭐ 对多模态长链推理的根本性insight，具有广泛适用性
