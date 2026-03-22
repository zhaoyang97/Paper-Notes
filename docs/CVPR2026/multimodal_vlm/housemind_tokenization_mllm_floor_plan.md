# HouseMind: Tokenization Allows MLLMs to Understand, Generate and Edit Architectural Floor Plans

**会议**: CVPR 2026  
**arXiv**: [2603.11640](https://arxiv.org/abs/2603.11640)  
**代码**: [housemind.github.io](https://housemind.github.io/)  
**领域**: 多模态VLM / 建筑设计 / 空间推理  
**关键词**: floor plan, MLLM, VQ-VAE tokenization, room-instance tokens, spatial reasoning, controllable generation  

## 一句话总结
提出HouseMind——通过VQ-VAE将建筑平面图离散化为房间级token，让轻量级LLM（Qwen3-0.6B）在统一框架中同时完成平面图理解、生成和编辑，在所有三项任务上全面超越现有扩散和VLM方法，且可单卡部署。

## 背景与动机
建筑平面图设计需同时推理几何（房间形状尺寸）、语义（功能类别）和空间层次（邻接/连通性），是AI系统的主要挑战。现有方法：(1) 布局合成=纯视觉过程，缺少房间级推理导致全局不连贯；(2) 大模型黑盒生成，空间可控性差；(3) 理解/生成/编辑三任务无法统一；(4) 计算开销大难以本地部署。

## 核心问题
如何让MLLM具备结构化空间推理能力，在一个轻量框架中统一平面图的理解、生成和编辑？

## 方法详解

### 整体框架
HouseMind = (1) Room-Instance Tokenization用层级VQ-VAE将平面图分解为outline tokens+room tokens + (2) 三阶段训练管线让LLM处理空间token和文本的混合序列。

### 关键设计
1. **层级VQ-VAE空间Token化**: 轮廓分支编码建筑外轮廓（8x8 grid, codebook=256），房间分支条件编码每个房间（输入=房间mask+轮廓mask保留邻接关系）。平面图表示为交错序列 Z = [z_o, label_r1, z_r1, ..., label_rN, z_rN]
2. **三阶段训练**: Stage 1 将VQ-VAE codebook嵌入LLM词表；Stage 2 在文本-空间token配对数据上自回归预训练；Stage 3 在理解/生成/编辑三类指令上SFT
3. **统一任务建模**: 理解=从Z推断拓扑；生成=给定文本+轮廓自回归输出Z；编辑=给定原始Z和指令输出修改版Z

### 损失函数 / 训练策略
VQ-VAE标准损失（重建+commitment），轮廓50ep lr=3e-4，房间30ep lr=1e-4。LLM用自回归next-token prediction，cosine schedule + 10% warmup，Qwen3-0.6B + FlashAttention-2，RTX 5090单卡。

## 实验关键数据
**理解**: HouseMind-U RMR=0.998, LocAcc=0.969, AreaDiff=0.549m2, AdjAcc=0.990, RelAcc=0.808（3秒）。对比Qwen3-VL-8B仅0.698/0.347/5.837/0.382/0.128（8秒），MiniCPM-V 4.5仅0.904/0.492/13.765/0.597/0.208（14秒）

**生成**: HouseMind-G Micro IoU=0.709, FID=1.91, GED=1.01, Node F1=0.994, Edge Ovl=0.880（2秒）。ChatHouseDiffusion仅0.589/11.3/2.36/0.985/0.710（30秒）

**编辑**: HouseMind-E Delta IoU=0.608, Node F1=0.998, Edge Ovl=0.934。FLUX.1-Kontext仅0.053/0.765/0.222

### 消融实验要点
- 三阶段缺一不可：w/o Stage1&2 Loss=0.0729，w/o Stage1=0.0659，w/o Stage2=0.0712，Full=0.0644
- Codebook大小256/512/1024几乎无差异，VQ-VAE非信息瓶颈
- Pixel-Structure耦合：HouseMind r=0.57 vs FloorPlanLLaMA r=0.70，room token化实现部分解耦

## 亮点
- Token化让0.6B小模型碾压8B级VLM，设计理念干净优雅
- 首个统一理解+生成+编辑的平面图框架，Omni变体不弱于单任务模型
- 对标GPT-5和Gemini 2.5 Pro，HouseMind在结构准确性上仍更优
- RTX 3090单卡2-3秒/样本，有实际工程价值

## 局限性
- 编辑仅支持简单操作（加/删房间），不支持复杂拓扑变换
- 未建模门窗家具，限制室内设计深度
- 未对齐人类设计偏好/美学约束
- 仅RPLAN数据集（中国住宅），其他建筑类型泛化性未知

## 与相关工作的对比
- **ChatHouseDiffusion**: 扩散+语言条件，简单布局OK但复杂空间失败；HouseMind room-level推理保持全局一致性
- **FloorPlanLLaMA**: VQ-VAE+LLM但编码整图为单一序列，缺房间级控制；HouseMind条件room tokenization保邻接关系
- **MaskPLAN**: VQ-VAE attributes+masked transformer，仅单任务；HouseMind统一三任务支持文本指令

## 启发与关联
- 领域结构化token化是让小模型做大事的通用范式
- 条件编码保留空间上下文的设计有普适价值

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐
