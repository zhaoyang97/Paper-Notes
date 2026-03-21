# ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding

**会议**: CVPR 2026  
**arXiv**: [2602.16412](https://arxiv.org/abs/2602.16412)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 长视频理解, 视频压缩表示, 运动向量, 状态空间模型, 光流精化

## 一句话总结
提出 ReMoRa，直接操作视频压缩表示（I帧 + 运动向量），通过 Refined Motion Representation (RMR) 模块将粗糙的块级运动向量精化为接近光流的细粒度运动表征，再用 Hierarchical Motion State Space (HMSS) 模块进行线性时间的长程时间建模，在 LongVideoBench、NExT-QA、MLVU 等基准上超越基线。

## 研究背景与动机
1. **领域现状**：视频 MLLM 在短视频上取得了显著进展，但长视频理解（分钟到小时级别）仍是重大挑战。
2. **现有痛点**：
   - 均匀帧采样面临固有 trade-off：稀疏采样遗漏关键事件，密集采样因二次注意力复杂度计算不可行
   - 基于帧的方法反复编码冗余内容（如静态背景），效率极低
   - 密集采样后的 token 压缩（pooling/reduction）会模糊细粒度细节和运动线索
3. **核心矛盾**：长视频需要密集的时间覆盖以捕捉短暂但重要的事件，但密集帧处理的计算代价过高。
4. **本文要解决什么**：利用视频压缩格式（H.264）中天然的外观-运动分解，以极低成本实现密集时间覆盖。
5. **切入角度**：现代视频编码已经做了关键帧选择和运动补偿，运动向量是光流的廉价近似，可以直接利用而非解码全部帧。
6. **核心idea**：保留少量 I 帧用于外观，用运动向量替代中间帧用于时间动态，通过 RMR 模块弥补运动向量的噪声和粗糙性。

## 方法详解

### 整体框架
视频 → H.264 压缩编码 → 每个 GOP 提取 (I帧, 运动向量序列) → Image Encoder 编码 I 帧 + RMR 精化运动 → HMSS 融合外观与运动并跨 GOP 时间建模 → LLM 生成回答。

### 关键设计

1. **Refined Motion Representation (RMR) 模块**:
   - 输入：P/B 帧的块级运动向量（codec 原生，无需解码）
   - 问题：这些运动向量是块级的、稀疏的、有噪声的、时间不一致的
   - 方法：预训练一个模块将粗糙运动向量映射为密集光流场，用 Co-Tracker3 生成的密集光流作为监督目标，$L_2$ 损失
   - 微调时作为特征编码器，输出运动嵌入 $E_M^{(k,t)} \in \mathbb{R}^{N_m \times d_s}$
   - 设计动机：运动向量的计算成本极低但质量差，精化后能接近光流的信息量同时保持效率

2. **Hierarchical Motion State Space (HMSS) 模块**:
   - 两层结构，镜像 GOP 的层次化组织：
   - **Codec-aware Selective Scan（GOP内）**：双向 Mamba 块融合 I 帧外观嵌入与运动嵌入，然后提取前 $N_p$ 个 token 作为运动增强的 I 帧表示
   - **Bidirectional Token Mixer（GOP间）**：将所有 GOP 的摘要向量接后，用双向 Mamba 进行全局长程建模
   - 序列长度比 naive 展平缩短 $L_g/N_p$ 倍，实现线性时间复杂度的长程依赖建模
   - 设计动机：避免 >100K token 的二次注意力，同时保持局部（GOP内）和全局（GOP间）的时间上下文

3. **场景自适应 GOP 构建**:
   - 用 ffmpeg 的场景自适应检测重新编码视频，在视觉不连续处动态插入 I 帧
   - 比固定间隔采样更符合内容结构，相当于隐式关键帧提取

### 损失函数 / 训练策略
RMR 模块先预训练（$L_2$ 光流重建），整体模型用标准 cross-entropy 指令微调。LLM backbone 用 LoRA，视觉编码器冻结。

## 实验关键数据

### 主实验

| 方法 | LLM | LongVideoBench | NExT-QA | MLVU | VideoMME | Avg |
|------|-----|:---------:|:-------:|:----:|:--------:|:---:|
| LLaVA-OneVision | Qwen2-7B | 56.5 | 79.4 | 64.7 | 58.2 | 63.2 |
| BIMBA | Qwen2-7B | 59.5 | 83.2 | 70.6 | 63.1 | 68.9 |
| LLaVA-Video | Qwen2-7B | 58.2 | 83.2 | 70.8 | 63.3 | 68.7 |
| **ReMoRa** | **Qwen2-7B** | **60.8** | **84.2** | **72.1** | **64.4** | **69.8** |

### 开放式 VideoQA

| 方法 | ActivityNet-QA Acc | ActivityNet-QA Score |
|------|:---------:|:---------:|
| EMA | 52.1 | 3.5 |
| **ReMoRa** | **60.5** | **3.7** |

### 关键发现
- ReMoRa 在 LongVideoBench (+1.3)、NExT-QA (+1.0)、MLVU (+1.3) 上均取得最佳成绩
- ActivityNet-QA 上准确率超过次优 8.4 个百分点，体现运动表示对时间推理的关键作用
- 与使用相同 codec 信息的 EMA 相比，RMR 精化运动表示带来显著提升
- 定性分析显示 ReMoRa 在需要细粒度动作理解的问题上明显优于 LLaVA-Video

## 亮点与洞察
- **利用视频压缩格式的天然结构**：不解码全部帧而直接操作编码域，这打破了"必须均匀采样RGB帧"的范式。运动向量虽然质量差，但数量可以非常密集，配合精化模块后成为高质量的时间线索
- **RMR 模块的巧妙设计**：预训练时学习"粗运动→密集光流"的映射，微调时作为特征编码器，既享受了光流的信息量又避免了光流的计算开销
- **HMSS 的层次化设计**：GOP 内融合（类似 segment-level attention）→ GOP 间建模（类似 video-level attention），完美匹配编码结构，且线性复杂度

## 局限性 / 可改进方向
- 依赖 H.264 编码器，对不同编码格式（HEVC、AV1）的适应性未验证
- GOP 最大长度固定为 32 帧，极长静态场景可能导致运动信息稀疏
- RMR 预训练需要 Co-Tracker3 生成的光流监督，增加了数据准备成本
- VideoMME 上不是最优（64.4 vs 65.1），说明在部分场景下运动信息可能不是关键

## 相关工作与启发
- **vs Video-LaVIT**：也使用 codec 信息但只做简单的 appearance-motion tokenization，缺少运动精化和层次化建模
- **vs EMA**：EMA 引入了 GOP 编码器但不精化运动向量，ReMoRa 的 RMR 模块填补了这个质量差距
- **vs LongVU（token pruning）**：LongVU 仍处理 RGB 帧只是减少 token 数量，ReMoRa 从源头上避免了冗余帧的处理

## 补充分析
- ReMoRa 使用 200K 的指令微调数据，来自 LLaVA-Video-178K 数据集，涵盖开放式QA、选择题和caption三种任务
- 场景自适应 GOP 构建：ffmpeg 的 scene-adaptive detection 自动在场景切换处插入 I 帧，比固定间隔GOP更合理
- 每个 GOP 最大长度 32 帧，宏块大小 4×4，保证细粒度运动向量分辨率
- 视觉编码器采用 SigLIP ViT-SO 且冻结，LLM backbone 用 LoRA 微调，参数效率高
- 在 ActivityNet-QA 上准确率领先次优（EMA）8.4 个点，主要因为 ActivityNet 中长时间跨度的动作理解对细粒度运动信息依赖更强

## 评分
- 新颖性: ⭐⭐⭐⭐ 利用压缩域信息的视频MLLM，方向有前景
- 实验充分度: ⭐⭐⭐⭐ 6个benchmark全面评估
- 写作质量: ⭐⭐⭐⭐ 背景铺垫充分，方法描述清楚
- 价值: ⭐⭐⭐⭐ 为长视频MLLM提供了高效的新范式
