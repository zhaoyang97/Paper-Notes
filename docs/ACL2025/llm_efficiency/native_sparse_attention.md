# Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

**会议**: ACL 2025 (Best Paper Award)  
**arXiv**: [2502.11089](https://arxiv.org/abs/2502.11089)  
**代码**: 无（DeepSeek内部）  
**领域**: 模型压缩 / 高效推理 / 注意力机制  
**关键词**: 稀疏注意力, 长上下文建模, 硬件对齐, 端到端训练, Token压缩与选择  

## 一句话总结
DeepSeek提出NSA——一种原生可训练的稀疏注意力机制，通过"压缩+选择+滑动窗口"的层次化稀疏策略和硬件对齐的Triton kernel设计，在27B参数模型上实现了超越Full Attention的性能，同时在64k序列上获得前向9倍、解码11.6倍的加速。

## 背景与动机
长上下文建模是下一代LLM的核心能力需求（深度推理如DeepSeek-R1需要超长思维链，代码生成需要仓库级上下文）。标准注意力在64k序列解码时占总延迟70-80%，效率瓶颈显著。现有稀疏注意力方法存在两大困境：
1. **"高效推理幻觉"**：很多方法虽然减少了理论计算量，但因与GQA/MQA等架构不兼容、非连续内存访问、仅加速推理的某个阶段（prefilling或decoding），实际加速远低于理论值
2. **"可训练稀疏性迷思"**：大多数方法仅在推理阶段应用稀疏性，导致模型被迫偏离预训练优化轨迹；少数尝试训练的方法因包含不可微操作（如k-means聚类、哈希选择）或低效反向传播而不实用

## 核心问题
如何设计一种稀疏注意力机制，既能在训练和推理的所有阶段都实现真实加速，又能通过端到端预训练让模型原生学习最优的稀疏模式？

## 方法详解

### 整体框架
NSA对每个query通过三条并行注意力路径处理所有前序的key-value：
1. **压缩注意力**(Compressed)：粗粒度token压缩，捕获全局上下文
2. **选择注意力**(Selected)：细粒度关键token块选择，保留重要细节
3. **滑动窗口**(Sliding Window)：局部上下文处理

三路输出通过learn的gate（MLP+sigmoid）加权融合。核心约束：激活的总token数N_t ≪ t（序列长度）。

### 关键设计

1. **Token压缩**
   - 将连续的key序列分成长度l=32的块，用可学习MLP（含块内位置编码）压缩每块为单个compressed key/value
   - 使用滑动步长d=16（d<l），块间有重叠以减少信息碎片化
   - 压缩token的注意力分数还被复用为选择模块的block importance scores——零开销的重要性估计

2. **Blockwise Token选择**
   - 以l'=64为块大小将key/value分块，利用压缩注意力分数推导每块的重要性得分
   - Top-n选择（n=16，含1个初始块+2个局部块）最重要的块参与精细注意力计算
   - GQA场景下跨同组头聚合重要性得分，确保一致的块选择以最小化KV cache加载

3. **滑动窗口分离设计**
   - 独立的w=512窗口处理局部上下文
   - 关键洞察：如果不分离，局部模式会"捷径学习"，压缩和选择分支无法学到有意义的长程模式
   - 三路使用独立的key/value投影，进一步防止梯度干扰

4. **硬件对齐Kernel设计**
   - 选择注意力的Forward kernel：以GQA组为外循环（而非时间连续的query块），因为同组query共享稀疏KV块
   - 块式连续内存访问，最大化Tensor Core利用率
   - 反向传播也有专用kernel，支持完整的端到端训练
   - 基于Triton实现，与Triton FlashAttention-2公平比较

## 实验关键数据

| 评估维度 | NSA vs Full Attention |
|---------|----------------------|
| 通用基准平均 | 0.456 vs 0.443（NSA在9项中7项更优） |
| LongBench平均 | 0.469 vs 0.437（+0.032，**稀疏模型超过Full Attention**） |
| AIME 8k | 0.121 vs 0.046（+0.075，链式推理显著优势） |
| AIME 16k | 0.146 vs 0.092（+0.054） |
| 64k Forward加速 | 9.0× |
| 64k Backward加速 | 6.0× |
| 64k Decoding加速 | 11.6×（理论值） |
| Needle-in-Haystack | 100%准确率（64k全位置） |

### 消融实验要点
- **压缩+选择的协同**：压缩token的注意力分数直接驱动块选择，无需额外计算开销
- **块级vs token级选择**：块级选择既符合注意力分数的空间连续性特征，又适配GPU连续内存访问
- **替代方案对比**：基于聚类(ClusterKV)和基于启发式(Quest)的选择策略在训练loss上均不如NSA
- **辅助loss方案**（类似SeerAttention）引入不可微操作和额外开销，也逊于NSA
- **预训练loss曲线**：NSA全程低于Full Attention，说明稀疏架构反而帮助模型聚焦关键信息

## 亮点
- **"稀疏反而更好"的反直觉发现**：在通用基准和长上下文任务上NSA均超越Full Attention，挑战了"稀疏必然牺牲性能"的认知。可能的解释是稀疏注意力强迫模型过滤噪声，专注于真正重要的信息
- **压缩注意力分数的免费复用**：压缩分支产生的attention score被零开销地转化为选择分支的block importance score，这个设计既优雅又高效
- **GQA-aware的共享块选择**：意识到GQA架构下不同头独立选择KV块会导致union后的内存访问量反而不减少，这个洞察解释了很多现有方法在GQA模型上"假加速"的现象
- **AIME数学推理的巨大优势**（NSA-R在8k上比Full Attention-R高0.075绝对值）：说明稀疏注意力预训练让模型学到了更适合长推理链的注意力模式

## 局限性 / 可改进方向
- 仅在DeepSeek内部的27B模型上验证，未在更多公开模型上验证可迁移性
- Triton kernel的实现可能不如CUDA kernel最优化
- 超参数（块大小l=32/l'=64、选择数n=16等）的选择依据未充分讨论
- 未与Linear Attention族方法（如Mamba/RWKV）对比

## 与相关工作的对比
- **vs FlashAttention-2**：FlashAttention是精确全注意力的高效实现；NSA是稀疏注意力的原生训练，目标不同但在效率上远超FA2
- **vs Quest/InfLLM等推理期稀疏方法**：在LongBench上NSA (0.469) > Full Attention (0.437) > Quest (0.392) > InfLLM (0.383)，差距显著
- **vs SeerAttention**：概念上类似（均做blockwise选择），但SeerAttention依赖辅助loss和不可微操作，而NSA通过压缩注意力分数的复用实现了完全可微的端到端训练

## 启发与关联
- NSA的成功暗示：在scale up过程中，不必坚持Full Attention——原生稀疏架构可能是更好的选择
- "压缩+选择"的双层策略可迁移到视觉Transformer等其他领域
- 与弹性对齐论文的关联：如果稀疏注意力改变了模型处理信息的方式，对齐的弹性是否也会受影响？

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个在预训练中完整集成的硬件对齐稀疏注意力架构
- 实验充分度: ⭐⭐⭐⭐⭐ 27B参数模型+260B tokens预训练，通用、长上下文、推理三维评估
- 写作质量: ⭐⭐⭐⭐ 技术细节清晰，但Section 2的问题分析特别精彩
- 价值: ⭐⭐⭐⭐⭐ 来自DeepSeek的实战经验，对工业级长上下文LLM设计有直接指导意义
