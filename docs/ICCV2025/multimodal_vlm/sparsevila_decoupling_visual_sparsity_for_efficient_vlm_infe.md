# SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference

**会议**: ICCV 2025  
**arXiv**: [2510.17777](https://arxiv.org/abs/2510.17777)  
**代码**: 无  
**领域**: 多模态VLM / 高效推理 / Token压缩  
**关键词**: visual token pruning, decoupled sparsity, prefill-decode, query-aware retrieval, multi-turn conversation  

## 一句话总结
提出SparseVILA，将VLM推理时的视觉token稀疏化解耦为两个阶段——prefill阶段做query-agnostic剪枝（去冗余）、decode阶段做query-aware检索（精选相关token），在长视频任务上实现4.0×prefill加速、2.5×decode加速、2.6×端到端加速，同时在视频理解benchmark上精度不降反升。

## 研究背景与动机
1. **领域现状**：VLM的视觉token数量庞大（尤其是长视频/高分辨率场景），主导了推理延迟。现有加速方法分为query-agnostic（基于视觉显著性剪枝，如VisionZip、PruMerge）和query-aware（基于文本-视觉关系剪枝，如FastV、SparseVLM）两类。
2. **现有痛点**：(1) Query-agnostic方法在高稀疏率下会丢失细粒度信息；(2) Query-aware方法在多轮对话中严重退化——第一轮问题剪掉的token在后续轮次中可能非常重要，且无法恢复。实验显示即使是query-aware的oracle（理论上限），多轮对话精度也会急剧下降。(3) 现有方法只关注prefill加速，忽视了decode阶段在长生成场景中的延迟瓶颈。
3. **核心矛盾**：Query-agnostic保稳定但不精准，query-aware精准但不可逆。且实际推理中decode阶段往往比prefill更耗时（尤其是长生成任务），但现有方法几乎不优化decode。
4. **本文要解决什么**：设计一个同时加速prefill和decode、且支持多轮对话的统一稀疏化框架。
5. **切入角度**：观察到prefill和decode有不同的计算特性——prefill算一次且compute-bound，decode反复迭代且memory-bound。因此应该在prefill做轻量剪枝（保留大部分visual KV cache），在decode做激进稀疏（每轮根据query选择最相关的子集）。
6. **核心idea一句话**：解耦prefill（query-agnostic轻度剪枝保覆盖）和decode（query-aware激进检索提效率）两个阶段的视觉稀疏化策略，兼顾多轮保真和推理加速。

## 方法详解

### 整体框架
Prefill阶段：用视觉编码器的self-attention map估计token显著性，剪掉低显著性的冗余token（如60-75%），保留的token构建完整KV cache。Decode阶段：对每个新问题，用query embedding与visual KV cache的attention强度估计相关性，只激活最相关的子集（如保留10-25%）参与decode attention，其余token保留在cache中供后续轮次使用。

### 关键设计

1. **Prefill阶段：Query-Agnostic Token显著性剪枝**:
   - 做什么：在视觉编码器输出后、送入LLM之前，基于self-attention map移除冗余视觉token。
   - 核心思路：对有CLS token的编码器（如CLIP），用每个token对CLS的attention贡献作为显著性；对多summary token的编码器（如RADIO），取对所有summary token的平均attention；对无summary token的编码器（如SigLIP、QwenVL），取所有token间的平均intra-attention。
   - 高效实现：自定义Triton kernel流式计算softmax+显著性累积，无需显式形成完整attention矩阵。对SigLIP加速3×，QwenVL加速10×。
   - 设计动机：只做轻度剪枝（60-75%），保留足够多的token以确保后续多轮对话的信息完整性。

2. **Decode阶段：Query-Aware Token检索**:
   - 做什么：在每轮问题到来时，从保留的visual KV cache中只激活与当前query最相关的token子集用于decode attention。
   - 核心思路：用query embedding和cached visual token之间的attention强度作为相关性分数，取top-k最相关token打包到连续内存区域用于高效decode。关键是不删除未激活的token——它们仍在cache中，下一轮问题可以重新激活。
   - 高效实现：检索计算与FlashAttention2路径并行执行，Triton kernel加速1.5×。选中token紧密打包避免稀疏访问。
   - 设计动机：decode阶段是memory-bound的，减少KV cache大小直接降低内存带宽需求。且query-aware选择在此阶段更合理，因为问题已知。

3. **解耦稀疏框架**:
   - 做什么：将稀疏化分配到decode（主瓶颈）而非全部堆在prefill。
   - 核心思路：同等端到端加速下，"70% prefill + 85% decode"比"90% prefill + 0% decode"效果更好。实验显示前者在RoboVQA上89.1%，后者只有80.0%。
   - 设计动机：decode在长生成任务中占总延迟的主要部分（如视频描述可达70%+），prefill-only的方法加速有限。

4. **RoPE位置编码处理**:
   - 做什么：处理token剪枝后位置编码不连续的问题。
   - 核心思路：对unified RoPE（如LLaVA-NeXT），保留对应位置索引范围；对multimodal RoPE（如Qwen2.5-VL），重构最小连续位置网格后平移文本位置。

### Visual Sink vs Retrieval Token分析
通过attention map分析发现两类token：Visual Sink Token（跨query稳定，浅层主导，类似attention sink）和Visual Retrieval Token（随query动态变化，深层主导）。SparseVILA的解耦设计天然保留了两者。

## 实验关键数据

### 图像任务（LLaVA-NeXT-7B，多轮评估）

| 方法 | Prefill稀疏 | Decode稀疏 | E2E加速 | DocVQA | ChartQA | GQA | POPE |
|------|-----------|-----------|--------|--------|---------|-----|------|
| 基线 | 0% | 0% | 1.0× | 63.6 | 53.0 | 63.5 | 84.5 |
| FastV | 80% | 0% | 1.2× | 33.5 | 31.6 | 55.3 | 76.7 |
| VisionZip | 80% | 0% | 1.2× | 48.5 | 38.2 | 60.3 | 84.1 |
| SparseVLM | 75% | 0% | 1.2× | 41.8 | 39.9 | 59.7 | 83.4 |
| **SparseVILA** | **60%** | **75%** | **1.2×** | **58.0** | **47.8** | **62.7** | **85.8** |

### 视频理解（LongVILA-7B，256帧）

| 方法 | Prefill稀疏 | Decode稀疏 | E2E加速 | LVB | MLVU | NExT-QA | Video-MME |
|------|-----------|-----------|--------|-----|------|---------|-----------|
| 基线 | 0% | 0% | 1.0× | 53.8 | 64.9 | 78.6 | 58.8 |
| VisionZip | 95% | 0% | 2.1× | 47.0 | 60.4 | 75.5 | 52.2 |
| PruMerge | 95% | 0% | 2.1× | 47.9 | 60.9 | 75.7 | 52.0 |
| **SparseVILA** | **75%** | **90%** | **2.1×** | **54.1** | **65.3** | **79.0** | **58.7** |

物理推理（Cosmos-Reason1-7B，24fps）：SparseVILA 75.9% vs 基线71.4%，**加速1.9×同时精度提升4.5%**。

### 消融实验

| Prefill稀疏 | Decode稀疏 | E2E加速 | RoboVQA |
|------------|-----------|--------|---------|
| 0% | 0% | 1.0× | 86.4 |
| 90% | 0% | 1.4× | 80.0 |
| **70%** | **85%** | **1.4×** | **89.1** |

### 关键发现
- **解耦比统一稀疏更好**：同等1.4×加速下，解耦（70%P+85%D）比prefill-only（90%P）高9.1个点。核心原因是prefill-only的高稀疏率永久丢失了关键信息。
- **视频任务上SparseVILA精度不降反升**：归因于decode阶段的query-aware检索让模型更聚焦于相关帧，类似StreamingLLM的发现——更小的active context有助于推理聚焦。
- **多轮对话优势显著**：FastV和SparseVLM在多轮评估中迅速退化（DocVQA从63.6降到33.5/41.8），SparseVILA仅降到58.0，因为它不永久删除token。
- **V-NIAH检索测试**：SparseVLM和FastV在32帧以上就OOM或精度崩溃，SparseVILA在200帧仍保持完美检索。
- **开销极低**：SparseVILA的Triton kernel开销仅94.9ms（LongVILA-7B），而VisionZip 206.3ms，PruMerge 448.3ms。
- 解码attention kernel在长视频场景最高可加速11.4×。

## 亮点与洞察
- **Prefill-Decode解耦是核心创新**：之前所有方法都把稀疏化看作一个统一操作（在prefill中完成），SparseVILA首次将其分为两个独立决策——prefill保覆盖、decode选精准。这种设计既加速了真正的瓶颈（decode），又保留了多轮对话的灵活性。
- **"不删除，只冻结"的设计哲学**：decode阶段未被选中的token不被删除，只是"休眠"在KV cache中。下一轮问题可以唤醒不同的子集。这是对现有query-aware方法"一删了之"的根本改进。
- **Sink Token + Retrieval Token的发现**：浅层的稳定sink token和深层的动态retrieval token共存，直接支持了解耦设计的合理性。这个发现可以指导未来更精细的层级稀疏策略。
- **工程价值极高**：training-free、architecture-agnostic，基于AWQ量化pipeline，Triton kernel加速，在单张A6000上实测2.6×加速。

## 局限性 / 可改进方向
- 每层使用统一稀疏率，更精细的layer-wise或head-aware策略可能进一步提升效果。
- 文档理解任务（DocVQA、ChartQA）上仍有明显掉点（58.0 vs 63.6），因为文档中每个文字都可能重要，不适合高稀疏率。
- 仅在8B以下模型验证，更大模型的效果和开销需要进一步评估。
- Decode阶段的query-aware检索为每层都做一次选择，更高效的跨层共享策略值得探索。

## 相关工作与启发
- **vs FastV**: FastV用LLM早期层的attention做query-aware prefill剪枝，多轮退化严重（DocVQA 33.5 vs SparseVILA 58.0）。因为FastV一删了之，无法恢复。
- **vs VisionZip**: VisionZip用token merging做query-agnostic压缩，开销大（206ms vs 95ms）且精度不如SparseVILA。
- **vs Quest/DuoAttention**: 这些是LLM KV cache压缩方法，面向纯文本。SparseVILA专门针对视觉token的空间-时间冗余设计，互补而非替代。
- **vs ShortV（同批论文）**: ShortV在ineffective layers中冻结视觉token KV cache，SparseVILA在所有层做decode-time检索。两者从不同角度（层级 vs 阶段）优化视觉token效率，可以结合。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ prefill-decode解耦稀疏是全新范式，首次系统性地优化VLM的decode阶段视觉效率
- 实验充分度: ⭐⭐⭐⭐⭐ 9个图像+4个视频+2个推理benchmark，5个VLM架构，多轮评估，V-NIAH，kernel分析，overhead分析极其全面
- 写作质量: ⭐⭐⭐⭐⭐ 问题分析深入，sink/retrieval token的发现有insight，图表丰富
- 价值: ⭐⭐⭐⭐⭐ training-free + architecture-agnostic + 实测加速，对VLM部署有直接实用价值
