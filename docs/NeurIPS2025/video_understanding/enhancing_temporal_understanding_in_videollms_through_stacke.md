# Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders

## 基本信息
- **arXiv**: 2510.26027
- **会议**: NeurIPS 2025
- **作者**: Ali Rasekh, Erfan Bagheri Soula, Omid Daliran, Simon Gottschalk, Mohsen Fayyaz
- **机构**: Leibniz University Hannover / L3S Research Center, Microsoft
- **代码**: https://alirasekh.github.io/STAVEQ2/

## 一句话总结
提出 STAVEQ2，在 Vision Encoder 中堆叠参数高效的时序注意力模块（STA），解决现有 Video-LLM 在细粒度时序理解（如区分"从左到右拉"和"从右到左拉"）上的根本性架构缺陷，在 VITATECS/MVBench/Video-MME 上提升最高 5.5%。

## 背景与动机
现有 Video-LLM 在时序理解上存在根本缺陷：
- **Qwen2-VL**：Vision Encoder 中只有空间注意力，将时序理解完全委托给 LLM
- **InternVideo2-Chat**：有联合时空注意力但仍无法可靠区分时间方向性动作
- 实验证明：在 SSv2-T10（时序对立动作对）上，Qwen2-VL 7B 零样本仅 21.91%，InternVideo2 仅 30.60%
- In-context learning 不但没帮助，反而降低性能——说明这是架构缺陷而非数据问题

## 核心问题
如何在不改变 LLM 的前提下，通过增强 Vision Encoder 的时序建模能力来提升 Video-LLM 的时间理解？

## 方法详解

### STAVEQ2 架构
在 Qwen2-VL 的 ViT 每个 transformer block 中，spatial attention 之后插入 temporal attention。

**空间注意力**（原有）：每帧内 $N$ 个 patch 之间做 self-attention
$$S_t^{(m)} = A_t^{(m)} V_t^{(m)} + X_t^{(m-1)}$$

**时序注意力**（新增）：每个 patch 跨 $T$ 帧做 self-attention
$$Z_i^{(m)} = A_i'^{(m)} V_i'^{(m)} + Y_i^{(m)}$$
其中 $Y_i^{(m)} = [S_{1,i}^{(m)}, \ldots, S_{T,i}^{(m)}]^\top$

最终：$X^{(m)} = \text{MLP}(\text{LN}(Z^{(m)})) + Z^{(m)}$

### 关键设计
1. **参数高效**：时序注意力头数仅为空间的 1/4（head scale = 0.25），大幅减少参数
2. **1D RoPE**：时序注意力使用 1D RoPE（vs. 空间的 2D RoPE）编码时间位置
3. **零初始化**：输出投影层初始化为零，初始状态等价于原始模型
4. **两阶段训练**：
   - Stage 1：冻结所有参数，仅训练时序注意力块 + LayerNorm
   - Stage 2：加入 LoRA adapter 联合训练整个模型
5. **全层部署**：所有 32 个 transformer block 都加入 STA 效果最好

## 实验关键数据

### SSv2 动作识别 (Vision-only)
| 模型 | SSv2 Acc. |
|---|---|
| InternVideo2 1B | 77.1% |
| InternVideo2 6B | 77.5% |
| **InternVideo2 1B + STA** | **78.0% (+0.5%)** |

→ 1.3B 模型超越 6B 模型！

### InternVideo2-Chat + STA (SSv2-T10)
| 方法 | Acc. |
|---|---|
| InternVideo2-Chat 8B | 84.17% |
| + STA | **95.18% (+11.01%)** |

### STAVEQ2 在 Video-LLM Benchmarks
| 模型 | VITATECS Dir. | MVBench | Video-MME (wo/w sub) |
|---|---|---|---|
| Qwen2-VL 7B | 86.6 | 67.0 | 63.3 / 69.0 |
| Qwen2.5-VL 7B | 80.0 | 69.6 | 65.1 / 71.6 |
| **STAVEQ2 7B** | **87.6** | **70.1** | **66.8 / 71.8** |
| Qwen2-VL 72B | 87.8 | 73.6 | 71.2 / 77.8 |
| **STAVEQ2 72B** | **90.1** | **74.5** | **73.9 / 79.9** |
| GPT-4o | – | – | 71.9 / 77.2 |

→ STAVEQ2 72B 在 Video-MME 上超越 GPT-4o (+2.0/+2.7)

### 跨模型泛化
- STAVEQ2.5 (Qwen2.5-VL + STA)：进一步提升
- VideoRoPE + STA：互补增益
- InternVideo2.5-Chat + STA：MVBench 75.7→76.8

## 亮点
1. **问题分析透彻**：系统性证明时序理解是架构缺陷而非数据问题（zero-shot + ICL + fine-tune 对比）
2. **简洁且有效**：仅在 ViT 中堆叠轻量时序注意力，不改 LLM
3. **广泛泛化**：在 Qwen2-VL/Qwen2.5-VL/InternVideo2/VideoRoPE 上均有效
4. **新 SOTA**：SSv2 动作识别新 SOTA（1.3B 超 6B），Video-MME 超 GPT-4o
5. **Divided space-time attention 的复兴**：证明了 TimeSformer 式分离注意力在 Video-LLM 中的价值

## 局限性
1. 受资源限制，未从头预训练，仅做了微调验证
2. 最大模型仅到 72B（尽管已超越了很多更大模型）
3. STA 增加了推理延迟（每层多一个时序注意力）
4. WebVid-QA 数据集质量可能限制训练效果

## 与相关工作的对比
- **vs. Qwen2-VL**：Qwen2-VL 完全依赖 LLM 做时序理解，STAVEQ2 证明这不够
- **vs. InternVideo2**：即使有联合时空注意力，也无法解决细粒度时间方向——需要专门的 divided attention
- **vs. ST-LLM**：ST-LLM 将时空建模委托给 LLM，STAVEQ2 7B 在 MVBench 上高出 15.2 分
- **vs. TG-Vid**：TG-Vid 用时间门控，效果有限且效率低；STAVEQ2 超出 13.7 分
- **vs. FastVID**：FastVID 关注效率（剪枝 token），STAVEQ2 关注能力（增强时序），两者互补

## 启发与关联
- **Vision Encoder 是瓶颈**：不能把所有时序理解甩给 LLM——在 token 送入 LLM 之前就应编码好时序信息
- **Divided vs. Joint Space-Time Attention**：进一步验证了 TimeSformer 的分离注意力在 Video-LLM 场景中比 joint attention 更可控
- **与 Eyes Wide Open 的互补**：Eyes Wide Open 做流式视频处理的 temporal KV cache 管理，STAVEQ2 做 encoder 级时序建模——可组合

## 评分
- 新颖性：★★★☆☆ — 分离时空注意力是已知方法，创新在于应用到 Video-LLM 的 encoder
- 技术深度：★★★★☆ — 问题分析深入，消融充分
- 实验完整度：★★★★★ — 4 模型 × 多 benchmark × 消融 × 注意力可视化
- 写作质量：★★★★☆ — 动机分析部分很有说服力
