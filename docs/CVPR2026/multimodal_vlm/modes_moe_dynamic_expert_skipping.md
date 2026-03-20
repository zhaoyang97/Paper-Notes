# MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

**会议**: CVPR 2026  
**arXiv**: [2511.15690](https://arxiv.org/abs/2511.15690)  
**代码**: [https://github.com/ModelTC/MoDES](https://github.com/ModelTC/MoDES)  
**领域**: 多模态VLM / MoE加速 / 高效推理  
**关键词**: MoE, 专家跳过, 双模态阈值, 全局调制局部门控, 多模态大模型加速  

## 一句话总结
首个针对MoE多模态大模型的专家跳过框架MoDES,通过全局调制局部门控(GMLG)将层级重要性融入路由概率、双模态阈值(DMT)对文本/视觉token分别设定跳过策略、前沿搜索高效优化阈值,在Qwen3-VL-MoE-30B上88%专家跳过仍保留97.33%精度,prefill加速2.16×。

## 背景与动机
MoE MLLM(如Kimi-VL、Qwen3-VL-MoE)通过稀疏激活部分专家来降低计算成本,但仍存在效率瓶颈——固定的top-k路由为所有token激活相同数量的专家。现有专家跳过方法(NAEE、MC-MoE、DiEP)针对文本LLM设计,直接迁移到MLLM导致>10%性能下降(83%跳过率下)。分析揭示两个被忽视的因素：(1) **全局贡献失配**——浅层专家对最终输出影响远大于深层(error explosion效应);(2) **模态差异**——视觉token与FFN权重更正交(角度→90°),专家对视觉token更新幅度更小,冗余度更高。

## 核心问题
如何为MoE MLLM设计模态感知、层级感知的专家跳过策略,在极端跳过率(>80%)下仍保持近baseline精度?

## 方法详解

### 整体框架
MoDES由两个核心组件构成：(1) GMLG估计每个token-expert对的重要性分数；(2) DMT根据token模态选择不同的跳过阈值,阈值通过前沿搜索算法高效确定。整个流程training-free。

### 关键设计
1. **全局调制局部门控 (GMLG)**：重要性分数$s_i^{(l)} = \alpha^{(l)} \cdot \pi_i^{(l)}$,其中$\pi_i^{(l)}$是标准路由概率(局部信号),$\alpha^{(l)}$是通过离线校准计算的层级全局权重——对第$l$层所有专家跳过后量测KL散度。$\alpha^{(l)}$在浅层高、深层低,确保浅层专家更少被跳过。校准仅需1024样本,约20min-4hr(20-30B模型)。

2. **双模态阈值 (DMT)**：分别为文本token和视觉token设定不同跳过阈值$\tau_t$和$\tau_v$。决策：$\{Expert_i^{(l)} | s_i^{(l)} < \tau_t \cdot \mathbb{I}_t + \tau_v \cdot \mathbb{I}_v\}$被跳过。可视化显示:实际策略中视觉token在所有层被跳过的比例远高于文本token(>90% vs 50-70%),验证了视觉experts冗余度更高的洞察。

3. **前沿搜索 (Frontier Search)**：在$(\tau_t, \tau_v)$的二维网格$\mathcal{B}^2$上优化,利用$f$(KL散度)和$g$(跳过率)的单调性,将搜索从$O(ND^2)$降至$O(ND)$——实测搜索时间减少~45×。有严格的正确性和最优性证明(Lemma 1-4 + Proposition 1-2)。

### 损失函数 / 训练策略
完全training-free。离线校准$\alpha^{(l)}$和前沿搜索$(\tau_t^*, \tau_v^*)$均在1024个GQA样本上完成。推理时仅需在MoE层的router kernel中添加branch-free的masked comparison,无额外kernel launch。

## 实验关键数据
| 模型 | 跳过率 | MoDES | MC-MoE | DiEP | NAEE | 直接减k |
|--------|------|------|----------|------|------|------|
| Kimi-VL-A3B | 50% | **99.91%** | 97.69 | 98.17 | 96.44 | 95.93 |
| Kimi-VL-A3B | 67% | **98.46%** | 95.45 | 94.81 | 94.03 | 93.88 |
| Kimi-VL-A3B | 83% | **96.25%** | 88.32 | 87.58 | 82.81 | 71.60 |
| Qwen3-VL-MoE-30B | 88% | **97.33%** | 86.66 | 85.30 | 80.60 | 60.11 |
| InternVL-3.5-30B | 88% | **97.03%** | 86.20 | 83.26 | 78.88 | 59.63 |

推理加速(Qwen3-VL-MoE-30B): prefill **2.16×**, decode **1.26×**。

与量化兼容: 2.5-bit量化+MoDES在Qwen3上保留94.43%精度(MC-MoE 89.58%)。

### 消融实验要点
- **GMLG和DMT均关键**：跳过83%专家时,单纯Thresholding 82.81%→+GMLG 84.48%→+DMT 85.50%→**GMLG+DMT 96.25%**
- **模态差异真实存在**：减少视觉token的top-k=1性能仅微降,减少文本token则严重下降——视觉experts冗余远高
- **数据选择不敏感**：GQA/COCO/VMMMU校准结果几乎一致(均~96%)
- **$\alpha^{(l)}$模式一致**：不同数据集上层级KL散度分布相似,浅层>深层
- **前沿搜索vs穷举**：精度几乎相同(96.24 vs 96.25%)但时间减少45×

## 亮点
- **首个MoE MLLM专家跳过框架**——之前方法全部针对单模态LLM,直接迁移大幅失效
- "视觉token对experts的冗余度更高"的发现与V2Drop/ApET的"大量视觉token冗余"一脉相承——只是这次是在expert维度而非token维度
- 前沿搜索有完整数学证明(单调性→可行域结构→最优性),理论功底扎实
- 88%跳过率保留97%精度是惊人的——说明MoE模型本身就大量过度分配experts
- 与量化正交可组合——未来可以MoDES+量化+token压缩三管齐下

## 局限性 / 可改进方向
- 阈值通过离线搜索确定,不同任务/输入可能需要不同阈值——输入自适应的动态阈值值得探索
- 仅验证了3个MoE MLLM (Kimi-VL/Qwen3-VL/InternVL3.5),更多架构待测
- 解码阶段加速有限(1.26×),主要因为解码是memory-bound且只处理文本token
- $\alpha^{(l)}$校准需要forward pass每层跳过的模型,计算开销随层数线性增长
- 未探索动态改变top-k(而非固定top-k后跳过)的策略

## 与相关工作的对比
- **vs NAEE/MC-MoE (LLM expert skipping)**：为单模态LLM设计,迁移到MLLM后83%跳过率精度<89%。MoDES 96.25%——差距巨大
- **vs DiEP (可微专家剪枝)**：DiEP在training-aware框架中做专家相似度+路由概率剪枝,但忽略层级差异和模态差异,MoDES training-free且更好
- **vs V2Drop/DUET-VLM (token压缩)**：正交互补——V2Drop压缩视觉token数量,MoDES压缩每个token激活的expert数量。二者可以组合使用
- **vs ApET (近似误差压缩)**：ApET从信息论角度减少token,MoDES从expert角度减少计算。思路不同但目标相同

## 启发与关联
- MoDES的"浅层更重要"发现与Overthinking论文的"中间层到深层的hypotheses不稳定→幻觉"互相印证——两者都指向"不是所有层同等重要"
- 与`ideas/model_compression/20260316_adaptive_model_routing.md`相关——该idea探索自适应路由,MoDES提供了模态感知路由的具体实例
- **组合idea**: MoDES(expert跳过) + V2Drop(token丢弃) + ApET(token合并) 三层压缩 → 极致VLM推理加速

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将expert skipping适配到多模态MoE,两个洞察(层级+模态)和前沿搜索算法均为新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 3个MoE模型系列、13个基准、多跳过率、量化组合、详细消融、数学证明
- 写作质量: ⭐⭐⭐⭐⭐ 动机→分析→方法→验证的逻辑完美,附录包含完整证明
- 价值: ⭐⭐⭐⭐⭐ MoE MLLM已成主流(Kimi-VL/DeepSeek/Qwen3都用MoE),该方法直接可落地
