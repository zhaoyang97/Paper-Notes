# One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers

**会议**: CVPR 2026  
**arXiv**: [2603.12245](https://arxiv.org/abs/2603.12245)  
**代码**: [https://snap-research.github.io/elit](https://snap-research.github.io/elit) (有)  
**领域**: 图像生成 / 模型压缩 / 扩散模型  
**关键词**: Diffusion Transformer, 弹性推理, 潜在接口, 可变计算预算, 自适应计算分配  

## 一句话总结
提出ELIT（Elastic Latent Interface Transformer），通过在DiT中插入可变长度的潜在token接口和轻量级Read/Write交叉注意力层，将计算量与输入分辨率解耦，使单一模型支持多种推理预算，在ImageNet-1K 512px上FID和FDD分别提升35.3%和39.6%。

## 背景与动机
DiT类扩散模型在图像/视频生成中取得了SOTA质量，但存在两个根本性的计算效率问题：(1) 每步计算量被锁定为输入分辨率的固定函数，无法灵活调整延迟-质量权衡；(2) 计算在所有空间token上均匀分配，不管某些区域是否简单或不重要。作者通过一个精巧的实验证实了第二点：给真实图像补零扩展token数后训练DiT，尽管FLOPs翻倍，生成质量丝毫未提升——注意力图显示零值token主要attend彼此，说明DiT无法将计算从简单区域重新分配到困难区域。

## 核心问题
如何让DiT类模型在不改变训练目标和主体架构的前提下，(a) 将计算集中在重要/困难区域而非均匀分布，(b) 用单一模型支持从低到高多种推理计算预算？现有方法或改架构太大（如RIN/FIT偏离DiT设计导致难以迁移），或仅加速训练而推理不变（如MaskDiT等masking方法），或为training-free但受限于基线质量上界（token merging方法）。

## 方法详解

### 整体框架
ELIT在DiT的transformer block栈中插入一个"潜在接口"——一组可学习的可变长度latent token。整体pipeline分三段：
- **Spatial Head**（$B_{in}$个block）：对patchified输入做初步处理，避免直接从原始噪声patch读取
- **Latent Core**（$B_{core}$个block）：先通过Read层将空间token信息拉入latent token，在latent域上执行标准transformer block，再通过Write层将更新广播回空间token。这是计算的主体
- **Spatial Tail**（$B_{out}$个block）：补全空间细节处理，产生最终速度场预测

输入输出不变，训练目标就是标准的Rectified Flow loss，无需任何辅助损失。

### 关键设计
1. **Read/Write交叉注意力层**：Read层以latent token为Query、空间token为Key/Value做交叉注意力，将信息从空间域拉入latent域，自然地优先关注loss更高的困难空间区域。Write层完全对称，将latent域的更新广播回空间token。两层均采用adaLN-Zero做时间步调制，加QK normalization保稳定性，MLP不做hidden维度扩展以减少开销。

2. **分组交叉注意力（Grouped Cross-Attention）**：将空间token划分为$G$个不重叠的组（如2D网格），latent token也对应分组为每组$J=K/G$个。交叉注意力仅在对应组内计算，复杂度从$O(NK)$降为$O(NK/G)$。latent token从一组共享的可学习位置编码初始化，使模型对输入分辨率变化鲁棒（增加分辨率只改变$G$和$N$，不改变每组latent数$J$）。

3. **随机尾部丢弃的多预算训练**：训练时每个iteration随机采样$\tilde{J} \sim \mathrm{Uniform}\{J_{\min}, \ldots, J_{\max}\}$，仅保留每组前$\tilde{J}$个latent token并丢弃其余。这使得前面的latent被训练得更频繁，被迫存储最重要的全局信息，形成"重要性排序"的层次结构。推理时用户选择$\tilde{J}$即可直接控制每步计算量，天然支持从低到高的计算预算。

4. **CCFG（Cheap Classifier-Free Guidance）**：多预算模型天然提供了一个"弱版本"的自己（低预算版本），可直接用于autoguidance而无需额外训练。CCFG进一步在guidance项中同时去掉类别条件，结合了autoguidance和CFG的优势，在不增加训练成本的前提下降低约33%的推理FLOPs同时提升质量。

### 损失函数 / 训练策略
- 训练目标：标准Rectified Flow loss $\mathcal{L}_{RF} = \mathbb{E}\|G(X_t, t) - (X_1 - X_0)\|_2^2$，无辅助损失
- 时间步采样：logit-normal分布
- 多预算训练时，为补偿低token数迭代的计算节省，将batch size从256增大到384以匹配训练FLOPs
- DiT-XL/2主实验：500k步，lr=1e-4，10k warmup，Adam，EMA β=0.9999，gradient clipping=1.0
- 大规模实验（Qwen-Image 20B）：采用RF loss + 蒸馏loss（20x缩放匹配量级），先512px训60k步再1024px训60k步

## 实验关键数据
| 数据集 | 指标 | 本文(ELIT-DiT-XL MB) | 之前SOTA(DiT-XL) | 提升 |
|--------|------|------|----------|------|
| ImageNet-1K 256px | FID↓ (+G) | 3.8 | 5.7 | 33% |
| ImageNet-1K 256px | FDD↓ (+G) | 124.5 | 232.9 | 47% |
| ImageNet-1K 512px | FID↓ (+G) | 4.9 | 9.5 | 48% |
| ImageNet-1K 512px | FDD↓ (+G) | 106.1 | 233.6 | 55% |
| ImageNet-1K 512px (CCFG) | FID↓ | 4.9 | 9.5(CFG) | 48% + 33% FLOPs节省 |
| Kinetics-700 256px | FID↓ (+G) | 10.7 | 11.3 | 5.3% |
| Qwen-Image 1024px | DPG-Bench Avg | 90.45(100%tok) → 88.02(25%tok) | 91.27 | 最多节省63% FLOPs |

- 在DiT、U-ViT、HDiT三种架构上：512px FID分别降低53%、28%、23%
- 收敛加速：256px上3.3×，512px上4.0×
- 随模型规模增大，ELIT增益变大而相对开销比例降低

### 消融实验要点
- **分组大小**：4×4（16组）在256px和512px上最优；1×1退化为一对一映射效果差，16×16覆盖全图也不好。适度分组提供粗粒度空间正则化+组内灵活重分配
- **Block分配**：约67-71%的block放在latent core最优（DiT-B: 3-6-3或4-4-4, DiT-XL: 4-20-4）
- **尾部丢弃 vs 随机丢弃**：尾部丢弃显著优于随机token丢弃，说明重要性排序的层次结构至关重要
- **多预算联合训练 vs 单预算独立训练**：联合训练在所有预算点上均优于独立训练，说明多预算训练本身起到了正则化效果
- **Read/Write设计**：单层交叉注意力优于Q-Former式和全自注意力；增加Write或FFN容量有提升但增加开销

## 亮点
- **极简但有效**：仅添加两个轻量级交叉注意力层+一组可学习latent token，不改训练目标、不加辅助损失，却获得全面大幅提升
- **补零实验**揭示DiT的计算浪费问题极为精巧——用合成实验证明DiT无法跨区域重分配计算
- **CCFG混合引导**巧妙：一个多预算模型天然内置弱版本，直接实现autoguidance + CFG混合，白嫖33%加速
- **Read注意力的可视化**直观展示了重要性排序：前面的latent关注全局结构，后面的关注细节纹理
- **通用兼容性**强：在DiT/U-ViT/HDiT/MM-DiT四种架构和图像/视频两种任务上都有效

## 局限性 / 可改进方向
- 大规模从头训练的效果尚未验证（Qwen-Image实验是蒸馏微调而非从头训练）
- CCFG比CFG更容易导致图像过饱和，需要使用较低的guidance scale
- 未探索跨采样步的预算调度（不同噪声水平可能需要不同token数），作者自己也提到这是future work
- per-group自适应token分配实验失败（用loss map预测每组重要性并不优于统一分配），说明Read操作已隐式实现了这一点

## 与相关工作的对比
- **vs FlexiDiT/多patch训练**：FlexiDiT用多patchification大小实现可变计算，但仍在空间域均匀分配计算；ELIT在latent域重分配计算，效果显著更好（消融实验中多patch训练甚至不如标准DiT）
- **vs RIN/FIT**：RIN/FIT也使用latent token做read/write交互，但偏离DiT设计太多（需专用优化器如LAMB），且推理预算固定。ELIT是drop-in即插即用，保持DiT架构+RF训练不变
- **vs Token Merging (ToMe/SDTM)**：training-free方法以DiT质量为上界；ELIT在仅用25%token时（FID=14.2）仍优于DiT基线（FID=20.9）

## 启发与关联
- latent接口的思路可以迁移到其他transformer架构（如视觉理解中的ViT），用于自适应计算分配
- 重要性排序+尾部丢弃可以作为通用的可变预算推理策略
- 与 `ideas/model_compression/20260316_task_aware_token_compression.md` 相关：ELIT的Read层天然学到了task-aware的token重要性
- 与 `ideas/self_supervised/20260317_supervised_query_for_open_world_attention.md` 相关：ELIT的Read层类似于用latent query来做注意力重分配

## 评分
- 新颖性: ⭐⭐⭐⭐ 核心idea并非全新（latent token + read/write在RIN/FIT中已有），但将其极简化并无缝融入DiT+多预算弹性推理是重要贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 四种架构、两种任务、多种分辨率、大规模模型实验、详尽消融、failed experiments都报了
- 写作质量: ⭐⭐⭐⭐⭐ 动机实验精巧，方法描述清晰，图表质量高，appendix极详细
- 价值: ⭐⭐⭐⭐⭐ 极具实用价值，drop-in设计使其可直接应用于现有DiT系统
