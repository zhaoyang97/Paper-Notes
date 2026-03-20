# Scaling Language-Free Visual Representation Learning

**会议**: ICCV 2025  
**arXiv**: [2504.01017](https://arxiv.org/abs/2504.01017)  
**代码**: [https://davidfan.io/webssl/](https://davidfan.io/webssl/)  
**领域**: 自监督学习 / 表示学习  
**关键词**: visual self-supervised learning, CLIP对比, scaling law, VQA评估, 无语言监督  

## 一句话总结
通过在MetaCLIP的20亿web图像上训练DINOv2/MAE系列模型（1B-7B参数），系统性地证明纯视觉自监督学习在模型和数据规模上展现优于CLIP的scaling behavior，5B+参数时在VQA平均性能上超越CLIP——包括传统认为需要语言监督的OCR/Chart任务。

## 背景与动机
视觉表示学习沿两条路径发展：CLIP用图文对训练，SSL（如DINOv2、MAE）只用图像。尽管SSL在分类和分割等传统视觉任务上已经很强，但在多模态LLM（如VQA）中远不如CLIP，尤其是OCR和图表理解任务。社区普遍认为这是因为SSL缺乏"语言监督"带来的语义信息。但一个被忽视的事实是：CLIP用的是数十亿级web数据，而SSL通常只用数百万级的ImageNet。那么SSL落后于CLIP，到底是因为"没有语言"还是因为"数据不同"？

## 核心问题
视觉自监督方法在多模态场景下落后于CLIP，究竟是语言监督的缺失导致的，还是训练数据规模和分布的差异？如果在相同数据上训练，SSL能否匹配甚至超越CLIP？

## 方法详解

### 整体框架
Web-SSL是一系列纯视觉自监督模型（Web-DINO = DINOv2在web数据上训练，Web-MAE = MAE在web数据上训练）。训练数据：MetaCLIP的20亿web图像（MC-2B），只用图像不用文本。模型规模从1B到7B。评估方式：冻结视觉编码器，通过Cambrian-1的视觉指令微调流程（2阶段MLP+LLM），在16个VQA benchmark上评估。

### 关键设计
1. **公平对比的实验设计**：这不是一个方法论文，而是一个实验研究。核心创新在于控制变量：SSL和CLIP在完全相同的MC-2B数据上训练，相同的模型架构（ViT-1B到7B），相同的评估流程（Cambrian-1 + Llama-3 8B）。这样就消除了"数据差异"这个混淆因素。

2. **SSL scaling比CLIP更好的发现**：Web-DINO的VQA性能随模型增大近似log-linear增长且未饱和（到7B仍在提升），而CLIP在3B后就基本饱和了。在数据维度上，Web-DINO 7B从1B到8B训练样本持续提升，尤其OCR & Chart持续改善。这意味着SSL有显著的未被挖掘的scaling潜力。

3. **数据过滤的OCR/Chart提升**：通过SmolVLM2过滤MC-2B中包含文本的图像，发现仅使用1.3%的图表/文档重图像就能让Web-DINO在OCR & Chart上超越全量数据训练的CLIP（+4.3%），同时平均VQA也更高。这证明数据组成（而非语言监督）才是OCR能力的关键。

4. **涌现的语言对齐性**：通过计算SSL编码器特征与LLM（Llama-3.1 8B/70B）文本特征的内在对齐度，发现随着模型增大和数据增多，SSL编码器自然地学到了与语言更对齐的视觉特征——无需任何语言监督。这呼应了"柏拉图表示假说"。

### 损失函数 / 训练策略
- Web-DINO：DINOv2的标准训练recipe，batch size 3072，lr 3.5e-4，warmup 100K
- Web-MAE：标准MAE recipe，大模型降低lr至1.6e-3防止发散
- CLIP：MetaCLIP标准recipe，batch size 32768
- 所有模型在224x224分辨率，单epoch（2B图像看一次）

## 实验关键数据
| 模型 | 参数 | 数据 | Avg VQA | OCR & Chart | Vision-Centric | IN1k linear |
|------|------|------|---------|-------------|----------------|-------------|
| MetaCLIP ViT-G(HF) | 1B | 12.8B | 54.8 | 37.3 | 58.4 | 86.4 |
| SigLIP SO400M | 400M | 45.0B | 55.4 | 39.5 | 58.9 | 86.5 |
| DINOv2 ViT-g(HF) | 1B | 1.9B | 47.9 | 21.2 | 55.3 | 86.0 |
| **Web-DINO 7B** | **7B** | **8B** | **55.2→59.9** | **39.4→55.1** | **59.1→60.8** | **86.5** |

- Web-DINO 5B+超越同数据的CLIP在Average VQA上
- Web-DINO 7B (8B data, 518px)达到59.9 Avg VQA，与SigLIP2 384px的62.0相当，且只用5x更少的数据
- 1.3%文本过滤后的Web-DINO 2B在OCR & Chart上比全数据CLIP 2B高4.3%
- 传统视觉任务上：Web-DINO在IN1k linear probe达86.5%，ADE20K分割超越MetaCLIP

### 消融实验要点
- 在ImageNet上训练的Web-DINO不展现scaling behavior——数据多样性是scaling的前提
- MAE也展现类似的scaling趋势，且在OCR & Chart上更强，说明这不是DINOv2特有现象
- 高分辨率适配（224→378→518）持续提升OCR & Chart性能
- CLIP在3B后所有VQA类别都饱和，SSL则持续提升

## 亮点
- **挑战了"视觉SSL需要语言监督"的共识**：这是一个paradigm shift级别的发现——语言监督并非必须，数据规模和分布才是关键
- **Bitter Lesson的视觉版本**：减少inductive bias（不用语言监督），依靠scale，反而能获得更好的scaling behavior
- **实验设计极其严谨**：控制变量做得非常好，所有模型同数据同架构同评估，结论有说服力
- **数据过滤的发现很有启发性**：仅用1.3%的文本密集图像就能超越CLIP的OCR能力，打开了数据组成优化的新方向
- **涌现对齐**：SSL模型在scale up后自然与LLM对齐，这是对Platonic Representation Hypothesis的实证支持

## 局限性 / 可改进方向
- SSL不支持zero-shot分类（需要通过MLLM间接实现或用LiT-style adapter）
- 仅在Llama-3 8B上验证VQA，更大LLM可能改变结论
- 7B encoder很大，实际部署需要考虑效率
- 还没探索超过7B和超过8B数据的regime
- 数据过滤用的是MLLM打标签（有点用语言），纯无语言的数据策划方法值得探索

## 与相关工作的对比
- **vs. DINOv2**：同方法但不同数据——Web-DINO在web数据上训练显著优于ImageNet上的DINOv2在VQA上的表现，证明数据是瓶颈
- **vs. CLIP/SigLIP**：公平对比下SSL scaling更好；CLIP在小模型时可能更data-efficient，但大模型时优势消失
- **vs. Cambrian-1**：使用其评估框架但提供了新的vision encoder洞察

## 启发与关联
- 对多模态社区有深远影响：未来MLLM可以考虑不依赖CLIP，转而使用scaling up的SSL encoder
- 数据组成优化（文本密集图像比例）的思路可以迁移到其他SSL预训练
- 与ideas/self_supervised/和ideas/multimodal_vlm/中关于VFM和表示学习的idea高度相关

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ paradigm shift级别的发现，挑战了CLIP主导的视觉预训练范式
- 实验充分度: ⭐⭐⭐⭐⭐ 1B-7B模型×1B-8B数据的全面scaling study，16个VQA+传统benchmark，对比CLIP/MAE/DINOv2，消融详尽
- 写作质量: ⭐⭐⭐⭐⭐ 以5个Research Questions组织发现，逻辑清晰，图表精美直观
- 价值: ⭐⭐⭐⭐⭐ 对视觉表示学习社区的认知产生重要影响，计划开源模型，引领SSL新方向
