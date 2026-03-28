# BioBench: A Blueprint to Move Beyond ImageNet for Scientific ML Benchmarks

**会议**: NeurIPS 2025  
**arXiv**: [2511.16315](https://arxiv.org/abs/2511.16315)  
**代码**: [github.com/samuelstevens/biobench](https://github.com/samuelstevens/biobench)  
**领域**: model_compression / benchmark  
**关键词**: 生态图像, 视觉Benchmark, ImageNet局限性, 迁移学习, AI for Science  

## 一句话总结
提出 BioBench——一个统一 9 个生态视觉任务、4 个分类界、6 种图像模态、310 万张图像的基准，证明 ImageNet top-1 准确率仅解释 34% 的生态任务方差，在 >75% 精度的前沿模型中 30% 的排名是错误的。

## 研究背景与动机

1. **领域现状**：视觉研究仍以 ImageNet-1K、MS COCO、ADE20K 为评估中心，新模型（ViT、DINOv2、CLIP）的 SOTA 声明都基于这些排行榜。实际科学领域——放射科、组织病理学、微生物学、生态学——的图像与网络照片有根本差异。
2. **现有痛点**：
   - ImageNet 的 RGB 网络照片与相机陷阱红外、多光谱无人机图像、显微镜载片等科学图像在频谱、噪声、分布上完全不同；
   - 科学任务是细粒度(fine-grained)且长尾的——生态学家需区分上千种昆虫，ImageNet 的 1000 个类别几乎没有此类细微差异；
   - 因此 ImageNet 准确率越高并不意味着科学任务越好。
3. **核心矛盾**：在 46 个现代视觉 Transformer checkpoint 上，ImageNet top-1 与生态任务的 Spearman ρ 仅 0.55（整体），>75% 时降至 0.42——即前沿模型中约 30% 的排名是反的。
4. **本文要解决什么？**
   - 构建一个统一、可复现的生态视觉基准，让研究者能在"真正重要的任务"上评估模型
   - 量化 ImageNet 作为科学 AI 代理评估的失效程度
   - 提供可推广到医学、制造等其他科学领域的 benchmark 设计蓝图
5. **切入角度**：生态学拥有丰富的公开数据和 CV4Ecology 挑战赛积累的标注任务，是理想的测试场。
6. **核心 idea 一句话**：将 9 个分散的生态视觉任务统一为 BioBench，证明 ImageNet 排行榜对科学 AI 已失去预测力，并提供 benchmark 设计的标准范式。

## 方法详解

### 整体框架

- **输入**：任意视觉模型的 frozen embedding $f: \text{image} \to \mathbb{R}^d$
- **评估**：在 9 个生态任务上拟合轻量线性/逻辑回归探针
- **输出**：每个任务的 macro-F1（FishNet/FungiCLEF 用领域特定指标），附 bootstrap 置信区间

### 关键设计

1. **任务覆盖**：
   - 做什么：统一 9 个公开生态任务
   - 核心覆盖：4 个分类界（动物/植物/真菌/原生生物）× 6 种图像模态（无人机 RGB、网络视频、显微镜、自然环境照片、标本照片、相机陷阱帧），共 310 万张图像
   - 任务类型：物种分类（iNat2021、Pl@ntNet、FungiCLEF、Herbarium19、iWildCam、Plankton）、个体重识别（BelugaID）、行为识别（KABR、MammalNet）、功能性状预测（FishNet）
   - 设计动机：覆盖不同的分类学、分布、任务类型，确保基准能暴露模型在不同维度上的弱点

2. **统一评估协议**：
   - 做什么：标准化评估流程，消除"锦标赛效应"
   - 核心思路：所有模型只需实现一个 embedding 接口 $f: \text{image} \to \mathbb{R}^d$。评估用 frozen features + 线性探针，报告 macro-F1 + bootstrap 置信区间。ViT-L 在单 A6000 GPU 上 6 小时完成全部评估
   - 设计动机：linear probe 隔离表征质量 vs 任务工程，macro-F1 对长尾类别公平

3. **预测力分析**：
   - 做什么：定量度量 ImageNet 对 BioBench 的预测能力
   - 核心思路：计算 46 个 checkpoint 上 ImageNet top-1 vs BioBench 的 $R^2$（0.34）和 Spearman ρ（0.55），以及 >75% 阈值时的 ρ（0.42）。"误排率" = $\frac{1}{2}(1-\rho) = 30\%$
   - 设计动机：不是主观声称 ImageNet 不够用，而是用统计数据严格证明

## 实验关键数据

### 主实验——46 个模型在 BioBench 上的排名

| 模型家族 | ImageNet top-1 | BioBench 均值 | 最佳任务 | 最差任务 |
|---------|---------------|-------------|---------|---------|
| SigLIP 2 (ViT-1B) | 88.9 | **43.5** | MammalNet 73.9 | BelugaID 3.6 |
| SigLIP (SO400M) | 87.8 | 42.0 | FishNet 69.0 | BelugaID 4.0 |
| BioCLIP 2 (ViT-L) | 80.0 | 41.7 | Herbarium19 73.1 | BelugaID 3.0 |
| DINOv2 (ViT-g) | 86.7 | 41.7 | FishNet 75.2 | BelugaID 4.5 |
| AIMv2 (ViT-3B) | 86.7 | 36.9 | MammalNet 68.8 | BelugaID 1.7 |
| CLIP (ViT-L) | 83.9 | 36.7 | FishNet 64.4 | BelugaID 2.8 |
| BioCLIP (ViT-B) | 58.5 | 34.3 | FishNet 62.6 | BelugaID 4.6 |

### 消融实验——ImageNet 预测力

| 分析 | $R^2$ | Spearman ρ | 误排率 |
|------|-------|-----------|--------|
| 全部 46 checkpoint | 0.34 | 0.55 | 22% |
| >75% ImageNet | — | 0.42 | **30%** |
| 单任务 (Herbarium19) | 更低 | <0.25 | >37% |

### 关键发现

- **ImageNet 在前沿区域完全失效**：>75% 准确率的模型中，ImageNet 排名与 BioBench 排名 30% 是反的——意味着基于 ImageNet 选模型有近 1/3 概率选到更差的。
- **通用模型 vs 领域模型**：BioCLIP 2（80% ImageNet）的 BioBench 均值 41.7 匹配 DINOv2（86.7% ImageNet），说明领域预训练比纯 ImageNet 指标更重要。
- **BelugaID 是所有模型的短板**：个体重识别任务对所有 frozen feature 方法都极难（3-9%），暗示当前表征缺乏个体级别的细粒度区分能力。
- **进步停滞**：Figure 3 显示大量新模型发布但 BioBench 得分未提升，只有 CLIP/SigLIP 系列真正推进了 SOTA。

## 亮点与洞察

- **"排名悬崖"现象**的严格量化：不是定性地说 ImageNet 不好，而是精确计算了误排概率 30%。这个数字本身就是一个有说服力的 argument，可推动社区关注领域特定评估。
- **"最小嵌入接口"设计哲学**：只要求模型输出 frozen embedding，极大降低了参与成本。这一设计可作为其他科学领域 benchmark 的模板。
- **macro-F1 作为默认指标**：显式奖励尾部类别性能，比 top-1 更符合生态学需求（稀有物种识别往往更重要）。
- **Benchmark 设计蓝图的方法论价值**：论文不只是贡献了一个 benchmark，更提供了 "如何为科学领域构建 benchmark" 的原则性指南——分布多样性、长尾指标、bootstrap 统计检验。

## 局限性 / 可改进方向

- **仅生态领域**：医学、制造等领域可能需要检测/分割等不同任务类型。
- **Frozen features 低估微调收益**：linear probe 隔离了表征质量，但实际部署时微调可能改变排名。
- **指标单一**：macro-F1 对某些应用不合适（如需要 precision@recall 曲线操作点的场景）。
- **缺少训练数据分析**：未分析预训练数据量/组成对 BioBench 性能的影响。
- **改进方向**：
  - 扩展到医学影像（PathBench？）和遥感（GeoBench？）
  - 加入 few-shot 和 fine-tuning 评估模式
  - 纳入检测/分割任务

## 相关工作与启发

- **vs VTAB/Taskonomy**：通用迁移学习基准包含少量科学内容，无法捕获生态学特有的挑战（细粒度分类、极端长尾、环境变异）。
- **vs iNaturalist**：iNat 提供大规模物种分类但缺少行为识别、性状预测等多样化任务。
- **vs WILDS**：WILDS 包含 iWildCam 但将生态监测视为众多领域之一，未深入探索其多面性。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个统一的生态视觉 AI 基准，"ImageNet 排名失效"的量化分析有冲击力
- 实验充分度: ⭐⭐⭐⭐⭐ 46 个模型、11 个家族、9 个任务的全面评估
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，统计分析严谨
- 价值: ⭐⭐⭐⭐⭐ 对 AI for Science 的 benchmark 方法论有深远影响，实际工具（代码+API）已可用
