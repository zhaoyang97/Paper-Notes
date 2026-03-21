# AID-AppEAL: Automatic Image Dataset and Algorithm for Content Appeal Enhancement and Assessment Labeling

**会议**: ECCV 2024  
**arXiv**: [2407.05546](https://arxiv.org/abs/2407.05546)  
**代码**: https://github.com/SherryXTChen/AID-Appeal (有)  
**领域**: 图像质量评估 / 推荐系统  
**关键词**: Image Content Appeal, Aesthetics Assessment, Dataset Creation, Stable Diffusion, Textual Inversion

## 一句话总结
首次提出图像内容吸引力评估（ICAA）任务，区别于传统美学评估（IAA），设计了一套自动化数据集生成 + 吸引力估计 + 吸引力增强的完整 pipeline，用 Stable Diffusion + Textual Inversion 实现零人工标注的大规模数据集构建。

## 研究背景与动机
1. **领域现状**：图像质量评估已有 IQA（失真质量）和 IAA（美学质量）两个成熟方向。但"内容吸引力"这个维度被忽视了——一张拍得很专业（高美学分）的发霉汉堡照片，在内容上其实毫无吸引力。

2. **现有痛点**：
   - 现有 IAA 方法（DIAA、MPADA、NIMA）会给专业拍摄的低吸引力内容打高分，混淆了"拍得好"和"内容吸引人"
   - 没有专门的 ICAA 数据集——现有数据集只有"interesting content"等粗粒度标签
   - 人工标注大规模图像评估数据集极其昂贵（如 IQA 数据集标注需要大量人力）

3. **核心矛盾**：需要构建大规模 ICAA 数据集来训练模型，但人工标注成本不现实；且"内容吸引力"这个概念需要与"美学"严格区分

4. **本文要解决什么？** (a) 定义并形式化 ICAA 任务；(b) 自动化构建 ICAA 数据集；(c) 训练吸引力评估和增强模型

5. **切入角度**：利用 Stable Diffusion 的 Textual Inversion 学习"吸引/不吸引"的嵌入表示，通过线性插值控制吸引力等级，生成连续梯度的合成数据

6. **核心 idea 一句话**：用 Textual Inversion 学习 appealing/unappealing 的文本嵌入，通过 $\alpha$ 线性插值控制合成图像的吸引力等级，训练 Siamese 比较器自动标注真实图像，再训练绝对分数估计器

## 方法详解

### 整体框架
Pipeline 分四步：(1) 自动搜索收集领域图像（如食物/房间）；(2) 用 Textual Inversion 学习吸引力嵌入，SD inpainting 生成不同吸引力等级的合成数据；(3) 训练相对吸引力比较器（Siamese CLIP），自动标注大规模真实图像；(4) 训练绝对吸引力估计器 + 吸引力增强器

### 关键设计

1. **Domain-Relevancy Map 生成**:
   - 做什么：精确定位图像中与领域相关的区域（如食物区域），只在这些区域操作吸引力
   - 核心思路：BLIP 生成图像描述 → NLTK 提取名词短语 → WordNet 匹配领域词（如 noun.food）→ CLIPSeg 分割出领域相关区域 $M_D(I)$
   - 设计动机：确保只修改内容区域不改背景，避免背景对吸引力评估的干扰

2. **Textual Inversion 嵌入 + 合成数据**:
   - 做什么：学习代表"appealing"和"unappealing"的文本嵌入向量 $z_D^+$ 和 $z_D^-$
   - 核心思路：从搜索结果中选最匹配的正/负图像集，用 Textual Inversion 学习嵌入。合成时通过 $f(\alpha) = \alpha z_D^+ + (1-\alpha)z_D^-$ 插值控制吸引力等级，用 SD inpainting 只修改 $M_D(I)$ 区域：$I' = SD(I, BLIP(I) + f(\alpha), M_D(I), seed())$
   - 同时随机化背景以增加多样性：$I'' = SD(I', \text{" "}, 1-M_D(I), seed())$
   - 设计动机：线性插值假设简单但有效——嵌入空间中的中间点对应中间吸引力

3. **相对吸引力比较器 → 自动标注**:
   - 做什么：训练 Siamese CLIP 网络预测两张图像的吸引力差值，然后用投票机制给真实图像打绝对分
   - 核心思路：训练集是同一源图的不同 $\alpha$ 变体对，标签为 $\alpha_1 - \alpha_2$。标注时每张图与一组 exemplar 图像比较，取平均得分，缩放到 1-10
   - 设计动机：相对比较比绝对打分更容易学——这与人类判断也一致

4. **吸引力增强（Content Appeal Enhancement）**:
   - 做什么：定位图像中不吸引人的区域并增强
   - 核心思路：用滑动窗口生成吸引力热图 $M_D^H(I)$（低吸引力区域权重高），然后 SD inpainting 用 $z_D^+$ 嵌入在热图引导下增强：$SD(I, BLIP(I) + z_D^+, M_D^H(I), seed())$
   - 设计动机：只增强需要改善的区域，避免过度修改

### 损失函数 / 训练策略
- 相对比较器：MAE 损失 $|A_{pred}(I_1, I_2) - (\alpha_1 - \alpha_2)|$，CLIP backbone 两阶段训练
- 绝对估计器：MAE 损失 $|A_{pred}(I) - A(I)|$，同样两阶段
- 增强用 SD v2.1 inpainting + depth-guided ControlNet

## 实验关键数据

### 主实验：ICAA 与 IAA 无关联

| 指标 | DIAA (Food) | MPADA (Food) | NIMA (Food) | DIAA (Room) | MPADA (Room) | NIMA (Room) |
|------|-------------|--------------|-------------|-------------|--------------|-------------|
| PLCC | 0.168 | 0.005 | 0.01 | -0.123 | -0.012 | -0.147 |
| SRCC | 0.162 | -0.015 | 0.003 | -0.121 | -0.017 | -0.149 |

→ 内容吸引力与美学分数几乎零相关，验证了 ICAA 是独立于 IAA 的新维度

### 用户研究

| 问题 | Food | Room |
|------|------|------|
| 增强后内容更吸引人 | 76.3% | 79.2% |
| 增强后图像更真实 | 65.8% | 72.1% |

### 消融实验

| 数据集 | 图像数 | 估计器 MAE |
|--------|--------|-----------|
| Food (𝕀_F) | 78,917 | 0.6756 |
| Room (𝕀_R) | 75,287 | 0.6332 |

### 关键发现
- **ICAA 和 IAA 确实是正交的**：相关系数接近零，甚至负相关（Room 领域的 NIMA PLCC=-0.147）
- **自动标注pipeline有效**：在 1-10 分范围内 MAE<0.7，考虑到任务主观性这是可接受的
- **76%+ 用户偏好增强图像**：证明吸引力估计匹配人类感知
- **跨领域可泛化**：同一 pipeline 适用于 food 和 room 两个差异很大的领域

## 亮点与洞察
- **提出了一个新的评估维度**：将"content appeal"与"aesthetics"明确分离，填补了图像质量评估体系的空白。这个 insight 对电商、外卖平台、酒店图片等应用场景有直接价值
- **全自动零标注 pipeline**：从搜索查询→合成数据→自动标注→训练，整个过程无需人工标注。核心巧思是用 Textual Inversion 学习吸引力的连续表示
- **Domain-relevancy map 思路**：BLIP+WordNet+CLIPSeg 的组合精确定位领域相关内容，确保只操作与吸引力相关的区域，很实用

## 局限性 / 可改进方向
- **线性插值假设可能过于简化**：嵌入空间中的线性路径不一定对应吸引力的线性变化
- **仅测试了 food 和 room 两个领域**：对人像、服装等更复杂领域的泛化需要验证
- **Textual Inversion 的表达能力有限**：单个嵌入向量可能无法捕捉所有维度的"吸引/不吸引"
- **背景影响被忽略**：论文假设背景不影响内容吸引力，但实际上餐桌环境会影响食物吸引力
- **Base images 来自 stock 网站的低分辨率缩略图**：经 ESRGAN 超分至 512×512，图像质量可能不稳定

## 相关工作与启发
- **vs NIMA/DIAA/MPADA**: 这些 IAA 方法评估的是美学（构图、光线），本文评估的是内容吸引力——两者几乎没有相关性，不能互相替代
- **vs CrowdCLIP/AFreeCA**: 有趣的类比——这两篇用 SD 合成数据做计数（上一篇笔记），本文用 SD 合成数据做吸引力评估，都是"用生成模型造数据训判别模型"的范式

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次形式化 ICAA 问题，全自动 pipeline 设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 有用户研究、跨领域验证、IAA 对比，但消融偏少
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，pipeline 逻辑流畅，图表直观
- 价值: ⭐⭐⭐⭐ 打开了内容吸引力评估的新方向，对商业应用有直接价值
