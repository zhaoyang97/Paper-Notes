# Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems

**会议**: ICLR 2026  
**arXiv**: [2509.20989](https://arxiv.org/abs/2509.20989)  
**代码**: [GitHub](https://github.com/BDML-lab/RCE-KD)  
**领域**: 推荐系统 / 知识蒸馏 / 模型压缩  
**关键词**: knowledge distillation, cross-entropy, NDCG, recommender system, ranking, partial NDCG

## 一句话总结
理论证明 CE 损失在推荐系统 KD 中最大化 NDCG 下界需满足"闭合性假设"（子集需包含学生 top 项目），但实际目标是蒸馏教师 top 项目的排序——两者冲突导致 vanilla CE 表现差。据此提出 RCE-KD：将教师 top-K 项目按是否在学生 top-K 中分两组，分别用精确 CE 和采样近似闭合性 CE，自适应融合权重随训练动态调整。

## 研究背景与动机
1. **领域现状**：知识蒸馏在推荐系统中用于将大教师模型压缩为小学生模型。Response-based KD（CE 损失、RRD、CD 等）是主流。CE 损失在 CV/NLP 的 KD 中极为成功。
2. **现有痛点**：
   - CE 损失在推荐 KD 中表现**出人意料地差**——在 MF→MF、LightGCN→LightGCN、HSTU→HSTU 三种设置中，vanilla CE 一致劣于所有基线（CD、RRD、HetComp 等）
   - 推荐 KD 有两个独特特点：（1）关注排序而非精确分数，尤其是教师 top 项目的排序，（2）由于项目集极大（百万级），CE 只能在小子集上计算
   - 已有的 CE-NDCG 理论连接仅适用于二值标签和全项目场景，不覆盖推荐 KD 的实际设定
3. **核心矛盾**：CE 约束 partial NDCG 需"闭合性假设"——子集必须包含学生排名最高的项目。但 KD 的目标是蒸馏教师 top 项目的排序，而学生和教师的 top 项目在训练初期几乎不重叠。
4. **核心 idea**：将教师 top-K 分裂为两组（与学生 top-K 交集 / 差集），对第一组在学生 top-K 上直接算 CE（精确满足闭合性），对第二组用自适应采样策略近似满足闭合性

## 方法详解

### 整体框架
预训练大教师 + 小学生 → 获取教师预测分数 $\mathbf{r}_u^T$ → 将教师 top-K $\mathcal{Q}_u^T$ 分裂为 $(\mathcal{Q}_u^T)_1 = \mathcal{Q}_u^T \cap \mathcal{Q}_u^S$ 和 $(\mathcal{Q}_u^T)_2 = \mathcal{Q}_u^T \setminus (\mathcal{Q}_u^T)_1$ → 分别计算 $\mathcal{L}_1$（精确）和 $\mathcal{L}_2$（采样近似）→ 自适应融合 → 联合 base loss 训练学生

### 关键设计

1. **理论基础：CE → NDCG 在 KD 中的推广**
   - 做什么：证明 CE 损失与 NDCG 在推荐 KD 场景中的连接
   - **定理 4.1（全项目 KD）**：在全项目集上最小化 CE 等价于最大化 NDCG 的下界，其中相关性分数 $y_i = \log_2(\sigma(r_{ui}^T) + 1)$。这为推荐 KD 使用 CE 提供了理论动机
   - **定理 4.4（部分项目 KD）**：在子集 $\mathcal{J}^u$ 上最小化 CE 约束 partial NDCG，但前提是 $\mathcal{J}^u$ 满足**闭合性假设（Assumption 4.3）**：子集中每个项目，学生排名更高的所有项目也必须在子集中
   - 设计动机：全项目 KD 不实际（项目太多），但部分项目 KD 的理论保证需要闭合性——这揭示了 vanilla CE 为何失败

2. **分裂策略**
   - 做什么：将教师 top-K 按是否也在学生 top-K 中分为两组
   - $(\mathcal{Q}_u^T)_1 = \mathcal{Q}_u^T \cap \mathcal{Q}_u^S$：教师和学生都认为重要的项目 → 在 $\mathcal{Q}_u^S$ 上直接计算 $\mathcal{L}_1$（$\mathcal{Q}_u^S$ 天然满足闭合性）
   - $(\mathcal{Q}_u^T)_2 = \mathcal{Q}_u^T \setminus (\mathcal{Q}_u^T)_1$：教师认为重要但学生排名低的项目 → 需要采样策略近似满足闭合性
   - 设计动机：直接在 $\mathcal{Q}_u^T$ 上算 CE 不满足闭合性；直接加学生 top 项目会导致子集过大。分裂后分别处理更高效

3. **自适应采样策略（for $\mathcal{L}_2$）**
   - 做什么：对 $(\mathcal{Q}_u^T)_2$ 中的每个项目 $i$，找到学生排名高于 $i$ 的所有项目并提高其采样概率，采样 $L$ 个与 $(\mathcal{Q}_u^T)_2$ 合并
   - 核心思路：采样概率 $p_j \propto e^{z_j/\tau}$，$z_j$ 是学生排名高于 $(\mathcal{Q}_u^T)_2$ 中某项目的累计计数。当学生对教师 top 项目排名低时，近似均匀采样覆盖更多项目；随训练进行学生提升排名后，采样集中在关键项目上
   - 设计动机：精确满足闭合性需加入太多项目；自适应采样在效率和精度间取得平衡

4. **自适应损失融合**
   - 融合权重 $\gamma = \exp(-\beta \cdot |(\mathcal{Q}_u^T)_1| / |\mathcal{Q}_u^T|)$，每 epoch 更新
   - 交集小时（学生还没学好）→ $\gamma$ 大 → 重点推学生提升 $(\mathcal{Q}_u^T)_2$ 中项目的排名
   - 交集大时（学生已经学得不错）→ $\gamma$ 小 → 重点精细化 $(\mathcal{Q}_u^T)_1$ 中的排序

### 损失函数 / 训练策略
- 总损失：$\mathcal{L} = \mathcal{L}_{Base} + \lambda \cdot \mathcal{L}_{RCE-KD}$，其中 $\mathcal{L}_{RCE-KD} = (1-\gamma)\mathcal{L}_1 + \gamma \mathcal{L}_2$
- 教师预测预先保存，训练时只加载不重新推理教师
- 采样 $\tau=10$ 固定，每 epoch 重新采样

## 实验关键数据

### 主实验（三数据集 × 三 backbone × 两指标）

| 数据集 | backbone | 方法 | Recall@20 | NDCG@20 |
|--------|----------|------|-----------|---------|
| CiteULike | MF→MF | CD | 基线 | 基线 |
| | | RRD | 改进 | 改进 |
| | | HetComp | 次优 | 次优 |
| | | **RCE-KD** | **最优** | **最优** |
| Gowalla | 同上 | **RCE-KD** | **最优** | **最优** |
| Yelp | 同上 | **RCE-KD** | **最优** | **最优** |

RCE-KD 在所有 9 种设置（3 数据集 × 3 backbone：MF/LightGCN/HSTU）上一致最优，统计显著（p ≤ 0.05）。学生性能可接近甚至匹配教师。

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 仅 $\mathcal{L}_1$（学生 top 项目上的 CE） | 优于 vanilla CE | 满足闭合性的好处 |
| 仅 $\mathcal{L}_2$（采样近似闭合性） | 优于 vanilla CE | 近似闭合性生效 |
| $\mathcal{L}_1 + \mathcal{L}_2$ 固定权重 | 不如自适应 | 自适应 $\gamma$ 的必要性 |
| **Full RCE-KD** | **最优** | 分裂+采样+自适应缺一不可 |

### 训练效率
| 方法 | 相对 Student 训练时间 | GPU 内存 |
|------|---------------------|---------|
| RCE-KD | ~1.1-1.3× | 与 CE 相当 |
| RRD | ~2-3× | 显著更高 |
| HetComp | ~2-4× | 显著更高 |

RCE-KD 仅在 CE 基础上增加随机采样开销，训练效率最高。

### 关键发现
- Vanilla CE 在推荐 KD 中表现差的根本原因是闭合性假设不满足——训练初期学生和教师 top 项目的重叠率极低（~10-20%）
- 训练过程中 NDCG 演化可视化验证了 RCE-KD 成功约束了 NDCG（符合理论预期）
- RCE-KD 的泛化性：在序列推荐和多模态推荐中也有效

## 亮点与洞察
- **闭合性假设的发现**是最有价值的贡献——它精确解释了一个令人困惑的实验现象（CE 在推荐 KD 中为何差），且直接指导了算法设计
- **理论驱动 → 算法设计**的范式很优雅：先分析 CE 的理论适用条件 → 识别实际场景的违反 → 设计算法修复违反
- **自适应 $\gamma$ 调度**巧妙地利用了学生和教师 top 项目重叠度作为训练进度的代理指标

## 局限性 / 可改进方向
- 闭合性假设的近似程度缺乏理论量化——采样 $L$ 个项目能多好地近似闭合性？
- 采样温度 $\tau=10$ 固定，可能在不同数据集上不是最优
- 仅在隐式反馈的排序任务上验证，是否适用于评分预测等其他推荐任务未知
- partial NDCG 只关注子集内排序，不保证子集外项目的排列质量

## 相关工作与启发
- **vs RRD (Kang 2020)**：RRD 用 ListMLE 损失蒸馏，不分析 CE 的理论基础；RCE-KD 基于 CE 的理论分析设计，训练更高效
- **vs CD (Lee 2019)**：CD 用 point-wise loss 对齐预测，效果一般；本文表明 list-wise CE 在正确使用时更优
- **vs CE-NDCG 理论 (Bruch 2019)**：只适用于二值标签和全项目场景；本文推广到连续标签（教师分数）和部分项目 KD

## 评分
- 新颖性: ⭐⭐⭐⭐ 闭合性假设的理论发现新颖，分裂+采样+自适应融合的设计由理论驱动
- 实验充分度: ⭐⭐⭐⭐⭐ 3 数据集 × 3 backbone × 多 KD 设置 + 充分消融 + 效率对比 + 泛化验证
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，从问题→理论→方法→实验的逻辑链完整
- 价值: ⭐⭐⭐⭐ 为推荐 KD 中 CE 的使用提供了理论指导和实践方法
