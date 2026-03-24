# CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation

**会议**: ICLR 2026  
**arXiv**: [2601.19178](https://arxiv.org/abs/2601.19178)  
**代码**: 待确认  
**领域**: 推荐系统 / 模型压缩  
**关键词**: KV缓存压缩, 跨用户共享, 协同信号, 序列推荐, SVD分析

## 一句话总结
观察到序列推荐中不同用户的 KV cache 具有显著跨用户相似性（协同信号），提出 CollectiveKV 将 KV 分解为低维用户特有部分和从全局 KV 池检索的高维共享部分，实现 0.8% 的压缩率且性能不降。

## 研究背景与动机

1. **领域现状**：序列推荐模型（SIM、HSTU 等）采用 Transformer 注意力机制提升性能，为降低推理延迟引入了 KV cache 技术预计算并缓存 K/V。
2. **现有痛点**：推荐系统用户基数庞大（亿级），每个用户可能有很长的行为历史，KV cache 总量很快超过 GPU 显存容量，必须卸载到 CPU/外存，引入巨大传输延迟。
3. **核心矛盾**：LLM 的 KV 压缩方法（如 token 裁剪、MLA 降维）只压缩单用户序列，忽视了推荐场景独有的跨用户协同信号。
4. **本文要解决什么**：利用跨用户 KV 相似性实现极致压缩——把大部分信息放入全局共享池，每用户只存极低维度的个性化 KV。
5. **切入角度**：通过 SVD 分解 K/V，发现主成分（>90% 信息）跨用户相关性强，残差（<10% 信息）是用户特有的——这给出了"什么可以共享"的定量依据。
6. **核心 idea 一句话**：用可学习的全局 KV 池存储跨用户共享信息，每用户仅缓存低维个性化 KV + 全局索引，实现 0.8% 极端压缩率。

## 方法详解

### 整体框架
分 prefill 和 decode 两阶段：prefill 阶段将用户序列线性投影为低维用户特有 KV（$d_u$ 维度），同时通过 router 网络计算全局 KV 索引并缓存。decode 阶段从缓存取出索引，从 GPU 常驻的全局 KV 池检索高维共享 KV（$d_g$ 维度），拼接后计算注意力。

### 关键设计

1. **KV 分解：用户特有 + 集体共享**:
   - 做什么：将 KV 分为低维 $\mathbf{K}_u \in \mathbb{R}^{n \times d_u}$ 和高维 $\mathbf{K}_c \in \mathbb{R}^{n \times d_g}$
   - 核心思路：$\mathbf{K}_u = \mathbf{S} W_k + b_k$（线性投影降维），$\mathbf{K}_c[i] = P_k[\mathbf{I}_k[i]]$（从全局池按索引检索），最终拼接 $\mathbf{K} = \text{concat}(\mathbf{K}_u, \mathbf{K}_c)$
   - 设计动机：SVD 分析表明主成分可跨用户共享、残差是个性化的，故用共享池承载高维主信息，低维投影保留个性化

2. **CollectiveKV Router**:
   - 做什么：将序列 embedding 映射为每个 item 的全局 KV 池索引
   - 核心思路：$\mathbf{M} = \mathbf{S} W_r + b_r$，$\mathbf{I}_k[i] = \arg\max_j \mathbf{M}_{ij}$。训练时用 sigmoid 门控保证梯度可传播：$\mathbf{K}_c[i] = \sigma(\mathbf{M}[i, \mathbf{I}_k[i]]) \cdot P_k[\mathbf{I}_k[i]]$
   - 设计动机：argmax 不可微，sigmoid 门控+peak loss 确保训练推理一致性

3. **全局 KV 池**:
   - 做什么：$P_k, P_v \in \mathbb{R}^{m \times d_g}$ 常驻 GPU 显存，所有用户共享
   - 设计动机：池大小 $m$ 远小于用户数 × 序列长度，极大减少存储；高维 $d_g$ 保证信息容量

### 损失函数 / 训练策略
- 原始推荐损失 + peak loss $\mathcal{L}_{\text{peak}} = -\frac{1}{n}\sum_i \log\sigma(\mathbf{M}[i, \mathbf{I}_k[i]])$（保证 sigmoid 输出接近 1）
- load balance loss（KL 散度使池中每个 key 被均匀选择）
- 端到端训练，pool/router/投影层联合优化

## 实验关键数据

### 主实验（5 模型 × 3 数据集）

| 模型 | 数据集 | GAUC（原始→+ours） | AUC（原始→+ours） | 压缩率 CR |
|------|--------|------------------|------------------|---------|
| SIM | MicroVideo | 0.6954→**0.6973** | 0.6933→**0.7057** | **1.6%** |
| SDIM | MicroVideo | 0.6857→**0.6883** | 0.6749→**0.6871** | **1.2%** |
| SIM | KuaiVideo | 0.6577→**0.6604** | 0.6798→**0.6900** | **1.2%** |
| HSTU | MicroVideo | - | - | **0.8%** |

### 消融实验

| 配置 | AUC | 说明 |
|------|-----|------|
| 完整 CollectiveKV | 0.7057 | 最佳 |
| 仅用户特有 KV | ~0.69 | 缺少共享信息 |
| 仅集体 KV | ~0.69 | 缺少个性化 |
| 无 peak loss | ~0.70 | 训练推理不一致 |
| 无 balance loss | ~0.70 | 池利用率低 |

### 关键发现
- **0.8% 压缩率不降反升**：5 个模型 × 3 个数据集上成绩持平或提升，说明共享 KV 起到了正则化/信息增强效果
- SVD 分析提供了可解释的压缩依据——主成分跨用户强相关、残差用户特有
- 推理延迟大幅降低——外存传输量缩小 50-100x，GPU 内索引操作延迟可忽略

## 亮点与洞察
- **跨用户 KV 共享是推荐系统独有的压缩维度**：LLM 的 KV 压缩无此维度（每次推理只服务一个序列），但推荐系统天然具有协同信号——这是一个被忽视但潜力巨大的方向
- **SVD 分解提供了"什么能共享"的理论分析工具**：主成分 vs 残差的跨用户相似度对比非常直观有说服力
- **router 设计的 sigmoid 门控+peak loss** 优雅解决了离散索引不可微的问题

## 局限性 / 可改进方向
- 全局 KV 池常驻 GPU 显存，池大小 $m$ 不能太大——大规模场景的 $m$ 如何选择？
- 仅验证了 CTR 预测任务，未在排序/生成式推荐上验证
- router 采用简单线性层，更复杂的路由策略是否能进一步提升？

## 相关工作与启发
- **vs MLA (DeepSeek)**：MLA 降维压缩单用户 KV；CollectiveKV 利用跨用户共享实现更极端压缩
- **vs Token pruning (Loki/Quest)**：裁剪 token 丢弃信息；CollectiveKV 不丢弃，而是将信息转移到共享池
- **vs HSTU**：HSTU 引入 KV cache 到推荐但未压缩；CollectiveKV 在此基础上实现 0.8% 压缩

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 跨用户 KV 共享是全新视角，SVD 分析提供理论支撑
- 实验充分度: ⭐⭐⭐⭐ 5 模型 × 3 数据集覆盖广，但缺少更多消融细节
- 写作质量: ⭐⭐⭐⭐ SVD 分析可视化清晰，整体逻辑通顺
- 价值: ⭐⭐⭐⭐⭐ 0.8% 压缩率有巨大工业部署价值
