# Converge Faster, Talk Less: Hessian-Informed Federated Zeroth-Order Optimization

**会议**: ICLR2026  
**arXiv**: [2506.02370](https://arxiv.org/abs/2506.02370)  
**代码**: 待确认  
**领域**: 优化/理论  
**关键词**: federated learning, zeroth-order optimization, Hessian preconditioning, LLM fine-tuning, communication efficiency

## 一句话总结
提出 HiSo，在联邦零阶优化中利用全局对角 Hessian 近似加速收敛，同时严格保持标量通信（不传输任何二阶信息），理论证明收敛速率独立于 Lipschitz 常数 $L$ 和模型维度 $d$，在 LLM 微调中通信轮次比 SOTA 零阶方法快 1-5 倍。

## 研究背景与动机

1. **领域现状**：联邦 LLM 微调的通信瓶颈巨大（FedAvg 需 ~1-5 TB/客户端）。DeComFL 用零阶梯度的标量表示实现维度无关通信（TB→MB）。
2. **现有痛点**：零阶 FL 收敛极慢——ZO-SGD 使用各向同性随机方向，忽略了 LLM 参数空间的异构曲率。传统 Hessian 信息需要 $O(d)$ 或 $O(d^2)$ 通信，直接违背标量通信目标。
3. **核心矛盾**：曲率信息能加速收敛但会破坏标量通信框架。如何在不传输二阶信息的前提下利用 Hessian？
4. **切入角度**：观察到全局聚合的零阶梯度标量本身可以重构 Adam 风格的对角 Hessian 近似——无需额外通信。
5. **核心idea一句话**：用 Hessian 逆平方根扭曲随机扰动方向使其沿高曲率方向更精细搜索 + 从免费的全局梯度标量计算对角 Hessian，零额外通信成本。

## 方法详解

### 整体框架
通用标量通信 FL 框架：客户端上传标量梯度→服务端聚合→客户端重构模型。HiSo 在此框架内替换 ZO-SGD 的各向同性扰动为 Hessian 信息引导的扰动 $z \sim \mathcal{N}(0, H_r^{-1})$。

### 关键设计

1. **Hessian 引导的零阶更新**:
   - 扰动方向从 $u \sim \mathcal{N}(0, I)$ 改为 $z = H_r^{-1/2} u$
   - 更新：$\Delta x = \frac{1}{\mu}[f(x + \mu z) - f(x)] \cdot z$
   - 期望值：$\mathbb{E}[\Delta x] \approx H_r^{-1} \nabla f(x)$——Newton 风格下降
   - 仍然只产生一个标量 $g$，维度无关通信保持不变

2. **零通信成本的 Hessian 学习**:
   - 对角 Hessian 通过全局聚合梯度标量的 EMA 估计：$H_{r+1} = (1-\nu)H_r + \nu \text{Diag}([\Delta x_{r}]^2 + \epsilon I)$
   - $\Delta x_r$ 可从标量重构（已用于模型重构）→零额外通信
   - 类似 Adam/RMSProp 的自适应学习率

### 理论保证
非凸设置下收敛速率独立于 $L$ 和 $d$（在低有效秩和白化假设下）——首个零阶 FL 的此类结果。支持多步本地更新。

## 实验关键数据

### 主实验（LLM 微调）

| 方法 | 通信类型 | OPT-1.3B Acc | 通信轮次比 |
|------|---------|-------------|-----------|
| FedAvg（一阶） | $O(d)$ 向量 | 基线 | 1× |
| DeComFL（ZO-SGD） | 标量 | 低于 FedAvg | 1× |
| **HiSo** | **标量** | **高于 DeComFL** | **1-5× 更快** |

### 关键发现
- HiSo 比 DeComFL 快 1-5× 达到相同精度
- 通信节省可达 9000 万倍 vs 一阶方法
- 多步本地更新有效（DeComFL 理论不支持）

## 亮点与洞察
- **"免费的午餐"**：Hessian 信息从已有的全局标量中提取，零额外通信——将约束转化为优势
- **理论突破**：首个维度无关+Lipschitz无关的零阶 FL 收敛率
- **通用标量通信框架**：解耦了标量通信与特定优化器，允许集成各种算法

## 局限性 / 可改进方向
- 对角 Hessian 是粗糙近似，非对角曲率信息被忽略
- 低有效秩假设可能不适用于所有 LLM 层
- 仅标量通信限制了每步信息量

## 相关工作与启发
- **vs DeComFL**：同为标量通信，但 DeComFL 绑定 ZO-SGD。HiSo 框架更通用且加速显著
- **vs FedAdam/FedYogi**：自适应 FL 但需 $O(d)$ 通信。HiSo 实现类似自适应效果但保持标量通信
- **vs Hessian-aware ZO**：单机设置有效。HiSo 首次将 Hessian ZO 推广到 FL 标量通信

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 标量通信+Hessian引导的完美结合，理论贡献突出
- 实验充分度: ⭐⭐⭐⭐ LLM 微调多任务验证
- 写作质量: ⭐⭐⭐⭐ 框架推导清晰
- 价值: ⭐⭐⭐⭐⭐ 对通信受限的联邦 LLM 微调有直接实用价值
