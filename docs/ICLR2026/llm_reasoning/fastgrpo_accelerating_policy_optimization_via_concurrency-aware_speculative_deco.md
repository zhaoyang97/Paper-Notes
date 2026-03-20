# FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning

**会议**: ICLR2026  
**arXiv**: [2509.21792](https://arxiv.org/abs/2509.21792)  
**代码**: [GitHub](https://github.com/yedaotian9/GRPO_speculative)  
**领域**: llm_reasoning  
**关键词**: GRPO加速, 投机解码, 并发感知, 在线draft学习, 强化学习训练效率  

## 一句话总结
针对GRPO训练中生成阶段占91%-98%时间的瓶颈，提出并发感知的投机解码策略（动态调整draft树大小）和在线draft模型学习（持续适配目标模型分布），实现2.35x-2.72x端到端加速。

## 背景与动机
1. GRPO是提升LLM推理能力的重要RL框架，但训练极慢，生成阶段占总时间91%-98%
2. 标准投机解码在高并发场景下加速效果有限，甚至可能减速（speedup<1.0x）
3. GRPO生成阶段的有效并发度动态变化：从高batch size逐渐降到接近1（序列长度不均）
4. 训练过程中目标模型持续更新，与固定draft模型的分布差距逐渐增大，接受率下降
5. 现有投机解码方法(EAGLE-2/HASS/EAGLE-3)在GRPO中仅获1.1x-1.3x加速
6. 低奖励方差的rollout导致数据浪费，进一步加重推理成本

## 方法详解
**并发感知投机解码**：核心思想是让验证阶段的有效batch size始终等于硬件最优并发度$C_{\text{peak}}$。
- 验证token数：$N_{\text{verify}} = C_{\text{peak}} / B$，随batch size B降低而增大
- Draft扩展宽度：$K_{\text{draft}} = \min(N_{\text{verify}}-1, K_{\text{draft}}^{\max})$
- Draft扩展深度：$L_{\text{draft}} = \min(\lfloor\log_2(N_{\text{verify}}/\alpha)\rfloor, L_{\text{draft}}^{\max})$，α编码draft模型质量
- 效果：高并发时保守投机避免计算瓶颈，低并发时激进投机最大化加速

**在线Draft学习**：在GRPO每轮迭代中，用目标模型当前生成的响应(含hidden states)更新draft模型，使其持续对齐目标模型分布。额外计算开销仅2%-3%（因hidden states在生成阶段已自然产生）。

## 实验关键数据
| 模型 | 方法 | GSM8K E2E SR | SimpleRL E2E SR | DAPO E2E SR | 平均 E2E SR |
|------|------|-------------|----------------|------------|------------|
| Qwen2.5-7B-I | EAGLE-3 | 1.26x | 1.20x | 1.13x | 1.20x |
| Qwen2.5-7B-I | **FastGRPO** | **2.43x** | **2.52x** | **2.53x** | **2.49x** |
| Llama3.1-8B-I | EAGLE-3 | 1.31x | 1.28x | 1.23x | 1.27x |
| Llama3.1-8B-I | **FastGRPO** | **2.51x** | **2.69x** | **2.67x** | **2.62x** |

- 在线draft学习贡献约0.7x-0.9x的额外生成加速比
- 训练准确率与标准GRPO基本一致(加速不损害训练质量)
- 5个模型×3个数据集全面验证

## 亮点
- 发现并利用GRPO生成阶段并发度动态变化的特性，设计自适应策略
- 理论分析operational intensity连接硬件特性与投机解码超参
- 在线draft学习几乎零额外开销（利用已有hidden states）
- 相比最强baseline(EAGLE-3)提升约2x，实际可部署性强

## 局限性 / 可改进方向
- $C_{\text{peak}}$需要针对每种GPU/模型组合做empirical profiling
- 仅在数学推理任务上验证，未测试代码/通用推理等场景
- Draft模型架构固定为EAGLE系列，未探索其他draft方案
- α超参需要手动调节
- 未讨论多节点分布式训练场景下的效果

## 与相关工作的对比
- EAGLE-2/HASS/EAGLE-3在GRPO中仅1.1x-1.3x，FastGRPO达2.4x-2.7x
- 关键差异：考虑了GRPO特有的动态并发变化(非静态推理场景)
- 在线draft学习比离线预训练draft模型保持更好的接受率(acceptance length持续上升vs下降)

## 评分
- 新颖性: ⭐⭐⭐⭐ (并发感知+在线学习的组合方案针对性强)
- 实验充分度: ⭐⭐⭐⭐⭐ (5模型3数据集，消融充分)
- 写作质量: ⭐⭐⭐⭐ (动机-观察-方法逻辑清晰)
- 价值: ⭐⭐⭐⭐⭐ (直接降低GRPO训练成本，实用性极高)
