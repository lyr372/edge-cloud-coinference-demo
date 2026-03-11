# Edge-Cloud CoInference Demo（Qwen 端云协同推理框架）

这个仓库给出一个**可落地扩展的框架骨架**，用于实现 Qwen 系列模型在“端侧 + 云侧”的协同推理，支持两种协同范式：

- **Token 级协同**：端侧先行生成/判断，低置信度时把上下文交给云侧继续。
- **任务级协同**：根据任务类型（如 code/reasoning/long_context）直接路由到端侧或云侧。

你可以自由指定端侧和云侧模型规模（例如 `Qwen2.5-1.5B-Instruct` + `Qwen2.5-72B-Instruct`），框架会自动解析参数量并参与策略决策。

---

## 1. 设计目标

- **一套接口，双模式协同**：统一 `CoInferenceEngine`，内部支持 token/task 两类协同策略。
- **模型可替换**：当前实现是可运行的模拟执行器；你可以平滑替换成真实的 `transformers/vLLM/TGI` 推理后端。
- **策略可演进**：路由与置信度逻辑集中在协调器，方便引入 SLA、成本、拥塞控制、A/B 策略。
- **工程可维护**：配置、模型画像、执行器、协调器、CLI 拆分清晰，便于后续服务化。

---

## 2. 目录结构

```text
src/edge_cloud_coinference/
├── __init__.py
├── cli.py            # 命令行入口
├── config.py         # 部署与运行时配置
├── coordinator.py    # 协同策略（token/task）
├── executors.py      # 端侧/云侧执行器（当前为可运行模拟）
├── models.py         # Qwen 参数量解析与模型画像
└── types.py          # 请求/响应数据结构

tests/
└── test_engine.py    # 核心协同行为测试
```

---

## 3. 快速开始

### 3.1 环境

- Python 3.10+

### 3.2 运行 Token 级协同

```bash
PYTHONPATH=src python -m edge_cloud_coinference.cli \
  --mode token \
  --prompt "请给出边缘计算场景下端云协同推理的关键指标" \
  --edge-model Qwen2.5-1.5B-Instruct \
  --cloud-model Qwen2.5-14B-Instruct
```

### 3.3 运行任务级协同

```bash
PYTHONPATH=src python -m edge_cloud_coinference.cli \
  --mode task \
  --task-type code \
  --prompt "请生成一个带重试和超时控制的异步HTTP客户端" \
  --edge-model Qwen2.5-3B-Instruct \
  --cloud-model Qwen2.5-32B-Instruct
```

输出是 JSON，包含路由结果、置信度、总时延和推理文本。

---

## 4. 核心机制说明

### 4.1 Token 级协同（`mode=token`）

1. 端侧先推理并给出置信度。  
2. 若 `edge_confidence >= threshold`，直接端侧返回。  
3. 否则触发“端 -> 云”接力，云侧继续并合并结果。  

适合：
- 日常高频、短文本、对时延敏感的请求优先留在端侧；
- 长尾复杂请求自动上云兜底。

### 4.2 任务级协同（`mode=task`）

按任务类型和输入复杂度直接路由：
- `reasoning/code/long_context/multilingual` -> 云侧
- 其他默认 -> 端侧

适合：
- 有明确业务类型标签；
- 对可解释路由策略有要求的场景。

---

## 5. 如何接入真实 Qwen 推理后端

当前 `executors.py` 是“可运行模拟器”，目的是先把协同框架打通。上线时建议按下面方式替换：

1. 保留 `BaseExecutor` 接口（`infer(prompt) -> ExecutorOutput`）。
2. 新建 `QwenHFExecutor`（transformers）或 `QwenVLLMExecutor`（vLLM）。
3. 在 `CoInferenceEngine` 初始化时注入真实执行器。
4. 把 `_simulate_confidence` 替换为：
   - token-level logprob统计、
   - 校验器分数（例如规则/判别模型）、
   - 或业务指标回归模型。
5. 把 `_simulate_latency` 替换为：
   - 实际RT（端设备 + 网络 + 云推理）采样、
   - 滑动窗口P95/P99估计。

---

## 6. 工程扩展建议

- **策略层**：增加成本感知路由（按 token 成本、GPU 队列、网络抖动动态决策）。
- **缓存层**：在端侧加入 prompt 前缀缓存和结果缓存，提高命中率。
- **观测层**：接入 tracing + metrics（route ratio、fallback ratio、P95 latency、cloud spend）。
- **安全层**：端侧先做轻量审查，违规请求直接拦截或强制上云审查。
- **多租户**：按租户级别配置不同阈值与模型组合。

---

## 7. 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

当前测试覆盖：
- token 模式下高置信直接端侧返回；
- token 模式下低置信触发端到云回退；
- task 模式下按任务类型路由云侧。
