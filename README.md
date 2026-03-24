# Edge-Cloud CoInference Demo（Qwen 端云协同推理框架）

这个仓库提供一个可扩展的端云协同推理原型，并新增以下能力：

- **TEE 安全卸载（Intel TDX 模拟）**：低置信或复杂任务自动触发“端 -> 云”安全卸载，返回 attestation 元数据。
- **端侧轻量化推理底座**：支持 `MNN / ONNX Runtime` 两种端侧运行时配置，支持 `INT4/INT8` 量化与蒸馏开关。
- **规则 + 轻量分类器路由**：简单查询优先端侧，复杂任务（长文本/代码/多轮）自动切换云端。
- **本地 KV Cache 持久化与增量更新**：高频 Query 缓存落盘并优先命中，降低重复推理延迟。
- **流水线协同**：`端侧首 Token -> 轻量预判 -> 本地缓存优先 -> 云侧复杂生成/推理`。

---

## 1. 目录结构

```text
src/edge_cloud_coinference/
├── __init__.py
├── cache.py          # 本地持久化 KV Cache
├── cli.py            # 命令行入口
├── config.py         # 部署与运行时配置
├── coordinator.py    # 路由策略与协同编排
├── executors.py      # 端侧/云侧执行器（模拟）
├── models.py         # Qwen 参数量解析
└── types.py          # 请求/响应数据结构

tests/
└── test_engine.py
```

---

## 2. 快速开始

### 2.1 Token 协同 + Intel TDX 安全卸载

```bash
PYTHONPATH=src python -m edge_cloud_coinference.cli \
  --mode token \
  --prompt "请分析该系统在复杂负载下的调度行为" \
  --edge-model Qwen2.5-1.5B-Instruct \
  --cloud-model Qwen2.5-14B-Instruct \
  --edge-runtime onnxruntime \
  --quantization int8
```

### 2.2 Task 协同 + MNN 端侧轻量化

```bash
PYTHONPATH=src python -m edge_cloud_coinference.cli \
  --mode task \
  --task-type general \
  --prompt "请根据上文继续完成代码重构并解释设计" \
  --edge-runtime mnn \
  --quantization int4
```

### 2.3 查看缓存落盘

默认缓存路径：`.cache/kv_cache.json`。

---

## 3. 协同策略

### 3.1 Token 模式

1. 端侧先返回首 Token 时延指标。  
2. 若置信度高于阈值：端侧直接返回。  
3. 若置信度不足：触发云侧，并在 metadata 标注 `secure_offload=intel_tdx:attested`。  

### 3.2 Task 模式（规则 + 分类器）

- **规则路由**：`reasoning/code/long_context/multi_turn` 默认云侧。
- **轻量分类器**：基于输入长度、代码特征、多轮上下文关键词估算复杂度。
- **决策**：短输入 + 低复杂度留在端侧，其余走云侧。

### 3.3 KV Cache

- 请求到达先查本地 KV Cache；命中则直接返回（`route=edge_cache`）。
- 未命中时执行推理并增量更新缓存。

---

## 4. 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
