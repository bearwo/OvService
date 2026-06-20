# OvService — 智能对话平台开发文档

## 项目概述

基于 OpenVINO GenAI 构建的本地智能对话平台，支持多模态交互（文本、图片），通过 CLI 和 API 两种方式提供服务。核心特点是完全本地化运行，数据不离开本机，同时提供 OpenAI 兼容 API 接口，可接入 Codex 等第三方工具。

### 核心能力
- **多模态对话**：文本聊天 + 图片理解（VLM）
- **动态上下文管理**：自动检测模型上下文长度，智能压缩历史对话
- **分层记忆系统**：对话摘要 → 数据库压缩 → 长期记忆注入
- **乱码检测与恢复**：自动检测输出异常，截断上下文重试
- **OpenAI 兼容 API**：标准 `/v1/chat/completions` 端点，支持 Codex 接入

## 技术栈

| 层 | 技术 | 说明 |
|---|------|------|
| 推理引擎 | OpenVINO GenAI (VLMPipeline) | Intel GPU 加速推理 |
| 当前模型 | Qwen3.6-35B-A3B-int4-ov | MoE 架构，256K 上下文 |
| 后端框架 | FastAPI + Uvicorn | 异步 API 服务 |
| CLI 交互 | Rich + Prompt Toolkit | 终端美化 + 流式输出 |
| 数据存储 | SQLite | 对话历史、知识库、摘要 |
| 文件处理 | Pillow + PyMuPDF | 图片加载 + PDF 解析 |
| 硬件 | Intel Arc B390 (24GB UMA) | 统一内存架构 GPU |

## 项目结构

```
D:\AISpace\Workspace\OvService\
├── app.py                  # FastAPI 入口 + OpenAI 兼容路由
├── cli.py                  # 命令行入口（Rich + 流式输出）
├── config.py               # 配置管理（模型路径、环境变量、上下文阈值）
├── core/
│   ├── base.py             # 模型适配器基类 + 生成配置
│   ├── engine.py           # 模型调度引擎（单例）
│   ├── conversation.py     # 多轮对话管理 + 动态压缩
│   ├── stream.py           # 流式输出处理
│   └── optimize.py         # 性能优化（PerfStats、LRU、推理参数）
├── adapters/
│   └── chat.py             # Qwen3.6 VLMPipeline 适配器
├── features/
│   ├── memory.py           # 对话历史 + 分层摘要 + 数据库压缩
│   ├── knowledge.py        # 知识库（文件导入、检索、上下文注入）
│   ├── image.py            # 图片处理（OpenVINO Tensor）
│   └── file_parser.py      # 文件解析（PDF/TXT/MD）+ 文本分块
├── api/
│   ├── routes.py           # 业务 API 路由
│   ├── openai_compat.py    # OpenAI 兼容 API（/v1/chat/completions）
│   ├── schemas.py          # Pydantic 请求/响应模型
│   ├── session.py          # 会话管理（隔离、超时清理）
│   └── task_queue.py       # 任务队列（排队、取消）
├── data/
│   ├── conversations.db    # SQLite 数据库
│   ├── knowledge/          # 知识库存储
│   └── uploads/            # 上传文件暂存
├── requirements.txt
└── .gitignore
```

## 核心架构设计

### 1. 多模型调度架构

```
用户请求 → CLI/API 路由
  ├── "聊天/问答" → ChatAdapter (Qwen3.6 VLMPipeline)
  ├── "图片理解"  → ChatAdapter + images 参数
  ├── "知识库问答" → 知识库检索 + ChatAdapter
  └── "生成图片"  → image_gen 适配器（预留）
```

**适配器模式**：每个模型实现 `BaseModelAdapter` 接口，支持 `load/unload/generate/generate_stream`。

**单例引擎**：`ModelEngine` 管理所有适配器的生命周期，同一模型全局只有一个实例。

### 2. 动态上下文管理

```
对话进行中
  ↓
达到 33% 上下文 (自动检测 max_position_embeddings)
  ↓
自动生成摘要 → 存入 DB (level=0)
  ↓
裁剪对话保留最近 4 条
  ↓
加载摘要作为系统上下文
  ↓
继续对话
```

**数据库压缩**：当摘要累积 ≥ 200 条 OR ≥ 50% 上下文时，合并多条摘要为 1 条精简摘要 (level=1)。

### 3. 乱码检测与恢复

```
生成输出 → _is_garbled() 检测
  ├─ 正常 → 返回
  └─ 乱码 → 截断为最近 4 条消息 → 重新生成 → 返回
```

**检测标准**：字母比例 < 40% 或特殊字符密度 > 10%。

### 4. 流式输出实现

```
ChatAdapter.generate_stream()
  → threading.Thread 启动 generate()
  → _TextCollector 逐 token 接收
  → sys.stdout.write() + flush 实时输出
  → 返回完整文本
```

**Windows 兼容**：需要 `python -u` 和 `PYTHONUNBUFFERED=1` 环境变量。

### 5. OpenAI 兼容 API

| 端点 | 说明 |
|------|------|
| `GET /v1/models` | 列出可用模型 |
| `POST /v1/chat/completions` | 聊天补全（支持 stream） |
| `GET /health` | 健康检查（含数据库和模型状态） |

**请求格式**：标准 OpenAI 格式，支持 `model`, `messages`, `temperature`, `max_tokens`, `stream`, `stop`。

## 关键技术要点

### Qwen3.6 模型适配
- 使用 `apply_chat_template(msgs, add_generation_prompt=True, extra_context={"enable_thinking": bool})` 构建 prompt
- 思考模式控制：`enable_thinking=True/False`（模型层面不支持完全关闭，为已知限制）
- max_new_tokens 设为 8192，适应思考过程消耗

### Intel Arc B390 UMA 架构
- GPU 和 CPU 共享 24GB 物理内存，无独立显存
- 模型权重加载到共享内存，GPU 通过 USM 直接访问，零拷贝
- oneDNN OpenCL 警告可通过 `ONEDNN_VERBOSE=0` 抑制

### SQLite 并发安全
- 使用 WAL 模式 + `check_same_thread=False`
- 知识库和对话历史使用同一数据库，不同表

### 乱码输出处理
- Qwen3.6 int4 量化在长上下文时可能产生乱码
- 检测：字母比例 < 40% 或特殊字符密度 > 10%
- 恢复：截断上下文为最近 4 条消息，重新生成

## 开发进度

### Phase 1：模型调度引擎 ✅
- [x] 独立虚拟环境
- [x] BaseModelAdapter 抽象基类
- [x] ModelEngine 单例调度器
- [x] ChatAdapter (Qwen3.6 VLMPipeline)
- [x] DLL 路径配置
- [x] 流式生成 + GPU 推理

### Phase 2：CLI 交互界面 ✅
- [x] Rich 美化终端
- [x] 多轮对话管理
- [x] 流式打字机输出
- [x] 命令系统（/clear, /history, /config, /image, /think, /nothink, /model, /sessions）
- [x] 性能指标显示

### Phase 3：API 服务 ✅
- [x] FastAPI + SSE 流式响应
- [x] 并发架构 (Semaphore)
- [x] 会话隔离 + 超时清理
- [x] 任务队列
- [x] CORS + 限流

### Phase 4：对话历史 + 自动总结 ✅
- [x] SQLite 持久化
- [x] 33% 上下文自动摘要
- [x] 200条/50% 数据库压缩
- [x] 会话恢复 + 记忆注入

### Phase 5：图片/文件处理 ✅
- [x] 图片理解 (/image + API)
- [x] PDF/TXT/MD 解析
- [x] 文本分块 + 知识库集成

### Phase 6：知识库 ✅
- [x] 文件导入 + 分块存储
- [x] 关键词检索 (AND 逻辑 + 通配符转义)
- [x] 上下文自动注入

### Phase 7：推理优化 ✅
- [x] 乱码检测 + 自动重试
- [x] LRU 模型卸载
- [x] Token 计数
- [x] OpenAI 兼容 API

### Phase 8：安全与稳定性 ✅
- [x] CORS 限制
- [x] 文件读取路径验证
- [x] 上传路径穿越防护
- [x] 输入验证
- [x] SQLite WAL 模式
- [x] 连接管理 + 上下文管理器

## 部署配置

- API 监听：`0.0.0.0:8000`
- 模型：`D:\AISpace\Models\Qwen3.6-35B-A3B-int4-ov` (~18GB)
- 上下文：256K tokens (自动检测)
- 设备：GPU (Intel Arc B390, 24GB UMA)
- 虚拟环境：`D:\AISpace\Envs\OvService\`

**启动命令**：
```powershell
$env:OPENVINO_LIB_PATHS = "D:\AISpace\Tools\openvino_genai\runtime\bin\intel64\Release;D:\AISpace\Tools\openvino_genai\runtime\3rdparty\tbb\bin"
python -u D:\AISpace\Workspace\OvService\cli.py   # CLI 模式
python -u D:\AISpace\Workspace\OvService\app.py   # API 模式
```

## 已知限制

1. `/nothink` 命令无法完全关闭 Qwen3.6 的思考模式（模型层面限制）
2. 流式 API 端点为模拟流式（全量生成后分块输出）
3. 知识库检索仅支持关键词匹配，无向量搜索
4. 单 GPU 限制，无法多卡并行

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0 | 2026-06-19 | 初始版本，7个开发阶段 |
| 1.1 | 2026-06-20 | 模型切换为 Qwen3.6-35B，动态上下文管理 |
| 1.2 | 2026-06-20 | OpenAI 兼容 API，安全修复，稳定性优化 |
