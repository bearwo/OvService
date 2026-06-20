# OvService — 智能对话平台

基于 OpenVINO GenAI 构建的本地智能对话平台，支持多模态交互（文本、图片、文档），完全本地化运行，数据不离开本机。

## 功能特性

- **多模态对话**：文本聊天 + 图片理解（VLM）+ 文档上传
- **动态上下文管理**：自动检测模型上下文长度，智能压缩历史对话
- **分层记忆系统**：对话摘要 → 数据库压缩 → 长期记忆注入
- **乱码检测与恢复**：自动检测输出异常，截断上下文重试
- **OpenAI 兼容 API**：标准 `/v1/chat/completions` 端点，支持 Codex 接入
- **可插拔 Web UI**：独立模块，不影响核心功能
- **多 Session 并发**：支持多个会话同时使用，互不干扰

## 技术栈

| 层 | 技术 |
|---|------|
| 推理引擎 | OpenVINO GenAI (VLMPipeline) |
| 当前模型 | Qwen3.6-35B-A3B-int4-ov (256K 上下文) |
| 后端框架 | FastAPI + Uvicorn |
| CLI | Rich + Prompt Toolkit |
| Web UI | 纯静态 HTML/CSS/JS + Python 代理 |
| 数据存储 | SQLite |
| 硬件 | Intel Arc B390 (24GB UMA) |

## 快速开始

### 环境要求

- Python 3.10+
- Intel Arc GPU 或支持 OpenVINO 的 CPU
- OpenVINO GenAI 运行时

### 安装

```bash
# 克隆仓库
git clone https://github.com/bearwo/OvService.git
cd OvService

# 安装依赖
pip install -r requirements.txt

# 安装 OpenVINO GenAI（从本地 wheel）
pip install D:\AISpace\Tools\openvino_genai\python\openvino\openvino-2026.2.1.0-3123-7dea0459b2a-cp314-cp314-win_amd64.whl
```

### 启动服务

**方式一：Web UI（推荐）**
```powershell
# 设置 OpenVINO 环境变量
$env:OPENVINO_LIB_PATHS = "D:\AISpace\Tools\openvino_genai\runtime\bin\intel64\Release;D:\AISpace\Tools\openvino_genai\runtime\3rdparty\tbb\bin"

# 启动（同时启动 API + Web UI）
python webui\server.py
# 或双击 webui.bat
```

访问 http://localhost:3000

**方式二：API 服务**
```powershell
$env:OPENVINO_LIB_PATHS = "D:\AISpace\Tools\openvino_genai\runtime\bin\intel64\Release;D:\AISpace\Tools\openvino_genai\runtime\3rdparty\tbb\bin"
python app.py
```

访问 http://localhost:8000/docs 查看 API 文档

**方式三：CLI**
```powershell
$env:OPENVINO_LIB_PATHS = "D:\AISpace\Tools\openvino_genai\runtime\bin\intel64\Release;D:\AISpace\Tools\openvino_genai\runtime\3rdparty\tbb\bin"
python -u cli.py
```

## API 接口

### OpenAI 兼容 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/models` | 列出可用模型 |
| POST | `/v1/chat/completions` | 聊天补全（支持 stream） |
| GET | `/health` | 健康检查 |

**请求示例：**
```json
{
  "model": "chat",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7,
  "max_tokens": 4096,
  "stream": true
}
```

### 业务 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | 同步聊天 |
| POST | `/chat/stream` | 流式聊天（SSE） |
| POST | `/chat/image/upload` | 图片理解 |
| GET | `/sessions` | 列出会话 |
| GET | `/sessions/{id}/messages` | 获取会话消息 |
| POST | `/knowledge/add` | 添加知识库文档 |
| POST | `/knowledge/search` | 搜索知识库 |
| POST | `/knowledge/upload` | 上传文档到知识库 |
| GET | `/models` | 列出模型 |
| POST | `/models/{name}/load` | 加载模型 |

## CLI 命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助 |
| `/clear` | 清空对话 |
| `/history` | 查看历史 |
| `/config` | 查看配置 |
| `/image <path>` | 图片理解 |
| `/think` | 开启思考模式 |
| `/nothink` | 关闭思考模式 |
| `/sessions` | 列出所有会话 |
| `/session new` | 开始新会话 |
| `/session load <id>` | 加载历史会话 |
| `/session export` | 导出会话 |
| `/model list` | 列出模型 |
| `/model load <name>` | 加载模型 |
| `/model unload <name>` | 卸载模型 |
| `/quit` | 退出 |

## 配置说明

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENVINO_LIB_PATHS` | - | OpenVINO DLL 路径（必须设置） |
| `OVSERVICE_MAX_CONCURRENT` | 2 | 最大并发推理数 |
| `OVSERVICE_CORS_ALLOW_ALL` | 1 | CORS 允许所有来源（生产环境设为 0） |

### config.py 关键配置

```python
# 模型路径
CHAT_MODEL = MODELS_DIR / "Qwen3.6-35B-A3B-int4-ov"

# 上下文管理
MAX_HISTORY_TURNS = 50
COMPRESS_CONTEXT_RATIO = 0.33  # 上下文压缩阈值（33%）
DB_COMPRESS_MAX_COUNT = 200     # 数据库压缩条数阈值
DB_COMPRESS_MAX_RATIO = 0.50   # 数据库压缩比例阈值

# API 配置
API_HOST = "0.0.0.0"
API_PORT = 8000
DEFAULT_DEVICE = "GPU"
```

## 项目结构

```
OvService/
├── app.py                  # FastAPI 入口
├── cli.py                  # 命令行入口
├── config.py               # 配置管理
├── core/
│   ├── base.py             # 模型适配器基类
│   ├── engine.py           # 模型调度引擎
│   ├── conversation.py     # 对话管理 + 动态压缩
│   └── optimize.py         # 性能优化
├── adapters/
│   └── chat.py             # Qwen3.6 VLMPipeline 适配器
├── features/
│   ├── memory.py           # 对话历史 + 分层摘要
│   ├── knowledge.py        # 知识库
│   ├── image.py            # 图片处理
│   └── file_parser.py      # 文件解析
├── api/
│   ├── routes.py           # 业务 API
│   ├── openai_compat.py    # OpenAI 兼容 API
│   ├── schemas.py          # 数据模型
│   ├── session.py          # 会话管理
│   └── task_queue.py       # 任务队列
├── webui/                  # 可插拔 Web UI
│   ├── server.py           # Web 服务器
│   ├── templates/
│   └── static/
├── data/                   # 数据目录
└── requirements.txt
```

## 架构设计

### 动态上下文管理

```
对话进行中 → 达到 33% 上下文 → 自动生成摘要 → 裁剪对话 → 加载摘要 → 继续对话
数据库压缩：摘要累积 ≥ 200 条 OR ≥ 50% 上下文时合并
```

### 乱码检测与恢复

```
生成输出 → _is_garbled() 检测 → 正常返回 / 乱码截断重试
```

### VLMPipeline 并发安全

- 每次 `generate()` 后调用 `finish_chat()` 清除 KV cache
- `_gen_lock` 确保同一时刻只有一个请求使用管道
- `finish_chat()` 必须在主线程调用

### OpenAI 兼容 API

支持标准 OpenAI 格式，可直接接入 Codex、LangChain 等工具。

## 已知限制

1. `/nothink` 命令无法完全关闭 Qwen3.6 的思考模式
2. 知识库检索仅支持关键词匹配，无向量搜索
3. 单 GPU 限制，无法多卡并行
4. Web UI 图片回复使用伪流式（生成完成后逐字显示）

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0 | 2026-06-19 | 初始版本，7个开发阶段 |
| 1.1 | 2026-06-20 | 模型切换为 Qwen3.6-35B，动态上下文管理 |
| 1.2 | 2026-06-20 | OpenAI 兼容 API，安全修复，稳定性优化 |
| 1.3 | 2026-06-20 | Web UI 模块，流式输出修复，全面代码审查优化 |
| 1.4 | 2026-06-20 | 多 session 并发修复，图片理解修复，80+ 问题修复 |

## 许可证

MIT License
