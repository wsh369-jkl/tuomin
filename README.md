# 合同脱敏系统

本项目是一个本地运行的合同脱敏系统，后端基于 FastAPI，前端基于 Vue 3 + Element Plus，识别链路已经统一到新的脱敏引擎。

截至 2026-03-08，本轮修复后已经完成并验证的主流程包括：

- 单文件上传 -> 敏感实体识别 -> 脱敏处理 -> 结果下载
- 批量文件上传 -> 批量识别结果查看
- 自定义关键词规则与正则规则的新增、编辑、删除、测试
- 历史记录、任务详情、统计信息查看
- 系统设置、默认识别开关、高级 `operator_config`、配置模板管理
- 开发模式桌面启动器联动后端与前端

## 技术架构

- `backend/`
  - FastAPI API
  - 新脱敏引擎
  - 自定义规则服务
  - 历史记录与模板存储
- `frontend/`
  - Vue 3
  - Vue Router
  - Element Plus
- `desktop/`
  - Python 启动器
  - 启动后端并在开发模式下拉起前端页面

## 当前支持能力

### 文件输入

- `PDF`（文本型 PDF）
- `DOCX`
- `TXT`

### 识别来源

- 内置规则识别
- 自定义关键词 / 正则规则
- LLM 识别（默认使用 Ollama）

### 脱敏操作

- `mask`
- `replace`
- `redact`
- `hash`
- `encrypt`

### 可用页面

- 文档脱敏
- 批量处理
- 自定义规则
- 历史记录
- 系统设置

## 当前限制

- 暂不支持扫描版 PDF/OCR
- 暂不支持旧版 `.doc`
- 批量处理当前为同步串行处理，结果一次性返回
- 批量处理前端目前只支持多文件上传，不支持文件夹或压缩包入口
- 若 Ollama 不可用，系统仍可启动，但 LLM 识别会退化为空结果

## 快速开始

### 方式 1：完整启动

```bat
install_dependencies.bat
start_all.bat
```

启动后访问：

- 前端：<http://localhost:5173>
- 后端：<http://localhost:8000>
- API 文档：<http://localhost:8000/docs>

### 方式 2：分别启动

```bat
start_backend.bat
start_frontend.bat
```

### 方式 3：启动桌面入口

```bat
start_desktop.bat
```

说明：

- 后端脚本会优先使用 `backend\venv\Scripts\python.exe`
- 如果虚拟环境不存在，脚本会自动创建
- 前端若未安装依赖，脚本会自动执行 `npm install`

## LLM 配置

默认使用 Ollama：

```env
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:4b
```

如果需要本地拉取模型，可以使用：

```bat
download_ollama_model.bat
```

或者手动执行：

```bat
ollama pull qwen3.5:4b
```

注意：`backend/.env` 会覆盖代码中的默认值。

## 页面说明

### 文档脱敏

- 上传单个合同文件
- 配置是否启用大模型和自定义规则
- 查看实体列表、文本预览、统计信息
- 执行脱敏并下载结果

### 批量处理

- 一次上传多个文件
- 使用默认识别配置批量执行分析
- 查看每个文件的实体结果和统计摘要

### 自定义规则

- 管理关键词规则
- 管理正则规则
- 为规则填写说明、上下文、评分
- 用测试文本即时验证规则效果

### 历史记录

- 查看任务列表
- 查看任务详情
- 查看任务统计
- 删除历史记录

### 系统设置

- 设置默认识别开关
- 编辑高级 `operator_config`
- 保存、应用、删除配置模板
- 查看当前引擎与模型信息

## 目录结构

```text
contract-desensitize/
├─ backend/                 后端服务
├─ frontend/                前端界面
├─ desktop/                 桌面启动器
├─ install_dependencies.bat 依赖安装脚本
├─ start_all.bat            一键启动前后端
├─ start_backend.bat        启动后端
├─ start_frontend.bat       启动前端
├─ start_desktop.bat        启动桌面入口
├─ FEATURE_STATUS.md        当前功能状态
└─ README.md                项目说明
```

## 已验证项

本轮修复已完成以下验证：

- 前端构建通过：`frontend\npm run build`
- 后端测试脚本通过：
  - `backend\tests\test_custom.py`
  - `backend\tests\test_presidio.py`
  - `backend\tests\test_hybrid.py`
- API 主流程通过：
  - 上传文件
  - 分析实体
  - 执行脱敏
  - 查询历史任务

## 后续可继续增强

- OCR 与扫描件支持
- 更完整的格式保留
- 批量任务异步队列
- 更细粒度的前端脱敏策略编辑器
