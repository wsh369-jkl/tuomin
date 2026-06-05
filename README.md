# 合同脱敏系统

本项目当前定位为一个 `macOS 本地客户端` 形态的合同脱敏成品。

## 一键启动

直接双击仓库根目录的 [start.command](/Users/wendyhan/Desktop/contract-desensitize/start.command) 即可进入系统。

首次运行时会自动完成这些准备工作：

- 创建 `backend/venv`
- 安装后端依赖
- 当前端静态资源缺失时自动构建 `frontend/dist`
- 启动本地桌面启动器，并自动打开系统入口

系统默认以本地运行方式工作：

- 前端界面由后端内嵌或开发态代理提供
- 后端服务运行在本机 `127.0.0.1`
- 大模型能力默认走本地 `Ollama`
- 文档、日志、数据库都保存在本机私有目录

当前产品已经收敛，不再包含批量处理页和历史记录页。

## 当前架构

- `desktop/`
  - Python 桌面启动器
  - 负责探测 Ollama、启动后端、打开浏览器入口
  - macOS 打包后会生成 `.app`
- `backend/`
  - FastAPI API
  - 脱敏引擎、识别器注册表、操作器注册表
  - 文档解析、上下文替换、导出
- `frontend/`
  - Vue 3 + Element Plus
  - 当前保留的页面：
    - 启动检查
    - 文档脱敏
    - 自定义规则
    - 系统设置
- `build/`
  - PyInstaller 打包脚本
  - macOS DMG 打包脚本

## 当前功能

### 文档输入

- `PDF`
  - 优先抽取原生文本
  - 对低文本页可按配置触发 OCR 回退
- `DOCX`
- `TXT`

### 识别来源

- 合同字段标签识别
- 正则规则识别
- 自定义关键词 / 正则规则
- LLM 语义识别

### 脱敏输出

- `DOCX`
  - 尽量保留结构和格式
- `TXT`
- `PDF`
  - 当前导出为重建后的 `DOCX`
  - 无法重建时回退为 `TXT`

### 当前前端页面

- `启动检查`
  - 检查 Ollama、固定模型和运行环境状态
- `文档脱敏`
  - 上传文档、识别实体、生成脱敏结果、下载导出文件
- `自定义规则`
  - 管理关键词规则和正则规则
- `系统设置`
  - 默认识别配置、匿名策略、模板管理、运行时信息

## 固定运行路线

当前成品路线默认锁定本地稳定模型：

```env
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3.5:4b
```

如果本机未安装模型，可执行：

```bash
ollama pull qwen3.5:4b
```

或在打包产物中运行：

```bash
./download_ollama_model.command
```

## macOS 客户端打包

### 1. 准备依赖

后端依赖：

```bash
python3 -m pip install -r backend/requirements.txt
python3 -m pip install -r build/requirements-build.txt
```

前端依赖：

```bash
cd frontend
npm install
cd ..
```

### 2. 生成发布目录

```bash
python3 build/build.py
```

输出目录：

```text
release-macos/
```

其中会包含：

- `contract-desensitize.app`
- `start.command`
- `download_ollama_model.command`
- `USAGE.txt`
- `MAC_QUICK_START.txt`

### 3. 生成 DMG

```bash
python3 build/package_macos_installer.py
```

输出文件：

```text
release-macos/ContractDesensitize-macOS.dmg
```

### 4. 生成对外分发元数据

```bash
python3 build/prepare_macos_distribution.py
```

附加产物：

- `release-macos/ContractDesensitize-macOS-portable.tar.gz`
- `release-macos/DISTRIBUTION_MANIFEST.txt`
- `release-macos/SHA256SUMS.txt`

建议对外分发时优先发送：

- `ContractDesensitize-macOS.dmg`

内部备份或手工回传时可附带：

- `ContractDesensitize-macOS-portable.tar.gz`
- `SHA256SUMS.txt`

## 开发模式

### 启动后端

```bash
cd backend
python3 main.py
```

### 启动前端

```bash
cd frontend
npm run dev
```

开发态默认地址：

- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8000`
- API 文档：`http://127.0.0.1:8000/docs`

## 当前限制

- 不支持旧版 `.doc`
- 扫描版 PDF 的效果仍依赖 OCR 与模型状态
- 任务状态当前以进程内任务表为主，不是完整的持久化任务中心
- 产品已去除批量处理和历史记录页面

## 仓库说明

仓库里仍可能存在一些历史文档或 Windows 辅助脚本，它们不代表当前主交付路径。当前以本 README、`build/build.py` 和 macOS 打包脚本为准。

## 对外分发建议

如果要发给其他 Mac 用户直接安装使用，建议目标形态是：

1. 已签名
2. 已 notarize
3. 已 staple 的 `ContractDesensitize-macOS.dmg`

仓库中的 GitHub Actions macOS workflow 已经按这个方向准备，可在打 `v*` tag 时直接生成发布资产。
