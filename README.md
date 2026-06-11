# 本地文档工作台

这是一个面向中文合同、法律文书和业务文档的本地处理系统。当前产品形态包含两个相互独立的功能区：

- 文本脱敏：上传 TXT、DOCX、PDF 后完成主体识别、脱敏替换、结果导出和脱敏目录生成。
- PDF 转 Word 核查：上传原始 PDF 和 WPS 转换后的 DOCX，对转换底稿进行 OCR/版面/文本一致性复核，并输出带批注的 Word 文档、审查报告和证据包。

系统默认在本机运行，后端绑定 `127.0.0.1`，运行数据、上传文件、输出文件、任务状态和日志均保存在本地目录，不作为源码提交内容。

## 功能概览

### 文本脱敏

文本脱敏功能负责对合同、法律文书、业务材料中的敏感主体进行识别、分组、替换和导出。当前默认路线是高质量低内存脱敏流程，重点面向本地环境稳定运行。

主要能力：

- 支持 `TXT`、`DOCX`、`PDF` 输入。
- DOCX 尽量保留原文档结构、段落、表格、页眉页脚、批注等可回写区域。
- PDF 优先抽取原生文本；必要时使用 OCR 与文档结构修复能力补充识别。
- 默认主体分类以规则层要求为基础，面向人员、组织机构、地名、官方机构等主体进行识别与归并。
- 数字类内容按默认数字脱敏策略处理，日期和金额等可按规则保留。
- 识别阶段包含规则召回、结构回填、主体台账、边界修复、上下文分组、模型复核和最终导出。
- 导出脱敏文件，并同时生成脱敏目录，目录展示实际参与脱敏的替换项。

核心设计：

- `backend/app/rules/`：默认线规则层，包括格式识别、主体规则、边界修复、误识别过滤、主体台账和目录质量检查。
- `backend/app/services/coverage_first/`：以最终覆盖和回写为中心的候选、目录、替换和验证链路。
- `backend/app/services/contextual_desensitization_service.py`：上下文分组、替换编号和最终脱敏实体准备。
- `backend/app/processors/document_exporter.py`：脱敏文件和目录导出。
- `backend/app/processors/docx_xml_utils.py`：DOCX 可见文本抽取、精确回写和包级 XML 处理。

### PDF 转 Word 核查

PDF 转 Word 核查功能用于复核“原始 PDF”和“WPS 转换 DOCX”之间的转换质量。它不是脱敏流程的一部分，而是独立的转换审查工具。

典型使用场景：

- 原 PDF 经过 WPS 转换成 Word 后，需要确认文字是否漏识别、错识别、顺序错乱或表格错位。
- 需要保留 WPS 转换后的 Word 版面，只在 DOCX 中写入批注，不直接改正文。
- 需要输出可追溯证据包，方便人工复核和定位问题页。

主要能力：

- 同时上传原始 `PDF` 和 WPS 转换后的 `DOCX`。
- 对 PDF 页面渲染、OCR 文本、DOCX 文本单元进行证据抽取。
- 对正文、表格、图片页、阅读顺序、缺失文本、疑似替换错误进行分流审查。
- 结合本机 OCR、规则检查、表格专项审查、视觉/文本模型门控等模块生成风险项。
- 输出带批注的 reviewed DOCX、JSON 审查报告和 evidence zip 证据包。
- 前端展示页面风险、表格摘要、覆盖摘要、审查项和人工复核队列。

核心设计：

- `backend/app/api/pdf_word_audit.py`：PDF 转 Word 核查 API、任务状态、下载入口。
- `backend/app/services/pdf_word_audit_v4/`：v4 审查主线，包括预检、渲染、OCR 证据、DOCX 证据、页面映射、表格审查、视觉门控、报告构建和证据包输出。
- `frontend/src/views/PdfWordAudit.vue`：前端上传、进度展示、结果查看和下载入口。

## 系统架构

```text
contract-desensitize/
├── backend/                  # FastAPI 后端、文档处理、识别、审查、导出
│   ├── app/api/              # API 路由
│   ├── app/processors/       # DOCX/PDF/TXT 解析与导出
│   ├── app/rules/            # 默认脱敏规则层
│   ├── app/services/         # 脱敏服务、模型服务、PDF 转 Word 审查服务
│   └── app/workers/          # 后台任务 worker
└── frontend/                 # Vue 3 + TypeScript + Element Plus 前端
```

后端负责文档解析、实体识别、模型审查、脱敏替换、PDF 转 Word 核查、结果导出和证据产物生成。前端提供工作台、文本脱敏、PDF 转 Word 核查、自定义规则和系统设置页面。

## 前端页面

当前主界面包含：

- 工作台：进入文本脱敏或 PDF 转 Word 核查。
- 启动检查：检查本机运行环境、模型和 OCR 能力状态。
- 文本脱敏：上传文档、查看识别结果、生成脱敏文件和脱敏目录。
- PDF 转 Word 核查：上传原 PDF 和 WPS DOCX，查看审查结果并下载批注文档和证据包。
- 自定义规则：维护脱敏识别使用的自定义关键词、正则和模板。
- 系统设置：配置默认识别与脱敏参数。

在 `VITE_HIGH_QUALITY_ONLY=1` 的构建模式下，前端可以收敛为只展示高质量脱敏入口。

## API 概览

默认后端地址：

```text
http://127.0.0.1:8000
```

健康检查：

```text
GET /health
```

主要 API 前缀：

```text
/api/v1
```

文本脱敏相关：

- `POST /api/v1/desensitize/upload`
- `POST /api/v1/desensitize/process`
- `GET /api/v1/desensitize/status/{task_id}`
- `GET /api/v1/desensitize/result/{task_id}`
- `GET /api/v1/desensitize/processed-result/{task_id}`
- `GET /api/v1/desensitize/download/{task_id}`
- `GET /api/v1/desensitize/download/mapping/{task_id}`
- `GET /api/v1/desensitize/runtime-status`
- `GET /api/v1/desensitize/models`
- `GET /api/v1/desensitize/info`

PDF 转 Word 核查相关：

- `POST /api/v1/pdf-word-audit/upload`
- `GET /api/v1/pdf-word-audit/status/{audit_id}`
- `GET /api/v1/pdf-word-audit/result/{audit_id}`
- `GET /api/v1/pdf-word-audit/download/{audit_id}`
- `GET /api/v1/pdf-word-audit/report/{audit_id}`
- `GET /api/v1/pdf-word-audit/evidence/{audit_id}`

自定义规则和模板相关：

- `GET /api/v1/custom/config`
- `POST /api/v1/custom/keywords`
- `POST /api/v1/custom/patterns`
- `POST /api/v1/custom/reload`
- `GET /api/v1/custom/test`
- `GET /api/v1/history/templates`
- `POST /api/v1/history/templates`
- `DELETE /api/v1/history/templates/{template_id}`

## 本地运行

### 启动后端

```bash
cd backend
python3 -m pip install -r requirements.txt
python3 main.py
```

后端默认地址：

```text
http://127.0.0.1:8000
```

API 文档：

```text
http://127.0.0.1:8000/docs
```

### 启动前端

```bash
cd frontend
npm install
npm run dev
```

前端默认地址：

```text
http://127.0.0.1:5173
```

开发服务器默认将 `/api` 代理到后端 `http://127.0.0.1:8000`。

## 运行数据和 Git 仓库边界

以下内容属于本地运行数据，不应提交到 GitHub：

- `backend/uploads/`
- `backend/outputs/`
- `backend/logs/`
- `backend/task_state/`
- `backend/desensitize.db`
- `backend/analysis-review-*.json`
- `backend/pdf_word_audit/`
- `backend/pdf_normalized/`
- `backend/tmp/`
- `backend/models/`
- `.env`
- `backend/.env`
- `*.log`
- `*.pid`

这些路径已经通过 `.gitignore` 排除。对于曾经被 Git 跟踪过的运行数据，需要使用 `git rm --cached` 从索引中移除，保留本地文件但不再上传。

## 当前限制

- 不支持旧版 `.doc` 文件直接处理，需要先转换为 `.docx`。
- 扫描版 PDF 的识别质量依赖本机 OCR、页面质量和模型配置。
- PDF 转 Word 核查只对 WPS 转换 DOCX 写入批注和证据，不把审查建议直接改写进正文。
- 本地任务状态用于运行期恢复和状态展示，不等同于完整的云端任务历史系统。
- 仓库中可能保留历史设计文档或旧路线文档，当前实际功能以代码、根 README 和前端工作台入口为准。

## 项目定位

本项目不是在线 SaaS，而是本地化文档处理工具。设计目标是将敏感文档处理、OCR、模型审查、脱敏导出和转换核查尽量留在本机环境中完成，并为复杂合同/法律文书提供可追溯、可复核的处理产物。
