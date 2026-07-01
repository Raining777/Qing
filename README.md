# 清（Qing）智能学习助手

清是一个面向学习与复习的智能助手。它可以帮助你整理课程资料、生成重点总结、构建知识地图、设计复习计划，并围绕你的材料进行追问、练习和查漏补缺。

如果你手上有课件、PDF、笔记、讲义、题库或复习资料，清会把它们变成更容易理解、记忆和复盘的学习内容。

---

## 核心能力

| 能力 | 说明 |
|---|---|
| 资料总结 | 自动阅读学习资料，提炼章节重点、核心概念、易错点和考试高频内容 |
| 分层复习 | 从课程、章节、主题三个层级生成复习材料，适合快速回顾和系统复盘 |
| **多轮知识问答** | 基于上传资料进行多轮对话，**支持上下文记忆**，可以追问、深挖、跨话题连贯交流 |
| 思维导图 | 将复杂知识整理成 Mermaid 思维导图，帮助建立知识结构 |
| 练习生成 | 根据资料生成选择题、简答题、编程题等练习内容 |
| 闪卡记忆 | 自动生成记忆卡片，支持导出为 Anki CSV |
| 公式整理 | 提取公式并生成公式表，适合理工科课程复习 |
| 概念对比 | 对容易混淆的概念生成对比表，例如 TCP vs UDP、进程 vs 线程 |
| 考前冲刺 | 诊断薄弱点，生成快速复习清单、陷阱题和重点回顾 |
| 多模型支持 | 支持 Anthropic、DeepSeek、OpenAI、Ollama 等模型来源 |

---

## 适合场景

- 期末考试前快速梳理重点
- 读完课件后生成章节总结
- 把零散笔记整理成结构化知识体系
- 根据自己的资料生成复习题
- 遇到概念不清时直接向资料提问，并持续追问深挖
- 为长期学习生成闪卡和复习计划

---

## 快速开始

### 环境要求

- Python 3.10+
- 至少准备一个可用的模型 API Key：
  - Anthropic：适合视觉识别和复杂推理
  - DeepSeek：性价比较高
  - OpenAI：可选
  - Ollama：可用于本地模型

### 一键启动

```bash
# Windows
双击 清.bat

# macOS / Linux
./清.sh
```

启动脚本会自动完成：

1. 创建虚拟环境
2. 安装依赖
3. 打开浏览器访问 `http://localhost:7860`
4. 在设置页填写 API Key
5. 上传资料并开始学习

---

## 推荐学习流程

1. 上传课件、讲义、PDF、笔记或题库。
2. 让清自动分类资料，按课程或主题整理。
3. 生成章节总结和知识地图，先建立整体框架。
4. 针对不懂的地方继续追问，清会记住对话上下文，像真人导师一样连续交流。
5. 生成练习题、闪卡和考前复习清单。
6. 根据薄弱点反复复习，逐步补齐知识漏洞。

---

## 功能概览

| 功能 | 描述 |
|---|---|
| 批量上传 | 支持一次上传多份文件，并显示处理进度 |
| 自动分类 | 读取资料内容，自动归入对应课程或主题 |
| 知识总结 | 生成章节摘要、重点清单、知识关联和复习提示 |
| **多轮 RAG 问答** | 向量检索 + 对话历史 Query 扩展 + 重排 + 上下文压缩，支持连贯追问 |
| 图片识别 | 支持图表、公式、手写笔记、扫描版 PDF 的识别能力 |
| 学习计划 | 根据考试时间和复习节奏生成计划 |
| 题目生成 | 从材料中生成选择题、简答题、代码题等 |
| 闪卡导出 | 生成可翻看的记忆卡片，并支持导出 |
| 公式表 | 自动整理公式，支持 LaTeX 渲染 |
| 重点冲刺 | 考前生成薄弱点诊断、速记清单和易错提醒 |

---

## 项目结构

```text
qing/
├── 清.bat / 清.sh          # 一键启动脚本
├── requirements.txt        # Python 依赖
├── Dockerfile              # Docker 部署
├── docker-compose.yml
├── app/
│   ├── main.py             # FastAPI 入口
│   ├── config.py           # 全局配置
│   ├── graph/              # LangGraph 工作流
│   │   ├── state.py        # 对话状态定义（含 expanded_query）
│   │   ├── builder.py      # 图编译 + InMemory 检查点
│   │   └── nodes/          # 节点实现
│   │       ├── route.py    # 意图路由 + 追问 Query 扩展
│   │       ├── retrieve.py # 混合检索 + 重排 + 上下文压缩
│   │       ├── answer.py   # RAG 流式回答（10 轮对话历史）
│   │       ├── summary.py  # 分层知识总结
│   │       └── generate.py # 思维导图/计划/练习/闪卡/公式/对比/冲刺
│   ├── services/           # 核心服务
│   │   ├── llm.py          # 多模型统一接口（Anthropic/DeepSeek/OpenAI/Ollama）
│   │   ├── embedder.py     # bge-large-en-v1.5 本地嵌入 / Voyage API
│   │   ├── vectordb.py     # ChromaDB 向量库（混合检索 + 关键词）
│   │   ├── chunker.py      # 代码感知文本切分
│   │   └── image_utils.py  # 图片预处理
│   ├── routers/            # API 路由
│   │   ├── chat.py         # 多轮对话（LangGraph checkpoint 持久化）
│   │   ├── actions.py      # 总结/导图/练习等动作（跨意图状态共享）
│   │   ├── upload.py       # 文件上传 + 课程管理
│   │   └── setup.py        # API Key 配置
│   └── static/             # 前端
│       ├── index.html
│       ├── css/style.css
│       └── js/             # chat.js / app.js / upload.js / 等
└── data/                   # 运行时数据（不上传 Git）
```

---

## 架构说明

### 多轮对话

```
用户提问 → LangGraph StateGraph
                │
                ├─ route: 意图识别 + 追问 Query 扩展
                │   检测短追问（"那它怎么用？"），用对话历史补全检索词
                │
                ├─ retrieve: 混合检索（语义 70% + 关键词 30%）
                │   使用扩展后的 query 匹配资料
                │
                ├─ rerank: LLM 重排取 Top 3
                │
                ├─ compress: 提取关键信息
                │
                └─ answer_stream: 流式生成回答
                    带上最近 10 轮对话历史 + 连续对话提示词

回答完成后 → graph.aupdate_state() 保存到 InMemory 检查点
             add_messages reducer 自动累积消息
```

### 关键技术点

- **Query 扩展**：检测"那"/"这个"/"怎么用"等短追问标记，用最近 6 条消息构建上下文注入检索词
- **状态持久化**：LangGraph `InMemorySaver` + `add_messages` reducer，消息自动跨轮累积
- **跨意图共享**：总结生成的 `chapter_summaries` 可被思维导图/闪卡/公式表复用
- **混合检索**：语义（ChromaDB cosine）70% + 关键词（BM25）30%，去重后按加权分排序

---

## Docker 启动

```bash
docker build -t qing .
docker run -p 7860:7860 -v $(pwd)/data:/app/data qing
```

---

## 技术栈

| 模块 | 技术 |
|---|---|
| Agent 框架 | LangGraph + InMemorySaver checkpointing |
| 后端 | FastAPI + SSE 流式 |
| 前端 | 原生 HTML / CSS / JavaScript |
| 文件解析 | PyMuPDF、python-docx、python-pptx、Claude Vision |
| 嵌入模型 | bge-large-en-v1.5 ONNX / Voyage AI |
| 向量数据库 | ChromaDB（hybrid: cosine + BM25） |
| 模型提供方 | Anthropic、DeepSeek、OpenAI、Ollama |
| 数学公式 | KaTeX |
| 代码高亮 | highlight.js |
| 图表渲染 | Mermaid.js |

---

## 常见问题

### 我的资料会被上传到哪里？

文件会优先在本地处理。问答时，清会根据需要把相关文本片段发送给你配置的模型服务，用于生成回答。

### 支持中文资料吗？

支持。清可以处理中文课件、笔记和题目，也可以用中文生成总结、复习计划和练习题。

### 可以离线使用吗？

可以部分离线。使用 Ollama 和本地嵌入模型时，部分问答能力可以在本地运行；但视觉识别、云端模型推理等能力仍需要网络和 API。

### 适合哪些学科？

尤其适合理工科、计算机、数学、工程类课程，也可以用于语言、历史、医学、法律等需要大量资料整理和复习的场景。

---

## License

MIT
