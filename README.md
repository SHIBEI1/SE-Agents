# SE-Agents · 软件开发技术设计工作台

输入一个软件想法，工作台由模型规划需求、架构和实施，并输出可下载的中文技术报告。产品经理、技术架构师、测试与交付视角、质疑者分阶段工作。LangGraph 管理流程与暂停，SQLite 保存任务、问答、已校验的阶段结果和用量。

## 配置模型

正式任务需要真实模型接口。将 `.env.example` 复制为项目内的 `.env`，在本机填写 `DEEPSEEK_API_KEY`。已有深度求索配置可直接使用。可通过 `DEEPSEEK_MODEL` 指定模型。

使用其他支持 OpenAI 兼容聊天接口及 JSON 对象输出的服务时，同时设置 `MODEL_BASE_URL`、`MODEL_API_KEY` 和 `MODEL_NAME`。三项配置优先于深度求索配置。接口地址通常包含服务商指定的版本路径。修改配置后重启工作台。模型请求会产生服务商用量。

[深度求索接口说明](https://api-docs.deepseek.com/) · [结构化输出说明](https://api-docs.deepseek.com/guides/json_mode/)

## 启动与使用

小型网页和 API 项目默认使用 Python 标准库及 SQLite。构建时可启用本机验收，工作台复用现有 Python 运行项目测试和 HTTP 业务检查。用户明确指定的技术栈仍按需求实现。

Windows、macOS 和 Linux 的完整安装和 API 配置步骤见 [使用说明](使用说明.md)。每台电脑在项目目录执行 `uv sync --locked --no-dev` 安装自己的运行环境。

Windows PowerShell 从项目目录运行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

macOS / Linux 从项目目录运行：

```bash
./.venv/bin/streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

首次安装依赖时可使用 `uv sync --locked`。

1. 打开本地工作台，在聊天框发一句软件想法。任务名称可选。
2. 助手最多集中问一轮会明显改变设计的问题。知道多少答多少，也可以回复「你决定」。
3. 工作台生成详细需求、默认设计及依据、技术选型、数据模型、接口、并发与故障处理、实施部署和验收测试。未确认的事项列出验证方法与替代路径。
4. 展开「查看完整技术报告」或点击「下载技术报告」。存在重大审查问题时，报告标为待复核草案。
5. 审阅报告后展开「确认方案并构建项目」。填写项目目录，例如 `~/Downloads/我的软件项目`，确认这版方案并开始构建。目录不存在时会自动创建。若生成文件与已有文件同名，工作台会提示冲突路径。
6. 构建任务会生成源码、依赖与配置、测试、README，并保留已确认方案与审查记录。按新项目的 README 启动和运行测试。

旧记录仍可查看。点击「用原始想法生成技术报告」会以原始描述新建真实模型任务。

## 命令行

```bash
./.venv/bin/se-agents start examples/campus-study.md
./.venv/bin/se-agents show <任务编号>
./.venv/bin/se-agents reply <任务编号> 你决定
./.venv/bin/se-agents export <任务编号> --format markdown
```

## 开发与验证

- 初版论证流程与课程设计基线见 [v2.0 设计报告](docs/design-v2.md)。新版将候选方向比较扩展为面向开发的需求、架构、实施和审查流程。
- 需求编号和验收用例引用由程序校验，实际系统的实现与测试应按报告执行。
- 数据默认保存在 `data/`，密钥保存在本地 `.env`。二者均被忽略。可通过 `SE_AGENTS_DATA_DIR` 或 CLI `--data-dir` 指定数据目录。
- 从历史任务点击「继续任务」或执行 `resume` 可从检查点恢复，已经通过校验的阶段结果会复用。

```bash
./.venv/bin/pytest -q
./.venv/bin/ruff check app.py src tests
```
