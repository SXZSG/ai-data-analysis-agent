# ai-data-analysis-agent

一个面向真实数据分析和算法建模场景的 AI Data Analysis Agent。项目支持上传 CSV / Excel 数据集，自动完成数据读取、数据质量检查、可视化分析、机器学习 baseline 建模、模型评估和报告生成。

它不是论文助手，也不是岗位匹配助手，而是一个可以在简历和 GitHub 中展示 Agent Workflow、Tool Calling、Python 数据分析、机器学习建模和 AI 报告生成能力的端到端项目。

## 项目背景

很多数据分析项目的第一阶段都包含相似流程：读取数据、理解字段、检查缺失值、判断建模任务、构建 baseline、评估模型并输出分析报告。本项目把这些步骤封装成一个清晰的 Agent Workflow，让用户通过 Streamlit 页面上传数据并触发自动分析。

## 项目功能

- 支持上传 CSV / Excel 文件。
- 支持用户选择目标列 target column。
- 自动判断 classification / regression 任务。
- 自动生成数据画像：字段类型、缺失率、唯一值数量、数值统计、类别 top values。
- 自动清洗和编码：数值缺失值 median 填充，类别缺失值 mode 填充，数值特征 StandardScaler，类别特征 OneHotEncoder。
- 自动生成可视化图片：缺失值、数值分布、相关性热力图、目标变量分布。
- 自动训练 baseline 模型。
- 自动评估模型并选择最佳 baseline。
- 支持 OpenAI-compatible API 生成 AI Summary。
- 未配置 API key 时自动回退到规则模板报告。
- 导出 Markdown 报告，可选导出 Word 报告。

## Agent Workflow 流程图

```text
User Upload Dataset
        |
        v
Step 1: Load Dataset
        |
        v
Step 2: Profile Data
        |
        v
Step 3: Detect Task Type
        |
        v
Step 4: Clean and Encode Data
        |
        v
Step 5: Generate Visualizations
        |
        v
Step 6: Train Baseline Models
        |
        v
Step 7: Evaluate Models
        |
        v
Step 8: Generate AI Summary
        |
        v
Step 9: Export Report
```

## 技术栈

- Python
- Streamlit
- pandas / numpy
- scikit-learn
- matplotlib
- openpyxl
- python-docx
- python-dotenv
- OpenAI-compatible API

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

打开页面后：

1. 上传 `examples/sample_classification.csv` 或 `examples/sample_regression.csv`。
2. 选择目标列，例如 `churn` 或 `price`。
3. 点击 `Run Agent`。
4. 在页面查看执行日志、数据画像、模型结果、图表和报告。

## 本地测试

语法检查：

```bash
python -m py_compile app.py
python -m py_compile src\config.py src\llm_client.py src\agent_orchestrator.py src\prompts.py src\utils.py
python -m py_compile tools\data_loader.py tools\data_profiler.py tools\data_cleaner.py tools\visualizer.py tools\task_detector.py tools\model_trainer.py tools\model_evaluator.py tools\report_exporter.py
```

逻辑测试建议使用 `examples/` 下的两个虚构数据集分别验证 classification 和 regression 工作流，确认 `outputs/data_analysis_report.md` 与可视化图片可以生成。

## .env 配置方法

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

填写 OpenAI-compatible API 配置：

```env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

如果不填写 `LLM_API_KEY`，项目不会报错，会使用规则模板生成基础报告。

## Demo 数据说明

- `examples/sample_classification.csv`：虚构客户流失数据，目标列为 `churn`。
- `examples/sample_regression.csv`：虚构房价数据，目标列为 `price`。

示例数据均为虚构数据，不包含真实隐私信息。

## 项目结构

```text
ai-data-analysis-agent/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── llm_client.py
│   ├── agent_orchestrator.py
│   ├── prompts.py
│   └── utils.py
├── tools/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_profiler.py
│   ├── data_cleaner.py
│   ├── visualizer.py
│   ├── task_detector.py
│   ├── model_trainer.py
│   ├── model_evaluator.py
│   └── report_exporter.py
├── examples/
│   ├── sample_classification.csv
│   └── sample_regression.csv
├── outputs/
│   └── .gitkeep
└── assets/
    └── .gitkeep
```

## 简历写法

可以写成：

> 构建 AI Data Analysis Agent，基于 Streamlit、pandas、scikit-learn 和 OpenAI-compatible API，实现 CSV / Excel 数据集的自动读取、数据画像、任务类型识别、特征预处理、baseline 建模、模型评估和 Markdown / Word 报告导出。项目采用 Agent Workflow + Tool Calling 结构，将数据加载、清洗、可视化、建模和报告生成拆分为独立工具模块，支持无 API key 时的规则模板回退。

## 面试讲法

面试时可以重点讲：

- 为什么把流程设计成 Agent Workflow，而不是一个长函数。
- 每个工具模块的输入输出边界，例如 data_loader、data_profiler、model_trainer。
- 如何判断 classification / regression。
- 如何使用 ColumnTransformer 统一处理数值和类别特征。
- baseline 模型为什么选择 LogisticRegression、LinearRegression 和 RandomForest。
- 无 LLM API key 时如何保证系统仍然可用。
- 后续如何扩展为多轮分析 Agent，例如支持用户追问、自动特征工程和模型解释。

## Limitations and Future Work

- 当前只支持本地上传的 CSV / Excel 文件。
- 当前 task detector 使用规则判断，复杂业务标签仍需要人工确认。
- 当前建模只覆盖 baseline，不包含超参数搜索、交叉验证和模型解释。
- 当前报告为静态报告，未来可扩展为交互式 HTML 报告。
- 未来可以加入 SHAP、特征重要性解释、自动特征工程、异常值检测和多轮对话式分析。

## 上传 GitHub

```bash
git init
git add .
git commit -m "Initial commit: AI data analysis agent"
git branch -M main
git remote add origin https://github.com/<your-username>/ai-data-analysis-agent.git
git push -u origin main
```

上传前不要提交 `.env`、真实业务数据或 `outputs/` 下的运行产物。
