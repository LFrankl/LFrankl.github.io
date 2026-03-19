---
layout: post
title: "AI Agent 系列 ⑧：MCP & Skills —— Agent 能力的标准化与生态"
date: 2026-03-19 15:00:00 +0800
categories: [AI, Agent]
tags: [AI Agent, MCP, Model Context Protocol, Skills, Anthropic, Tool Use]
math: false
mermaid: true
---

> 系列文章：
> [① Agent 概览](../ai-agent-01-overview) ·
> [② Tool Use](../ai-agent-02-tool-use) ·
> [③ RAG](../ai-agent-03-rag) ·
> [④ Memory](../ai-agent-04-memory) ·
> [⑤ Planning](../ai-agent-05-planning) ·
> [⑥ Multi-Agent](../ai-agent-06-multi-agent) ·
> [⑦ Agent Engineering](../ai-agent-07-engineering) ·
> **⑧ MCP & Skills（本篇）**

---

② 篇讲 Tool Use 的时候，工具是直接硬编码在 Agent 里的——你写一个函数，注册进去，完事。

这样可以跑 demo，但扩展性很差：
- 换个 Agent，工具得重新注册一遍
- 工具更新了，所有用到它的 Agent 都要改
- 没有标准接口，不同框架的工具不能互用

**MCP（Model Context Protocol）** 就是为了解决这个问题——把工具、数据源、提示模板标准化成一套协议，Agent 通过统一接口接入，工具开发者和 Agent 开发者完全解耦。

---

## 1. MCP 是什么

MCP 是 Anthropic 于 2024 年 11 月开源的协议，定义了 AI 模型和外部能力之间的标准通信方式。

核心思路：

```mermaid
graph LR
    subgraph "之前：紧耦合"
        A1[Agent A] -->|硬编码| T1[工具1]
        A1 -->|硬编码| T2[工具2]
        A2[Agent B] -->|硬编码| T1
        A2 -->|硬编码| T3[工具3]
    end
```

```mermaid
graph LR
    subgraph "MCP：松耦合"
        A1[Agent A] -->|MCP 协议| S1[MCP Server: 搜索]
        A1 -->|MCP 协议| S2[MCP Server: 数据库]
        A2[Agent B] -->|MCP 协议| S1
        A2 -->|MCP 协议| S3[MCP Server: 文件系统]
    end
```

MCP 定义了三类能力：

| 类型 | 描述 | 例子 |
|------|------|------|
| **Tools** | Agent 可以调用的函数 | 执行 SQL、发 HTTP 请求、读文件 |
| **Resources** | Agent 可以读取的数据 | 数据库 schema、文档内容、文件列表 |
| **Prompts** | 预定义的提示模板 | 代码审查模板、报告生成模板 |

---

## 2. 协议架构

```mermaid
graph TD
    subgraph "MCP Client（通常是 Agent / IDE）"
        C[Client]
    end

    subgraph "传输层"
        T1[stdio<br/>本地进程]
        T2[HTTP + SSE<br/>远程服务]
    end

    subgraph "MCP Server"
        S[Server]
        S --> Tools[Tools 列表]
        S --> Resources[Resources 列表]
        S --> Prompts[Prompts 列表]
    end

    C -->|JSON-RPC 2.0| T1
    C -->|JSON-RPC 2.0| T2
    T1 --> S
    T2 --> S
```

通信用的是 **JSON-RPC 2.0**，传输支持两种模式：
- **stdio**：本地子进程，Client 通过 stdin/stdout 和 Server 通信，适合本地工具
- **HTTP + SSE**：远程 Server，Client 发 HTTP 请求，Server 用 SSE 推流式响应，适合云端服务

---

## 3. 从零实现一个 MCP Server

用官方的 Python SDK `mcp`：

```bash
pip install mcp
```

### 3.1 最简单的 MCP Server

```python
# server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import asyncio
import json
import httpx


# 创建 Server 实例
server = Server("my-tools-server")


# ===== 注册 Tools =====

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """返回这个 Server 提供的所有工具"""
    return [
        types.Tool(
            name="web_search",
            description="搜索网页，返回相关结果摘要",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_weather",
            description="获取指定城市的当前天气",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京'、'Shanghai'",
                    },
                },
                "required": ["city"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """处理工具调用请求"""
    if name == "web_search":
        results = await _do_web_search(
            query=arguments["query"],
            num_results=arguments.get("num_results", 5),
        )
        return [types.TextContent(type="text", text=json.dumps(results, ensure_ascii=False))]

    elif name == "get_weather":
        weather = await _get_weather(city=arguments["city"])
        return [types.TextContent(type="text", text=json.dumps(weather, ensure_ascii=False))]

    else:
        raise ValueError(f"未知工具: {name}")


# ===== 注册 Resources =====

@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """返回可读取的资源列表"""
    return [
        types.Resource(
            uri="config://app-settings",
            name="应用配置",
            description="当前应用的配置信息",
            mimeType="application/json",
        ),
        types.Resource(
            uri="db://schema",
            name="数据库 Schema",
            description="所有表的结构定义",
            mimeType="text/plain",
        ),
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """读取资源内容"""
    if uri == "config://app-settings":
        return json.dumps({
            "version": "1.0.0",
            "environment": "production",
            "max_retries": 3,
        }, ensure_ascii=False)

    elif uri == "db://schema":
        return """
Table: users
  - id: INTEGER PRIMARY KEY
  - name: VARCHAR(100)
  - email: VARCHAR(200) UNIQUE
  - created_at: TIMESTAMP

Table: orders
  - id: INTEGER PRIMARY KEY
  - user_id: INTEGER REFERENCES users(id)
  - amount: DECIMAL(10,2)
  - status: ENUM('pending','paid','shipped','done')
"""
    raise ValueError(f"未知资源: {uri}")


# ===== 注册 Prompts =====

@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="code_review",
            description="代码审查提示模板",
            arguments=[
                types.PromptArgument(
                    name="language",
                    description="编程语言",
                    required=True,
                ),
                types.PromptArgument(
                    name="focus",
                    description="审查重点：security / performance / style",
                    required=False,
                ),
            ],
        ),
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> types.GetPromptResult:
    if name == "code_review":
        language = (arguments or {}).get("language", "Python")
        focus = (arguments or {}).get("focus", "全面审查")
        return types.GetPromptResult(
            description=f"{language} 代码审查",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"""请对以下 {language} 代码进行审查，重点关注：{focus}。

检查项目：
1. 潜在 bug 和边界条件
2. 安全漏洞（SQL 注入、XSS 等）
3. 性能问题（N+1 查询、内存泄漏等）
4. 代码可读性和命名规范
5. 是否有缺失的错误处理

请给出具体的改进建议和示例代码。""",
                    ),
                )
            ],
        )
    raise ValueError(f"未知 Prompt: {name}")


# ===== 工具实现 =====

async def _do_web_search(query: str, num_results: int) -> list[dict]:
    # 实际接入 Tavily / Bing API
    # 这里 mock
    return [
        {"title": f"结果 {i}: {query}", "url": f"https://example.com/{i}", "snippet": f"关于 {query} 的内容..."}
        for i in range(num_results)
    ]


async def _get_weather(city: str) -> dict:
    # 实际接入天气 API
    return {
        "city": city,
        "temperature": 22,
        "condition": "晴",
        "humidity": 45,
    }


# ===== 启动 Server =====

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
```

### 3.2 启动方式

```bash
# stdio 模式（本地）
python server.py

# 或者通过 mcp CLI 启动
mcp run server.py
```

---

## 4. MCP Client：在 Agent 里接入 MCP Server

```python
# client.py
import asyncio
import json
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import anthropic


class MCPAgent:
    """
    集成 MCP 的 Agent：
    - 自动发现 MCP Server 提供的工具
    - 把工具转换成 Anthropic tool_use 格式
    - 执行 LLM 决策 → MCP 工具调用 循环
    """

    def __init__(self, server_script: str):
        self.server_script = server_script
        self.client = anthropic.Anthropic()
        self.session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()

        # 从 MCP Server 发现的工具
        self._mcp_tools: list[dict] = []
        # MCP 工具名 -> Anthropic tool schema
        self._tool_schemas: list[dict] = []

    async def connect(self):
        """连接到 MCP Server，加载工具列表"""
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script],
        )

        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(*stdio_transport)
        )

        await self.session.initialize()

        # 拉取工具列表并转换格式
        tools_result = await self.session.list_tools()
        self._mcp_tools = tools_result.tools

        # MCP Tool schema → Anthropic tool schema
        self._tool_schemas = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in self._mcp_tools
        ]

        print(f"[MCPAgent] 已连接，发现 {len(self._mcp_tools)} 个工具:")
        for tool in self._mcp_tools:
            print(f"  - {tool.name}: {tool.description}")

    async def disconnect(self):
        await self._exit_stack.aclose()

    async def run(self, goal: str) -> str:
        """ReAct 循环：LLM 决策 → 工具调用 → 观察 → 继续"""
        messages = [{"role": "user", "content": goal}]

        for step in range(20):
            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                tools=self._tool_schemas,
                messages=messages,
            )

            # 没有工具调用，直接返回
            if response.stop_reason == "end_turn":
                text_blocks = [b for b in response.content if b.type == "text"]
                return text_blocks[0].text if text_blocks else ""

            # 处理工具调用
            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    print(f"  [Step {step+1}] 调用工具: {block.name}({block.input})")

                    # 通过 MCP 调用工具
                    result = await self.session.call_tool(block.name, block.input)
                    result_text = result.content[0].text if result.content else ""

                    print(f"  [Step {step+1}] 结果: {result_text[:100]}...")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

                messages.append({"role": "user", "content": tool_results})

        return "超过最大步数限制"

    async def list_resources(self) -> list[dict]:
        """列出 MCP Server 的可用资源"""
        result = await self.session.list_resources()
        return [{"uri": r.uri, "name": r.name, "description": r.description} for r in result.resources]

    async def read_resource(self, uri: str) -> str:
        """读取 MCP Resource"""
        result = await self.session.read_resource(uri)
        return result.contents[0].text if result.contents else ""

    async def get_prompt(self, name: str, arguments: dict = None) -> str:
        """获取 MCP Prompt 模板"""
        result = await self.session.get_prompt(name, arguments or {})
        return result.messages[0].content.text if result.messages else ""


# 使用
async def main():
    agent = MCPAgent("server.py")
    await agent.connect()

    try:
        # 正常 Agent 任务
        result = await agent.run("搜索 Python 异步编程的最佳实践，并给出总结")
        print("\n结果:", result)

        # 读取资源
        schema = await agent.read_resource("db://schema")
        print("\nDB Schema:", schema[:200])

        # 获取 Prompt 模板
        prompt = await agent.get_prompt("code_review", {"language": "Python", "focus": "security"})
        print("\n代码审查模板:", prompt[:200])

    finally:
        await agent.disconnect()


asyncio.run(main())
```

---

## 5. 多 Server 接入

一个 Agent 可以同时接入多个 MCP Server，能力叠加：

```python
class MultiMCPAgent:
    """同时接入多个 MCP Server"""

    def __init__(self, servers: dict[str, str]):
        """
        servers: {server_name: script_path}
        例: {"search": "servers/search_server.py", "db": "servers/db_server.py"}
        """
        self.servers = servers
        self.client = anthropic.Anthropic()
        self._sessions: dict[str, ClientSession] = {}
        self._all_tools: list[dict] = []
        self._tool_to_server: dict[str, str] = {}  # tool_name -> server_name
        self._exit_stack = AsyncExitStack()

    async def connect_all(self):
        for name, script in self.servers.items():
            session = await self._connect_server(name, script)
            self._sessions[name] = session

            # 加载工具并记录归属
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                self._all_tools.append({
                    "name": tool.name,
                    "description": f"[{name}] {tool.description}",
                    "input_schema": tool.inputSchema,
                })
                self._tool_to_server[tool.name] = name

        print(f"[MultiMCPAgent] 共接入 {len(self._sessions)} 个 Server，{len(self._all_tools)} 个工具")

    async def call_tool(self, tool_name: str, args: dict) -> str:
        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            raise ValueError(f"工具 '{tool_name}' 不存在于任何 Server")

        session = self._sessions[server_name]
        result = await session.call_tool(tool_name, args)
        return result.content[0].text if result.content else ""

    async def _connect_server(self, name: str, script: str) -> ClientSession:
        params = StdioServerParameters(command="python", args=[script])
        transport = await self._exit_stack.enter_async_context(stdio_client(params))
        session = await self._exit_stack.enter_async_context(ClientSession(*transport))
        await session.initialize()
        print(f"  [Connect] {name} ({script})")
        return session

    async def run(self, goal: str) -> str:
        messages = [{"role": "user", "content": goal}]

        for _ in range(20):
            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                tools=self._all_tools,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                texts = [b.text for b in response.content if b.type == "text"]
                return texts[0] if texts else ""

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue
                result_text = await self.call_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

            messages.append({"role": "user", "content": tool_results})

        return "超过最大步数"

    async def close(self):
        await self._exit_stack.aclose()
```

---

## 6. HTTP 模式的 MCP Server

上面的例子都是 stdio（本地进程）。如果要把 MCP Server 部署成云端服务，用 HTTP + SSE：

```python
# http_server.py
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.requests import Request
import uvicorn

# Server 定义和 stdio 版完全一样，只改传输层
server = Server("remote-tools-server")

# ... 同上注册 tools / resources / prompts ...

# HTTP 传输
sse_transport = SseServerTransport("/messages/")


async def handle_sse(request: Request):
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0],
            streams[1],
            server.create_initialization_options(),
        )


app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse_transport.handle_post_message),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Client 端连接远程 Server：

```python
from mcp.client.sse import sse_client

async def connect_remote():
    async with sse_client("http://your-server:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print([t.name for t in tools.tools])
```

---

## 7. Skills：Agent 能力的模块化封装

MCP 解决了**工具**的标准化，而 **Skills** 解决的是**更高层的能力复用**——把一组工具 + 提示逻辑 + 执行策略打包成一个可插拔的模块。

```mermaid
graph TD
    subgraph "Skill: 竞品分析"
        P[Prompt 模板] --> E[执行逻辑]
        T1[Tool: web_search] --> E
        T2[Tool: read_url] --> E
        T3[Tool: write_file] --> E
        E --> O[结构化输出]
    end

    Agent -->|挂载 Skill| S["Skill: 竞品分析"]
    S --> Agent
```

### 7.1 Skill 基类

```python
from dataclasses import dataclass, field
from typing import Any, Callable
import anthropic


@dataclass
class SkillInput:
    """Skill 的输入规范"""
    name: str
    type: str           # "string" / "integer" / "list" / "dict"
    description: str
    required: bool = True
    default: Any = None


@dataclass
class SkillOutput:
    """Skill 的输出规范"""
    success: bool
    data: Any
    error: str | None = None
    metadata: dict = field(default_factory=dict)


class BaseSkill:
    """所有 Skill 的基类"""

    name: str = ""
    description: str = ""
    inputs: list[SkillInput] = []

    def __init__(self, tools: dict[str, Callable] = None):
        self.tools = tools or {}
        self.client = anthropic.Anthropic()

    def validate_inputs(self, **kwargs) -> dict:
        """校验并填充默认值"""
        result = {}
        for inp in self.inputs:
            val = kwargs.get(inp.name, inp.default)
            if inp.required and val is None:
                raise ValueError(f"Skill '{self.name}' 缺少必填参数: {inp.name}")
            result[inp.name] = val
        return result

    async def execute(self, **kwargs) -> SkillOutput:
        """子类实现"""
        raise NotImplementedError

    def __repr__(self):
        return f"Skill({self.name}: {self.description})"
```

### 7.2 实现一个竞品分析 Skill

```python
class CompetitorAnalysisSkill(BaseSkill):
    name = "competitor_analysis"
    description = "分析竞争对手，输出结构化对比报告"
    inputs = [
        SkillInput("topic", "string", "分析主题，如 'AI 编程工具'", required=True),
        SkillInput("competitors", "list", "竞争对手列表", required=True),
        SkillInput("dimensions", "list", "对比维度", required=False,
                   default=["功能", "定价", "用户评价", "市场份额"]),
    ]

    ANALYZE_PROMPT = """你是一个专业的商业分析师。

分析主题: {topic}
竞争对手: {competitors}
对比维度: {dimensions}

已收集的资料:
{research_data}

请输出一份结构化的竞品分析报告，包含：
1. 各维度对比表格（Markdown 格式）
2. 各竞品的核心优势和劣势
3. 市场机会和威胁
4. 结论和建议

要求：客观、数据支撑、有可操作的结论。"""

    async def execute(self, **kwargs) -> SkillOutput:
        params = self.validate_inputs(**kwargs)
        topic = params["topic"]
        competitors = params["competitors"]
        dimensions = params["dimensions"]

        try:
            # Step 1: 并行搜索每个竞品
            import asyncio
            search_tasks = [
                self._research_competitor(c, topic, dimensions)
                for c in competitors
            ]
            research_results = await asyncio.gather(*search_tasks)

            # Step 2: 汇总 + 生成报告
            research_data = "\n\n".join(
                f"### {comp}\n{data}"
                for comp, data in zip(competitors, research_results)
            )

            prompt = self.ANALYZE_PROMPT.format(
                topic=topic,
                competitors=", ".join(competitors),
                dimensions=", ".join(dimensions),
                research_data=research_data[:6000],
            )

            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )

            return SkillOutput(
                success=True,
                data={
                    "report": response.content[0].text,
                    "topic": topic,
                    "competitors": competitors,
                },
                metadata={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )

        except Exception as e:
            return SkillOutput(success=False, data=None, error=str(e))

    async def _research_competitor(self, competitor: str, topic: str, dimensions: list) -> str:
        """搜索单个竞品信息"""
        if "web_search" not in self.tools:
            return f"{competitor}: 搜索工具不可用"

        query = f"{competitor} {topic} {' '.join(dimensions[:2])}"
        result = await self.tools["web_search"](query=query, num_results=3)
        return str(result)[:1000]
```

### 7.3 Skill Registry：统一管理和发现

```python
class SkillRegistry:
    """Skill 注册中心"""

    def __init__(self):
        self._skills: dict[str, type[BaseSkill]] = {}

    def register(self, skill_class: type[BaseSkill]):
        """注册 Skill"""
        self._skills[skill_class.name] = skill_class
        return skill_class  # 支持装饰器用法

    def get(self, name: str) -> type[BaseSkill] | None:
        return self._skills.get(name)

    def list_skills(self) -> list[dict]:
        return [
            {
                "name": cls.name,
                "description": cls.description,
                "inputs": [
                    {"name": i.name, "type": i.type, "required": i.required}
                    for i in cls.inputs
                ],
            }
            for cls in self._skills.values()
        ]

    def instantiate(self, name: str, tools: dict = None) -> BaseSkill:
        cls = self._skills.get(name)
        if not cls:
            raise ValueError(f"Skill '{name}' 未注册")
        return cls(tools=tools)


# 全局注册中心
registry = SkillRegistry()

# 注册
registry.register(CompetitorAnalysisSkill)

# 可以用装饰器
@registry.register
class SummarySkill(BaseSkill):
    name = "summarize"
    description = "将长文本压缩成结构化摘要"
    inputs = [
        SkillInput("text", "string", "要总结的文本", required=True),
        SkillInput("max_words", "integer", "摘要最大字数", required=False, default=200),
        SkillInput("format", "string", "输出格式: bullet/paragraph", required=False, default="bullet"),
    ]

    async def execute(self, **kwargs) -> SkillOutput:
        params = self.validate_inputs(**kwargs)
        prompt = f"""将以下文本总结为{params['format']}格式的摘要，不超过{params['max_words']}字：

{params['text'][:4000]}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return SkillOutput(success=True, data=response.content[0].text)
```

### 7.4 把 Skill 挂到 Agent 上

```python
class SkillEnabledAgent:
    """
    支持 Skill 的 Agent：
    - LLM 可以决定调用哪个 Skill
    - Skill 内部封装了工具调用逻辑
    """

    def __init__(
        self,
        skill_registry: SkillRegistry,
        mcp_agent: MCPAgent | None = None,
    ):
        self.registry = skill_registry
        self.mcp = mcp_agent
        self.client = anthropic.Anthropic()

    def _skills_as_tools(self) -> list[dict]:
        """把 Skill 暴露为 LLM 工具"""
        tools = []
        for skill_info in self.registry.list_skills():
            props = {
                inp["name"]: {
                    "type": inp["type"],
                    "description": f"参数: {inp['name']}",
                }
                for inp in skill_info["inputs"]
            }
            required = [
                inp["name"] for inp in skill_info["inputs"] if inp["required"]
            ]
            tools.append({
                "name": f"skill_{skill_info['name']}",
                "description": skill_info["description"],
                "input_schema": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            })
        return tools

    async def run(self, goal: str) -> str:
        tools = self._skills_as_tools()
        # 也可以把 MCP 工具加进来
        if self.mcp:
            tools += self.mcp._tool_schemas

        messages = [{"role": "user", "content": goal}]

        for _ in range(10):
            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                tools=tools,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                texts = [b.text for b in response.content if b.type == "text"]
                return texts[0] if texts else ""

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                args = block.input

                if tool_name.startswith("skill_"):
                    # 调用 Skill
                    skill_name = tool_name[len("skill_"):]
                    mcp_tools = {}
                    if self.mcp:
                        mcp_tools = {
                            t["name"]: lambda n=t["name"], **kw: self.mcp.session.call_tool(n, kw)
                            for t in self.mcp._tool_schemas
                        }
                    skill = self.registry.instantiate(skill_name, tools=mcp_tools)
                    output = await skill.execute(**args)
                    result_text = str(output.data) if output.success else f"Skill 失败: {output.error}"
                elif self.mcp:
                    # 直接调用 MCP 工具
                    result_text = await self.mcp.call_tool(tool_name, args)
                else:
                    result_text = f"未知工具: {tool_name}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

            messages.append({"role": "user", "content": tool_results})

        return "超过最大步数"
```

---

## 8. MCP vs 直接 Function Calling

什么时候用 MCP，什么时候直接写工具函数？

| 维度 | 直接 Function Calling | MCP |
|------|----------------------|-----|
| 开发速度 | 快，直接写函数 | 慢，需要实现 Server |
| 复用性 | 只能在当前 Agent 用 | 任意 MCP Client 都能用 |
| 跨语言 | 受限（同进程） | 支持（任何语言实现 Server） |
| 部署隔离 | 工具和 Agent 同进程 | Server 独立部署、独立扩容 |
| 生态 | 自己维护 | 可以直接用社区 MCP Server |
| 调试 | 直接打断点 | 需要跨进程调试 |

**结论**：
- 原型 / 一次性工具 → 直接 Function Calling
- 需要复用 / 跨团队共享 / 需要独立部署 → MCP
- 复杂业务逻辑 + 多步骤 → Skill（在 MCP 之上封装）

---

## 9. 社区 MCP Server 生态

MCP 开放以来，社区和各大公司贡献了大量现成的 Server：

```bash
# 文件系统
npx @modelcontextprotocol/server-filesystem /path/to/allowed

# GitHub
npx @modelcontextprotocol/server-github

# PostgreSQL
npx @modelcontextprotocol/server-postgres postgresql://user:pass@host/db

# Slack
npx @modelcontextprotocol/server-slack

# Brave Search
npx @modelcontextprotocol/server-brave-search

# Puppeteer（浏览器控制）
npx @modelcontextprotocol/server-puppeteer
```

在 Claude Desktop 的配置文件里（`claude_desktop_config.json`）加入：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/projects"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "<your-token>"
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
    }
  }
}
```

配置完重启 Claude Desktop，Claude 就能直接操作文件系统、读写 GitHub、查数据库。

---

## 10. 小结

```mermaid
graph TD
    A[用户目标] --> B[SkillEnabledAgent]
    B --> C{需要什么能力?}
    C -->|复杂多步任务| D[Skill<br/>封装好的能力模块]
    C -->|原子工具调用| E[MCP Server<br/>标准化工具接口]
    D --> E
    E --> F[实际执行<br/>搜索/数据库/文件/API]
    F --> B
    B --> G[最终结果]
```

三层能力架构：
- **MCP Tools**：原子操作，标准化协议，可跨 Agent 复用
- **Skills**：业务逻辑封装，组合多个工具，有输入输出规范
- **Agent**：接入 Skills 和 MCP，通过 LLM 动态决策调用哪个

MCP 最大的价值不是技术复杂性（协议本身很简单），而是**生态效应**：你实现一个 MCP Server，全世界所有支持 MCP 的 Agent 都能用它。这是 Agent 能力标准化的基础。

---

*AI Agent 系列完结（共 8 篇）。*
