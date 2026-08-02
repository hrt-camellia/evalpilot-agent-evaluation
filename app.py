from __future__ import annotations

import ast
import asyncio
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
TASKS_FILE = DATA_DIR / "tasks.json"
RUNS_DIR = DATA_DIR / "runs"
STATIC_DIR = APP_DIR / "static"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="EvalPilot Real Evaluation", version="0.7.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

JOBS: dict[str, dict[str, Any]] = {}
JOB_LOCK = asyncio.Lock()

METRICS = [
    "completion",
    "correctness",
    "instruction",
    "intervention",
    "efficiency",
    "usability",
]

DEFAULT_WEIGHTS = {
    "completion": 0.20,
    "correctness": 0.25,
    "instruction": 0.15,
    "intervention": 0.10,
    "efficiency": 0.10,
    "usability": 0.20,
}

FAILURE_TYPES = [
    "无可执行代码",
    "语法或导入错误",
    "单元测试未全部通过",
    "全部单元测试失败",
    "输出重复",
    "输出疑似截断",
    "关键约束未覆盖",
    "受安全策略阻止",
    "执行超时",
    "执行工具异常",
]


PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "mock": {"protocol": "mock", "base_url": ""},
    "openai": {"protocol": "openai_responses", "base_url": "https://api.openai.com/v1"},
    "anthropic": {"protocol": "anthropic", "base_url": "https://api.anthropic.com"},
    "google_gemini": {"protocol": "openai_compatible", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai"},
    "deepseek": {"protocol": "openai_compatible", "base_url": "https://api.deepseek.com"},
    "xai": {"protocol": "openai_compatible", "base_url": "https://api.x.ai/v1"},
    "qwen_dashscope": {"protocol": "openai_compatible", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
    "kimi": {"protocol": "openai_compatible", "base_url": "https://api.moonshot.cn/v1"},
    "zhipu_glm": {"protocol": "openai_compatible", "base_url": "https://open.bigmodel.cn/api/paas/v4"},
    "minimax": {"protocol": "openai_compatible", "base_url": "https://api.minimaxi.com/v1"},
    "volcengine_ark": {"protocol": "openai_compatible", "base_url": "https://ark.cn-beijing.volces.com/api/v3"},
    "baidu_qianfan": {"protocol": "openai_compatible", "base_url": "https://qianfan.baidubce.com/v2"},
    "tencent_hunyuan": {"protocol": "openai_compatible", "base_url": "https://api.hunyuan.cloud.tencent.com/v1"},
    "openrouter": {"protocol": "openai_compatible", "base_url": "https://openrouter.ai/api/v1"},
    "siliconflow": {"protocol": "openai_compatible", "base_url": "https://api.siliconflow.cn/v1"},
    "ollama": {"protocol": "openai_compatible", "base_url": "http://localhost:11434/v1"},
    "lm_studio": {"protocol": "openai_compatible", "base_url": "http://localhost:1234/v1"},
    "custom_openai": {"protocol": "openai_compatible", "base_url": "http://localhost:11434/v1"},
}

LOCAL_NO_KEY_PROVIDERS = {"ollama", "lm_studio"}


class ProviderConfig(BaseModel):
    provider: Literal[
        "mock",
        "openai",
        "anthropic",
        "google_gemini",
        "deepseek",
        "xai",
        "qwen_dashscope",
        "kimi",
        "zhipu_glm",
        "minimax",
        "volcengine_ark",
        "baidu_qianfan",
        "tencent_hunyuan",
        "openrouter",
        "siliconflow",
        "ollama",
        "lm_studio",
        "custom_openai",
    ]
    model: str = Field(min_length=1)
    api_key: str = ""
    base_url: str = ""
    max_output_tokens: int = Field(default=1800, ge=128, le=16000)
    temperature: float | None = Field(default=0.2, ge=0, le=2)
    label: str = ""


class TaskModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(min_length=1)
    task_type: str
    difficulty: Literal["简单", "中等", "困难"] = "中等"
    language: str = "python"
    prompt: str = Field(min_length=1)
    expected: str = ""
    required_terms: list[str] = []
    test_code: str = ""
    timeout_seconds: int = Field(default=8, ge=1, le=60)
    tags: list[str] = []


class TaskCreate(BaseModel):
    name: str
    task_type: str
    difficulty: Literal["简单", "中等", "困难"] = "中等"
    language: str = "python"
    prompt: str
    expected: str = ""
    required_terms: list[str] = []
    test_code: str = ""
    timeout_seconds: int = 8
    tags: list[str] = []


class RunRequest(BaseModel):
    name: str = "AI Coding Agent 结构化输出与自修复评测"
    agent_a: ProviderConfig
    agent_b: ProviderConfig
    judge: ProviderConfig | None = None
    task_ids: list[str] = []
    execution_mode: Literal["disabled", "local", "docker"] = "local"
    weights: dict[str, float] = DEFAULT_WEIGHTS
    parallelism: int = Field(default=1, ge=1, le=4)
    repeat_count: int = Field(default=3, ge=1, le=5)
    repair_attempts: int = Field(default=1, ge=0, le=1)


class ConnectionTestRequest(BaseModel):
    provider: ProviderConfig


class ModelListRequest(BaseModel):
    provider: ProviderConfig


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_tasks() -> list[dict[str, Any]]:
    if not TASKS_FILE.exists():
        return []
    return json.loads(TASKS_FILE.read_text(encoding="utf-8"))


def save_tasks(tasks: list[dict[str, Any]]) -> None:
    TASKS_FILE.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def sanitize_provider(config: ProviderConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    data = config.model_dump()
    data["api_key"] = ""
    return data


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    safe = {key: max(0.0, float(weights.get(key, 0.0))) for key in METRICS}
    total = sum(safe.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    return {key: value / total for key, value in safe.items()}


def weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    normalized = normalize_weights(weights)
    return round(
        sum(float(scores.get(key, 0)) * normalized[key] for key in METRICS), 2
    )


def provider_meta(config: ProviderConfig) -> dict[str, str]:
    return PROVIDER_DEFAULTS[config.provider]


def provider_base(config: ProviderConfig) -> str:
    return (config.base_url or provider_meta(config)["base_url"]).rstrip("/")


def append_endpoint(base: str, path: str) -> str:
    normalized_path = "/" + path.lstrip("/")
    if base.endswith(normalized_path):
        return base
    return base + normalized_path


def provider_endpoint(config: ProviderConfig) -> str:
    protocol = provider_meta(config)["protocol"]
    base = provider_base(config)
    if protocol == "openai_responses":
        return append_endpoint(base, "responses")
    if protocol == "anthropic":
        if base.endswith("/v1"):
            return append_endpoint(base, "messages")
        return append_endpoint(base, "v1/messages")
    if protocol == "openai_compatible":
        return append_endpoint(base, "chat/completions")
    return "mock://local"


def model_list_endpoint(config: ProviderConfig) -> str:
    protocol = provider_meta(config)["protocol"]
    base = provider_base(config)
    if protocol == "anthropic":
        if base.endswith("/v1"):
            return append_endpoint(base, "models")
        return append_endpoint(base, "v1/models")
    return append_endpoint(base, "models")


def extract_openai_response_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    chunks: list[str] = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text", "")
                if isinstance(text, str):
                    chunks.append(text)
                elif isinstance(text, dict):
                    chunks.append(str(text.get("value", "")))
    return "\n".join(filter(None, chunks)).strip()


async def call_provider(
    config: ProviderConfig,
    prompt: str,
    system_prompt: str = "",
    response_mode: Literal["text", "code_json"] = "text",
) -> dict[str, Any]:
    if config.provider == "mock":
        started = time.perf_counter()
        await asyncio.sleep(0.15)
        code = """```python
def solve(value):
    return value
```"""
        if response_mode == "code_json":
            code = "def solve(value):\n    return value"
        return {
            "text": code,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "input_tokens": max(1, len(prompt) // 4),
            "output_tokens": 20,
            "load_duration_ms": 0,
            "generation_duration_ms": 0,
            "done_reason": "mock",
            "structured_output": response_mode == "code_json",
            "raw": {"mock": True},
        }

    if not config.api_key and config.provider not in LOCAL_NO_KEY_PROVIDERS:
        raise ValueError(f"{config.label or config.model} 未填写 API Key")

    timeout = httpx.Timeout(180.0, connect=20.0)
    started = time.perf_counter()

    if config.provider == "ollama":
        base = provider_base(config)
        if base.endswith("/v1"):
            base = base[:-3]
        endpoint = append_endpoint(base, "api/chat")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        options: dict[str, Any] = {
            "num_predict": config.max_output_tokens,
            "repeat_penalty": 1.20,
            "repeat_last_n": 128,
        }
        if config.temperature is not None:
            options["temperature"] = config.temperature
        payload = {
            "model": config.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "15m",
            "options": options,
        }
        if response_mode == "code_json":
            payload["format"] = {
                "type": "object",
                "properties": {
                    "code": {"type": "string"}
                },
                "required": ["code"],
                "additionalProperties": False,
            }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
            )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        if response.status_code >= 400:
            raise RuntimeError(
                f"Ollama 请求失败：HTTP {response.status_code}；"
                f"{response.text[:1500]}"
            )
        data = response.json()
        text = ((data.get("message") or {}).get("content") or "").strip()
        structured_ok = False
        if response_mode == "code_json" and text:
            try:
                structured = json.loads(text)
                candidate = structured.get("code", "")
                if isinstance(candidate, str) and candidate.strip():
                    text = candidate.strip()
                    structured_ok = True
            except json.JSONDecodeError:
                structured_ok = False
        if not text:
            raise RuntimeError("Ollama 返回成功，但未解析到文本输出")
        return {
            "text": text,
            "structured_output": structured_ok,
            "latency_ms": latency_ms,
            "input_tokens": int(data.get("prompt_eval_count", 0) or 0),
            "output_tokens": int(data.get("eval_count", 0) or 0),
            "load_duration_ms": round(
                float(data.get("load_duration", 0) or 0) / 1_000_000, 2
            ),
            "generation_duration_ms": round(
                float(data.get("eval_duration", 0) or 0) / 1_000_000, 2
            ),
            "done_reason": str(data.get("done_reason", "")),
            "raw": data,
        }

    endpoint = provider_endpoint(config)
    protocol = provider_meta(config)["protocol"]
    async with httpx.AsyncClient(timeout=timeout) as client:
        if protocol == "openai_responses":
            headers = {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            }
            full_prompt = (
                f"{system_prompt.strip()}\n\n{prompt}".strip()
                if system_prompt else prompt
            )
            payload = {
                "model": config.model,
                "input": full_prompt,
                "max_output_tokens": config.max_output_tokens,
            }
            response = await client.post(endpoint, headers=headers, json=payload)
        elif protocol == "anthropic":
            headers = {
                "x-api-key": config.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload: dict[str, Any] = {
                "model": config.model,
                "max_tokens": config.max_output_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if config.temperature is not None:
                payload["temperature"] = config.temperature
            if system_prompt:
                payload["system"] = system_prompt
            response = await client.post(endpoint, headers=headers, json=payload)
        else:
            headers = {"Content-Type": "application/json"}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"
            if config.provider == "openrouter":
                headers["HTTP-Referer"] = "http://127.0.0.1:8000"
                headers["X-Title"] = "EvalPilot"
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload: dict[str, Any] = {
                "model": config.model,
                "messages": messages,
                "max_tokens": config.max_output_tokens,
            }
            if config.temperature is not None and config.provider != "kimi":
                temperature = config.temperature
                if config.provider == "minimax":
                    temperature = min(1.0, max(0.01, temperature))
                payload["temperature"] = temperature
            response = await client.post(endpoint, headers=headers, json=payload)
            if response.status_code in {400, 422}:
                retry = await client.post(
                    endpoint,
                    headers=headers,
                    json={"model": config.model, "messages": messages},
                )
                if retry.status_code < 400:
                    response = retry

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    if response.status_code >= 400:
        raise RuntimeError(
            f"API 请求失败：HTTP {response.status_code}；{response.text[:1500]}"
        )
    data = response.json()
    if protocol == "openai_responses":
        text = extract_openai_response_text(data)
        usage = data.get("usage", {}) or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
    elif protocol == "anthropic":
        text = "\n".join(
            block.get("text", "")
            for block in data.get("content", []) or []
            if block.get("type") == "text"
        )
        usage = data.get("usage", {}) or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
    else:
        choice = (data.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "")
        usage = data.get("usage", {}) or {}
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    if not text:
        raise RuntimeError("API 返回成功，但未解析到文本输出")
    return {
        "text": str(text),
        "latency_ms": latency_ms,
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "load_duration_ms": 0,
        "generation_duration_ms": 0,
        "done_reason": "",
        "structured_output": False,
        "raw": data,
    }




def _candidate_python_blocks(text: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    # Accept fenced blocks even when the language marker is followed by spaces
    # rather than an immediate newline.
    for match in re.finditer(
        r"```(?:python|py)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        block = match.group(1).strip()
        if block:
            candidates.append(("fenced", block))

    # Fallback: many small models add prose before otherwise valid code.
    lines = text.strip().splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^\s*(?:from\s+\w+\s+import|import\s+\w+|def\s+\w+|class\s+\w+|@\w+)", line):
            block = "\n".join(lines[index:]).strip()
            block = re.sub(r"\n```.*$", "", block, flags=re.DOTALL).strip()
            if block:
                candidates.append(("raw_slice", block))
            break

    raw = text.strip()
    if raw:
        candidates.append(("raw", raw))
    return candidates


def extract_code(
    text: str,
    language: str = "python",
    required_terms: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    required_terms = required_terms or []
    candidates = _candidate_python_blocks(text)
    best_code = ""
    best_meta: dict[str, Any] = {
        "method": "none",
        "candidate_count": len(candidates),
        "syntax_ok": None,
        "syntax_error": "",
        "required_term_ratio": 0.0,
    }
    best_score = float("-inf")

    for method, code in candidates:
        syntax_ok = None
        syntax_error = ""
        if language.lower() == "python":
            try:
                ast.parse(code)
                syntax_ok = True
            except SyntaxError as exc:
                syntax_ok = False
                syntax_error = f"{exc.msg}（第 {exc.lineno} 行）"
        term_ratio = (
            sum(term.lower() in code.lower() for term in required_terms)
            / len(required_terms)
            if required_terms
            else 1.0
        )
        code_marker = bool(re.search(r"(?m)^\\s*(?:def|class|import|from)\\s+", code))
        score = (
            (100 if syntax_ok is True else 0)
            + (20 if code_marker else 0)
            + 20 * term_ratio
            - (len(code) / 100000)
        )
        if method == "fenced":
            score += 5
        if score > best_score:
            best_score = score
            best_code = code
            best_meta = {
                "method": method,
                "candidate_count": len(candidates),
                "syntax_ok": syntax_ok,
                "syntax_error": syntax_error,
                "required_term_ratio": round(term_ratio, 4),
            }
    return best_code, best_meta


def analyze_output(
    text: str,
    language: str,
    output_tokens: int,
    max_output_tokens: int,
    required_terms: list[str] | None = None,
) -> dict[str, Any]:
    code, extraction = extract_code(text, language, required_terms)
    fence_count = text.count("```")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated_output = False
    repetition_ratio = 0.0
    if len(lines) >= 8:
        counts: dict[str, int] = {}
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
        max_count = max(counts.values(), default=1)
        repetition_ratio = max_count / len(lines)
        repeated_output = max_count >= 4 or repetition_ratio >= 0.30
    likely_truncated = (
        fence_count % 2 == 1
        or (
            max_output_tokens > 0
            and output_tokens >= max(1, int(max_output_tokens * 0.98))
        )
    )
    stripped_code = code.strip()
    program_marker = bool(re.search(
        r"(?m)^\s*(?:async\s+def|def|class|import|from)\s+",
        stripped_code,
    ))
    placeholder_output = bool(re.fullmatch(
        r"(?is)\s*(?:#\s*)?(?:your|write\s+your)\s+code\s+here[.!]?\s*",
        stripped_code,
    ))
    path_like_output = bool(re.fullmatch(
        r"(?is)\s*(?:[A-Za-z]:[\\/]|[\\/])[^\r\n]+\s*",
        stripped_code,
    ))
    empty_json_wrapper = bool(re.fullmatch(
        r'(?is)\s*\{\s*["\']code["\']\s*:\s*["\']\s*["\']\s*\}\s*',
        stripped_code,
    ))
    has_real_code = bool(
        stripped_code
        and program_marker
        and not placeholder_output
        and not path_like_output
        and not empty_json_wrapper
    )
    return {
        "code": code,
        "has_code": has_real_code,
        "program_marker": program_marker,
        "placeholder_output": placeholder_output,
        "path_like_output": path_like_output,
        "empty_json_wrapper": empty_json_wrapper,
        "extraction_method": extraction["method"],
        "candidate_count": extraction["candidate_count"],
        "required_term_ratio": extraction["required_term_ratio"],
        "fence_count": fence_count,
        "unclosed_fence": fence_count % 2 == 1,
        "repeated_output": repeated_output,
        "repetition_ratio": round(repetition_ratio, 4),
        "likely_truncated": likely_truncated,
        "syntax_ok": extraction["syntax_ok"],
        "syntax_error": extraction["syntax_error"],
    }


def deterministic_evaluation(
    task: dict[str, Any],
    output: str,
    execution: dict[str, Any],
    latency_ms: float,
) -> dict[str, Any]:
    required = [term for term in task.get("required_terms", []) if term]
    hit_count = sum(1 for term in required if term.lower() in output.lower())
    term_ratio = hit_count / len(required) if required else 1.0
    diagnostics = execution.get("diagnostics", {})
    test_ratio = execution.get("test_case_pass_rate")
    syntax_ok = diagnostics.get("syntax_ok") is not False
    format_ok = diagnostics.get("fence_count", 0) == 2

    if test_ratio is not None:
        objective = float(test_ratio)
        completion = 15 + 70 * objective + 15 * float(syntax_ok)
        correctness = 100 * objective
        instruction = 20 + 50 * max(term_ratio, objective) + 30 * float(format_ok)
        intervention = 10 + 90 * objective
        usability = 10 + 90 * objective
    else:
        has_code = bool(diagnostics.get("has_code"))
        completion = 55 if has_code else 10
        correctness = 45 if has_code and syntax_ok else 0
        instruction = 30 + 50 * term_ratio + 20 * float(format_ok)
        intervention = 40 if has_code else 10
        usability = 45 if has_code and syntax_ok else 10

    efficiency = 90.0
    if diagnostics.get("repeated_output"):
        efficiency -= 45
    if diagnostics.get("likely_truncated"):
        efficiency -= 25
    efficiency = max(10.0, efficiency)

    failures: list[str] = []
    status = execution.get("status")
    if status == "no_code":
        failures.append("无可执行代码")
    if status == "syntax_error":
        failures.append("Python语法错误")
    if status == "test_error":
        details = "\n".join(execution.get("error_details", []) or [])
        failures.append(
            "测试导入或入口错误" if "ImportError" in details else "测试运行错误"
        )
    if test_ratio is not None and test_ratio < 1:
        failures.append("单元测试未全部通过")
    if test_ratio == 0:
        failures.append("全部单元测试失败")
    if diagnostics.get("repeated_output"):
        failures.append("输出重复")
    if diagnostics.get("likely_truncated"):
        failures.append("输出疑似截断")
    if term_ratio < 1:
        failures.append("静态关键字未命中")
    if status == "blocked":
        failures.append("受安全策略阻止")
    if status == "timeout":
        failures.append("执行超时")
    if status == "tool_error":
        failures.append("执行工具异常")

    scores = {
        "completion": round(min(100, completion), 1),
        "correctness": round(min(100, correctness), 1),
        "instruction": round(min(100, instruction), 1),
        "intervention": round(min(100, intervention), 1),
        "efficiency": round(min(100, efficiency), 1),
        "usability": round(min(100, usability), 1),
    }
    return {
        "scores": scores,
        "failure_types": list(dict.fromkeys(failures)),
        "summary": (
            f"自动测试通过 {execution.get('tests_passed', 0)}/"
            f"{execution.get('tests_run', 0)}；状态 {status}。"
        ),
        "source": "deterministic",
        "objective_test_rate": test_ratio,
    }




def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start < 0:
        raise ValueError("未找到 JSON 对象")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : i + 1])
    raise ValueError("JSON 对象不完整")


async def judge_output(
    judge: ProviderConfig,
    task: dict[str, Any],
    output: str,
    execution: dict[str, Any],
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    judge_prompt = f"""
你是 AI Coding Agent 评测员。请依据任务目标、成功标准、Agent 输出与确定性测试结果评分。
必须只输出一个 JSON 对象，不要使用 Markdown。

任务名称：{task.get("name")}
任务类型：{task.get("task_type")}
难度：{task.get("difficulty")}
用户任务：
{task.get("prompt")}

成功标准：
{task.get("expected")}

Agent 输出：
{output}

确定性执行结果：
{json.dumps(execution, ensure_ascii=False)}

确定性初评分：
{json.dumps(deterministic, ensure_ascii=False)}

请输出：
{{
  "scores": {{
    "completion": 0到100,
    "correctness": 0到100,
    "instruction": 0到100,
    "intervention": 0到100,
    "efficiency": 0到100,
    "usability": 0到100
  }},
  "failure_types": ["从以下类型选择，可为空：{",".join(FAILURE_TYPES)}"],
  "summary": "不超过120字的评测结论",
  "evidence": ["最多3条证据"]
}}

评分要求：
1. 单元测试失败时，correctness 通常不得超过60。
2. 不得仅凭文风给高分，要检查任务约束和可执行性。
3. failure_types 只能使用给定中文枚举。
4. intervention 分数越高，表示越少需要人工补充或修正。
"""
    result = await call_provider(
        judge,
        judge_prompt.strip(),
        "你负责严格、可复核的 AI 产品评测，并稳定返回合法 JSON。",
    )
    parsed = parse_json_object(result["text"])
    scores = parsed.get("scores", {})
    validated_scores = {
        key: round(max(0, min(100, float(scores.get(key, 0)))), 1)
        for key in METRICS
    }
    failure_types = [
        item
        for item in parsed.get("failure_types", [])
        if item in FAILURE_TYPES
    ]
    return {
        "scores": validated_scores,
        "failure_types": failure_types,
        "summary": str(parsed.get("summary", ""))[:500],
        "evidence": [str(item)[:300] for item in parsed.get("evidence", [])[:3]],
        "source": "llm_judge",
        "judge_usage": {
            "latency_ms": result["latency_ms"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
        },
    }


BANNED_IMPORT_ROOTS = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests",
    "httpx", "urllib", "ftplib", "telnetlib", "multiprocessing", "ctypes",
    "importlib", "webbrowser",
}
BANNED_CALL_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "input", "breakpoint",
}
BANNED_ATTRIBUTE_CALLS = {
    "system", "popen", "spawn", "fork", "kill", "remove", "unlink",
    "rmdir", "rmtree", "rename", "replace", "chmod", "chown", "connect",
    "request", "urlopen", "run", "Popen",
}


def scan_local_code_safety(code: str) -> list[str]:
    """Conservative static gate for local execution; not a full sandbox."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BANNED_IMPORT_ROOTS:
                    violations.append(f"禁止导入模块：{root}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BANNED_IMPORT_ROOTS:
                violations.append(f"禁止导入模块：{root}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALL_NAMES:
                violations.append(f"禁止调用：{node.func.id}")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in BANNED_ATTRIBUTE_CALLS:
                violations.append(f"禁止调用属性：{node.func.attr}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            violations.append(f"禁止访问双下划线属性：{node.attr}")
    return list(dict.fromkeys(violations))


def run_code(
    task: dict[str, Any],
    output: str,
    mode: str,
    output_tokens: int = 0,
    max_output_tokens: int = 0,
) -> dict[str, Any]:
    diagnostics = analyze_output(
        output,
        task.get("language", "python"),
        output_tokens,
        max_output_tokens,
        task.get("required_terms", []),
    )
    base = {
        "stdout": "", "stderr": "", "duration_ms": 0,
        "safety_violations": [], "tests_run": 0, "tests_passed": 0,
        "tests_failed": 0, "tests_error": 0,
        "test_case_pass_rate": None, "evaluable": False,
        "diagnostics": diagnostics,
    }
    if mode == "disabled":
        return {**base, "status": "disabled", "passed": None}
    if not task.get("test_code"):
        return {**base, "status": "no_tests", "passed": None}
    if task.get("language", "python").lower() != "python":
        return {**base, "status": "unsupported_language", "passed": None,
                "stderr": "当前自动执行仅支持 Python。"}
    if not diagnostics["has_code"]:
        return {**base, "status": "no_code", "passed": False,
                "stderr": "未提取到可执行代码。"}
    if diagnostics["syntax_ok"] is False:
        return {**base, "status": "syntax_error", "passed": False,
                "stderr": diagnostics["syntax_error"]}

    code = diagnostics["code"]
    timeout_seconds = int(task.get("timeout_seconds", 8))
    started = time.perf_counter()
    if mode == "local":
        violations = scan_local_code_safety(code)
        if violations:
            return {**base, "status": "blocked", "passed": False,
                    "stderr": "受限本地执行已阻止潜在危险代码。",
                    "safety_violations": violations}

    runner_code = r"""
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
suite = unittest.defaultTestLoader.loadTestsFromName("test_solution")
result = unittest.TestResult()
suite.run(result)
payload = {
    "tests_run": result.testsRun,
    "tests_failed": len(result.failures),
    "tests_error": len(result.errors),
    "failure_details": [text[-2000:] for _, text in result.failures],
    "error_details": [text[-2000:] for _, text in result.errors],
}
print("__EVALPILOT_RESULT__" + json.dumps(payload, ensure_ascii=False))
raise SystemExit(0 if not result.failures and not result.errors else 1)
""".strip()

    with tempfile.TemporaryDirectory(prefix="evalpilot_") as temp:
        temp_path = Path(temp)
        (temp_path / "solution.py").write_text(code, encoding="utf-8")
        (temp_path / "test_solution.py").write_text(task["test_code"], encoding="utf-8")
        (temp_path / "runner.py").write_text(runner_code, encoding="utf-8")
        if mode == "docker":
            if not shutil.which("docker"):
                return {**base, "status": "tool_error", "passed": False,
                        "stderr": "未检测到 Docker。请改用 local 或安装 Docker。"}
            command = [
                "docker", "run", "--rm", "--network", "none",
                "--memory", "256m", "--cpus", "0.5", "--pids-limit", "64",
                "--read-only", "-e", "PYTHONDONTWRITEBYTECODE=1",
                "-v", f"{temp_path.resolve()}:/workspace:ro", "-w", "/workspace",
                "python:3.12-slim", "python", "-I", "runner.py",
            ]
            cwd = None
        else:
            command = [sys.executable, "-I", "runner.py"]
            cwd = temp_path
        env = {"PATH": os.environ.get("PATH", ""),
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
        try:
            completed = subprocess.run(
                command, cwd=cwd, capture_output=True, text=True,
                timeout=timeout_seconds, env=env
            )
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
            match = re.search(r"__EVALPILOT_RESULT__(\{.*\})", combined)
            if not match:
                return {**base, "status": "tool_error", "passed": False,
                        "stdout": (completed.stdout or "")[-6000:],
                        "stderr": "测试运行器未返回结构化结果。\n" +
                                  (completed.stderr or "")[-5000:],
                        "duration_ms": duration_ms}
            payload = json.loads(match.group(1))
            tests_run = int(payload.get("tests_run", 0))
            failed = int(payload.get("tests_failed", 0))
            errors = int(payload.get("tests_error", 0))
            passed_tests = max(0, tests_run - failed - errors)
            ratio = passed_tests / tests_run if tests_run else 0.0
            if tests_run == 0:
                status = "tool_error"
            elif errors:
                status = "test_error"
            elif failed:
                status = "partial" if passed_tests else "failed_tests"
            else:
                status = "passed"
            return {
                **base, "status": status, "passed": ratio == 1.0,
                "stdout": (completed.stdout or "")[-6000:],
                "stderr": (completed.stderr or "")[-6000:],
                "duration_ms": duration_ms, "tests_run": tests_run,
                "tests_passed": passed_tests, "tests_failed": failed,
                "tests_error": errors, "test_case_pass_rate": round(ratio, 4),
                "evaluable": tests_run > 0,
                "failure_details": payload.get("failure_details", []),
                "error_details": payload.get("error_details", []),
            }
        except subprocess.TimeoutExpired:
            return {**base, "status": "timeout", "passed": False,
                    "stderr": "执行超时",
                    "duration_ms": round((time.perf_counter()-started)*1000,2)}
        except Exception as exc:
            return {**base, "status": "tool_error", "passed": False,
                    "stderr": str(exc),
                    "duration_ms": round((time.perf_counter()-started)*1000,2)}




def candidate_prompt(task: dict[str, Any]) -> str:
    return f"""
完成下面的 Python 编程任务。

任务：
{task.get("prompt")}

成功标准：
{task.get("expected")}

必须包含的入口或关键项：
{", ".join(task.get("required_terms", []))}

输出要求：
- 返回完整可运行的 Python 实现；
- 不写解释、分析或测试结果；
- 不重复题目；
- 代码应能直接保存为 solution.py。
""".strip()


def repair_prompt(
    task: dict[str, Any],
    previous_output: str,
    execution: dict[str, Any],
) -> str:
    diagnostics = execution.get("diagnostics", {})
    error = execution.get("stderr") or diagnostics.get("syntax_error") or execution.get("status")
    return f"""
你上一次为下面任务生成的输出缺少可执行代码、必要入口，或存在 Python 语法/导入问题。请修复输出协议、缺失入口或语法，使其成为完整可运行代码；不要解释。

任务：
{task.get("prompt")}

成功标准：
{task.get("expected")}

必须包含的入口或关键项：
{", ".join(task.get("required_terms", []))}

上一次输出：
{previous_output[:8000]}

检测到的问题：
{error}

请返回修复后的完整 Python 代码。
""".strip()




async def evaluate_one_agent(
    config: ProviderConfig,
    task: dict[str, Any],
    judge: ProviderConfig | None,
    execution_mode: str,
    weights: dict[str, float],
    repair_attempts: int = 1,
) -> dict[str, Any]:
    generation = await call_provider(
        config,
        candidate_prompt(task),
        "你是严谨的 Python 代码生成器。只生成完整代码，不要重复。",
        response_mode="code_json" if config.provider == "ollama" else "text",
    )
    first_execution = await asyncio.to_thread(
        run_code, task, generation["text"], execution_mode,
        generation["output_tokens"], config.max_output_tokens,
    )
    first_pass = {
        "output": generation["text"],
        "usage": {
            "latency_ms": generation["latency_ms"],
            "input_tokens": generation["input_tokens"],
            "output_tokens": generation["output_tokens"],
            "load_duration_ms": generation.get("load_duration_ms", 0),
            "generation_duration_ms": generation.get("generation_duration_ms", 0),
            "done_reason": generation.get("done_reason", ""),
            "structured_output": generation.get("structured_output", False),
        },
        "execution": first_execution,
    }

    final_generation = generation
    final_execution = first_execution
    repair_attempted = False
    repair_succeeded = False
    repair_record = None

    # A standardized, single protocol/entrypoint/syntax repair is applied equally.
    # Placeholder comments and missing required entrypoints must not be treated as
    # valid code merely because Python can parse them.
    first_diag = first_execution.get("diagnostics", {})
    repairable_test_error = (
        first_execution.get("status") == "test_error"
        and float(first_diag.get("required_term_ratio", 0) or 0) < 1.0
        and int(first_execution.get("tests_passed", 0) or 0) == 0
    )
    should_repair = (
        not first_execution.get("evaluable")
        or first_execution.get("status") in {"no_code", "syntax_error"}
        or repairable_test_error
    )
    if repair_attempts > 0 and should_repair:
        repair_attempted = True
        repair_config = config.model_copy(update={
            "temperature": 0.0,
            "max_output_tokens": max(384, min(config.max_output_tokens, 768)),
        })
        repaired = await call_provider(
            repair_config,
            repair_prompt(task, generation["text"], first_execution),
            "修复上一次输出。只生成完整、可运行的 Python 代码。",
            # Some small/older Ollama models satisfy the JSON schema with an
            # empty placeholder. The standardized recovery therefore falls back
            # to plain code for both A and B.
            response_mode="text",
        )
        repaired_execution = await asyncio.to_thread(
            run_code, task, repaired["text"], execution_mode,
            repaired["output_tokens"], repair_config.max_output_tokens,
        )
        repair_succeeded = bool(repaired_execution.get("evaluable"))
        repair_record = {
            "output": repaired["text"],
            "usage": {
                "latency_ms": repaired["latency_ms"],
                "input_tokens": repaired["input_tokens"],
                "output_tokens": repaired["output_tokens"],
                "load_duration_ms": repaired.get("load_duration_ms", 0),
                "generation_duration_ms": repaired.get("generation_duration_ms", 0),
                "done_reason": repaired.get("done_reason", ""),
                "structured_output": repaired.get("structured_output", False),
            },
            "execution": repaired_execution,
        }
        final_generation = repaired
        final_execution = repaired_execution

    deterministic = deterministic_evaluation(
        task, final_generation["text"], final_execution,
        final_generation["latency_ms"]
    )
    final_eval = deterministic
    judge_error = ""
    if judge is not None:
        try:
            final_eval = await judge_output(
                judge, task, final_generation["text"], final_execution, deterministic
            )
        except Exception as exc:
            judge_error = str(exc)
            final_eval = deterministic

    diagnostic_score = weighted_score(final_eval["scores"], weights)
    test_ratio = final_execution.get("test_case_pass_rate")
    comparable_score: float | None = None
    if test_ratio is not None:
        comparable_score = min(diagnostic_score, 35 + 65 * float(test_ratio))

    total_latency = generation["latency_ms"]
    total_input = generation["input_tokens"]
    total_output = generation["output_tokens"]
    if repair_record:
        total_latency += repair_record["usage"]["latency_ms"]
        total_input += repair_record["usage"]["input_tokens"]
        total_output += repair_record["usage"]["output_tokens"]

    return {
        "provider": sanitize_provider(config),
        "output": final_generation["text"],
        "first_pass": first_pass,
        "repair_attempted": repair_attempted,
        "repair_succeeded": repair_succeeded,
        "repair": repair_record,
        "usage": {
            "latency_ms": round(total_latency, 2),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "load_duration_ms": final_generation.get("load_duration_ms", 0),
            "generation_duration_ms": final_generation.get("generation_duration_ms", 0),
            "done_reason": final_generation.get("done_reason", ""),
            "structured_output": final_generation.get("structured_output", False),
        },
        "execution": final_execution,
        "evaluation": final_eval,
        "deterministic_evaluation": deterministic,
        "judge_error": judge_error,
        "overall_score": round(comparable_score, 2) if comparable_score is not None else None,
        "diagnostic_score": round(diagnostic_score, 2),
    }




def mean_or_zero(values: list[float]) -> float:
    return round(statistics.fmean(values), 2) if values else 0.0


def std_or_zero(values: list[float]) -> float:
    return round(statistics.stdev(values), 2) if len(values) >= 2 else 0.0


def median_or_zero(values: list[float]) -> float:
    return round(statistics.median(values), 2) if values else 0.0


def _stage_rates(trials: list[dict[str, Any]]) -> dict[str, float]:
    total = len(trials) or 1
    def rate(predicate) -> float:
        return round(sum(bool(predicate(trial)) for trial in trials) / total, 4)
    return {
        "response_received": 1.0 if trials else 0.0,
        "first_syntax_valid": rate(
            lambda t: t.get("first_pass", {}).get("execution", {}).get("diagnostics", {}).get("syntax_ok") is True
        ),
        "first_tests_started": rate(
            lambda t: int(t.get("first_pass", {}).get("execution", {}).get("tests_run", 0)) > 0
        ),
        "repair_attempted": rate(lambda t: t.get("repair_attempted")),
        "repair_succeeded": rate(lambda t: t.get("repair_succeeded")),
        "final_code_extracted": rate(
            lambda t: t["execution"].get("diagnostics", {}).get("has_code")
        ),
        "final_syntax_valid": rate(
            lambda t: t["execution"].get("diagnostics", {}).get("syntax_ok") is True
        ),
        "final_tests_started": rate(
            lambda t: int(t["execution"].get("tests_run", 0)) > 0
        ),
        "any_test_passed": rate(
            lambda t: int(t["execution"].get("tests_passed", 0)) > 0
        ),
        "all_tests_passed": rate(
            lambda t: t["execution"].get("test_case_pass_rate") == 1
        ),
    }


def aggregate_agent_trials(
    trials: list[dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, Any]:
    if not trials:
        raise ValueError("没有可聚合的有效试验")
    metric_scores = {
        key: mean_or_zero([
            float(trial["evaluation"]["scores"].get(key, 0))
            for trial in trials
        ])
        for key in METRICS
    }
    comparable_scores = [
        float(trial["overall_score"])
        for trial in trials
        if trial.get("overall_score") is not None
    ]
    diagnostic_scores = [
        float(trial.get("diagnostic_score", 0)) for trial in trials
    ]
    test_rates = [
        float(trial["execution"]["test_case_pass_rate"])
        for trial in trials
        if trial["execution"].get("test_case_pass_rate") is not None
    ]
    full_pass_rate = (
        round(sum(rate == 1 for rate in test_rates) / len(test_rates), 4)
        if test_rates else None
    )
    test_case_pass_rate = (
        round(statistics.fmean(test_rates), 4) if test_rates else None
    )
    evaluable_rate = round(len(test_rates) / len(trials), 4)
    failure_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for trial in trials:
        status = str(trial["execution"].get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        for failure in trial["evaluation"].get("failure_types", []):
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
    representative = max(
        trials,
        key=lambda t: (
            t["execution"].get("test_case_pass_rate")
            if t["execution"].get("test_case_pass_rate") is not None
            else -1,
            t.get("diagnostic_score", 0),
        ),
    )
    latency_values = [float(t["usage"].get("latency_ms", 0)) for t in trials]
    load_values = [float(t["usage"].get("load_duration_ms", 0)) for t in trials]
    pipeline = _stage_rates(trials)
    return {
        "provider": representative["provider"],
        "output": representative["output"],
        "usage": {
            "latency_ms": median_or_zero(latency_values),
            "median_latency_ms": median_or_zero(latency_values),
            "median_load_duration_ms": median_or_zero(load_values),
            "input_tokens": round(statistics.fmean(
                [int(t["usage"].get("input_tokens", 0)) for t in trials]
            )),
            "output_tokens": round(statistics.fmean(
                [int(t["usage"].get("output_tokens", 0)) for t in trials]
            )),
        },
        "execution": {
            "status": "aggregate",
            "passed": full_pass_rate == 1 if full_pass_rate is not None else None,
            "full_pass_rate": full_pass_rate,
            "test_case_pass_rate": test_case_pass_rate,
            "evaluable_rate": evaluable_rate,
            "trial_count": len(trials),
            "diagnostics": representative["execution"].get("diagnostics", {}),
            "status_counts": status_counts,
            "pipeline": pipeline,
        },
        "evaluation": {
            "scores": metric_scores,
            "failure_types": [
                name for name, _ in sorted(
                    failure_counts.items(), key=lambda x: (-x[1], x[0])
                )
            ],
            "failure_counts": failure_counts,
            "summary": (
                f"基于 {len(trials)} 次运行聚合；可评测率 "
                f"{evaluable_rate * 100:.1f}%；测试用例通过率 "
                f"{(test_case_pass_rate or 0) * 100:.1f}%。"
            ),
            "source": (
                "llm_judge"
                if any(t["evaluation"].get("source") == "llm_judge" for t in trials)
                else "deterministic"
            ),
        },
        "overall_score": (
            mean_or_zero(comparable_scores) if comparable_scores else None
        ),
        "score_std": (
            std_or_zero(comparable_scores) if len(comparable_scores) >= 2 else 0.0
        ),
        "diagnostic_score": mean_or_zero(diagnostic_scores),
        "diagnostic_score_std": std_or_zero(diagnostic_scores),
        "full_pass_rate": full_pass_rate,
        "test_case_pass_rate": test_case_pass_rate,
        "evaluable_rate": evaluable_rate,
        "median_latency_ms": median_or_zero(latency_values),
        "median_load_duration_ms": median_or_zero(load_values),
        "status_counts": status_counts,
        "pipeline": pipeline,
        "trials": trials,
    }


def assess_evidence_level(
    request: RunRequest,
    selected: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    valid = [x for x in results if x.get("agent_a") and x.get("agent_b")]
    different_agents = (
        request.agent_a.provider, request.agent_a.model
    ) != (
        request.agent_b.provider, request.agent_b.model
    )
    all_a_trials = [
        trial for item in valid for trial in item["agent_a"].get("trials", [])
    ]
    all_b_trials = [
        trial for item in valid for trial in item["agent_b"].get("trials", [])
    ]
    evaluable_rate_a = (
        statistics.fmean(bool(t.get("execution", {}).get("evaluable")) for t in all_a_trials)
        if all_a_trials else 0.0
    )
    evaluable_rate_b = (
        statistics.fmean(bool(t.get("execution", {}).get("evaluable")) for t in all_b_trials)
        if all_b_trials else 0.0
    )
    min_evaluable = min(evaluable_rate_a, evaluable_rate_b)
    comparable_tasks = [
        item for item in valid
        if item["agent_a"].get("test_case_pass_rate") is not None
        and item["agent_b"].get("test_case_pass_rate") is not None
    ]
    paired_trials = 0
    total_trial_pairs = 0
    for item in valid:
        a_trials = item["agent_a"].get("trials", [])
        b_trials = item["agent_b"].get("trials", [])
        for a_trial, b_trial in zip(a_trials, b_trials):
            total_trial_pairs += 1
            if (
                a_trial.get("execution", {}).get("test_case_pass_rate") is not None
                and b_trial.get("execution", {}).get("test_case_pass_rate") is not None
            ):
                paired_trials += 1
    checklist = {
        "real_models": request.agent_a.provider != "mock" and request.agent_b.provider != "mock",
        "different_agents": different_agents,
        "execution_enabled": request.execution_mode != "disabled",
        "repeat_count": request.repeat_count >= 3,
        "task_coverage": len(selected) >= 3,
        "tests_available": bool(selected) and all(t.get("test_code") for t in selected),
        "run_complete": len(valid) == len(selected),
        "outputs_evaluable": min_evaluable >= 0.80,
        "paired_comparability": (
            total_trial_pairs > 0 and paired_trials >= math.ceil(total_trial_pairs * 0.80)
        ),
    }
    if all(checklist.values()):
        level = "可复核真实对比"
        note = "A/B在同一批任务上均形成可执行结果，可以比较客观测试通过率。"
    elif checklist["real_models"] and checklist["execution_enabled"] and not checklist["paired_comparability"]:
        level = "不可直接比较：输出可执行性不对称"
        note = (
            "至少一侧大量输出未进入自动测试。当前只能比较输出可执行性，"
            "不能把未执行视为0%并计算代码正确性提升。"
        )
    elif checklist["real_models"] and checklist["execution_enabled"]:
        level = "初步真实评测"
        note = "已产生真实输出，但覆盖、重复或完整性不足。"
    else:
        level = "流程演示"
        note = "当前结果不能作为真实能力结论。"
    return {
        "level": level,
        "note": note,
        "checklist": checklist,
        "minimum_evaluable_rate": round(min_evaluable, 4),
        "evaluable_rate_a": round(evaluable_rate_a, 4),
        "evaluable_rate_b": round(evaluable_rate_b, 4),
        "paired_trial_count": paired_trials,
        "total_trial_pairs": total_trial_pairs,
        "comparable_task_count": len(comparable_tasks),
        "selected_task_count": len(selected),
    }


async def warm_up_provider(config: ProviderConfig) -> dict[str, Any] | None:
    if config.provider != "ollama":
        return None
    warm_config = config.model_copy(update={"max_output_tokens": 8, "temperature": 0.0})
    try:
        result = await call_provider(warm_config, "只回复 OK。", "这是模型预热请求。")
        return {
            "ok": True,
            "latency_ms": result["latency_ms"],
            "load_duration_ms": result.get("load_duration_ms", 0),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def execute_run(run_id: str, request: RunRequest) -> None:
    async with JOB_LOCK:
        JOBS[run_id]["status"] = "running"
        JOBS[run_id]["started_at"] = utc_now()

    all_tasks = load_tasks()
    selected = (
        [task for task in all_tasks if task["id"] in request.task_ids]
        if request.task_ids else all_tasks
    )
    warmups = {
        "agent_a": await warm_up_provider(request.agent_a),
        "agent_b": await warm_up_provider(request.agent_b),
    }
    semaphore = asyncio.Semaphore(request.parallelism)
    results: list[dict[str, Any]] = []
    progress_lock = asyncio.Lock()

    async def advance_progress() -> None:
        async with progress_lock:
            async with JOB_LOCK:
                JOBS[run_id]["completed_units"] = JOBS[run_id].get("completed_units", 0) + 1

    async def run_pair(task: dict[str, Any], repeat_index: int):
        both_local = request.agent_a.provider == "ollama" and request.agent_b.provider == "ollama"
        if not both_local:
            return await asyncio.gather(
                evaluate_one_agent(request.agent_a, task, request.judge, request.execution_mode, request.weights, request.repair_attempts),
                evaluate_one_agent(request.agent_b, task, request.judge, request.execution_mode, request.weights, request.repair_attempts),
            )
        if repeat_index % 2 == 0:
            a = await evaluate_one_agent(request.agent_a, task, request.judge, request.execution_mode, request.weights, request.repair_attempts)
            b = await evaluate_one_agent(request.agent_b, task, request.judge, request.execution_mode, request.weights, request.repair_attempts)
        else:
            b = await evaluate_one_agent(request.agent_b, task, request.judge, request.execution_mode, request.weights, request.repair_attempts)
            a = await evaluate_one_agent(request.agent_a, task, request.judge, request.execution_mode, request.weights, request.repair_attempts)
        return a, b

    async def process_task(index: int, task: dict[str, Any]) -> None:
        async with semaphore:
            trials_a, trials_b, trial_errors = [], [], []
            for repeat_index in range(request.repeat_count):
                try:
                    a, b = await run_pair(task, repeat_index)
                    a["repeat_index"] = repeat_index + 1
                    b["repeat_index"] = repeat_index + 1
                    trials_a.append(a)
                    trials_b.append(b)
                except Exception:
                    trial_errors.append(traceback.format_exc(limit=4))
                finally:
                    await advance_progress()
            item = {"task": task, "agent_a": None, "agent_b": None,
                    "error": "", "trial_errors": trial_errors}
            if trials_a and trials_b:
                aa = aggregate_agent_trials(trials_a, request.weights)
                bb = aggregate_agent_trials(trials_b, request.weights)
                item["agent_a"], item["agent_b"] = aa, bb
                if aa.get("overall_score") is not None and bb.get("overall_score") is not None:
                    item["delta"] = round(
                        bb["overall_score"] - aa["overall_score"], 2
                    )
                else:
                    item["delta"] = None
                if (
                    aa.get("test_case_pass_rate") is not None
                    and bb.get("test_case_pass_rate") is not None
                ):
                    item["objective_delta"] = round(
                        bb["test_case_pass_rate"] - aa["test_case_pass_rate"],
                        4,
                    )
                else:
                    item["objective_delta"] = None
            else:
                item["error"] = trial_errors[-1] if trial_errors else "所有重复试验均失败"
            results.append(item)
            async with JOB_LOCK:
                JOBS[run_id]["completed_tasks"] = len(results)

    try:
        await asyncio.gather(*(process_task(i, t) for i, t in enumerate(selected)))
        order = {t["id"]: i for i, t in enumerate(selected)}
        results.sort(key=lambda x: order[x["task"]["id"]])
        valid = [x for x in results if x.get("agent_a") and x.get("agent_b")]
        a_scores = [
            float(x["agent_a"]["overall_score"])
            for x in valid
            if x["agent_a"].get("overall_score") is not None
        ]
        b_scores = [
            float(x["agent_b"]["overall_score"])
            for x in valid
            if x["agent_b"].get("overall_score") is not None
        ]
        all_a = [t for x in valid for t in x["agent_a"].get("trials", [])]
        all_b = [t for x in valid for t in x["agent_b"].get("trials", [])]

        def avg_execution(trials, field):
            vals = [t["execution"].get(field) for t in trials
                    if t["execution"].get(field) is not None]
            return round(statistics.fmean(float(v) for v in vals), 4) if vals else None

        evidence = assess_evidence_level(request, selected, results)
        summary = {
            "task_count": len(selected),
            "repeat_count": request.repeat_count,
            "total_agent_runs": len(selected) * request.repeat_count * 2,
            "successful_task_count": len(valid),
            "average_a": mean_or_zero(a_scores),
            "average_b": mean_or_zero(b_scores),
            "std_a": std_or_zero(a_scores),
            "std_b": std_or_zero(b_scores),
            "delta": (
                round(mean_or_zero(b_scores) - mean_or_zero(a_scores), 2)
                if a_scores and b_scores else None
            ),
            "full_pass_rate_a": avg_execution(all_a, "passed"),
            "full_pass_rate_b": avg_execution(all_b, "passed"),
            "test_case_pass_rate_a": avg_execution(all_a, "test_case_pass_rate"),
            "test_case_pass_rate_b": avg_execution(all_b, "test_case_pass_rate"),
            "evaluable_rate_a": avg_execution(all_a, "evaluable"),
            "evaluable_rate_b": avg_execution(all_b, "evaluable"),
            "median_latency_a": median_or_zero([float(t["usage"].get("latency_ms",0)) for t in all_a]),
            "median_latency_b": median_or_zero([float(t["usage"].get("latency_ms",0)) for t in all_b]),
            "median_load_duration_a": median_or_zero([float(t["usage"].get("load_duration_ms",0)) for t in all_a]),
            "median_load_duration_b": median_or_zero([float(t["usage"].get("load_duration_ms",0)) for t in all_b]),
        }
        run_record = {
            "id": run_id, "name": request.name,
            "created_at": JOBS[run_id]["created_at"],
            "started_at": JOBS[run_id].get("started_at"),
            "finished_at": utc_now(), "status": "completed",
            "execution_mode": request.execution_mode,
            "repeat_count": request.repeat_count,
            "repair_attempts": request.repair_attempts,
            "weights": normalize_weights(request.weights),
            "agent_a": sanitize_provider(request.agent_a),
            "agent_b": sanitize_provider(request.agent_b),
            "judge": sanitize_provider(request.judge),
            "warmups": warmups, "evidence": evidence,
            "summary": summary, "results": results,
        }
        (RUNS_DIR / f"{run_id}.json").write_text(
            json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        async with JOB_LOCK:
            JOBS[run_id].update(run_record)
    except Exception:
        async with JOB_LOCK:
            JOBS[run_id]["status"] = "failed"
            JOBS[run_id]["error"] = traceback.format_exc()




def load_run(run_id: str) -> dict[str, Any]:
    if run_id in JOBS:
        return JOBS[run_id]
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="评测记录不存在")
    return json.loads(path.read_text(encoding="utf-8"))


def make_markdown_report(run: dict[str, Any]) -> str:
    summary = run.get("summary", {})
    evidence = run.get("evidence", {})
    checklist = evidence.get("checklist", {})
    agent_a = run.get("agent_a") or {}
    agent_b = run.get("agent_b") or {}

    def pct(value: float | None) -> str:
        return "不可计算" if value is None else f"{value * 100:.1f}%"

    def score(value: float | None, std: float | None = None) -> str:
        if value is None:
            return "不适用"
        return f"{value:.2f}" if std is None else f"{value:.2f} ± {float(std or 0):.2f}"

    def agent_name(agent: dict[str, Any]) -> str:
        label = agent.get("label") or "未命名"
        return f"{label}（{agent.get('provider')} / {agent.get('model')}）"

    checklist_names = {
        "real_models": "A/B均为真实模型",
        "different_agents": "A/B模型配置不同",
        "execution_enabled": "启用代码执行",
        "repeat_count": "每任务至少重复3次",
        "task_coverage": "至少覆盖3个任务",
        "tests_available": "全部任务具有自动测试",
        "run_complete": "全部任务调用完成",
        "outputs_evaluable": "A/B可评测输出率均达到80%",
        "paired_comparability": "至少80%的任务可进行A/B配对比较",
    }

    lines = [
        f"# {run.get('name', 'EvalPilot 评测报告')}",
        "",
        f"> 证据等级：**{evidence.get('level', '未知')}**",
        f"> {evidence.get('note', '')}",
        "",
        "## 1. 评测配置",
        f"- Agent A：{agent_name(agent_a)}",
        f"- Agent B：{agent_name(agent_b)}",
        f"- A参数：max_tokens={agent_a.get('max_output_tokens')}，temperature={agent_a.get('temperature')}",
        f"- B参数：max_tokens={agent_b.get('max_output_tokens')}，temperature={agent_b.get('temperature')}",
        f"- 执行模式：{run.get('execution_mode')}",
        f"- 任务数：{summary.get('task_count', 0)}",
        f"- 每任务重复次数：{summary.get('repeat_count', 1)}",
        f"- 不可评测输出的标准化修复次数：{run.get('repair_attempts', 0)}",
        f"- 基础模型调用次数：{summary.get('total_agent_runs', 0)}（修复调用另计）",
        "",
        "### 证据条件检查",
    ]
    for key, label in checklist_names.items():
        lines.append(f"- {'✅' if checklist.get(key) else '❌'} {label}")

    lines.extend([
        "",
        "## 2. 输出进入测试的诊断漏斗",
    ])
    stage_names = {
        "response_received": "获得模型响应",
        "first_syntax_valid": "首轮Python语法有效",
        "first_tests_started": "首轮进入单元测试",
        "repair_attempted": "触发标准化修复",
        "repair_succeeded": "修复后成功进入测试",
        "final_code_extracted": "最终提取到代码",
        "final_syntax_valid": "最终Python语法有效",
        "final_tests_started": "最终进入单元测试",
        "any_test_passed": "至少通过1个测试用例",
        "all_tests_passed": "全部测试用例通过",
    }
    pipeline_a = {}
    pipeline_b = {}
    if run.get("results"):
        # Global funnel is averaged across task-level aggregates.
        for key in stage_names:
            av = [x["agent_a"].get("pipeline", {}).get(key, 0) for x in run["results"] if x.get("agent_a")]
            bv = [x["agent_b"].get("pipeline", {}).get(key, 0) for x in run["results"] if x.get("agent_b")]
            pipeline_a[key] = statistics.fmean(av) if av else 0
            pipeline_b[key] = statistics.fmean(bv) if bv else 0
    for key, label in stage_names.items():
        lines.append(f"- {label}：A {pct(pipeline_a.get(key))}；B {pct(pipeline_b.get(key))}")

    lines.extend([
        "",
        "## 3. 客观测试结果",
        f"- Agent A测试用例通过率：**{pct(summary.get('test_case_pass_rate_a'))}**",
        f"- Agent B测试用例通过率：**{pct(summary.get('test_case_pass_rate_b'))}**",
        f"- Agent A可评测输出率：**{pct(summary.get('evaluable_rate_a'))}**",
        f"- Agent B可评测输出率：**{pct(summary.get('evaluable_rate_b'))}**",
        f"- 可配对比较任务：**{evidence.get('comparable_task_count', 0)}/{evidence.get('selected_task_count', 0)}**",
        "",
        "当一侧为“不可计算”时，不将其按0%处理，也不计算客观提升或退化。",
        "",
        "## 4. 辅助评分",
        f"- Agent A可比较综合分：**{score(summary.get('average_a'), summary.get('std_a')) if summary.get('test_case_pass_rate_a') is not None else '不适用'}**",
        f"- Agent B可比较综合分：**{score(summary.get('average_b'), summary.get('std_b')) if summary.get('test_case_pass_rate_b') is not None else '不适用'}**",
        f"- B-A：**{score(summary.get('delta'))}**",
        "",
        "无可执行测试结果时，诊断分不参与A/B模型能力比较。",
        "",
        "## 5. 分任务结果",
    ])

    failures_a: dict[str, int] = {}
    failures_b: dict[str, int] = {}
    comparable_deltas: list[tuple[float, str]] = []
    for item in run.get("results", []):
        task = item.get("task", {})
        if item.get("error"):
            lines.append(f"- {task.get('name')}：调用失败。")
            continue
        a = item["agent_a"]
        b = item["agent_b"]
        obj_delta = item.get("objective_delta")
        delta_text = (
            f"{obj_delta * 100:+.1f}个百分点"
            if obj_delta is not None else "不可比较"
        )
        lines.append(
            f"- {task.get('name')}（{task.get('task_type')}）："
            f"A可评测率 {pct(a.get('evaluable_rate'))}、用例通过率 {pct(a.get('test_case_pass_rate'))}；"
            f"B可评测率 {pct(b.get('evaluable_rate'))}、用例通过率 {pct(b.get('test_case_pass_rate'))}；"
            f"客观差异 {delta_text}。"
        )
        if obj_delta is not None:
            comparable_deltas.append((obj_delta, task.get("name", "")))
        for f, count in a.get("evaluation", {}).get("failure_counts", {}).items():
            failures_a[f] = failures_a.get(f, 0) + int(count)
        for f, count in b.get("evaluation", {}).get("failure_counts", {}).items():
            failures_b[f] = failures_b.get(f, 0) + int(count)

    lines.extend(["", "## 6. 可观察失败对照"])
    lines.append("### Agent A")
    if failures_a:
        for f, count in sorted(failures_a.items(), key=lambda x: -x[1]):
            lines.append(f"- {f}：{count}次。")
    else:
        lines.append("- 暂无记录。")
    lines.append("### Agent B")
    if failures_b:
        for f, count in sorted(failures_b.items(), key=lambda x: -x[1]):
            lines.append(f"- {f}：{count}次。")
    else:
        lines.append("- 暂无记录。")

    lines.extend(["", "## 7. 当前结论"])
    if evidence.get("level") == "不可直接比较：输出可执行性不对称":
        lines.append(
            "Agent A已完成模型调用，但首轮或修复后的输出仍未形成可执行测试结果；问题位于代码生成/语法阶段，而不是Agent A未被调用。"
        )
        lines.append(
            "可以确认的产品事实仅是：在当前Prompt与解析规则下，Agent B更容易生成可进入测试的输出。"
        )
        lines.append(
            "下一步应根据Agent A的状态分布和代表输出，判断是重复生成、代码块提取、语法错误还是安全阻止。"
        )
    elif comparable_deltas:
        mean_delta = statistics.fmean([x[0] for x in comparable_deltas])
        if mean_delta > 0:
            lines.append("在可配对任务上，Agent B的测试用例通过率更高。")
        elif mean_delta < 0:
            lines.append("在可配对任务上，Agent A的测试用例通过率更高。")
        else:
            lines.append("在可配对任务上，A/B测试用例通过率相同。")
    else:
        lines.append("本轮没有可配对的客观测试结果，不能形成模型优劣结论。")

    lines.extend([
        "",
        "## 8. 适用边界",
        "结果仅适用于本报告中的模型ID、参数、任务、Prompt、解析器和本机执行环境。",
    ])
    return "\\n".join(lines)


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": "0.7.0",
        "docker_available": bool(shutil.which("docker")),
        "python": sys.version.split()[0],
    }


@app.get("/api/tasks")
async def get_tasks() -> list[dict[str, Any]]:
    return load_tasks()


@app.post("/api/tasks")
async def create_task(payload: TaskCreate) -> dict[str, Any]:
    tasks = load_tasks()
    task = TaskModel(**payload.model_dump()).model_dump()
    tasks.append(task)
    save_tasks(tasks)
    return task


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str) -> dict[str, bool]:
    tasks = load_tasks()
    filtered = [task for task in tasks if task["id"] != task_id]
    if len(filtered) == len(tasks):
        raise HTTPException(status_code=404, detail="任务不存在")
    save_tasks(filtered)
    return {"ok": True}


@app.post("/api/providers/models")
async def list_provider_models(request: ModelListRequest) -> dict[str, Any]:
    config = request.provider
    if config.provider == "mock":
        return {"models": ["mock-model"]}
    if not config.api_key and config.provider not in LOCAL_NO_KEY_PROVIDERS:
        raise HTTPException(status_code=400, detail="请先填写 API Key")
    try:
        if config.provider == "ollama":
            base = provider_base(config)
            if base.endswith("/v1"):
                base = base[:-3]
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(45.0, connect=15.0)
            ) as client:
                response = await client.get(append_endpoint(base, "api/tags"))
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Ollama 模型列表失败：HTTP {response.status_code}；"
                    f"{response.text[:800]}"
                )
            data = response.json()
            return {
                "models": sorted({
                    str(item.get("name"))
                    for item in data.get("models", [])
                    if item.get("name")
                })
            }

        headers: dict[str, str] = {"Accept": "application/json"}
        protocol = provider_meta(config)["protocol"]
        if protocol == "anthropic":
            headers.update({
                "x-api-key": config.api_key,
                "anthropic-version": "2023-06-01",
            })
        elif config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(45.0, connect=15.0)
        ) as client:
            response = await client.get(
                model_list_endpoint(config), headers=headers
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"模型列表请求失败：HTTP {response.status_code}；"
                f"{response.text[:800]}"
            )
        data = response.json()
        raw_models = data.get("data", data.get("models", []))
        models = []
        for item in raw_models or []:
            if isinstance(item, str):
                models.append(item)
            elif isinstance(item, dict):
                value = item.get("id") or item.get("name") or item.get("model")
                if value:
                    models.append(str(value))
        return {"models": sorted(set(models))}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/providers/test")
async def test_provider(request: ConnectionTestRequest) -> dict[str, Any]:
    try:
        result = await call_provider(
            request.provider,
            "请只回复：连接成功",
            "这是一次 API 连接测试。",
        )
        return {
            "ok": True,
            "text": result["text"][:300],
            "latency_ms": result["latency_ms"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/runs")
async def start_run(request: RunRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    tasks = load_tasks()
    selected_count = len(request.task_ids) if request.task_ids else len(tasks)
    if selected_count == 0:
        raise HTTPException(status_code=400, detail="没有可执行的评测任务")
    run_id = str(uuid.uuid4())
    JOBS[run_id] = {
        "id": run_id,
        "name": request.name,
        "status": "queued",
        "created_at": utc_now(),
        "completed_tasks": 0,
        "total_tasks": selected_count,
        "completed_units": 0,
        "total_units": selected_count * request.repeat_count,
        "repeat_count": request.repeat_count,
        "repair_attempts": request.repair_attempts,
        "agent_a": sanitize_provider(request.agent_a),
        "agent_b": sanitize_provider(request.agent_b),
        "judge": sanitize_provider(request.judge),
        "execution_mode": request.execution_mode,
    }
    background_tasks.add_task(execute_run, run_id, request)
    return {"run_id": run_id}


@app.get("/api/runs")
async def list_runs() -> list[dict[str, Any]]:
    records = []
    for path in sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                {
                    "id": run["id"],
                    "name": run["name"],
                    "created_at": run["created_at"],
                    "status": run["status"],
                    "summary": run.get("summary", {}),
                    "agent_a": run.get("agent_a"),
                    "agent_b": run.get("agent_b"),
                }
            )
        except Exception:
            continue
    for run_id, job in JOBS.items():
        if job.get("status") not in {"completed"}:
            records.insert(0, {
                key: job.get(key)
                for key in [
                    "id", "name", "created_at", "status", "summary",
                    "agent_a", "agent_b", "completed_tasks", "total_tasks",
                    "completed_units", "total_units", "repeat_count", "evidence"
                ]
            })
    return records[:30]


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    return load_run(run_id)


@app.get("/api/runs/{run_id}/report", response_class=PlainTextResponse)
async def get_report(run_id: str) -> str:
    run = load_run(run_id)
    if run.get("status") != "completed":
        raise HTTPException(status_code=409, detail="评测尚未完成")
    return make_markdown_report(run)
