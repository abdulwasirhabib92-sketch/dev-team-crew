"""
Multi-LLM Tool — lets any agent call ANY LLM provider on demand.
Supports: Gemini, Groq, OpenAI, Anthropic, OpenRouter, Hugging Face,
          DeepSeek, SiliconFlow (free Chinese LLMs), Qwen/DashScope.
"""
import os
import json
import requests
from crewai.tools import BaseTool


# ═══════════════════════════════════════════════════════════
# DIRECT API CALLERS
# ═══════════════════════════════════════════════════════════

def _call_gemini(prompt: str, system: str = "") -> str:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        return "Gemini API key not configured"
    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    resp = requests.post(url, json=body, timeout=60)
    if resp.status_code == 200:
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            return candidates[0]["content"]["parts"][0]["text"]
    return f"Gemini error {resp.status_code}: {resp.text[:200]}"


def _call_groq(prompt: str, system: str = "") -> str:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "Groq API key not configured"
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    url = "https://api.groq.com/openai/v1/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                         json={"model": model, "messages": messages, "temperature": 0.7}, timeout=60)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"Groq error {resp.status_code}: {resp.text[:200]}"


def _call_openai(prompt: str, system: str = "") -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "OpenAI API key not configured"
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    url = "https://api.openai.com/v1/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                         json={"model": model, "messages": messages, "temperature": 0.7}, timeout=60)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"OpenAI error {resp.status_code}: {resp.text[:200]}"


def _call_anthropic(prompt: str, system: str = "") -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "Anthropic API key not configured"
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    body = {"model": model, "max_tokens": 4096, "messages": [{"role": "user", "content": prompt}]}
    if system:
        body["system"] = system
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    if resp.status_code == 200:
        data = resp.json()
        content = data.get("content", [])
        if content:
            return content[0]["text"]
    return f"Anthropic error {resp.status_code}: {resp.text[:200]}"


def _call_openrouter(prompt: str, system: str = "") -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return "OpenRouter API key not configured"
    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    url = "https://openrouter.ai/api/v1/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                         json={"model": model, "messages": messages, "temperature": 0.7}, timeout=60)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"OpenRouter error {resp.status_code}: {resp.text[:200]}"


def _call_huggingface(prompt: str, system: str = "") -> str:
    api_key = os.getenv("HUGGINGFACE_API_KEY", "")
    if not api_key:
        return "Hugging Face API key not configured"
    model = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    url = f"https://api-inference.huggingface.co/models/{model}"
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                         json={"inputs": full_prompt, "parameters": {"temperature": 0.7, "max_new_tokens": 2048}}, timeout=120)
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", str(data[0]))
        return str(data)
    return f"HuggingFace error {resp.status_code}: {resp.text[:200]}"


# ═══════════════════════════════════════════════════════════
# CHINESE LLM PROVIDERS (free tiers available)
# ═══════════════════════════════════════════════════════════

def _call_deepseek(prompt: str, system: str = "") -> str:
    """DeepSeek — free API credits on signup. Excellent at code + reasoning.
    Get key: https://platform.deepseek.com/api_keys"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return "DeepSeek API key not configured"
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    url = "https://api.deepseek.com/v1/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                         json={"model": model, "messages": messages, "temperature": 0.7}, timeout=120)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"DeepSeek error {resp.status_code}: {resp.text[:200]}"


def _call_siliconflow(prompt: str, system: str = "") -> str:
    """SiliconFlow — free access to many Chinese models (Qwen, GLM, DeepSeek, Yi, etc.)
    Get key: https://siliconflow.cn/user/info"""
    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    if not api_key:
        return "SiliconFlow API key not configured"
    model = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    url = "https://api.siliconflow.cn/v1/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                         json={"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 2048}, timeout=120)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"SiliconFlow error {resp.status_code}: {resp.text[:200]}"


def _call_qwen(prompt: str, system: str = "") -> str:
    """Qwen / DashScope (Alibaba) — free tier available.
    Get key: https://dashscope.console.aliyun.com/apiKey"""
    api_key = os.getenv("QWEN_API_KEY", "")
    if not api_key:
        return "Qwen API key not configured"
    model = os.getenv("QWEN_MODEL", "qwen-turbo")
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                         json={"model": model, "messages": messages, "temperature": 0.7}, timeout=120)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"Qwen error {resp.status_code}: {resp.text[:200]}"


def _call_glm(prompt: str, system: str = "") -> str:
    """Zhipu AI / GLM (ChatGLM) — free tier available.
    Get key: https://open.bigmodel.cn/usercenter/apikeys"""
    api_key = os.getenv("GLM_API_KEY", "")
    if not api_key:
        return "GLM (Zhipu) API key not configured"
    model = os.getenv("GLM_MODEL", "glm-4-flash")
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                         json={"model": model, "messages": messages, "temperature": 0.7}, timeout=120)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"GLM error {resp.status_code}: {resp.text[:200]}"


def _call_moonshot(prompt: str, system: str = "") -> str:
    """Moonshot AI (Kimi) — free tier available.
    Get key: https://platform.moonshot.cn/console/api-keys"""
    api_key = os.getenv("MOONSHOT_API_KEY", "")
    if not api_key:
        return "Moonshot API key not configured"
    model = os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k")
    url = "https://api.moonshot.cn/v1/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                         json={"model": model, "messages": messages, "temperature": 0.7}, timeout=120)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"Moonshot error {resp.status_code}: {resp.text[:200]}"


# ═══════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════

LLM_CALLERS = {
    "gemini": _call_gemini, "groq": _call_groq, "openai": _call_openai,
    "anthropic": _call_anthropic, "openrouter": _call_openrouter,
    "huggingface": _call_huggingface,
    # Chinese LLMs
    "deepseek": _call_deepseek, "siliconflow": _call_siliconflow,
    "qwen": _call_qwen, "glm": _call_glm, "moonshot": _call_moonshot,
}

ENV_KEY_MAP = {
    "gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY", "huggingface": "HUGGINGFACE_API_KEY",
    # Chinese LLMs
    "deepseek": "DEEPSEEK_API_KEY", "siliconflow": "SILICONFLOW_API_KEY",
    "qwen": "QWEN_API_KEY", "glm": "GLM_API_KEY", "moonshot": "MOONSHOT_API_KEY",
}

# Provider display info for the UI
PROVIDER_INFO = {
    "gemini": {"name": "Google Gemini", "region": "US", "free": True, "url": "https://aistudio.google.com/apikey"},
    "groq": {"name": "Groq", "region": "US", "free": True, "url": "https://console.groq.com/keys"},
    "openai": {"name": "OpenAI GPT", "region": "US", "free": False, "url": "https://platform.openai.com/api-keys"},
    "anthropic": {"name": "Anthropic Claude", "region": "US", "free": False, "url": "https://console.anthropic.com/"},
    "openrouter": {"name": "OpenRouter", "region": "US", "free": True, "url": "https://openrouter.ai/keys"},
    "huggingface": {"name": "Hugging Face", "region": "US", "free": True, "url": "https://huggingface.co/settings/tokens"},
    "deepseek": {"name": "DeepSeek 深度求索", "region": "CN", "free": True, "url": "https://platform.deepseek.com/api_keys"},
    "siliconflow": {"name": "SiliconFlow 硅基流动", "region": "CN", "free": True, "url": "https://cloud.siliconflow.cn/user/info"},
    "qwen": {"name": "Qwen 通义千问 (Alibaba)", "region": "CN", "free": True, "url": "https://dashscope.console.aliyun.com/apiKey"},
    "glm": {"name": "GLM 智谱清言 (Zhipu)", "region": "CN", "free": True, "url": "https://open.bigmodel.cn/usercenter/apikeys"},
    "moonshot": {"name": "Moonshot 月之暗面 (Kimi)", "region": "CN", "free": True, "url": "https://platform.moonshot.cn/console/api-keys"},
}


def _has_valid_key(provider: str) -> bool:
    val = os.getenv(ENV_KEY_MAP.get(provider, ""), "")
    return bool(val) and val not in ["", "your_gemini_api_key_here"]


def get_available_providers() -> list:
    return [name for name in LLM_CALLERS if _has_valid_key(name)]


def get_provider_info() -> list:
    """Return info for all providers with their status."""
    result = []
    for name, info in PROVIDER_INFO.items():
        result.append({
            "id": name, "name": info["name"], "region": info["region"],
            "free": info["free"], "url": info["url"],
            "active": _has_valid_key(name),
        })
    return result


# ═══════════════════════════════════════════════════════════
# MULTI-LLM TOOLS FOR CREWAI AGENTS
# ═══════════════════════════════════════════════════════════

class AskLLMTool(BaseTool):
    name: str = "ask_llm"
    description: str = (
        "Call a specific LLM provider with a prompt. "
        "Input JSON: {\"provider\": \"gemini|groq|openai|anthropic|deepseek|siliconflow|qwen|glm|moonshot\", "
        "\"prompt\": \"your question\", \"system\": \"optional system instruction\"}. "
        "Use to get a second opinion from a different AI."
    )
    def _run(self, query: str) -> str:
        try:
            params = json.loads(query) if isinstance(query, str) else query
        except json.JSONDecodeError:
            return "Error: Provide valid JSON with 'provider' and 'prompt'."
        provider = params.get("provider", "").lower().strip()
        prompt = params.get("prompt", "")
        system = params.get("system", "")
        if provider not in LLM_CALLERS:
            available = get_available_providers()
            return f"Unknown provider '{provider}'. Available: {', '.join(available)}"
        if not _has_valid_key(provider):
            return f"{provider} API key not configured. Available: {', '.join(get_available_providers())}"
        result = LLM_CALLERS[provider](prompt, system)
        return f"[{provider}] {result}"


class AskAllLLMsTool(BaseTool):
    name: str = "ask_all_llms"
    description: str = (
        "Call ALL available LLM providers simultaneously with the same prompt. "
        "Great for diverse perspectives from US + Chinese models. "
        "Input JSON: {\"prompt\": \"your question\", \"system\": \"optional system instruction\"}."
    )
    def _run(self, query: str) -> str:
        try:
            params = json.loads(query) if isinstance(query, str) else query
        except json.JSONDecodeError:
            return "Error: Provide valid JSON with 'prompt'."
        prompt = params.get("prompt", "")
        system = params.get("system", "")
        available = get_available_providers()
        if not available:
            return "No LLM providers configured."
        results = []
        for provider in available:
            try:
                result = LLM_CALLERS[provider](prompt, system)
                results.append(f"--- {provider.upper()} ---\n{result}\n")
            except Exception as e:
                results.append(f"--- {provider.upper()} ---\nError: {str(e)}\n")
        return "\n".join(results)


class CompareLLMsTool(BaseTool):
    name: str = "compare_llms"
    description: str = (
        "Call multiple specified LLM providers and compare responses. "
        "Input JSON: {\"prompt\": \"your question\", \"providers\": [\"gemini\", \"deepseek\", \"openai\"], "
        "\"system\": \"optional system instruction\"}."
    )
    def _run(self, query: str) -> str:
        try:
            params = json.loads(query) if isinstance(query, str) else query
        except json.JSONDecodeError:
            return "Error: Provide valid JSON with 'prompt' and 'providers'."
        prompt = params.get("prompt", "")
        system = params.get("system", "")
        providers = params.get("providers", [])
        available = get_available_providers()
        to_call = [p for p in providers if p in available]
        if not to_call:
            to_call = available
        results = []
        for provider in to_call:
            try:
                result = LLM_CALLERS[provider](prompt, system)
                results.append(f"=== {provider.upper()} ===\n{result}\n")
            except Exception as e:
                results.append(f"=== {provider.upper()} ===\nError: {str(e)}\n")
        return "\n".join(results)


class ListLLMsTool(BaseTool):
    name: str = "list_available_llms"
    description: str = "List all available LLM providers that have valid API keys. No input required."
    def _run(self, query: str = "") -> str:
        available = get_available_providers()
        if not available:
            return "No LLM providers configured."
        return f"Available LLM providers ({len(available)}): {', '.join(available)}"


def get_multi_llm_tools():
    return [AskLLMTool(), AskAllLLMsTool(), CompareLLMsTool(), ListLLMsTool()]
