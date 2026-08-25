"""
Multi-LLM Tool — lets any agent call ANY LLM provider on demand.
Each agent has a primary LLM but can also invoke other LLMs as tools.
This gives every agent access to all providers simultaneously.
"""
import os
import json
import requests
from crewai.tools import BaseTool
from typing import Optional


# ═══════════════════════════════════════════════════════════
# DIRECT API CALLERS (no LangChain dependency needed)
# ═══════════════════════════════════════════════════════════

def _call_gemini(prompt: str, system: str = "") -> str:
    """Call Google Gemini API directly."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        return "Gemini API key not configured"
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    parts = []
    if system:
        parts.append({"text": system})
    parts.append({"text": prompt})
    body = {"contents": [{"parts": parts}]}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
        body["contents"] = [{"parts": [{"text": prompt}]}]
    resp = requests.post(url, json=body, timeout=60)
    if resp.status_code == 200:
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            return candidates[0]["content"]["parts"][0]["text"]
    return f"Gemini error {resp.status_code}: {resp.text[:200]}"


def _call_groq(prompt: str, system: str = "") -> str:
    """Call Groq API directly."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "Groq API key not configured"
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    url = "https://api.groq.com/openai/v1/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(url, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }, json={"model": model, "messages": messages, "temperature": 0.7}, timeout=60)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"Groq error {resp.status_code}: {resp.text[:200]}"


def _call_openai(prompt: str, system: str = "") -> str:
    """Call OpenAI API directly."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "OpenAI API key not configured"
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    url = "https://api.openai.com/v1/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(url, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }, json={"model": model, "messages": messages, "temperature": 0.7}, timeout=60)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"OpenAI error {resp.status_code}: {resp.text[:200]}"


def _call_anthropic(prompt: str, system: str = "") -> str:
    """Call Anthropic API directly."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "Anthropic API key not configured"
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
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
    """Call OpenRouter API directly."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return "OpenRouter API key not configured"
    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    url = "https://openrouter.ai/api/v1/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(url, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }, json={"model": model, "messages": messages, "temperature": 0.7}, timeout=60)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"OpenRouter error {resp.status_code}: {resp.text[:200]}"


def _call_huggingface(prompt: str, system: str = "") -> str:
    """Call Hugging Face Inference API directly."""
    api_key = os.getenv("HUGGINGFACE_API_KEY", "")
    if not api_key:
        return "Hugging Face API key not configured"
    model = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    url = f"https://api-inference.huggingface.co/models/{model}"
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    resp = requests.post(url, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }, json={"inputs": full_prompt, "parameters": {"temperature": 0.7, "max_new_tokens": 2048}}, timeout=120)
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", str(data[0]))
        return str(data)
    return f"HuggingFace error {resp.status_code}: {resp.text[:200]}"


# Registry of all direct API callers
LLM_CALLERS = {
    "gemini": _call_gemini,
    "groq": _call_groq,
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "openrouter": _call_openrouter,
    "huggingface": _call_huggingface,
}


def _has_valid_key(provider: str) -> bool:
    key_map = {
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
    }
    val = os.getenv(key_map.get(provider, ""), "")
    return bool(val) and val not in ["", "your_gemini_api_key_here"]


def get_available_providers() -> list:
    """Return list of providers with valid API keys."""
    return [name for name in LLM_CALLERS if _has_valid_key(name)]


# ═══════════════════════════════════════════════════════════
# MULTI-LLM TOOLS FOR CREWAI AGENTS
# ═══════════════════════════════════════════════════════════

class AskLLMTool(BaseTool):
    """Tool that lets an agent call ANY specific LLM provider."""
    name: str = "ask_llm"
    description: str = (
        "Call a specific LLM provider with a prompt. "
        "Input JSON: {\"provider\": \"gemini|groq|openai|anthropic|openrouter|huggingface\", "
        "\"prompt\": \"your question or instruction\", "
        "\"system\": \"optional system instruction\"}. "
        "Use this to get a second opinion from a different AI, "
        "or to leverage a model's specific strength."
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
    """Tool that calls ALL available LLMs and returns their combined responses."""
    name: str = "ask_all_llms"
    description: str = (
        "Call ALL available LLM providers simultaneously with the same prompt "
        "and get their combined responses. Great for getting diverse perspectives. "
        "Input JSON: {\"prompt\": \"your question\", \"system\": \"optional system instruction\"}. "
        "Returns each provider's response labeled."
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
    """Tool that calls multiple LLMs and compares their answers."""
    name: str = "compare_llms"
    description: str = (
        "Call multiple specified LLM providers with the same prompt and compare responses. "
        "Input JSON: {\"prompt\": \"your question\", "
        "\"providers\": [\"gemini\", \"openai\", \"anthropic\"], "
        "\"system\": \"optional system instruction\"}. "
        "Returns a side-by-side comparison."
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

        # Filter to only available providers
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
    """Tool that lists all available LLM providers."""
    name: str = "list_available_llms"
    description: str = (
        "List all available LLM providers that have valid API keys configured. "
        "No input required — pass an empty string."
    )

    def _run(self, query: str = "") -> str:
        available = get_available_providers()
        if not available:
            return "No LLM providers configured."
        return f"Available LLM providers ({len(available)}): {', '.join(available)}"


def get_multi_llm_tools():
    """Return all multi-LLM tools for agent use."""
    return [
        AskLLMTool(),
        AskAllLLMsTool(),
        CompareLLMsTool(),
        ListLLMsTool(),
    ]
