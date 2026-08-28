"""
Agent definitions for the Dev Team Crew.
Each agent is a CHARACTER — with personality, vibe, and working style.
11 LLM providers: 6 Western + 5 Chinese.
Per-agent Groq model spreading + automatic model fallback on 503/429.
"""
from crewai import Agent, LLM
from crewai_tools import SerperDevTool, FileReadTool
from supabase_tools import get_supabase_tools, is_supabase_configured
from multi_llm_tools import get_multi_llm_tools, get_available_providers, get_provider_info
from agent_identities import AGENT_IDENTITIES
import os
import logging
import litellm
import time

# Disable prompt caching — Groq doesn't support cache_breakpoint
litellm.disable_caching = True
litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# GROQ MODEL FALLBACK CHAIN
# If a model returns 503 (overloaded) or 429 (rate limit),
# automatically try the next model in the chain.
# All models verified working on 2026-08-28.
# ═══════════════════════════════════════════════════════════
GROQ_MODELS = [
    "groq/openai/gpt-oss-120b",     # Smartest but gets overloaded
    "groq/qwen/qwen3.8-27b",        # Excellent at code
    "groq/qwen/qwen3.6-27b",        # Good general purpose
    "groq/groq/compound",           # Balanced
    "groq/openai/gpt-oss-20b",      # Fast, lightweight
    "groq/allam-2-7b",               # Arabic-focused but works
]

# Per-agent primary model (index into GROQ_MODELS)
# Spread agents to start on different models
AGENT_MODEL_INDEX = {
    "researcher":  1,  # openai/gpt-oss-120b
    "architect":   4,  # qwen/qwen3.6-27b
    "implementer": 2,  # qwen/qwen3.8-27b
    "critic":      3,  # groq/compound
    "tester":      4,  # openai/gpt-oss-20b
    "devops":      1,  # qwen/qwen3.8-27b
}


# ═══════════════════════════════════════════════════════════
# LLM PROVIDER FACTORY
# ═══════════════════════════════════════════════════════════

def _make_gemini_llm():
    return LLM(model=f"gemini/{os.getenv('GEMINI_MODEL', 'gemini-3.5-flash')}",
               api_key=os.getenv("GEMINI_API_KEY"), temperature=0.7)

def _make_groq_llm(model=None, agent_name=None):
    """Create a Groq LLM with per-agent model selection."""
    if agent_name and agent_name in AGENT_MODEL_INDEX:
        idx = AGENT_MODEL_INDEX[agent_name]
        # Check for env override: RESEARCHER_GROQ_MODEL
        env_key = f"{agent_name.upper()}_GROQ_MODEL"
        env_model = os.getenv(env_key)
        if env_model:
            model = env_model
        elif not model:
            model = GROQ_MODELS[idx]
    if not model:
        model = os.getenv('GROQ_MODEL', 'groq/openai/gpt-oss-120b')
    return LLM(model=model,
               api_key=os.getenv("GROQ_API_KEY"),
               temperature=0.7)

def _make_openrouter_llm():
    return LLM(model=f"openrouter/{os.getenv('OPENROUTER_MODEL', 'meta-llama/llama-3.1-8b-instruct:free')}",
               api_key=os.getenv("OPENROUTER_API_KEY"), temperature=0.7,
               base_url="https://openrouter.ai/api/v1")

def _make_huggingface_llm():
    return LLM(model=f"huggingface/{os.getenv('HUGGINGFACE_MODEL', 'meta-llama/Llama-3.1-8B-Instruct')}",
               api_key=os.getenv("HUGGINGFACE_API_KEY"), temperature=0.7)

def _make_openai_llm():
    return LLM(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
               api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)

def _make_anthropic_llm():
    return LLM(model=f"anthropic/{os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022')}",
               api_key=os.getenv("ANTHROPIC_API_KEY"), temperature=0.7)

def _make_deepseek_llm():
    return LLM(model=f"deepseek/{os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')}",
               api_key=os.getenv("DEEPSEEK_API_KEY"), temperature=0.7)

def _make_siliconflow_llm():
    return LLM(model=os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
               api_key=os.getenv("SILICONFLOW_API_KEY"), temperature=0.7,
               base_url="https://api.siliconflow.cn/v1")

def _make_qwen_llm():
    return LLM(model=f"dashscope/{os.getenv('QWEN_MODEL', 'qwen-turbo')}",
               api_key=os.getenv("QWEN_API_KEY"), temperature=0.7)

def _make_glm_llm():
    return LLM(model=os.getenv("GLM_MODEL", "glm-4-flash"),
               api_key=os.getenv("GLM_API_KEY"), temperature=0.7,
               base_url="https://open.bigmodel.cn/api/paas/v4")

def _make_moonshot_llm():
    return LLM(model=os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k"),
               api_key=os.getenv("MOONSHOT_API_KEY"), temperature=0.7,
               base_url="https://api.moonshot.cn/v1")

LLM_PROVIDERS = {
    "gemini": _make_gemini_llm, "groq": _make_groq_llm,
    "openrouter": _make_openrouter_llm, "huggingface": _make_huggingface_llm,
    "openai": _make_openai_llm, "anthropic": _make_anthropic_llm,
    "deepseek": _make_deepseek_llm, "siliconflow": _make_siliconflow_llm,
    "qwen": _make_qwen_llm, "glm": _make_glm_llm, "moonshot": _make_moonshot_llm,
}

ENV_KEY_MAP = {
    "gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY", "huggingface": "HUGGINGFACE_API_KEY",
    "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY", "siliconflow": "SILICONFLOW_API_KEY",
    "qwen": "QWEN_API_KEY", "glm": "GLM_API_KEY", "moonshot": "MOONSHOT_API_KEY",
}

# Provider priority: Groq first (free, high quota), then Gemini
PRIORITY_ORDER = ["groq", "gemini", "deepseek", "openai",
                  "openrouter", "huggingface", "siliconflow", "qwen", "glm", "moonshot", "anthropic"]

# All agents on Groq (free, 14,400 req/day)
DEFAULT_AGENT_LLM = {
    "researcher": "groq",
    "architect": "groq",
    "implementer": "groq",
    "critic": "groq",
    "tester": "groq",
    "devops": "groq",
}

def _has_valid_key(provider: str) -> bool:
    val = os.getenv(ENV_KEY_MAP.get(provider, ""), "")
    return bool(val) and val not in ["", "your_gemini_api_key_here"]

def get_llm(provider: str = None, agent_name: str = None):
    """Get an LLM by provider name, with automatic fallback."""
    if provider and provider in LLM_PROVIDERS:
        if _has_valid_key(provider):
            logger.info(f"🧠 Using LLM: {provider}" + (f" for {agent_name}" if agent_name else ""))
            factory = LLM_PROVIDERS[provider]
            if provider == "groq":
                return factory(agent_name=agent_name)
            return factory()
        logger.warning(f"⚠️  {provider} API key not set, falling back")
    # Fallback: try providers in priority order
    for name in PRIORITY_ORDER:
        if name in LLM_PROVIDERS and _has_valid_key(name):
            logger.info(f"🧠 Fallback LLM: {name}")
            factory = LLM_PROVIDERS[name]
            if name == "groq":
                return factory(agent_name=agent_name)
            return factory()
    raise ValueError("No LLM API key found! Set at least one provider key.")

def get_agent_llm(agent_name: str):
    """Get LLM for a specific agent."""
    env_key = f"{agent_name.upper()}_LLM"
    provider = os.getenv(env_key, "").strip().lower()
    if not provider:
        provider = DEFAULT_AGENT_LLM.get(agent_name, "")
    return get_llm(provider, agent_name=agent_name)


# ═══════════════════════════════════════════════════════════
# TOOL ASSEMBLY
# ═══════════════════════════════════════════════════════════

def _get_agent_tools(agent_name: str = None):
    tools = []
    if agent_name in ["researcher", "architect", "critic"]:
        tools.append(SerperDevTool())
        tools.append(FileReadTool())
    if is_supabase_configured():
        tools.extend(get_supabase_tools())
    tools.extend(get_multi_llm_tools())
    return tools


def _build_identity(agent_key: str):
    ident = AGENT_IDENTITIES.get(agent_key, {})
    name = ident.get("name", agent_key.title())
    personality = ident.get("personality", "")
    working_style = ident.get("working_style", "")
    catchphrase = ident.get("catchphrase", "")
    vibe = ident.get("vibe", "")
    backstory = f"{personality}\n\nYour working style: {working_style}\n\nYour vibe: {vibe}\nYour catchphrase: \"{catchphrase}\""
    return name, backstory


# ═══════════════════════════════════════════════════════════
# AGENT DEFINITIONS
# ═══════════════════════════════════════════════════════════

def create_researcher():
    name, backstory = _build_identity("researcher")
    return Agent(
        role=f"Researcher ({name})",
        goal=("Gather comprehensive information and provide well-researched context. "
              "Use ask_all_llms to get diverse perspectives from all available AI models. "
              "Store findings in Supabase."),
        backstory=backstory, verbose=True, allow_delegation=False,
        llm=get_agent_llm("researcher"), tools=_get_agent_tools("researcher"),
    )

def create_architect():
    name, backstory = _build_identity("architect")
    return Agent(
        role=f"Architect ({name})",
        goal=("Design system architecture and break tasks into clear steps. "
              "Use compare_llms to get multiple AI perspectives before deciding. "
              "Store architecture plans in Supabase."),
        backstory=backstory, verbose=True, allow_delegation=True,
        llm=get_agent_llm("architect"), tools=_get_agent_tools("architect"),
    )

def create_implementer():
    name, backstory = _build_identity("implementer")
    return Agent(
        role=f"Implementer ({name})",
        goal=("Write clean, production-ready code. Use ask_llm to consult different "
              "models for specific challenges. Store code artifacts in Supabase."),
        backstory=backstory, verbose=True, allow_delegation=False,
        llm=get_agent_llm("implementer"), tools=_get_agent_tools("implementer"),
    )

def create_critic():
    name, backstory = _build_identity("critic")
    return Agent(
        role=f"Critic ({name})",
        goal=("Review code critically for bugs, security, and quality. "
              "Use ask_all_llms for independent review. Always provide a fix. "
              "Store review notes in Supabase."),
        backstory=backstory, verbose=True, allow_delegation=True,
        llm=get_agent_llm("critic"), tools=_get_agent_tools("critic"),
    )

def create_tester():
    name, backstory = _build_identity("tester")
    return Agent(
        role=f"Tester ({name})",
        goal=("Write comprehensive tests. Use ask_all_llms to get every model to "
              "suggest test cases. Report failures clearly. Use Supabase for test data."),
        backstory=backstory, verbose=True, allow_delegation=True,
        llm=get_agent_llm("tester"), tools=_get_agent_tools("tester"),
    )

def create_devops():
    name, backstory = _build_identity("devops")
    return Agent(
        role=f"DevOps Engineer ({name})",
        goal=("Handle deployment, CI/CD, and infrastructure. Automate everything. "
              "Track state in Supabase."),
        backstory=backstory, verbose=True, allow_delegation=False,
        llm=get_agent_llm("devops"), tools=_get_agent_tools("devops"),
    )

def create_all_agents():
    return {
        "researcher": create_researcher(),
        "architect": create_architect(),
        "implementer": create_implementer(),
        "critic": create_critic(),
        "tester": create_tester(),
        "devops": create_devops(),
    }

def list_team():
    team = []
    for key, ident in AGENT_IDENTITIES.items():
        idx = AGENT_MODEL_INDEX.get(key, 0)
        team.append({
            "codename": ident.get("name", key.title()),
            "role": key.title(),
            "vibe": ident.get("vibe", ""),
            "model": GROQ_MODELS[idx] if idx < len(GROQ_MODELS) else GROQ_MODELS[0],
            "provider": "groq",
        })
    return team

def list_available_providers():
    return get_available_providers()

def list_providers():
    return get_provider_info()
