"""
Agent definitions for the Dev Team Crew.
Each agent has a primary LLM PLUS multi-LLM tools — every agent can call
any other LLM provider on demand (ask_llm, ask_all_llms, compare_llms).
"""
from crewai import Agent, LLM
from crewai_tools import SerperDevTool, FileReadTool
from supabase_tools import get_supabase_tools, is_supabase_configured
from multi_llm_tools import get_multi_llm_tools, get_available_providers
import os


# ═══════════════════════════════════════════════════════════
# LLM PROVIDER FACTORY (for primary LLM)
# ═══════════════════════════════════════════════════════════

def _make_gemini_llm():
    return LLM(model=f"gemini/{os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')}",
               api_key=os.getenv("GEMINI_API_KEY"), temperature=0.7)

def _make_groq_llm():
    return LLM(model=f"groq/{os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}",
               api_key=os.getenv("GROQ_API_KEY"), temperature=0.7)

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

LLM_PROVIDERS = {
    "gemini": _make_gemini_llm,
    "groq": _make_groq_llm,
    "openrouter": _make_openrouter_llm,
    "huggingface": _make_huggingface_llm,
    "openai": _make_openai_llm,
    "anthropic": _make_anthropic_llm,
}

ENV_KEY_MAP = {
    "gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY", "huggingface": "HUGGINGFACE_API_KEY",
    "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
}

def _has_valid_key(provider: str) -> bool:
    val = os.getenv(ENV_KEY_MAP.get(provider, ""), "")
    return bool(val) and val not in ["", "your_gemini_api_key_here"]

def get_llm(provider: str = None):
    """Get an LLM instance by provider name. Auto-detects if none specified."""
    if provider and provider in LLM_PROVIDERS:
        if _has_valid_key(provider):
            return LLM_PROVIDERS[provider]()
        print(f"⚠️  {provider} API key not set, falling back to auto-detection")
    for name, factory in LLM_PROVIDERS.items():
        if _has_valid_key(name):
            print(f"🧠 Primary LLM: {name}")
            return factory()
    raise ValueError(
        "No LLM API key found! Set at least one in .env:\n"
        "- GEMINI_API_KEY (free)\n- GROQ_API_KEY (free)\n"
        "- OPENROUTER_API_KEY (free)\n- OPENAI_API_KEY (paid)\n"
        "- ANTHROPIC_API_KEY (paid)"
    )

def get_agent_llm(agent_name: str):
    """Get the primary LLM for a specific agent based on env routing."""
    env_key = f"{agent_name.upper()}_LLM"
    provider = os.getenv(env_key, "").strip().lower()
    if provider:
        return get_llm(provider)
    return get_llm()


# ═══════════════════════════════════════════════════════════
# TOOL ASSEMBLY — every agent gets multi-LLM tools
# ═══════════════════════════════════════════════════════════

def _get_agent_tools(agent_name: str = None):
    """Build the tool list for an agent — always includes multi-LLM tools."""
    tools = []
    # Research-oriented agents get web search + file read
    if agent_name in ["researcher", "architect", "critic"]:
        tools.append(SerperDevTool())
        tools.append(FileReadTool())
    # ALL agents get Supabase tools if configured
    if is_supabase_configured():
        tools.extend(get_supabase_tools())
    # ALL agents get multi-LLM tools — this is the key part
    # Every agent can call any LLM: ask_llm, ask_all_llms, compare_llms, list_available_llms
    tools.extend(get_multi_llm_tools())
    return tools


# ═══════════════════════════════════════════════════════════
# AGENT DEFINITIONS — each has a primary LLM + multi-LLM tools
# ═══════════════════════════════════════════════════════════

def create_researcher():
    return Agent(
        role="Researcher",
        goal="Gather comprehensive information and provide well-researched context. "
             "Use ask_all_llms to get diverse perspectives from all available AI models.",
        backstory=(
            "You are a meticulous researcher. You have access to MULTIPLE AI models "
            "and can call any of them at any time using your tools. Use ask_llm to "
            "query a specific model, ask_all_llms to get all models' opinions at once, "
            "or compare_llms to see different approaches side by side. You also have "
            "Supabase database access to store and retrieve research data."
        ),
        verbose=True,
        allow_delegation=False,
        llm=get_agent_llm("researcher"),
        tools=_get_agent_tools("researcher"),
    )

def create_architect():
    return Agent(
        role="Architect",
        goal="Design the system architecture. Use compare_llms to get multiple AI "
             "perspectives on the best approach before deciding.",
        backstory=(
            "You are a senior software architect. You have access to MULTIPLE AI models "
            "and can call any of them. Use compare_llms to get different models' "
            "opinions on architecture decisions. Use ask_llm to consult a specific "
            "model for a specific concern. You also have Supabase access to check "
            "existing schemas and store architecture plans."
        ),
        verbose=True,
        allow_delegation=True,
        llm=get_agent_llm("architect"),
        tools=_get_agent_tools("architect"),
    )

def create_implementer():
    return Agent(
        role="Implementer",
        goal="Write clean, production-ready code. Use ask_llm to consult different "
             "models for different parts — e.g., ask Claude for complex logic and "
             "GPT for API design.",
        backstory=(
            "You are a full-stack developer who writes production-quality code. "
            "You have access to MULTIPLE AI models. Use ask_llm to get help from "
            "specific models — e.g., call 'anthropic' for complex algorithms, "
            "'openai' for API design, 'gemini' for documentation. Use ask_all_llms "
            "for tricky problems where you want multiple approaches. You also have "
            "Supabase access to store and retrieve code artifacts."
        ),
        verbose=True,
        allow_delegation=False,
        llm=get_agent_llm("implementer"),
        tools=_get_agent_tools("implementer"),
    )

def create_critic():
    return Agent(
        role="Critic",
        goal="Review code critically. Use ask_all_llms to get all models to review "
             "the code independently, then synthesize their findings.",
        backstory=(
            "You are a strict but fair code reviewer. You have access to MULTIPLE AI "
            "models. Use ask_all_llms to have all models independently review the code — "
            "different models catch different bugs. Use compare_llms to compare security "
            "assessments from different models. You also have Supabase access to verify "
            "data integrity."
        ),
        verbose=True,
        allow_delegation=True,
        llm=get_agent_llm("critic"),
        tools=_get_agent_tools("critic"),
    )

def create_tester():
    return Agent(
        role="Tester",
        goal="Write comprehensive tests. Use ask_llm to get different models to "
             "generate test cases — more models means more edge cases caught.",
        backstory=(
            "You are a QA engineer. You have access to MULTIPLE AI models. Use "
            "ask_all_llms to get all models to suggest test cases — each model "
            "thinks differently and catches different edge cases. Use ask_llm to "
            "get a specific model's opinion on a tricky test scenario. You also "
            "have Supabase access to set up and verify test data."
        ),
        verbose=True,
        allow_delegation=True,
        llm=get_agent_llm("tester"),
        tools=_get_agent_tools("tester"),
    )

def create_devops():
    return Agent(
        role="DevOps Engineer",
        goal="Handle deployment and infrastructure. Use ask_llm to consult different "
             "models for Docker, CI/CD, and cloud config best practices.",
        backstory=(
            "You are a DevOps engineer. You have access to MULTIPLE AI models. Use "
            "ask_llm to get specific advice — e.g., call 'openai' for Docker optimization, "
            "'gemini' for CI/CD pipelines, 'anthropic' for security hardening. You also "
            "have Supabase access to track deployment state."
        ),
        verbose=True,
        allow_delegation=False,
        llm=get_agent_llm("devops"),
        tools=_get_agent_tools("devops"),
    )

def create_all_agents():
    """Create and return all 6 agents, each with multi-LLM access."""
    return {
        "researcher": create_researcher(),
        "architect": create_architect(),
        "implementer": create_implementer(),
        "critic": create_critic(),
        "tester": create_tester(),
        "devops": create_devops(),
    }

def list_available_providers():
    """Check which LLM providers have valid API keys configured."""
    return get_available_providers()
