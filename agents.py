"""
Agent definitions for the Dev Team Crew.
Each agent has a distinct role, backstory, and LLM provider.
Supports multiple API providers: Gemini, Groq, OpenRouter, Hugging Face, OpenAI, Anthropic.
Supabase tools available for data operations.
"""
from crewai import Agent, LLM
from crewai_tools import SerperDevTool, FileReadTool
from supabase_tools import get_supabase_tools, is_supabase_configured
import os


# ═══════════════════════════════════════════════════════════
# LLM PROVIDER FACTORY
# ═══════════════════════════════════════════════════════════

def _make_gemini_llm():
    return LLM(
        model=f"gemini/{os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')}",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7,
    )


def _make_groq_llm():
    return LLM(
        model=f"groq/{os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
    )


def _make_openrouter_llm():
    return LLM(
        model=f"openrouter/{os.getenv('OPENROUTER_MODEL', 'meta-llama/llama-3.1-8b-instruct:free')}",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.7,
        base_url="https://openrouter.ai/api/v1",
    )


def _make_huggingface_llm():
    return LLM(
        model=f"huggingface/{os.getenv('HUGGINGFACE_MODEL', 'meta-llama/Llama-3.1-8B-Instruct')}",
        api_key=os.getenv("HUGGINGFACE_API_KEY"),
        temperature=0.7,
    )


def _make_openai_llm():
    return LLM(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
    )


def _make_anthropic_llm():
    return LLM(
        model=f"anthropic/{os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022')}",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.7,
    )


# Registry of all available providers
LLM_PROVIDERS = {
    "gemini": _make_gemini_llm,
    "groq": _make_groq_llm,
    "openrouter": _make_openrouter_llm,
    "huggingface": _make_huggingface_llm,
    "openai": _make_openai_llm,
    "anthropic": _make_anthropic_llm,
}

# Map of env var names for each provider's API key
ENV_KEY_MAP = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _has_valid_key(provider: str) -> bool:
    key = ENV_KEY_MAP.get(provider, "")
    val = os.getenv(key, "")
    return bool(val) and val not in ["", "your_gemini_api_key_here"]


def get_llm(provider: str = None):
    """Get an LLM instance by provider name. Auto-detects if none specified."""
    if provider and provider in LLM_PROVIDERS:
        if _has_valid_key(provider):
            return LLM_PROVIDERS[provider]()
        print(f"⚠️  {provider} API key not set, falling back to auto-detection")

    for name, factory in LLM_PROVIDERS.items():
        if _has_valid_key(name):
            print(f"🧠 Using LLM provider: {name}")
            return factory()

    raise ValueError(
        "No LLM API key found! Set at least one in .env:\n"
        "- GEMINI_API_KEY (free: https://aistudio.google.com/apikey)\n"
        "- GROQ_API_KEY (free: https://console.groq.com/keys)\n"
        "- OPENROUTER_API_KEY (free: https://openrouter.ai/keys)\n"
        "- HUGGINGFACE_API_KEY (free: https://huggingface.co/settings/tokens)"
    )


def get_agent_llm(agent_name: str):
    """Get the LLM for a specific agent based on env routing."""
    env_key = f"{agent_name.upper()}_LLM"
    provider = os.getenv(env_key, "").strip().lower()
    if provider:
        return get_llm(provider)
    return get_llm()


def _get_agent_tools(agent_name: str = None):
    """Build the tool list for an agent."""
    tools = []
    # Research-oriented agents get web search + file read
    if agent_name in ["researcher", "architect", "critic"]:
        tools.append(SerperDevTool())
        tools.append(FileReadTool())
    # All agents get Supabase tools if configured
    if is_supabase_configured():
        tools.extend(get_supabase_tools())
    return tools


# ═══════════════════════════════════════════════════════════
# AGENT DEFINITIONS
# ═══════════════════════════════════════════════════════════

def create_researcher():
    return Agent(
        role="Researcher",
        goal="Gather comprehensive information, explore options, and provide "
        "well-researched context for the team to make informed decisions.",
        backstory=(
            "You are a meticulous researcher with years of experience in software "
            "development. You excel at finding the right documentation, comparing "
            "technologies, and presenting findings in a clear, structured way. "
            "You can also query the Supabase database to understand existing data "
        ),
        verbose=True,
        allow_delegation=False,
        llm=get_agent_llm("researcher"),
        tools=_get_agent_tools("researcher"),
    )


def create_architect():
    return Agent(
        role="Architect",
        goal="Design the overall system architecture, break tasks into actionable "
        "steps, and define the best technical approach for the team.",
        backstory=(
            "You are a senior software architect with 15+ years of experience. "
            "You think in terms of trade-offs, scalability, and maintainability. "
            "You can query Supabase to check existing schemas and plan accordingly."
        ),
        verbose=True,
        allow_delegation=True,
        llm=get_agent_llm("architect"),
        tools=_get_agent_tools("architect"),
    )


def create_implementer():
    return Agent(
        role="Implementer",
        goal="Write clean, efficient, well-documented code that follows the "
        "architecture plan and implements the required features.",
        backstory=(
            "You are a full-stack developer who writes production-quality code. "
            "You can use Supabase tools to store and retrieve project data, "
            "manage database records, and persist work across the team."
        ),
        verbose=True,
        allow_delegation=False,
        llm=get_agent_llm("implementer"),
        tools=_get_agent_tools("implementer"),
    )


def create_critic():
    return Agent(
        role="Critic",
        goal="Review code and plans critically, identify bugs, edge cases, "
        "security issues, and areas for improvement before anything ships.",
        backstory=(
            "You are a strict but fair code reviewer. You focus on correctness, "
            "security, performance, and code quality. You can query Supabase "
            "to verify data integrity and check for consistency."
        ),
        verbose=True,
        allow_delegation=True,
        llm=get_agent_llm("critic"),
        tools=_get_agent_tools("critic"),
    )


def create_tester():
    return Agent(
        role="Tester",
        goal="Write comprehensive tests covering unit, integration, and edge "
        "cases. Report failures clearly so the Implementer can fix them.",
        backstory=(
            "You are a QA engineer who believes tests are as important as the "
            "code itself. You can use Supabase to set up test data, verify "
            "database operations, and clean up after tests."
        ),
        verbose=True,
        allow_delegation=True,
        llm=get_agent_llm("tester"),
        tools=_get_agent_tools("tester"),
    )


def create_devops():
    return Agent(
        role="DevOps Engineer",
        goal="Handle deployment, CI/CD pipelines, environment setup, and "
        "infrastructure so the team's code runs reliably in production.",
        backstory=(
            "You are a DevOps engineer who has deployed hundreds of services. "
            "You know Docker, CI/CD, cloud platforms, and monitoring inside out. "
            "You can use Supabase to manage deployment configs and track "
            "infrastructure state in the database."
        ),
        verbose=True,
        allow_delegation=False,
        llm=get_agent_llm("devops"),
        tools=_get_agent_tools("devops"),
    )


def create_all_agents():
    """Create and return all 6 agents as a dictionary."""
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
    available = []
    for name in ENV_KEY_MAP:
        if _has_valid_key(name):
            available.append(name)
    return available
