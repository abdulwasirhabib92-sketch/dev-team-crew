"""
Agent definitions for the Dev Team Crew.
Each agent is a CHARACTER — with personality, vibe, and working style.
Designed like Elara's identity but tailored to each role.
Every agent has a primary LLM + multi-LLM tools (ask_llm, ask_all_llms, compare_llms).
"""
from crewai import Agent, LLM
from crewai_tools import SerperDevTool, FileReadTool
from supabase_tools import get_supabase_tools, is_supabase_configured
from multi_llm_tools import get_multi_llm_tools, get_available_providers
from agent_identities import AGENT_IDENTITIES
import os


# ═══════════════════════════════════════════════════════════
# LLM PROVIDER FACTORY
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
    "gemini": _make_gemini_llm, "groq": _make_groq_llm,
    "openrouter": _make_openrouter_llm, "huggingface": _make_huggingface_llm,
    "openai": _make_openai_llm, "anthropic": _make_anthropic_llm,
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
    if provider and provider in LLM_PROVIDERS:
        if _has_valid_key(provider):
            return LLM_PROVIDERS[provider]()
        print(f"⚠️  {provider} API key not set, falling back to auto-detection")
    for name, factory in LLM_PROVIDERS.items():
        if _has_valid_key(name):
            print(f"🧠 Primary LLM: {name}")
            return factory()
    raise ValueError("No LLM API key found!")

def get_agent_llm(agent_name: str):
    env_key = f"{agent_name.upper()}_LLM"
    provider = os.getenv(env_key, "").strip().lower()
    if provider:
        return get_llm(provider)
    return get_llm()


# ═══════════════════════════════════════════════════════════
# TOOL ASSEMBLY — every agent gets multi-LLM tools
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
    """Pull the personality, backstory, and vibe from agent_identities."""
    ident = AGENT_IDENTITIES.get(agent_key, {})
    name = ident.get("name", agent_key.title())
    personality = ident.get("personality", "")
    working_style = ident.get("working_style", "")
    catchphrase = ident.get("catchphrase", "")
    vibe = ident.get("vibe", "")

    backstory = (
        f"{personality}\n\n"
        f"Your working style: {working_style}\n\n"
        f"Your vibe: {vibe}\n"
        f"Your catchphrase: \"{catchphrase}\""
    )
    return name, backstory


# ═══════════════════════════════════════════════════════════
# AGENT DEFINITIONS — each is a character with personality
# ═══════════════════════════════════════════════════════════

def create_researcher():
    name, backstory = _build_identity("researcher")
    return Agent(
        role=f"Researcher ({name})",
        goal=(
            "Gather comprehensive information and provide well-researched context. "
            "Use ask_all_llms to get diverse perspectives from all available AI models — "
            "you're naturally curious and love consulting multiple experts. "
            "Store your findings in Supabase for the team."
        ),
        backstory=backstory,
        verbose=True,
        allow_delegation=False,
        llm=get_agent_llm("researcher"),
        tools=_get_agent_tools("researcher"),
    )

def create_architect():
    name, backstory = _build_identity("architect")
    return Agent(
        role=f"Architect ({name})",
        goal=(
            "Design the system architecture and break tasks into clear steps. "
            "Use compare_llms to get multiple AI perspectives before deciding — "
            "every decision is a trade-off and you want to see all angles. "
            "Store architecture plans in Supabase."
        ),
        backstory=backstory,
        verbose=True,
        allow_delegation=True,
        llm=get_agent_llm("architect"),
        tools=_get_agent_tools("architect"),
    )

def create_implementer():
    name, backstory = _build_identity("implementer")
    return Agent(
        role=f"Implementer ({name})",
        goal=(
            "Write clean, production-ready code. Use ask_llm to consult different "
            "models for different parts — Claude for complex logic, GPT for API "
            "design, Gemini for quick scaffolding. You take pride in your craft. "
            "Store code artifacts in Supabase."
        ),
        backstory=backstory,
        verbose=True,
        allow_delegation=False,
        llm=get_agent_llm("implementer"),
        tools=_get_agent_tools("implementer"),
    )

def create_critic():
    name, backstory = _build_identity("critic")
    return Agent(
        role=f"Critic ({name})",
        goal=(
            "Review code critically for bugs, security, and quality. "
            "Use ask_all_llms to have every model independently review the code — "
            "different models catch different bugs. Always provide a fix, "
            "never just a complaint. Store review notes in Supabase."
        ),
        backstory=backstory,
        verbose=True,
        allow_delegation=True,
        llm=get_agent_llm("critic"),
        tools=_get_agent_tools("critic"),
    )

def create_tester():
    name, backstory = _build_identity("tester")
    return Agent(
        role=f"Tester ({name})",
        goal=(
            "Write comprehensive tests. Use ask_all_llms to get every model to "
            "suggest test cases — each model thinks differently and catches "
            "different edge cases. Report failures with clear steps. "
            "Use Supabase for test data setup and cleanup."
        ),
        backstory=backstory,
        verbose=True,
        allow_delegation=True,
        llm=get_agent_llm("tester"),
        tools=_get_agent_tools("tester"),
    )

def create_devops():
    name, backstory = _build_identity("devops")
    return Agent(
        role=f"DevOps Engineer ({name})",
        goal=(
            "Handle deployment, CI/CD, and infrastructure. Use ask_llm to consult "
            "specific models — OpenAI for Docker, Gemini for CI/CD, Anthropic "
            "for security. Automate everything. Track deployment state in Supabase."
        ),
        backstory=backstory,
        verbose=True,
        allow_delegation=False,
        llm=get_agent_llm("devops"),
        tools=_get_agent_tools("devops"),
    )

def create_all_agents():
    """Create and return all 6 agents — each a character with personality."""
    return {
        "researcher": create_researcher(),
        "architect": create_architect(),
        "implementer": create_implementer(),
        "critic": create_critic(),
        "tester": create_tester(),
        "devops": create_devops(),
    }

def list_available_providers():
    return get_available_providers()

def list_team():
    """List all team members with their identities."""
    team = []
    for key, ident in AGENT_IDENTITIES.items():
        team.append({
            "codename": ident["name"],
            "role": key.title(),
            "vibe": ident["vibe"],
            "catchphrase": ident["catchphrase"],
        })
    return team
