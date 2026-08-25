"""
Agent definitions for the Dev Team Crew.
Each agent has a distinct role, backstory, and LLM provider.
Supports multiple API providers: Gemini, Groq, OpenRouter, Hugging Face, OpenAI, Anthropic.
"""
from crewai import Agent, LLM
from crewai_tools import SerperDevTool, FileReadTool
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
    from langchain_groq import ChatGroq
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


def get_llm(provider: str = None):
    """
    Get an LLM instance by provider name.
    If no provider is specified, auto-detect the first available one.
    """
    if provider and provider in LLM_PROVIDERS:
        key = provider.upper().replace("_", "") + "_API_KEY"
        # Groq uses GROQ_API_KEY, check if it's set
        env_key_map = {
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        expected_key = env_key_map.get(provider, "")
        if expected_key and os.getenv(expected_key) and os.getenv(expected_key) != "your_gemini_api_key_here":
            return LLM_PROVIDERS[provider]()
        else:
            print(f"⚠️  {provider} API key not set, falling back to auto-detection")

    # Auto-detect: find the first provider with a valid key
    for name, factory in LLM_PROVIDERS.items():
        env_key_map = {
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        key = env_key_map.get(name, "")
        if key and os.getenv(key) and os.getenv(key) not in ["", "your_gemini_api_key_here"]:
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
    """
    Get the LLM for a specific agent based on env routing.
    Falls back to auto-detection if no specific route is set.
    """
    env_key = f"{agent_name.upper()}_LLM"
    provider = os.getenv(env_key, "").strip().lower()
    if provider:
        return get_llm(provider)
    return get_llm()  # auto-detect


# ═══════════════════════════════════════════════════════════
# AGENT DEFINITIONS
# ═══════════════════════════════════════════════════════════

def create_researcher():
    """Gathers info, reads docs, explores options."""
    return Agent(
        role="Researcher",
        goal="Gather comprehensive information, explore options, and provide "
        "well-researched context for the team to make informed decisions.",
        backstory=(
            "You are a meticulous researcher with years of experience in software "
            "development. You excel at finding the right documentation, comparing "
            "technologies, and presenting findings in a clear, structured way. "
            "You leave no stone unturned."
        ),
        verbose=True,
        allow_delegation=False,
        llm=get_agent_llm("researcher"),
        tools=[SerperDevTool(), FileReadTool()],
    )


def create_architect():
    """Breaks the task into steps, picks the tech approach."""
    return Agent(
        role="Architect",
        goal="Design the overall system architecture, break tasks into actionable "
        "steps, and define the best technical approach for the team.",
        backstory=(
            "You are a senior software architect with 15+ years of experience. "
            "You've designed systems ranging from startups to enterprise scale. "
            "You think in terms of trade-offs, scalability, and maintainability. "
            "You create clear, step-by-step plans that any developer can follow."
        ),
        verbose=True,
        allow_delegation=True,
        llm=get_agent_llm("architect"),
    )


def create_implementer():
    """Writes the code."""
    return Agent(
        role="Implementer",
        goal="Write clean, efficient, well-documented code that follows the "
        "architecture plan and implements the required features.",
        backstory=(
            "You are a full-stack developer who writes production-quality code. "
            "You follow best practices, write meaningful comments, and always "
            "consider edge cases. You take pride in clean, readable code that "
            "others can easily maintain."
        ),
        verbose=True,
        allow_delegation=False,
        llm=get_agent_llm("implementer"),
    )


def create_critic():
    """Reviews for bugs, edge cases, quality."""
    return Agent(
        role="Critic",
        goal="Review code and plans critically, identify bugs, edge cases, "
        "security issues, and areas for improvement before anything ships.",
        backstory=(
            "You are a strict but fair code reviewer. You've caught countless "
            "bugs before they reached production. You focus on correctness, "
            "security, performance, and code quality. You never approve something "
            "you wouldn't deploy yourself."
        ),
        verbose=True,
        allow_delegation=True,
        llm=get_agent_llm("critic"),
    )


def create_tester():
    """Writes and runs tests, reports failures back to the Implementer."""
    return Agent(
        role="Tester",
        goal="Write comprehensive tests covering unit, integration, and edge "
        "cases. Report failures clearly so the Implementer can fix them.",
        backstory=(
            "You are a QA engineer who believes tests are as important as the "
            "code itself. You write tests that break things on purpose to find "
            "weaknesses. You report failures with clear reproduction steps and "
            "never mark something as tested unless you've verified it yourself."
        ),
        verbose=True,
        allow_delegation=True,
        llm=get_agent_llm("tester"),
    )


def create_devops():
    """Handles deployment, CI/CD, env setup."""
    return Agent(
        role="DevOps Engineer",
        goal="Handle deployment, CI/CD pipelines, environment setup, and "
        "infrastructure so the team's code runs reliably in production.",
        backstory=(
            "You are a DevOps engineer who has deployed hundreds of services. "
            "You know Docker, CI/CD, cloud platforms, and monitoring inside out. "
            "You automate everything and ensure zero-downtime deployments."
        ),
        verbose=True,
        allow_delegation=False,
        llm=get_agent_llm("devops"),
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
    env_key_map = {
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    for name, key in env_key_map.items():
        val = os.getenv(key, "")
        if val and val not in ["", "your_gemini_api_key_here"]:
            available.append(name)
    return available
