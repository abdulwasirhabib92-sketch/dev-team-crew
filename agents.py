"""
Agent definitions for the Dev Team Crew.
Each agent has a distinct role, backstory, and set of tools.
"""
from crewai import Agent, LLM
from crewai_tools import SerperDevTool, FileReadTool
import os


def get_llm():
    """Create a Google Gemini LLM instance."""
    return LLM(
        model=f"gemini/{os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')}",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7,
    )


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
        llm=get_llm(),
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
        llm=get_llm(),
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
        llm=get_llm(),
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
        llm=get_llm(),
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
        llm=get_llm(),
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
        llm=get_llm(),
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
