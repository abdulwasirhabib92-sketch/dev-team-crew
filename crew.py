"""
Crew assembly — defines the Dev Team Crew with all 6 agents and their tasks.
Supports multiple LLM providers with per-agent routing.
"""
from crewai import Crew, Process
from agents import create_all_agents, list_available_providers
from tasks import create_all_tasks


class DevTeamCrew:
    """A 6-agent development team powered by CrewAI with multi-LLM support."""

    def __init__(self, topic: str):
        self.topic = topic
        available = list_available_providers()
        if not available:
            raise ValueError(
                "No LLM API keys configured! Set at least one in .env.\n"
                "Free options: Gemini, Groq, OpenRouter, Hugging Face"
            )
        print(f"🔧 Available LLM providers: {', '.join(available)}")
        self.agents = create_all_agents()
        self.tasks = create_all_tasks(topic, self.agents)

    def run(self):
        """Execute the full dev team workflow."""
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff()

    def run_custom(self, custom_tasks):
        """Run with custom tasks instead of the default pipeline."""
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=custom_tasks,
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff()
