"""
Crew assembly — defines the Dev Team Crew with all 6 agents and their tasks.
Supports multiple LLM providers with per-agent routing.
Includes async execution for long-running tasks (stored in Supabase).
"""
from crewai import Crew, Process
from agents import create_all_agents, list_available_providers
from tasks import create_all_tasks
import threading
import traceback


class DevTeamCrew:
    """A 6-agent development team powered by CrewAI with multi-LLM support."""

    def __init__(self, topic: str):
        self.topic = topic
        available = list_available_providers()
        if not available:
            raise ValueError(
                "No LLM API keys configured! Set at least one in .env.\n"
                "Free options: Gemini, Groq, OpenRouter, Hugging Face, "
                "DeepSeek, SiliconFlow, Qwen, GLM, Moonshot"
            )
        print(f"🔧 Available LLM providers: {', '.join(available)}}")
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

    def run_async(self, task_id: str, supabase_client=None):
        """Run the crew in a background thread, storing results in Supabase."""
        def _run():
            try:
                result = self.run()
                result_str = str(result)
                if supabase_client:
                    supabase_client.table("crew_results").insert({
                        "task_id": task_id,
                        "topic": self.topic,
                        "status": "completed",
                        "result": result_str,
                    }).execute()
                    supabase_client.table("crew_tasks").update({
                        "status": "completed",
                    }).eq("task_id", task_id).execute()
                print(f"✅ Task {task_id} completed")
                return result_str
            except Exception as e:
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                print(f"❌ Task {task_id} failed: {error_msg}")
                if supabase_client:
                    supabase_client.table("crew_results").insert({
                        "task_id": task_id,
                        "topic": self.topic,
                        "status": "failed",
                        "result": error_msg,
                    }).execute()
                    supabase_client.table("crew_tasks").update({
                        "status": "failed",
                        "error": str(e),
                    }).eq("task_id", task_id).execute()
                return error_msg

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return task_id
