"""
Entry point — supports both CLI and web API modes.
Multi-LLM: Gemini, Groq, OpenRouter, Hugging Face, OpenAI, Anthropic.

CLI:  python main.py "Build a REST API for a todo app"
API:  uvicorn main:app --host 0.0.0.0 --port $PORT
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()


# ─── CLI Mode ──────────────────────────────────────────────────────────────
def run_cli():
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<your project description>\"")
        print('Example: python main.py "Build a REST API for a todo app"')
        sys.exit(1)

    topic = sys.argv[1]
    print(f"\n🚀 Starting Dev Team Crew for: {topic}\n")

    # Show which providers are active
    from agents import list_available_providers
    available = list_available_providers()
    print(f"🔧 Active LLM providers: {', '.join(available)}")
    print("=" * 60)

    from crew import DevTeamCrew
    crew = DevTeamCrew(topic)
    result = crew.run()

    print("=" * 60)
    print("✅ Dev Team Crew completed!\n")
    print(result)
    return result


# ─── Web API Mode ──────────────────────────────────────────────────────────
def create_app():
    """Create a FastAPI app for web deployment."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(
        title="Dev Team Crew API",
        description="A 6-agent AI development team with multi-LLM support: "
                    "Researcher, Architect, Implementer, Critic, Tester, DevOps",
        version="2.0.0",
    )

    class TaskRequest(BaseModel):
        topic: str

    @app.get("/")
    async def root():
        from agents import list_available_providers
        return {
            "service": "Dev Team Crew",
            "version": "2.0.0",
            "agents": [
                {"name": "Researcher", "role": "Gathers info, explores options"},
                {"name": "Architect", "role": "Designs system, creates plans"},
                {"name": "Implementer", "role": "Writes production code"},
                {"name": "Critic", "role": "Reviews for bugs, security, quality"},
                {"name": "Tester", "role": "Writes tests, reports failures"},
                {"name": "DevOps", "role": "Handles deployment and CI/CD"},
            ],
            "llm_providers": list_available_providers(),
            "supported_providers": ["gemini", "groq", "openrouter", "huggingface", "openai", "anthropic"],
            "status": "ready",
            "endpoints": {
                "run": "POST /run",
                "health": "GET /health",
            },
        }

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.post("/run")
    async def run_crew(req: TaskRequest):
        from crew import DevTeamCrew
        crew = DevTeamCrew(req.topic)
        result = crew.run()
        return {"topic": req.topic, "result": str(result)}

    return app


if __name__ == "__main__":
    run_cli()
else:
    app = create_app()
