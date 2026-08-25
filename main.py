"""
Entry point — supports both CLI and web API modes.
Multi-LLM: each agent can call any/all LLM providers on demand.
Each agent is a character with personality, vibe, and working style.

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
    print(f"\n🚀 Dev Team Crew assembling for: {topic}\n")

    from agents import list_available_providers, list_team
    available = list_available_providers()
    print(f"🔧 Active LLM providers: {', '.join(available) if available else 'NONE — add API keys!'}")
    print("=" * 60)
    print("👥 The Team:")
    for member in list_team():
        print(f"  {member['codename']} ({member['role']}) — {member['vibe']}")
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
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(
        title="Dev Team Crew API",
        description="A 6-agent AI dev team — each agent is a character with personality "
                    "and can call any/all LLM providers on demand.",
        version="3.0.0",
    )

    class TaskRequest(BaseModel):
        topic: str

    @app.get("/")
    async def root():
        from agents import list_available_providers, list_team
        return {
            "service": "Dev Team Crew",
            "version": "3.0.0",
            "team": list_team(),
            "llm_providers": list_available_providers(),
            "supported_providers": ["gemini", "groq", "openrouter", "huggingface", "openai", "anthropic"],
            "multi_llm_tools": ["ask_llm", "ask_all_llms", "compare_llms", "list_available_llms"],
            "status": "ready" if list_available_providers() else "needs_api_keys",
            "endpoints": {
                "team": "GET /team",
                "run": "POST /run",
                "health": "GET /health",
            },
        }

    @app.get("/team")
    async def get_team():
        from agents import list_team, list_available_providers
        return {
            "team": list_team(),
            "available_llms": list_available_providers(),
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
