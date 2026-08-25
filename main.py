"""
Entry point — supports both CLI and web app modes.
Web app serves a full dashboard UI + API.
Each agent is a character with personality and can call any/all LLM providers.

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


# ─── Web App Mode ──────────────────────────────────────────────────────────
def create_app():
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
    import pathlib

    app = FastAPI(
        title="Dev Team Crew",
        description="A 6-agent AI dev team — each agent is a character with personality "
                    "and can call any/all LLM providers on demand.",
        version="4.0.0",
    )

    # Serve static files (the dashboard UI)
    static_dir = pathlib.Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    class TaskRequest(BaseModel):
        topic: str

    @app.get("/")
    async def root():
        from agents import list_available_providers, list_team
        static_index = static_dir / "index.html"
        if static_index.exists():
            return FileResponse(str(static_index))
        return {
            "service": "Dev Team Crew", "version": "4.0.0",
            "team": list_team(), "llm_providers": list_available_providers(),
            "status": "ready" if list_available_providers() else "needs_api_keys",
        }

    @app.get("/api/info")
    async def info():
        from agents import list_team, list_available_providers
        return {
            "service": "Dev Team Crew", "version": "4.0.0",
            "team": list_team(),
            "llm_providers": list_available_providers(),
            "supported_providers": ["gemini", "groq", "openrouter", "huggingface", "openai", "anthropic"],
            "multi_llm_tools": ["ask_llm", "ask_all_llms", "compare_llms", "list_available_llms"],
            "status": "ready" if list_available_providers() else "needs_api_keys",
        }

    @app.get("/api/team")
    async def get_team():
        from agents import list_team, list_available_providers
        return {"team": list_team(), "available_llms": list_available_providers()}

    @app.get("/api/health")
    async def health():
        return {"status": "healthy"}

    @app.post("/api/run")
    async def run_crew(req: TaskRequest):
        from crew import DevTeamCrew
        crew = DevTeamCrew(req.topic)
        result = crew.run()
        return {"topic": req.topic, "result": str(result)}

    # Keep old endpoints for backward compat
    @app.get("/team")
    async def get_team_old():
        return await get_team()

    @app.get("/health")
    async def health_old():
        return await health()

    @app.post("/run")
    async def run_crew_old(req: TaskRequest):
        return await run_crew(req)

    return app


if __name__ == "__main__":
    run_cli()
else:
    app = create_app()
