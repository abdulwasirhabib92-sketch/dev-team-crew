"""
Entry point — supports both CLI and web app modes.
Web app serves a full dashboard UI + API with async task execution.
11 LLM providers: 6 Western + 5 Chinese (all free tiers available).
Each agent is a character with personality and can call any/all LLM providers.

CLI:  python main.py "Build a REST API for a todo app"
API:  uvicorn main:app --host 0.0.0.0 --port $PORT
"""
import os
import sys
import uuid
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def run_cli():
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<your project description>\"")
        print('Example: python main.py "Build a REST API for a todo app"')
        sys.exit(1)
    topic = sys.argv[1]
    print(f"\n🚀 Dev Team Crew assembling for: {topic}\n")
    from agents import list_available_providers, list_team
    available = list_available_providers()
    print(f"🔧 Active LLM providers ({len(available)}): {', '.join(available)}}")
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


def _get_supabase():
    """Get Supabase client for task storage."""
    try:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        if url and key:
            return create_client(url, key)
    except Exception:
        pass
    return None


def create_app():
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
    import pathlib

    app = FastAPI(
        title="Dev Team Crew",
        description="6-agent AI dev team — 11 LLM providers (6 Western + 5 Chinese). "
                    "Each agent is a character with personality and multi-LLM access. "
                    "Async task execution with Supabase storage.",
        version="6.0.0",
    )

    static_dir = pathlib.Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # In-memory task store (fallback when Supabase is not available)
    _task_store = {}

    class TaskRequest(BaseModel):
        topic: str

    @app.get("/")
    async def root():
        static_index = static_dir / "index.html"
        if static_index.exists():
            return FileResponse(str(static_index))
        return {"service": "Dev Team Crew", "version": "6.0.0"}

    @app.get("/api/info")
    async def info():
        from agents import list_team, list_available_providers, list_providers
        return {
            "service": "Dev Team Crew", "version": "6.0.0",
            "team": list_team(),
            "llm_providers": list_available_providers(),
            "all_providers": list_providers(),
            "multi_llm_tools": ["ask_llm", "ask_all_llms", "compare_llms", "list_available_llms"],
            "status": "ready" if list_available_providers() else "needs_api_keys",
        }

    @app.get("/api/team")
    async def get_team():
        from agents import list_team, list_available_providers
        return {"team": list_team(), "available_llms": list_available_providers()}

    @app.get("/api/providers")
    async def get_providers():
        from agents import list_providers
        return {"providers": list_providers()}

    @app.get("/api/health")
    async def health():
        return {"status": "healthy"}

    @app.post("/api/run")
    async def run_crew(req: TaskRequest):
        """Submit a task — runs in background, returns task_id for polling."""
        task_id = str(uuid.uuid4())[:8]
        supabase = _get_supabase()

        # Store task in Supabase or in-memory
        if supabase:
            try:
                supabase.table("crew_tasks").insert({
                    "task_id": task_id,
                    "topic": req.topic,
                    "status": "running",
                    "created_at": datetime.utcnow().isoformat(),
                }).execute()
            except Exception as e:
                print(f"⚠️ Supabase insert failed: {e}")
                supabase = None

        if not supabase:
            _task_store[task_id] = {
                "task_id": task_id,
                "topic": req.topic,
                "status": "running",
                "result": None,
                "created_at": datetime.utcnow().isoformat(),
            }

        # Run crew in background thread
        try:
            from crew import DevTeamCrew
            crew = DevTeamCrew(req.topic)

            def _run_bg():
                try:
                    result = str(crew.run())
                    if supabase:
                        try:
                            supabase.table("crew_results").insert({
                                "task_id": task_id,
                                "topic": req.topic,
                                "status": "completed",
                                "result": result,
                            }).execute()
                            supabase.table("crew_tasks").update({
                                "status": "completed"
                            }).eq("task_id", task_id).execute()
                        except Exception:
                            pass
                    else:
                        _task_store[task_id]["status"] = "completed"
                        _task_store[task_id]["result"] = result
                    print(f"✅ Task {task_id} completed ({len(result)} chars)")
                except Exception as e:
                    import traceback
                    error = f"Error: {str(e)}\n{traceback.format_exc()}"
                    if supabase:
                        try:
                            supabase.table("crew_results").insert({
                                "task_id": task_id,
                                "topic": req.topic,
                                "status": "failed",
                                "result": error,
                            }).execute()
                            supabase.table("crew_tasks").update({
                                "status": "failed"
                            }).eq("task_id", task_id).execute()
                        except Exception:
                            pass
                    else:
                        _task_store[task_id]["status"] = "failed"
                        _task_store[task_id]["result"] = error
                    print(f"❌ Task {task_id} failed: {str(e)[:100]}")

            import threading
            thread = threading.Thread(target=_run_bg, daemon=True)
            thread.start()
        except Exception as e:
            error = f"Failed to start crew: {str(e)}"
            if supabase:
                supabase.table("crew_tasks").update({"status": "failed", "error": error}).eq("task_id", task_id).execute()
            else:
                _task_store[task_id]["status"] = "failed"
                _task_store[task_id]["result"] = error
            return {"task_id": task_id, "status": "failed", "error": error}

        return {"task_id": task_id, "status": "running", "topic": req.topic}

    @app.get("/api/task/{task_id}")
    async def get_task_status(task_id: str):
        """Poll for task status and results."""
        supabase = _get_supabase()
        if supabase:
            try:
                # Check task status
                resp = supabase.table("crew_tasks").select("*").eq("task_id", task_id).execute()
                if resp.data:
                    task = resp.data[0]
                    result = None
                    if task.get("status") in ["completed", "failed"]:
                        rresp = supabase.table("crew_results").select("*").eq("task_id", task_id).execute()
                        if rresp.data:
                            result = rresp.data[0].get("result")
                    return {
                        "task_id": task_id,
                        "status": task.get("status", "unknown"),
                        "topic": task.get("topic"),
                        "result": result,
                    }
            except Exception as e:
                print(f"⚠️ Supabase query failed: {e}")

        # Fallback to in-memory
        task = _task_store.get(task_id)
        if task:
            return {
                "task_id": task_id,
                "status": task["status"],
                "topic": task["topic"],
                "result": task["result"],
            }
        return {"task_id": task_id, "status": "not_found"}

    @app.get("/api/tasks")
    async def list_tasks():
        """List all tasks."""
        supabase = _get_supabase()
        if supabase:
            try:
                resp = supabase.table("crew_tasks").select("*").order("created_at", desc=True).limit(20).execute()
                return {"tasks": resp.data}
            except Exception:
                pass
        return {"tasks": list(_task_store.values())}

    # Backward compat
    @app.get("/team")
    async def get_team_old(): return await get_team()
    @app.get("/health")
    async def health_old(): return await health()
    @app.post("/run")
    async def run_crew_old(req: TaskRequest): return await run_crew(req)

    return app


if __name__ == "__main__":
    run_cli()
else:
    app = create_app()
