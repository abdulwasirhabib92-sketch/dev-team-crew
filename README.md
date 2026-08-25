# Dev Team Crew 🤖

A 6-agent AI development team built with [CrewAI](https://github.com/crewAIInc/crewAI) and Google Gemini (free tier).

## Agents

| # | Agent | Role |
|---|-------|------|
| 1 | 🔍 **Researcher** | Gathers info, explores options, finds docs |
| 2 | 🏗️ **Architect** | Designs system, breaks tasks into steps |
| 3 | 💻 **Implementer** | Writes clean, production-ready code |
| 4 | 🔎 **Critic** | Reviews for bugs, security, quality |
| 5 | 🧪 **Tester** | Writes and runs tests, reports failures |
| 6 | 🚀 **DevOps** | Handles deployment, CI/CD, infrastructure |

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url>
cd dev-team-crew

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Add your Gemini API key from https://aistudio.google.com/apikey

# 4. Run via CLI
python main.py "Build a REST API for a todo app"

# 5. Or run as a web API
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and agent list |
| GET | `/health` | Health check |
| POST | `/run` | Run the crew with a task |

### Example API Call

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"topic": "Build a REST API for a todo app"}'
```

## How It Works

The crew runs sequentially:

```
Researcher → Architect → Implementer → Critic → Tester → DevOps
```

1. **Researcher** investigates the topic and finds best approaches
2. **Architect** designs the system and creates an implementation plan
3. **Implementer** writes the code based on the plan
4. **Critic** reviews the code for bugs, security, and quality
5. **Tester** writes tests and reports any failures
6. **DevOps** sets up deployment, Docker, and CI/CD

## Customizing Agents

Edit `agents.py` to change agent roles, goals, and backstories.
Edit `tasks.py` to change what each agent does.
Edit `crew.py` to change the workflow (sequential, hierarchical, etc.).

## Cost

**$0** — Uses Google Gemini's free tier API.

## Deployment

### Render (free tier)

1. Push to GitHub
2. Create a new Web Service on Render
3. Connect your GitHub repo
4. Render will auto-detect `render.yaml`
5. Add `GEMINI_API_KEY` as an environment variable
6. Deploy!

### Docker

```bash
docker build -t dev-team-crew .
docker run -p 10000:10000 -e GEMINI_API_KEY=your_key dev-team-crew
```

## License

MIT
