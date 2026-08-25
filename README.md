# Dev Team Crew 🤖

A 6-agent AI development team built with [CrewAI](https://github.com/crewAIInc/crewAI) and multiple free LLM providers.

## Agents

| # | Agent | Role |
|---|-------|------|
| 1 | 🔍 **Researcher** | Gathers info, explores options, finds docs |
| 2 | 🏗️ **Architect** | Designs system, breaks tasks into steps |
| 3 | 💻 **Implementer** | Writes clean, production-ready code |
| 4 | 🔎 **Critic** | Reviews for bugs, security, quality |
| 5 | 🧪 **Tester** | Writes and runs tests, reports failures |
| 6 | 🚀 **DevOps** | Handles deployment, CI/CD, infrastructure |

## Supported LLM Providers (all have free tiers!)

| Provider | Free Tier | Get API Key |
|----------|-----------|-------------|
| Google Gemini | ✅ | https://aistudio.google.com/apikey |
| Groq | ✅ | https://console.groq.com/keys |
| OpenRouter | ✅ (free models) | https://openrouter.ai/keys |
| Hugging Face | ✅ | https://huggingface.co/settings/tokens |
| OpenAI | ❌ (paid) | https://platform.openai.com/api-keys |
| Anthropic | ❌ (paid) | https://console.anthropic.com/ |

You can use **any combination** of providers. Each agent can use a different LLM via environment routing.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/abdulwasirhabib92-sketch/dev-team-crew.git
cd dev-team-crew

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment — add at least one API key
cp .env.example .env
# Edit .env with your API keys

# 4. Run via CLI
python main.py "Build a REST API for a todo app"

# 5. Or run as a web API
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Per-Agent LLM Routing

Control which provider each agent uses in `.env`:

```env
RESEARCHER_LLM=gemini
ARCHITECT_LLM=groq
IMPLEMENTER_LLM=gemini
CRITIC_LLM=groq
TESTER_LLM=gemini
DEVOPS_LLM=groq
```

Leave any unset and it auto-detects the first available provider.

## Why Multiple APIs?

- **Redundancy** — if one provider is down, others keep the team running
- **Cost optimization** — mix free and paid providers strategically
- **Model diversity** — different models excel at different tasks (e.g., Gemini for research, Groq for fast code generation)
- **Rate limit relief** — spread requests across providers to avoid hitting limits

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info, agent list, active providers |
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

## Deployment

### Render (free tier)

1. Push to GitHub
2. Create a new Web Service on Render
3. Connect your GitHub repo
4. Render auto-detects `render.yaml`
5. Add your API keys as environment variables
6. Deploy!

### Docker

```bash
docker build -t dev-team-crew .
docker run -p 10000:10000 \
  -e GEMINI_API_KEY=your_key \
  -e GROQ_API_KEY=your_key \
  dev-team-crew
```

## License

MIT
