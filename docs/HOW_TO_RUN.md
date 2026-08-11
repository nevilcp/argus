# ARGUS - Setup and Run Guide

## Prerequisites
- **Python 3.11+**
- **Docker & Docker Compose** (optional, for containerized deployments)
- **API Keys**: Groq, Google AI (Gemini), FRED, and NewsAPI

---

## Option 1: Local Development Setup

### 1. Create and Activate a Virtual Environment
Navigate to the project directory and set up an isolated Python environment:
```bash
cd /home/nevilcp/Desktop/Projects/argus
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
Install the project along with its development and testing dependencies:
```bash
pip install -e ".[dev,test]"
```

### 3. Configure Environment Variables
Copy the example environment file to set up your local secrets:
```bash
cp .env.example .env
```
Open the `.env` file and fill in the required API keys:
- `GROQ_API_KEY`: Drives Llama 3.3-70b (portfolio) and Llama 3.1-8b (sentiment/arbitration).
- `GOOGLE_AI_API_KEY`: Drives Gemini 3.1 Flash Lite (fundamental analysis).
- `FRED_API_KEY`: Drives Macroeconomic data (CPI, Yield Curve, etc).
- `NEWSAPI_KEY`: Drives News Sentiment analysis.


### 4. Start the Backend API (FastAPI)
In your terminal, start the FastAPI orchestrator:
```bash
uvicorn api.main:app --reload --port 8000
```
- The backend API will be available at **http://localhost:8000**.
- You can access the interactive Swagger documentation at **http://localhost:8000/docs**.

### 5. Start the Frontend UI (Streamlit)
Open a **second terminal window**, activate your virtual environment, and launch the UI:
```bash
cd /home/nevilcp/Desktop/Projects/argus
source .venv/bin/activate
streamlit run ui/app.py --server.port 8501
```
The dashboard will automatically open in your browser at **http://localhost:8501**.

---

## Option 2: Running via Docker

If you prefer to run the entire stack in isolated containers without managing local Python dependencies:

### 1. Configure Environment Variables
Ensure you have created and populated your `.env` file just like in the local setup:
```bash
cp .env.example .env
```

### 2. Build and Launch
Run Docker Compose in detached mode to build and start both the API and UI containers:
```bash
docker compose up -d --build
```

### 3. Access the Application
- **API:** http://localhost:8000
- **UI Dashboard:** http://localhost:8501

### 4. Shutting Down
To stop and remove the containers, run:
```bash
docker compose down
```

---

## Running Tests

To run the integration and unit tests locally (against the live statistical pipeline and mocked LLM agents):

```bash
# Ensure your virtual environment is activated
pytest tests/ -v --tb=short
```

## Useful Endpoints to Test First

Once the backend API is running, you can verify system health or trigger a manual test via terminal:

**1. Check API & Governor Health:**
```bash
curl http://localhost:8000/health
```

**2. Replay a recorded session (no live API endpoint — see ADR 0009):**
```bash
.venv/bin/python -m scripts.replay_backtest
```
