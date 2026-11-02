
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage


# =====================================================
# 1. IMPORT EXISTING URBANPULSE AGENT
# =====================================================

SCRIPTS_DIR = Path(__file__).resolve().parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from urbanpulse_agent import build_urbanpulse_agent


# =====================================================
# 2. APPLICATION LIFESPAN
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting UrbanPulse AI backend...")

    # Initialize LangGraph agent once at startup
    app.state.agent = build_urbanpulse_agent()

    print("UrbanPulse AI backend is ready.")

    yield

    # Cleanup during shutdown
    app.state.agent = None

    print("UrbanPulse AI backend stopped.")


# Create FastAPI application
app = FastAPI(
    title="UrbanPulse AI API",
    description="API for UrbanPulse AI Multimodal RAG Agent",
    version="1.0.0",
    lifespan=lifespan
)


# =====================================================
# 3. REQUEST AND RESPONSE MODELS
# =====================================================

class ChatRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question asked by the user"
    )

    session_id: str = Field(
        default="demo-session",
        min_length=1,
        max_length=100,
        description="Unique conversation identifier"
    )


class ChatResponse(BaseModel):

    answer: str
    session_id: str
    status: str


# =====================================================
# 4. HEALTH CHECK API
# =====================================================

@app.get("/health")
def health_check(request: Request):

    agent_ready = getattr(
        request.app.state,
        "agent",
        None
    ) is not None

    return {
        "status": "healthy" if agent_ready else "starting",
        "service": "UrbanPulse AI",
        "agent_ready": agent_ready
    }


# =====================================================
# 5. CHAT API
# =====================================================

@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request):

    agent = request.app.state.agent

    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="UrbanPulse AI agent is not ready."
        )

    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid question."
        )

    try:

        # Maintain separate LangGraph conversation threads
        config = {
            "configurable": {
                "thread_id": payload.session_id
            }
        }

        # Send question to the LangGraph agent
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config=config
        )

        answer = result.get(
            "answer",
            "No response generated."
        )

        return ChatResponse(
            answer=answer,
            session_id=payload.session_id,
            status="success"
        )

    except Exception as error:

        print(f"Chat API error: {error}")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question."
        )


# =====================================================
# 6. RUN USING UVICORN
# =====================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "scripts.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
