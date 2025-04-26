import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any, AsyncGenerator

from coding_task.server.config import settings
from coding_task.inference.predictor import TextClassifierPredictor
from coding_task.inference.config import InferenceConfig
from coding_task.logging_utils import get_logger

logger = get_logger("APIServer", str(settings.log_file))

# Pydantic models
class IntentRequest(BaseModel):
    text: str = Field(..., description="Input sentence for intent classification")

class Prediction(BaseModel):
    label: str = Field(..., description="Intent label name")
    confidence: float = Field(..., description="Probability for the predicted intent")

class IntentResponse(BaseModel):
    intents: List[Prediction] = Field(..., description="An array of top 3 intent prediction results.")

class ErrorResponse(BaseModel):
    label: str
    message: str

#  global app state
app_state: Dict[str, Any] = {"predictor": None, "model_ready": False}

# Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup (model loading) and shutdown."""
    logger.info("Server starting up...")
    try:
        logger.info(f"Attempting to load model from: {settings.model_path}")

        # NOTE: Ccnvert Path objects from Pydantic to strings if predictor expects strings
        inference_config = InferenceConfig(
            model_path=str(settings.model_path),
            device=settings.inference_device,
            log_file=str(settings.log_file)
            # e.g., batch_size=settings.inference_batch_size if settings.inference_batch_size else 8 # use default if not set
        )
        app_state["predictor"] = TextClassifierPredictor(inference_config)
        app_state["model_ready"] = True
        logger.info(f"Model loaded successfully using device: {settings.inference_device}. Server is ready.")
    except Exception as e:
        app_state["model_ready"] = False
        logger.error(f"FATAL: Failed to load model during startup: {e}", exc_info=True)
        # raise the exception to prevent server from starting fully if model load is critical
        raise RuntimeError(f"Failed to load model: {e}")

    yield

    # Shutdown logic
    logger.info("Server shutting down...")
    app_state["predictor"] = None
    app_state["model_ready"] = False
    logger.info("Resources released.")


app = FastAPI(
    title="Intent Classification API",
    description="API service for classifying text intent using a Desriminative PEFT model.",
    version="1.0.0",
    lifespan=lifespan
)

# Exception Handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(label="INTERNAL_ERROR", message=str(exc)).dict(),
    )

@app.get(
    "/ready",
    summary="Check if the service is ready",
    description="Returns OK if the model is loaded and ready to serve requests.",
    responses={
        status.HTTP_200_OK: {"description": "Service is ready", "content": {"text/plain": {"example": "OK"}}},
        status.HTTP_423_LOCKED: {"description": "Service is not ready (model not loaded)", "content": {"text/plain": {"example": "Not ready"}}},
    },
    tags=["Status"] # optional: group endpoints in Swagger UI
)
async def get_ready():
    if app_state.get("model_ready") and app_state.get("predictor") is not None:
        return JSONResponse(content="OK", status_code=status.HTTP_200_OK)
    else:
        return JSONResponse(content="Not ready", status_code=status.HTTP_423_LOCKED)


@app.post(
    "/intent",
    summary="Classify intent",
    description="Responds with intent classification results for the given query utterance.",
    response_model=IntentResponse,
    responses={
        status.HTTP_200_OK: {"description": "Successful classification", "model": IntentResponse},
        status.HTTP_400_BAD_REQUEST: {"description": "Input text is empty", "model": ErrorResponse},
        status.HTTP_423_LOCKED: {"description": "Service is not ready", "content": {"text/plain": {"example": "Not ready"}}},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error", "model": ErrorResponse},
    },
    tags=["Classification"] #opptional: group endpoints in Swagger UI
)
async def classify_intent(request: IntentRequest):
    predictor = app_state.get("predictor")
    model_ready = app_state.get("model_ready")

    if not model_ready or predictor is None:
         return JSONResponse(content="Not ready", status_code=status.HTTP_423_LOCKED)

    if not request.text or request.text.strip() == "":
        logger.warning("Received request with empty text.")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(label="TEXT_EMPTY", message='"text" is empty.').dict(),
        )

    try:
        logger.info(f"Received intent request for text: '{request.text[:100]}...'")
        all_predictions: List[Tuple[str, float]] = predictor.predict(request.text)
        top_3_predictions = all_predictions[:3]
        response_intents = [
            Prediction(label=label, confidence=round(confidence, 4))
            for label, confidence in top_3_predictions
        ]
        logger.info(f"Prediction successful. Top intent: {response_intents[0].label if response_intents else 'N/A'}")
        return IntentResponse(intents=response_intents)

    except Exception as e:
        logger.error(f"Error during prediction for text '{request.text[:100]}...': {e}", exc_info=True)
        raise e # Re-raise to be caught by the general handler


if __name__ == "__main__":
    # settings are loaded when config module is imported
    logger.info(f"Starting Uvicorn server on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        "coding_task.server.server:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=False,
        workers=2,
    )
