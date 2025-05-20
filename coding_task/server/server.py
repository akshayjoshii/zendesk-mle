import uvicorn
import asyncio
import redis # Added
import json # Added
import uuid # Added
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any, AsyncGenerator, Optional

from coding_task.server.config import settings
from coding_task.inference.predictor import TextClassifierPredictor
from coding_task.inference.config import InferenceConfig
from coding_task.logging_utils import get_logger

logger = get_logger("APIServer", str(settings.log_file))

# Pydantic models
class IntentRequest(BaseModel):
    text: str = Field(..., description="Input sentence for intent classification")

# QueueItem is removed

class Prediction(BaseModel):
    label: str = Field(..., description="Intent label name")
    confidence: float = Field(..., description="Probability for the predicted intent")

class IntentResponse(BaseModel):
    intents: List[Prediction] = Field(..., description="An array of top 3 intent prediction results.")

class ErrorResponse(BaseModel):
    label: str
    message: str

#  global app state
app_state: Dict[str, Any] = {
    "predictor": None,
    "model_ready": False,
    "redis_client": None, # For Redis client
    "redis_worker_task": None # For the Redis worker task
}

# Redis Worker Function
async def redis_worker_main(app_state: Dict[str, Any]):
    """Continuously processes requests from the Redis queue."""
    logger.info("Redis worker started.")
    redis_client = app_state.get("redis_client")
    predictor = app_state.get("predictor")

    if not redis_client:
        logger.error("Redis worker: Redis client not available. Worker cannot start.")
        return
    if not predictor:
        logger.error("Redis worker: Predictor not available. Worker cannot start.")
        return

    while True:
        try:
            # Get a task from Redis
            logger.info(f"Worker waiting for task on queue: {settings.redis_queue_name}")
            # BRPOP returns a tuple (list_name, item_json_payload)
            # decode_responses=True in Redis client means item_json_payload is already a string.
            task_data = await asyncio.to_thread(
                redis_client.brpop, settings.redis_queue_name
            ) 

            if task_data is None: # Should not happen with brpop without timeout
                logger.debug("Worker: BRPOP returned None, continuing.")
                continue

            _queue_name, json_payload = task_data
            
            try:
                payload = json.loads(json_payload)
                request_id = payload["request_id"] # Corrected key
                text_to_predict = payload["text"]
                result_channel = f"{settings.redis_result_channel_prefix}{request_id}"
                logger.info(f"Worker processing request_id: {request_id} for text: '{text_to_predict[:50]}...'")
            except json.JSONDecodeError as e:
                logger.error(f"Worker: Failed to decode JSON payload: {json_payload}. Error: {e}")
                continue # Skip this item
            except KeyError as e:
                logger.error(f"Worker: Missing key in JSON payload: {e}. Payload: {json_payload}")
                continue # Skip this item

            # Perform Prediction
            worker_response = {} # Define before try block
            try:
                if not app_state.get("predictor") or not app_state.get("model_ready"):
                    # This check is important in case the model becomes unavailable during runtime
                    logger.error(f"Worker: Predictor not available or model not ready for request_id {request_id}.")
                    raise Exception("Model predictor is not available or not ready.")
                
                # predictor.predict is synchronous, run in thread
                prediction_list: List[Tuple[str, float]] = await asyncio.to_thread(
                    app_state["predictor"].predict, text_to_predict
                )
                
                worker_response = {
                    "request_id": request_id,
                    "data": prediction_list 
                }
            except Exception as e:
                logger.error(f"Worker: Error during prediction for request_id {request_id}: {e}", exc_info=True)
                worker_response = {
                    "request_id": request_id,
                    "error": str(e),
                    "detail": "Error occurred during prediction processing in worker." # Optional detail
                }
            
            # Publish Result to Redis Pub/Sub
            try:
                json_worker_response = json.dumps(worker_response)
                redis_client.publish(result_channel, json_worker_response)
                logger.info(f"Worker published result for request_id {request_id} to channel {result_channel}")
            except redis.exceptions.RedisError as e:
                logger.error(f"Worker: RedisError publishing result for request_id {request_id} to {result_channel}: {e}", exc_info=True)
            except Exception as e: # Catch other potential errors during publish (e.g., json.dumps if result is complex)
                logger.error(f"Worker: Unexpected error publishing result for {request_id}: {e}", exc_info=True)

        except redis.exceptions.RedisError as e: # Errors from brpop itself
            logger.error(f"Worker: RedisError during BRPOP or connection issue: {e}. Retrying after a short delay.", exc_info=True)
            await asyncio.sleep(5) 
            # Re-check client connection, or rely on Redis client's auto-reconnect if configured
            if not redis_client.ping(): # Basic check if connection is alive
                logger.error("Worker: Redis connection lost. Attempting to re-establish or waiting for reconnect.")
                # Potentially implement more robust reconnection logic or rely on client's features
            continue
        except asyncio.CancelledError:
            logger.info("Redis worker stopping due to cancellation.")
            break # Exit the loop
        except Exception as e: # Catch any other unexpected error in the main worker loop
            logger.error(f"Worker: Unexpected error in main loop: {e}", exc_info=True)
            await asyncio.sleep(1) # Brief pause before trying to get next task
            continue
    logger.info("Redis worker stopped.")


# Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup (model loading, Redis connection) and shutdown."""
    logger.info("Server starting up...")
    # Initialize Model
    try:
        logger.info(f"Attempting to load model from: {settings.model_path}")
        inference_config = InferenceConfig(
            model_path=str(settings.model_path),
            device=settings.inference_device,
            log_file=str(settings.log_file)
        )
        app_state["predictor"] = TextClassifierPredictor(inference_config)
        app_state["model_ready"] = True
        logger.info(f"Model loaded successfully using device: {settings.inference_device}.")
    except Exception as e:
        app_state["model_ready"] = False
        logger.error(f"FATAL: Failed to load model during startup: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load model: {e}") # Model is critical

    # Initialize Redis Client
    try:
        logger.info(f"Connecting to Redis on {settings.redis_host}:{settings.redis_port}")
        app_state["redis_client"] = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True
        )
        app_state["redis_client"].ping() # Verify connection
        logger.info("Successfully connected to Redis.")
        
        # Start the Redis worker task if Redis connection is successful and model is ready
        if app_state["model_ready"] and app_state["redis_client"]:
            app_state["redis_worker_task"] = asyncio.create_task(redis_worker_main(app_state))
            logger.info("Redis request worker task started.")
        else:
            logger.warning("Redis worker task not started due to model or Redis not being ready.")

    except redis.exceptions.ConnectionError as e:
        logger.error(f"FATAL: Could not connect to Redis: {e}", exc_info=True)
        app_state["redis_client"] = None
        logger.warning("Server will start without Redis connection. Endpoints relying on Redis will fail, and worker will not start.")
    except Exception as e: # Catch other startup errors (e.g. if model loading failed before this)
        logger.error(f"FATAL: An error occurred during server startup sequence: {e}", exc_info=True)
        # Ensure critical resources are cleaned up if part of startup failed
        if app_state.get("redis_client"):
            app_state["redis_client"].close()
            app_state["redis_client"] = None
        app_state["model_ready"] = False
        app_state["predictor"] = None
        raise # Re-raise the exception to prevent server from starting in a bad state


    logger.info("Server is ready.")
    yield

    # Shutdown logic
    logger.info("Server shutting down...")

    # Cancel and cleanup Redis worker task
    redis_worker_task = app_state.get("redis_worker_task")
    if redis_worker_task:
        logger.info("Cancelling Redis request worker task...")
        redis_worker_task.cancel()
        try:
            await redis_worker_task
            logger.info("Redis request worker task successfully cancelled.")
        except asyncio.CancelledError:
            logger.info("Redis request worker task was cancelled as expected during shutdown.")
        except Exception as e:
            logger.error(f"Error during Redis worker task cancellation: {e}", exc_info=True)
    
    # Close Redis connection
    if app_state.get("redis_client"):
        try:
            app_state["redis_client"].close()
            logger.info("Redis client connection closed.")
        except Exception as e: 
            logger.error(f"Error closing Redis connection: {e}", exc_info=True)
    
    app_state["predictor"] = None
    app_state["model_ready"] = False
    app_state["redis_client"] = None
    app_state["redis_worker_task"] = None # Clear task from app_state
    logger.info("Resources released and server shutdown complete.")


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
    redis_client = app_state.get("redis_client")

    if not model_ready or predictor is None:
         logger.warning("Model not ready or predictor not available.")
         return JSONResponse(
            status_code=status.HTTP_423_LOCKED, # Locked
            content=ErrorResponse(label="NOT_READY", message="Service is not ready (model not loaded).").dict()
        )

    if redis_client is None:
        logger.error("Redis client not available. Cannot process request.")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, # Service Unavailable
            content=ErrorResponse(label="REDIS_UNAVAILABLE", message="Service temporarily unavailable due to Redis connection issue.").dict()
        )

    if not request.text or request.text.strip() == "":
        logger.warning("Received request with empty text.")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(label="TEXT_EMPTY", message='"text" is empty.').dict(),
        )

    request_id = str(uuid.uuid4())
    pubsub = None # Initialize pubsub to None for finally block

    try:
        payload = {
            "request_id": request_id,
            "text": request.text
        }
        json_payload = json.dumps(payload)

        try:
            logger.debug(f"Attempting to LPUSH request {request_id} to queue '{settings.redis_queue_name}'.")
            redis_client.lpush(settings.redis_queue_name, json_payload)
            logger.info(f"Request {request_id} for text '{request.text[:50]}...' enqueued.")
        except redis.exceptions.RedisError as e:
            logger.error(f"RedisError while enqueuing request {request_id}: {e}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(label="QUEUE_ERROR", message="Failed to enqueue request.").dict(),
            )

        # Implement Pub/Sub for waiting for the result
        pubsub = redis_client.pubsub(ignore_subscribe_messages=True) # ignore initial subscribe messages
        result_channel = f"{settings.redis_result_channel_prefix}{request_id}"
        
        logger.info(f"Subscribing to result channel: {result_channel} for request {request_id}")
        await asyncio.to_thread(pubsub.subscribe, result_channel) # Use to_thread for blocking subscribe

        prediction_result_json = None
        logger.info(f"Waiting for result for request {request_id} on {result_channel} (timeout: {settings.redis_request_timeout}s)")
        
        # Async-friendly loop for pubsub.listen() or get_message()
        # Using get_message with timeout in a loop is generally safer.
        # For this step, we'll use get_message within an asyncio.to_thread wrapper.
        # Note: redis-py's pubsub.listen() is a generator that blocks.
        # pubsub.get_message() is non-blocking or has a timeout.
        try:
            # This loop structure is for get_message, which is more explicit about timeouts.
            # If listen() was used, the timeout handling would be different (e.g., select or background task).
            # We are using a timeout in get_message itself.
            message = await asyncio.to_thread(pubsub.get_message, timeout=settings.redis_request_timeout)
            
            if message and message["type"] == "message":
                logger.info(f"Received result for {request_id} on channel {message['channel']}")
                prediction_result_json = message["data"]
            elif message: # e.g. subscribe confirmation if ignore_subscribe_messages=False
                 logger.debug(f"Received non-data message for {request_id}: {message}")
            # If message is None, it means timeout occurred with get_message

        except redis.exceptions.RedisError as e: # Catch Redis specific errors during get_message
            logger.error(f"RedisError while waiting for result for request {request_id}: {e}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(label="QUEUE_RESULT_ERROR", message="Error waiting for processing result from queue.").dict(),
            )
        # No specific asyncio.TimeoutError to catch here if get_message itself times out and returns None.
        # Timeout is handled by checking if prediction_result_json is None.

        if prediction_result_json is None:
            logger.warning(f"No result received for request {request_id} on {result_channel} within timeout ({settings.redis_request_timeout}s).")
            return JSONResponse(
                status_code=status.HTTP_408_REQUEST_TIMEOUT, # Request Timeout
                content=ErrorResponse(label="TIMEOUT", message="Request timed out waiting for processing result.").dict(),
            )

        # Deserialize the result
        logger.debug(f"Raw prediction result for {request_id}: {prediction_result_json}")
        prediction_output = json.loads(prediction_result_json) # Result from worker

        if "error" in prediction_output:
            error_msg = prediction_output.get("error", "Unknown worker error")
            error_detail = prediction_output.get("detail", "") # Optional: more detail from worker
            logger.error(f"Worker reported error for request {request_id}: {error_msg} - Detail: {error_detail}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # Or a more specific error if applicable
                content=ErrorResponse(label="WORKER_ERROR", message=error_msg).dict(),
            )

        # Process the successful result (prediction_output should contain "data" key)
        # Assuming worker sends: {"request_id": "...", "data": List[Tuple[str, float]]} or {"request_id": "...", "error": "..."}
        all_predictions: List[Tuple[str, float]] = prediction_output.get("data", [])
        
        top_3_predictions = all_predictions[:3]
        response_intents = [
            Prediction(label=label, confidence=round(float(confidence), 4)) # Ensure confidence is float
            for label, confidence in top_3_predictions
        ]
        logger.info(f"Prediction successful for {request_id}. Top intent: {response_intents[0].label if response_intents else 'N/A'}")
        return IntentResponse(intents=response_intents)

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError for request {request_id} when parsing result: {e}. Data: '{prediction_result_json}'", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(label="RESULT_PARSE_ERROR", message="Error parsing processing result.").dict(),
        )
    except Exception as e:
        # General catch-all for unexpected errors in this endpoint
        logger.error(f"Unexpected error in classify_intent for request {request_id}: {e}", exc_info=True)
        # Re-raise to be caught by the general_exception_handler for a generic 500 response
        # or return a specific JSONResponse
        raise e # This will be caught by the app's general exception handler
    finally:
        if pubsub:
            try:
                logger.debug(f"Unsubscribing and closing pubsub for request {request_id} from channel {result_channel}")
                # Unsubscribe can also be blocking, ensure it's handled if this becomes an issue
                await asyncio.to_thread(pubsub.unsubscribe, result_channel)
                await asyncio.to_thread(pubsub.close)
            except redis.exceptions.RedisError as e:
                logger.warning(f"RedisError during pubsub cleanup for request {request_id}: {e}", exc_info=True)
            except Exception as e:
                 logger.warning(f"Generic error during pubsub cleanup for request {request_id}: {e}", exc_info=True)


if __name__ == "__main__":
    # settings are loaded when config module is imported
    logger.info(f"Starting Uvicorn server on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        "coding_task.server.server:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=False,
        workers=1,
    )
