import os
from dotenv import load_dotenv
from pathlib import Path
import logging
import asyncio
import base64
import json
import time
import warnings
import uvicorn

from typing import Optional

# Load environment variables first
load_dotenv(override=True)

# =========================================================================
# TELEMETRY ACTIVATION (Must happen BEFORE any google or adk imports!)
# =========================================================================
from gemini_live_telemetry import activate, InstrumentationConfig

activate(InstrumentationConfig(
    project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
    
    # Export settings
    enable_gcp_export=True,                # Push metrics to Cloud Monitoring
    enable_json_export=True,               # Write local JSON file
    enable_dashboard=True,                 # Auto-create Cloud dashboard

    # Timing
    export_interval_s=15.0,               # OTel export interval (min 10s)
    json_flush_interval_s=30.0,            # JSON file flush interval

    # Cloud Monitoring
    metric_prefix="workload.googleapis.com",  # Metric type prefix
    dashboard_name="Gemini Live API Metrics",  # Dashboard display name

    # Audio config (for duration calculations)
    input_sample_rate=16000,               # Input audio sample rate
    output_sample_rate=24000,              # Output audio sample rate    
))
# =========================================================================

# =========================================================================
# Logging configuration so loggers in AgentConfig can be captured
# =========================================================================
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)   
logger = logging.getLogger(__name__)

# Structured CSV Usage Logger Setup
TRACE_FILE = "gemini_usage_trace.log"
CSV_HEADER = "timestamp,user_id,session_id,total_token_count,prompt_token_count,prompt_tokens_details,candidates_token_count,candidates_tokens_details,cached_content_token_count,cache_tokens_details,thoughts_token_count\n"

# OVERWRITE MODE: Unconditionally open with "w" to wipe old data and write a fresh header
with open(TRACE_FILE, "w", encoding="utf-8") as f:
    f.write(CSV_HEADER)

usage_logger = logging.getLogger("gemini_usage_trace")
usage_logger.setLevel(logging.INFO)
usage_logger.propagate = False  

# GUARD: Only attach handlers if they haven't been configured yet
if not usage_logger.handlers:
    # 1. Local CSV File Handler (Leave default mode='a' so entries accumulate DURING this server run)
    file_handler = logging.FileHandler(TRACE_FILE)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    usage_logger.addHandler(file_handler)

    # 2. Google Cloud Logging Handler
    try:
        from google.cloud import logging as cloud_logging
        from google.cloud.logging.handlers import CloudLoggingHandler
        
        cl_client = cloud_logging.Client()
        cl_handler = CloudLoggingHandler(cl_client, name="gemini-live-usage-trace")
        cl_handler.setFormatter(logging.Formatter("%(message)s"))
        usage_logger.addHandler(cl_handler)
        logger.info("Google Cloud Logging handler attached successfully.")
    except ImportError:
        logger.warning("google-cloud-logging package not found. Cloud Logging fallback active.")
        
# Suppress Pydantic serialization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
# =========================================================================

# =========================================================================
# Config imports after Logging Config
# =========================================================================

import config
from config import AgentConfig
from agents import get_aris_agent
from voiceconfig import VoiceConfig

app_name = config.APP_NAME
agent_config = config.agent_config
# =========================================================================

# =========================================================================
# Now it is safe to bring in your Google and ADK Core modules
# =========================================================================
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.genai.types import ProactivityConfig

# Remaining Third-Party App Frameworks (FastAPI, Uvicorn, etc.)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
# =========================================================================

def format_modality_details(details_list) -> str:
    """Flattens and alphabetically sorts a list of ModalityTokenCount objects 
    into a consistent safe CSV column format.
    """
    if not details_list:
        return ""
    items = []
    for item in details_list:
        modality = getattr(item, "modality", "UNKNOWN")
        token_count = getattr(item, "token_count", 0) or 0
        items.append(f"{modality}:{token_count}")
    
    # Alphabetically sort so AUDIO always precedes TEXT in your log columns
    items.sort()
    return "|".join(items)


app = FastAPI(title="Aris: The Smart Home Agent")
session_service = InMemorySessionService()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/live-smarthome-agent/static", StaticFiles(directory=static_dir), name="static")

# Define the headers once to keep things clean
NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}

@app.middleware("http")
async def add_cache_control_header(request: Request, call_next):
    response = await call_next(request)
    # Target only the images inside the static camview directory
    if request.url.path.startswith("/live-smarthome-agent/static/camview/") or request.url.path.endswith(".js"):
        response.headers.update(NO_CACHE_HEADERS)
    return response

# ========================================
# Front End Endpoints
# ========================================

@app.get("/live-smarthome-agent")
async def root():
    """Serve the index.html page."""
    return FileResponse(Path(__file__).parent / "static" / "index.html", headers=NO_CACHE_HEADERS)

@app.get("/live-smarthome-agent/initialize")
async def root():
    """Serve the initialize.html page."""
    return FileResponse(Path(__file__).parent / "static" / "initialize.html", headers=NO_CACHE_HEADERS)

@app.get("/live-smarthome-agent/camview")
async def root():
    """Serve the index.html page."""
    return FileResponse(Path(__file__).parent / "static" / "camview.html", headers=NO_CACHE_HEADERS)



# ========================================
# WebSocket Endpoint
# ========================================

@app.websocket("/live-smarthome-agent/ws/{user_id}/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    session_id: str,
    voice: Optional[str] = "aoede",              # Defaults to 'aoede'
    affective_dialog: Optional[bool] = False,    # Auto-converts "true" to True
    proactive_audio: Optional[bool] = False      # Auto-converts "false" to False    
) -> None:    
    
    logger.info(
        f"WebSocket connection request: user_id={user_id}, session_id={session_id}"
    )    
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    logger.info(f"Settings - Voice: {voice} / Gender: {VoiceConfig.get_gender(voice).name}, Affective: {affective_dialog}, Proactive: {proactive_audio}")

    # Fetch Agent Dynamically (Run in Threadpool so we don't block the event loop)
    try:
        agent = await run_in_threadpool(get_aris_agent, VoiceConfig.is_female(voice))
        logger.info(f"Successfully loaded agent profile for ARIS")
    except Exception as e:
        logger.error(f"Failed to load agent ARIS: {e}")
        await websocket.close(code=1008, reason=f"Agent load failed: {str(e)}")
        return

    # Initialize a localized Runner for this specific connection
    runner = Runner(app_name=app_name, agent=agent, session_service=session_service)

    # Get or create session
    session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    if not session:
        await session_service.create_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )

    # ========================================
    # Phase 2: Session Initialization 
    # ========================================

    response_modalities = ["AUDIO"]
        
    # 1. Define the base configuration arguments that apply to both Gemini and Vertex AI
    run_config_kwargs = {
        "streaming_mode": StreamingMode.BIDI,
        "response_modalities": response_modalities,
        "speech_config": types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice
                )
            )
        ),            
        "input_audio_transcription": types.AudioTranscriptionConfig(language_codes=['en-US', 'th-TH']),
        "output_audio_transcription": types.AudioTranscriptionConfig(language_codes=['en-US', 'th-TH']),
    }

    # 2. Conditionally inject features exclusive to the Vertex AI Live API
    if agent_config.IS_VERTEX_AI_LIVE_API:
        run_config_kwargs["session_resumption"] = types.SessionResumptionConfig()
        run_config_kwargs["proactivity"] = ProactivityConfig(proactive_audio=proactive_audio)
        run_config_kwargs["enable_affective_dialog"] = affective_dialog

    # 3. Instantiate the RunConfig by unpacking the dictionary
    run_config = RunConfig(**run_config_kwargs)

    live_request_queue = LiveRequestQueue()
    
    # Fix to optimize voice delivery
    opening_system_direction = types.Content(
        parts=[types.Part(text="System Instruction: Speak SLOWLY, use casual warm tone but SLOWLy. Dont respond to this message, wait for user input.")]
    )    
    live_request_queue.send_content(opening_system_direction)

    # ========================================
    # Phase 3: Active Session Tasks
    # ========================================

    async def upstream_task() -> None:
        """Receives messages from WebSocket and sends to LiveRequestQueue."""
        logger.info("upstream_task started")
        while True:
            try:
                message = await websocket.receive()

                # Cleanly break the loop if the frontend sends a disconnect signal
                if message.get("type") == "websocket.disconnect":
                    logger.info("Frontend explicitly closed the connection. Stopping upstream task.")
                    live_request_queue.close()
                    break

                if "bytes" in message:
                    audio_blob = types.Blob(
                        mime_type="audio/pcm;rate=16000", data=message["bytes"]
                    )
                    logger.debug("Frontend sent AUDIO.")
                    live_request_queue.send_realtime(audio_blob)

                elif "text" in message:
                    json_message = json.loads(message["text"])
                    
                    logger.info(f"Frontend sent TEXT - {json_message}")

                    if json_message.get("type") == "text":
                        content = types.Content(
                            parts=[types.Part(text=json_message["text"])]
                        )
                        live_request_queue.send_content(content)

                    elif json_message.get("type") == "image":
                        logger.info(f"Frontend sent IMAGE")
                        image_data = base64.b64decode(json_message["data"])
                        mime_type = json_message.get("mimeType", "image/jpeg")
                        image_blob = types.Blob(
                            mime_type=mime_type, data=image_data
                        )
                        live_request_queue.send_realtime(image_blob)
            except RuntimeError as e:
                if "disconnect message" in str(e):
                    logger.info("Caught disconnect RuntimeError, stopping upstream task.")
                    break
                logger.error(f"Unexpected RuntimeError in upstream_task: {e}")
                break
            
            except WebSocketDisconnect:
                logger.info("WebSocket disconnect exception caught.")
                break                        

    async def downstream_task() -> None:
        """Receives Events from run_live() and sends to WebSocket."""
        logger.info("downstream_task started")
        
        async for event in runner.run_live(
            user_id=user_id,
            session_id=session_id,
            live_request_queue=live_request_queue,
            run_config=run_config,
        ):

            event_json = event.model_dump_json(exclude_none=True, by_alias=True)

            event_dict = json.loads(event_json)
            
            event_type = None
            event_summary = None
            is_audio_stream = False
            
            if event.content and event.content.parts:
                part = event.content.parts[0]
                
                if part.inline_data:
                    event_summary = f"AUDIO {part.inline_data.mime_type} Received {len(part.inline_data.data)} bytes"
                elif part.text:
                    event_summary = f"TEXT {part.text} IS_PARTIAL {event.partial} TURN_COMPLETE {event.turn_complete}"
                for part in event.content.parts:
                    if part.function_call:
                        event_type = "function_call"
                        event_summary = f"MODEL FUNCTION CALL {part.function_call.name} INPUT PARAMS {part.function_call.args}"
                    elif part.function_response:
                        event_type = "function_response"
                        event_summary = f"USER FUNCTION CALL RESPONSE {part.function_response.name} OUTPUT PARAMS {part.function_response.response}"                        
                    
            if event.input_transcription:
                event_summary = f"🗣️ USER TALKING: {event.input_transcription.text} IS_FINISHED {event.input_transcription.finished} IS_PARTIAL {event.partial} TURN_COMPLETE {event.turn_complete}"                        
            elif event.output_transcription:
                event_summary = f"🤖 AI AGENT TALKING: {event.output_transcription.text} IS_FINISHED {event.output_transcription.finished} IS_PARTIAL {event.partial} TURN_COMPLETE {event.turn_complete}"                        
                
            # Uncomment for event logging
            if event_summary:
               print(f"++ {event_summary}", flush=True)
            else:
                print(f"xx UNTAGGED EVENT {event_dict}", flush=True)
            
            if event.input_transcription and event.input_transcription.finished:
                print("\n" + "-"*50)
                print(f"🗣️ USER FINISHED: {event.input_transcription.text}")
                print("-" *50 + "\n", flush=True)                        
            elif event.output_transcription and event.output_transcription.finished:
                print("\n" + "="*50)
                print(f"🤖 AI AGENT FINISHED: {event.output_transcription.text}")
                print("="*50 + "\n", flush=True)
                
            # =========================================================================
            # NEW: CSV Metrics Interception & Extract Logic
            # =========================================================================
            if hasattr(event, "usage_metadata") and event.usage_metadata:
                usage = event.usage_metadata
                    
                # 1. Clean details fields from internal lists
                p_details = format_modality_details(getattr(usage, "prompt_tokens_details", None))
                c_details = format_modality_details(getattr(usage, "cache_tokens_details", None))
                cand_details = format_modality_details(getattr(usage, "candidates_tokens_details", None))
                
                timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # 2. Extract values with fallback to avoid explicit None values from breaking formatting
                total_tokens = getattr(usage, 'total_token_count', 0) or 0
                prompt_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                cand_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                cached_tokens = getattr(usage, 'cached_content_token_count', 0) or 0
                thoughts_tokens = getattr(usage, 'thoughts_token_count', 0) or 0

                # 3. Spreadsheet-ready CSV Row for the actual background trace log file 
                csv_fields = [
                    str(timestamp_str), str(user_id), str(session_id),
                    str(total_tokens), str(prompt_tokens), f'"{p_details}"',
                    str(cand_tokens), f'"{cand_details}"', str(cached_tokens),
                    f'"{c_details}"', str(thoughts_tokens)
                ]
                csv_row = ",".join(csv_fields)
                
                usage_logger.info(f"usage_metadata {csv_row}")

                # 4. Human-readable visualization layout (Implicit multi-line string concatenation)
                # NOTE: Keep the commas strictly INSIDE the quotes to avoid creating a tuple.
                csv_row_display = (
                    f"{timestamp_str},{user_id},{session_id},"
                    f" total_token_count {total_tokens},"
                    f" prompt_token_count {prompt_tokens}; {p_details},"
                    f" candidates_token_count {cand_tokens}; {cand_details},"
                    f" cached_content_token_count {cached_tokens}; {c_details},"
                    f" thoughts_token_count {thoughts_tokens}"
                )                
                                
                logger.info(csv_row_display)
            # =========================================================================                                 
            
            # Always forward the raw event to the frontend (for audio), everything else is JSON
            if event.content and event.content.parts:
                part = event.content.parts[0]
                if part.inline_data:                                                
                    if hasattr(part, 'inline_data') and part.inline_data:
                        if hasattr(part.inline_data, 'data') and part.inline_data.data:
                            logger.debug(f"### SENDING AUDIO RESPONSE TO FRONTEND")                                
                            await websocket.send_bytes(part.inline_data.data)
                else:
                    if(event_type in ("function_call", "function_response")):                
                        logger.info(f"### RESPONSE TO FRONTEND - {event_json}")
                        if(event_type in ("function_call")):
                            usage_logger.info(f"function_call {len(json.dumps(event_json))}")
                        else:
                            usage_logger.info(f"function_response {len(json.dumps(event_json))}")
                    await websocket.send_text(event_json)                    
            else:                
                # logger.info(f"### RESPONSE TO FRONTEND - {event_json}")
                await websocket.send_text(event_json)

    # ========================================
    # Run the Concurrent Tasks
    # ========================================
    try:
        logger.info("Starting asyncio.gather for tasks")
        await asyncio.gather(
            upstream_task(), 
            downstream_task(),
        )
    except WebSocketDisconnect:
        logger.info("Client disconnected normally")
    except asyncio.CancelledError:
        logger.info("Server shutting down. Cancelling active WebSocket tasks...")        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        logger.info("Closing live_request_queue")
        live_request_queue.close()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)