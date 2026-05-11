import os
from dotenv import load_dotenv

import logging

logger = logging.getLogger(__name__)

# Load environment variables early
load_dotenv(override=True)

class AgentConfig:
    """
    Manages environment-based routing between Gemini AI Studio 
    and Vertex AI models dynamically for the Live Translator.
    """
    def __init__(self):
        if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() == "false":
            
            logger.info("Need to route to GEMINI API (Google AI Studio).")
            
            # # 1. Patch ADK to use v1beta for Gemini API live connections for 3.1 Flash.
            # # ADK (as of 1.32.0) still defaults `_live_api_version` to "v1alpha" for AI Studio, 
            # # but `gemini-3.1-flash-live-preview` is only served on v1beta.
            # from google.adk.models.google_llm import Gemini
            # from google import genai
            
            # # Patch ADK to use v1beta for Gemini API live connections for 3.1 Flash.
            # Gemini._live_api_version = "v1beta"
            
            self.ORCHESTRATOR_MODEL = os.getenv(
                "LIVEAGENT_GEMINI_MODEL", 
                "gemini-3.1-flash-live-preview"
            )            
            
            self.IS_VERTEX_AI_LIVE_API = False
            
        else:
            logger.info("Need to route to VERTEX AI API.")
            
            self.ORCHESTRATOR_MODEL = os.getenv(
                "LIVEAGENT_VERTEXAI_MODEL", 
                "gemini-live-2.5-flash-native-audio"
            )
            
            self.IS_VERTEX_AI_LIVE_API = True
            
        self.SUBAGENT_MODEL = os.getenv(
            "SUBAGENT_LITE_MODEL", 
            "gemini-3.1-flash-lite"
        )        

# App Configuration
APP_NAME = "aris_smarthome_agent"

# Initialize a singleton configuration object for importing across the app
agent_config = AgentConfig()