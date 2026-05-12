import os
import json
import logging
import asyncio
import functools
from dotenv import load_dotenv

# Import the unified Google Gen AI SDK
from google import genai
from google.genai import types

# Import the camera simulation metadata
from camsim import CAM_SIM

from config import agent_config

logger = logging.getLogger(__name__)

# Load environment variables from the .env file
load_dotenv()

def _generate_content_sync(client, model, contents, config):
    """Pure synchronous wrapper — safe to hand to run_in_executor."""
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

async def analyze_camera_feed(device_id: str, user_query: str) -> str:
    """
    Analyzes a camera feed image using Gemini Flash and answers a user query in JSON format.
    Strictly reads configuration from environment variables and injects camera metadata.
    
    Args:
        device_id (str): The ID of the camera (e.g., 'cam-1').
        user_query (str): The question the user is asking about the feed.
        
    Returns:
        str: A JSON formatted string containing the combined metadata and LLM response.
        
    Raises:
        ValueError: If required environment variables are not set.
        FileNotFoundError: If the specified camera image does not exist.
    """
    
    # 1. Strictly fetch from environment variables
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    
    missing_vars =[]
    if not project_id:
        missing_vars.append("GOOGLE_CLOUD_PROJECT")
        
    if missing_vars:
        error_msg = f"Missing required environment variables in .env file: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    # 2. Construct the image path & verify it exists
    image_path = f"./static/camview/{device_id}.jpg"
    
    if not os.path.exists(image_path):
        error_msg = f"Camera feed for device '{device_id}' could not be found at {image_path}."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # 3. Retrieve Metadata from mapping.json and CAM_SIM
    mapping_path = "./static/camview/mapping.json"
    metadata_json_str = "{}"
    llm_metadata = {} # Store metadata securely so we can use it for the final JSON payload
    
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
            
            target_key = f"{device_id}.jpg"
            original_filename = mapping.get(target_key)
            
            if original_filename:
                for sim in CAM_SIM:
                    if sim.get("filename") == original_filename:
                        llm_metadata = sim.copy()
                        # Remove the filename before passing to LLM so it doesn't get confused
                        llm_metadata.pop("filename", None)
                        metadata_json_str = json.dumps(llm_metadata, indent=2)
                        logger.info(f"Successfully loaded metadata for {device_id}")
                        break
            else:
                logger.warning(f"Key '{target_key}' not found in mapping.json")
        except Exception as e:
            logger.error(f"Error reading mapping or metadata: {e}")
    else:
        logger.warning(f"Mapping file not found at {mapping_path}")

    try:
        # 4. Initialize the new genai Client
        client = genai.Client(
            vertexai=True, 
            project=project_id, 
            location=agent_config.SUBAGENT_CLOUD_LOCATION
        )
        
        # 5. Read the local image as bytes
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            
        camera_image = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        )
        
        # system_instruction = """
        #     <role>
        #     You are a precise smart home AI assistant analyzing camera feeds and metadata. Your primary function is to give accurate, factual updates based strictly on the provided inputs.
        #     </role>

        #     <rules>
        #     1. STRICT GROUNDING: Answer ONLY using the visible image and the provided metadata. Do not guess, infer off-screen actions, or hallucinate details.
        #     2. AMBIGUITY: If user asks to show all cameras or show multiple cameras, assume and respond as if the query was for this specific camera.
        #     3. UNCERTAINTY & ABSENCES: If the camera feed shows no one, explicitly state that. If the user asks for information not visible in the image or metadata, state that it cannot be determined from the current view.
        #     4. IDENTIFICATION: When a subject is visible, identify them, their current activity, and the location. Use specific names from the metadata ONLY if they logically match the visual context (e.g., recognized faces).
        #     5. BOOLEAN STRICTNESS: You must accurately evaluate if the visual evidence or metadata fulfills the user's core request.
        #     - TRUE: The user issues a command to view the feed ("Show the porch", "Show cameras"), asks about a visible state ("Is the garage open?"), or asks about a visible subject ("Who is on the couch?"). Commands to show the camera are ALWAYS considered fulfilled (TRUE).
        #     - FALSE: The user asks about past events ("Who took the package?"), asks for data not present ("What's the temperature?"), or asks about a subject/object that is entirely out of frame or obscured.
        #     6. TONE: Be brief, factual, and direct. Omit conversational filler. Do not mention timestamps unless explicitly requested.
        #     </rules>

        #     <output_format>
        #     You must respond in valid JSON format using the following schema:
        #     {
        #     "thought_process": "Step 1: State the user's core request or intent. Step 2: Determine if the image/metadata successfully fulfills this request or provides the requested information. Step 3: Conclude true or false.",
        #     "response": "The brief, factual response to the user without your internal thinking. (e.g., 'Here is the current view of the porch.' or 'John is sitting on the couch.')",
        #     "is_user_query_addressed": true // strictly boolean based on Step 3 of your thought process.
        #     }
        #     </output_format>
        # """
        
        # """
        #         <output_format>
        #             <instructions>You must respond in valid JSON format using the following schema. Do not output markdown codeblocks around the JSON.</instructions>
        #             <schema>
        #                 {
        #                 "thought_process": "Step 1: Analyze the user's core request or intent. Step 2: Determine if the image/metadata successfully fulfills this request or provides the requested information. Follow the rules in describe_scene_rules strictly.",
        #                 "response": "The factual response to the user without your internal thinking. Follow the rules in describe_scene_rules strictly.
        #                 "is_user_query_addressed": true|false. Follow rules in is_user_query_addressed
        #                 }
        #             </schema>
        #         </output_format>        
        # """        
        
        system_instruction = """
            <system_prompt>
                <role>
                    You are a precise smart home AI assistant analyzing camera feeds and metadata. Your primary function is to give accurate, factual updates based strictly on the provided inputs.
                </role>

                <response_rules>
                    - Always identify in your responses - the scene and subjects describing them based on the image (also use metadata where available), describing their activity, and the location. Use the label from the metadata best matching the user query and name if available.
                    - When subjects are visible, identify them when possible, their current activity, and the location. Use specific names from the metadata ONLY if they logically match the visual context (e.g., recognized faces).
                    - If the camera feed shows no one, explicitly state that. If the user asks for information not visible in the image or metadata, state that it cannot be determined from the current view.
                    - When 'control_checkcamera' returns an image and metadata, use BOTH to describe the scene naturally. Example: "Grandmother and a delivery driver are at the front door."
                    - Answer ONLY using the visible image and the provided metadata. Do not guess, infer off-screen actions, or hallucinate details.
                    - If user asks to show all cameras or show multiple cameras, assume and respond as if the query was for this specific camera.               
                </describe_scene_rules>

                <response_rules>
                    - You must accurately evaluate if the visual evidence or metadata fulfills the user's core request.
                    - Set TRUE if:
                        a) The user issues a command to view the feed (e.g., "Show the porch", "Show cameras"). Commands to show the camera are ALWAYS considered TRUE.
                        b) The user asks about a visible state (e.g., "Is the garage open?").
                        c) The user asks about a visible subject (e.g., "Who is on the couch?").
                    - Set FALSE if:
                        a) The user asks about past events (e.g., "Who took the package?").
                        b) The user asks for data not present (e.g., "What's the temperature?").
                        c) The user asks about a subject/object that is entirely out of frame or obscured.
                </response_rules>
            </system_prompt>
        """        

        # 7. Keep the Prompt clean (just dynamic data)
        prompt = (
            f"CAMERA METADATA:\n{metadata_json_str}\n\n"
            f"USER QUESTION: \"{user_query}\""
        )

        # 8. Define the Expected JSON Output Schema for the LLM
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "thought_process": {"type": "STRING"},
                "response": {"type": "STRING"},
                "is_user_query_addressed": {"type": "BOOLEAN"}
            },
            "required": ["thought_process", "response", "is_user_query_addressed"]
        }

        # 9. Configure Generation (Enforcing JSON Output)
        generation_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json", # Forces JSON Response
            response_schema=response_schema,       # Enforces schema structure
            temperature=0.2, 
            max_output_tokens=256 
        )
        
        # 10. Generate content (non-blocking — runs SDK call in thread pool)
        logger.info(f"Sending prompt to Gemini for device '{device_id}'...")      
        
        loop = asyncio.get_running_loop()
        sync_fn = functools.partial(
            _generate_content_sync,
            client,
            agent_config.SUBAGENT_MODEL,
            [camera_image, prompt],
            generation_config,
        )
        response = await loop.run_in_executor(None, sync_fn)
        
        logger.info(f"ANALYZE CAMERA FEED - {device_id} \n LLM Response - {response}\n")
        
        # 11. Parse the LLM's JSON Response
        llm_response_dict = json.loads(response.text)        
        
        # 12. Combine with System Metadata
        final_payload = {
            "id": llm_metadata.get("id", device_id),
            "room": llm_metadata.get("room", "Unknown"),
            "timestamp": llm_metadata.get("timestamp", ""),
            "has_unidentified_face": llm_metadata.get("has_unidentified_face", False),
            "identified_faces": llm_metadata.get("identified_faces",[]),
            "response": llm_response_dict.get("response", ""),
            "is_user_query_addressed": llm_response_dict.get("is_user_query_addressed", False)
        }
        
        # 13. Return
        return final_payload
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON returned from Gemini: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while communicating with Google Gen AI: {str(e)}")
        raise e

# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    
    # Make sure asyncio is imported at the top of your file!
    import asyncio 
    import logging    
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    print("--- Camera Agent Tester ---")
    
    cam_id = input("Enter Camera ID (e.g., cam-1): ").strip()
    user_question = input("Enter your question: ").strip()
    
    if not cam_id or not user_question:
        print("\nError: Both Camera ID and Question are required. Exiting.")
    else:
        try:
            print("\nProcessing... please wait.")
            
            # Use asyncio.run() to properly await the coroutine
            response_json_dict = asyncio.run(analyze_camera_feed(cam_id, user_question))
            
            print("\n--- Final JSON Payload ---")
            # The function returns a dictionary, so we should convert it to a formatted string
            print(json.dumps(response_json_dict, indent=2))
            
        except Exception as ex:
            print(f"\nExecution failed: {ex}")