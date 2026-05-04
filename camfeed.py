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

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    project_location = os.getenv("GOOGLE_CLOUD_LOCATION")
    
    missing_vars =[]
    if not project_id:
        missing_vars.append("GOOGLE_CLOUD_PROJECT")
    if not project_location:
        missing_vars.append("GOOGLE_CLOUD_LOCATION")
        
    if missing_vars:
        error_msg = f"Missing required environment variables in .env file: {', '.join(missing_vars)}"
        logging.error(error_msg)
        raise ValueError(error_msg)
        
    # 2. Construct the image path & verify it exists
    image_path = f"./static/camview/{device_id}.jpg"
    
    if not os.path.exists(image_path):
        error_msg = f"Camera feed for device '{device_id}' could not be found at {image_path}."
        logging.error(error_msg)
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
                        logging.info(f"Successfully loaded metadata for {device_id}")
                        break
            else:
                logging.warning(f"Key '{target_key}' not found in mapping.json")
        except Exception as e:
            logging.error(f"Error reading mapping or metadata: {e}")
    else:
        logging.warning(f"Mapping file not found at {mapping_path}")

    try:
        # 4. Initialize the new genai Client
        client = genai.Client(
            vertexai=True, 
            project=project_id, 
            location=project_location
        )
        
        # 5. Read the local image as bytes
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            
        camera_image = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        )
        
        # 6. Define the strict System Instructions
        # system_instruction = (
        #         "You are a precise smart home AI assistant analyzing camera feeds and metadata. "
        #         "CRITICAL INSTRUCTIONS:\n"
        #         "1. GROUNDING: Answer STRICTLY using the visible image and the provided metadata. No guessing.\n"
        #         "2. ANALYZE: Analyze the image to determine the best response for the user query"
        #         "3. SUMMARIZE: If the feed shows a subject; identify the subject their activity, and the location. Use specific names from the metadata if they match the visual context.\n"
        #         "4. UNCERTAINTY: If the camera feed shows no one, state that. If the requested information is not visible in the image or metadata, state that contexually.'\n"
        #         "5. TONE: Be brief, factual, and direct. Omit conversational filler. Do not mention timestamps unless explicitly asked.\n"
        #         "6. IS USER QUERY ADDRESSED: Set is_user_query_addressed to true if the user's query is to show the camera or can user query can be partially or fully answered by the image/metadata context. Set to false if it cannot be addressed at all."
        # )
        
        system_instruction = """
            <role>
            You are a precise smart home AI assistant analyzing camera feeds and metadata. Your primary function is to give accurate, factual updates based strictly on the provided inputs.
            </role>

            <rules>
            1. STRICT GROUNDING: Answer ONLY using the visible image and the provided metadata. Do not guess, infer off-screen actions, or hallucinate details.
            2. AMBIGUITY: If user asks to show all cameras or show multiple cameras, assume and respond as if the query was for this specific camera.
            3. UNCERTAINTY & ABSENCES: If the camera feed shows no one, explicitly state that. If the user asks for information not visible in the image or metadata, state that it cannot be determined from the current view.
            4. IDENTIFICATION: When a subject is visible, identify them, their current activity, and the location. Use specific names from the metadata ONLY if they logically match the visual context (e.g., recognized faces).
            5. BOOLEAN STRICTNESS: You must accurately evaluate if the visual evidence or metadata resolves the user's core intent.
            - TRUE: The user asks to see a location ("Show the porch"), asks about a visible state ("Is the garage open?"), or asks about a visible subject ("Who is on the couch?").
            - FALSE: The user asks about past events ("Who took the package?"), asks for data not present ("What's the temperature?"), or asks about a subject/object that is entirely out of frame or obscured.
            6. TONE: Be brief, factual, and direct. Omit conversational filler. Do not mention timestamps unless explicitly requested.
            </rules>

            <output_format>
            You must respond in valid JSON format using the following schema:
            {
            "thought_process": "Step 1: State the user's core question. Step 2: Determine if the image/metadata provides the specific information required to answer it. Step 3: Conclude true or false.",
            "summary": "The brief, factual response to the user. (e.g., 'John is sitting on the living room couch reading a book.' or 'I cannot determine who took the package from this view.')",
            "is_user_query_addressed": true // strictly boolean based on Step 3 of your thought process.
            }
            </output_format>  
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
                "response": {"type": "STRING"},
                "is_user_query_addressed": {"type": "BOOLEAN"}
            },
            "required": ["response", "is_user_query_addressed"]
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
        logging.info(f"Sending prompt to Gemini for device '{device_id}'...")
        
        # response = client.models.generate_content(
        #     model="gemini-2.5-flash-lite",
        #     contents=[camera_image, prompt],
        #     config=generation_config
        # )
        
        # -------------------------------------------------------
        # THE FIX: run the blocking SDK call in a thread pool
        # -------------------------------------------------------        
        
        loop = asyncio.get_running_loop()
        sync_fn = functools.partial(
            _generate_content_sync,
            client,
            "gemini-2.5-flash-lite",
            [camera_image, prompt],
            generation_config,
        )
        response = await loop.run_in_executor(None, sync_fn)
        
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
        
        # 13. Return formatted JSON String
        return final_payload
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON returned from Gemini: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while communicating with Google Gen AI: {str(e)}")
        raise e

# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    print("--- Camera Agent Tester ---")
    
    cam_id = input("Enter Camera ID (e.g., cam-1): ").strip()
    user_question = input("Enter your question: ").strip()
    
    if not cam_id or not user_question:
        print("\nError: Both Camera ID and Question are required. Exiting.")
    else:
        try:
            print("\nProcessing... please wait.")
            
            response_json_string = analyze_camera_feed(cam_id, user_question)
            
            print("\n--- Final JSON Payload ---")
            print(response_json_string)
            
        except Exception as ex:
            print(f"\nExecution failed: {ex}")