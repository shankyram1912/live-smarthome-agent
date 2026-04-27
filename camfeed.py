import os
import json
import logging
import asyncio
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
        system_instruction = (
            "You are a precise smart home AI assistant analyzing camera feeds and metadata. "
            "CRITICAL INSTRUCTIONS:\n"
            "1. GROUNDING: Answer STRICTLY using the visible image and the provided metadata. No guessing.\n"
            "2. ANALYZE: Analyze the image to determine the best response for the user query"
            "3. SUMMARIZE: If the feed shows a subject; identify the subject their activity, and the location. Use specific names from the metadata if they match the visual context.\n"
            "4. UNCERTAINTY: If the camera feed shows no one, state that. If the requested information is not visible in the image or metadata, state that contexually.'\n"
            "5. TONE: Be brief, factual, and direct. Omit conversational filler. Do not mention timestamps unless explicitly asked.\n"
            "6. STATUS: Set 'is_user_query_addressed' to true if the user's query can be partially or fully answered by the image/metadata context. Set to false if it cannot be addressed."
        )

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
        
        # 10. Generate content
        logging.info(f"Sending prompt to Gemini for device '{device_id}'...")
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[camera_image, prompt],
            config=generation_config
        )
        
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