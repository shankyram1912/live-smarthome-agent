import os
import json
import logging
import config
from dotenv import load_dotenv

# NEW: Import the unified Google Gen AI SDK
from google import genai
from google.genai import types

# Import the camera simulation metadata
from camsim import CAM_SIM

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from the .env file
load_dotenv()

def ask_camera_agent(device_id: str, user_query: str) -> str:
    """
    Analyzes a camera feed image using Vertex AI Gemini Flash and answers a user query.
    Strictly reads configuration from environment variables and injects camera metadata.
    
    Args:
        device_id (str): The ID of the camera (e.g., 'cam-1').
        user_query (str): The question the user is asking about the feed.
        
    Returns:
        str: The text response from Gemini.
        
    Raises:
        ValueError: If required environment variables are not set.
        FileNotFoundError: If the specified camera image does not exist.
    """
    
    # 1. Strictly fetch from environment variables
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    project_location = os.getenv("GOOGLE_CLOUD_LOCATION")
    
    # 2. Check if env variables are missing: log error and throw exception
    missing_vars =[]
    if not project_id:
        missing_vars.append("GOOGLE_CLOUD_PROJECT")
    if not project_location:
        missing_vars.append("GOOGLE_CLOUD_LOCATION")
        
    if missing_vars:
        error_msg = f"Missing required environment variables in .env file: {', '.join(missing_vars)}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    else:
        logging.info(f"GOOGLE_CLOUD_PROJECT {project_id}, GOOGLE_CLOUD_LOCATION {project_location}")
        
    # 3. Construct the image path
    image_path = f"./static/camview/{device_id}.jpg"
    
    # 4. Verify the image actually exists (Log and throw if not)
    if not os.path.exists(image_path):
        error_msg = f"Camera feed for device '{device_id}' could not be found at {image_path}."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    # 5. Retrieve Metadata from mapping.json and CAM_SIM
    mapping_path = "./static/camview/mapping.json"
    metadata_json_str = "{}"
    
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
            
            # Retrieve the original filename mapped to this device's current image
            target_key = f"{device_id}.jpg"
            original_filename = mapping.get(target_key)
            
            if original_filename:
                # Find the matching dictionary in CAM_SIM
                for sim in CAM_SIM:
                    if sim.get("filename") == original_filename:
                        # Create a copy so we don't modify the original CAM_SIM dictionary
                        llm_metadata = sim.copy()
                        
                        # Remove the filename before passing to LLM
                        llm_metadata.pop("filename", None)
                        
                        # Convert the matched dict to a formatted JSON string
                        metadata_json_str = json.dumps(llm_metadata, indent=2)
                        logging.info(f"Successfully loaded metadata for {device_id} (excluded filename from prompt)")
                        break
            else:
                logging.warning(f"Key '{target_key}' not found in mapping.json")
        except Exception as e:
            logging.error(f"Error reading mapping or metadata: {e}")
    else:
        logging.warning(f"Mapping file not found at {mapping_path}")

    try:
        # 6. Initialize the new genai Client using Vertex AI parameters
        client = genai.Client(
            vertexai=True, 
            project=project_id, 
            location=project_location
        )
        
        # 7. Read the local image as bytes and create a Part object
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            
        camera_image = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        )
        
        # 8. Construct the strict prompt including the dynamic metadata
        prompt = (
            "You are a precise and helpful smart home AI assistant. "
            "Your task is to analyze the provided live camera feed image and answer the user's question.\n\n"
            "Below is the system metadata associated with this camera feed. Use this data to accurately identify people (e.g., by name/label), the activity they are doing and room they are in:\n"
            f"CAMERA METADATA:\n{metadata_json_str}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. GROUNDING: Base your answer STRICTLY on what is clearly visible in the image AND the provided CAMERA METADATA. Do not guess or hallucinate details outside of these two sources.\n"
            "2. ANSWER: Use the information from the camera feed image to answer the question following the ANSWER FORMAT, by contextually analyzing the data to summarize the situation."
            "3. ANSWER FORMAT: Subject A (based on user query, address by name where available) is doing X (activity) in Y (which room). If the requested information is not present in the image or metadata, respond appropriately in your response.\n"
            "4. TONE & LENGTH: Keep your response brief, factual, and direct. Dont reference time until user query response needs it. Avoid unnecessary conversational filler.\n"
            "5. FORMATTING: Output RAW TEXT ONLY. Do not use Markdown, asterisks, bolding, lists, quotes, or code blocks.\n\n"
            f"USER QUESTION: \"{user_query}\"\n"
            "ANSWER:"
        )

        # 9. Enforce Generation Configuration
        generation_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            temperature=0.2, # Low temperature makes the output highly deterministic and factual
            max_output_tokens=256 # Caps the length of the response
        )
        
        # 10. Generate content with the configuration applied
        logging.info(f"Sending prompt to Gemini for device '{device_id}'...")
        response = client.models.generate_content(
            model=config.SUBAGENT_MODEL,
            contents=[camera_image, prompt],
            config=generation_config
        )
        
        # 11. Return the text result, stripping any accidental trailing/leading whitespace
        return response.text.strip()
        
    except Exception as e:
        # Log any API or connection errors, then re-raise the exception
        logging.error(f"An error occurred while communicating with Google Gen AI: {str(e)}")
        raise e

# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    print("--- Camera Agent Tester ---")
    
    # Capture the camera ID from the user
    cam_id = input("Enter Camera ID (e.g., cam-1): ").strip()
    
    # Capture the question from the user
    user_question = input("Enter your question: ").strip()
    
    # Ensure both inputs were provided
    if not cam_id or not user_question:
        print("\nError: Both Camera ID and Question are required. Exiting.")
    else:
        try:
            print("\nProcessing... please wait.")
            # Pass the user's inputs dynamically to the function
            response_text = ask_camera_agent(cam_id, user_question)
            
            print("\n--- Gemini Response ---")
            print(response_text)
            
        except Exception as ex:
            print(f"\nExecution failed: {ex}")