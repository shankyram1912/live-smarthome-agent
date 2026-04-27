import os
import logging
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel, Image, GenerationConfig

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from the .env file
load_dotenv()

def ask_camera_agent(device_id: str, user_query: str) -> str:
    """
    Analyzes a camera feed image using Vertex AI Gemini 3 Flash and answers a user query.
    Strictly reads configuration from environment variables.
    
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
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    
    # 2. Check if env variables are missing: log error and throw exception
    missing_vars =[]
    if not project_id:
        missing_vars.append("GOOGLE_CLOUD_PROJECT")
    if not location:
        missing_vars.append("GOOGLE_CLOUD_LOCATION")
        
    if missing_vars:
        error_msg = f"Missing required environment variables in .env file: {', '.join(missing_vars)}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    else:
        logging.info(f"GOOGLE_CLOUD_PROJECT {GOOGLE_CLOUD_PROJECT}, GOOGLE_CLOUD_LOCATION {GOOGLE_CLOUD_LOCATION}")
        
    # 3. Construct the image path
    image_path = f"./static/camview/{device_id}.jpg"
    
    # 4. Verify the image actually exists (Log and throw if not)
    if not os.path.exists(image_path):
        error_msg = f"Camera feed for device '{device_id}' could not be found at {image_path}."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        # 5. Initialize Vertex AI with env variables
        vertexai.init(project=project_id, location=location)
        
        # 6. Initialize the requested Gemini 3 Flash model
        model = GenerativeModel("gemini-3-flash-preview")
        
        # 7. Load the local image file
        camera_image = Image.load_from_file(image_path)
        
        # 8. Construct the strict prompt (Added Rule #4)
        prompt = (
            "You are a precise and helpful smart home AI assistant. "
            "Your task is to analyze the provided live camera feed image and answer the user's question.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. GROUNDING: Base your answer STRICTLY on what is clearly visible in the image. Do not guess, assume, or hallucinate details outside the frame.\n"
            "2. UNCERTAINTY: If the requested information is not present, obscured, or unclear, explicitly state: 'I cannot clearly see that in the current camera feed.'\n"
            "3. TONE & LENGTH: Keep your response brief, factual, and direct. Avoid unnecessary conversational filler.\n"
            "4. FORMATTING: Output RAW TEXT ONLY. Do not use Markdown, asterisks, bolding, lists, quotes, or code blocks.\n\n"
            f"USER QUESTION: \"{user_query}\"\n"
            "ANSWER:"
        )

        # 9. Enforce Generation Configuration
        generation_config = GenerationConfig(
            response_mime_type="text/plain",
            temperature=0.2, # Low temperature makes the output highly deterministic and factual
            max_output_tokens=256 # Caps the length of the response
        )
        
        # 10. Generate content with the configuration applied
        logging.info(f"Sending prompt to Gemini for device '{device_id}'...")
        response = model.generate_content(
            [camera_image, prompt],
            generation_config=generation_config
        )
        
        # 11. Return the text result, stripping any accidental trailing/leading whitespace
        return response.text.strip()
        
    except Exception as e:
        # Log any API or connection errors, then re-raise the exception
        logging.error(f"An error occurred while communicating with Vertex AI: {str(e)}")
        raise e

# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    try:
        response_text = ask_camera_agent("cam-1", "Is there anyone at the front door?")
        print("\n--- Gemini Response ---")
        print(response_text)
    except Exception as ex:
        print(f"\nExecution failed: {ex}")