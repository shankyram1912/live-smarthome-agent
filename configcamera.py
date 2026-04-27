import os
import shutil
import json
from camsim import CAM_SIM

# Helper functions for the mapping file
def load_mapping(filepath):
    """Loads the mapping from a JSON file, or returns an empty dict if it doesn't exist."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_mapping(filepath, mapping):
    """Saves the mapping dictionary to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(mapping, f, indent=4)

def main():
    # Setup directories once before the loop starts
    src_dir = "./camconfig"
    dest_dir = "./static/camview"
    mapping_file = os.path.join(dest_dir, "mapping.json")
    
    # Create the destination directory if it doesn't already exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Load the current mappings into memory
    mapping = load_mapping(mapping_file)
    
    print("--- Camera Simulator Started ---")

    while True:
        print("\nPlease select a camera simulation:\n")
        
        # Display Option 0 and leave a blank line
        print("0. Delete All Cam Views\n")
        
        # 1. Display the options from CAM_SIM
        for i, sim in enumerate(CAM_SIM, start=1):
            room = sim.get('room', 'Unknown Room')
            filename = sim.get('filename', '')
            
            # Extract the label from the filename 
            try:
                display_name = filename.split('_')[1].split('.')[0]
            except IndexError:
                display_name = filename 
                
            print(f"{i}. {room} | {display_name}")

        # 2. Get user selection
        user_input = input("\nEnter the number of your choice (or 'q' to quit): ").strip().lower()
        
        # Check for termination command
        if user_input in ['q', 'quit', 'exit']:
            print("Terminating camera simulation. Goodbye!")
            break # Breaks out of the while loop, ending the program

        try:
            choice = int(user_input)
            if choice < 0 or choice > len(CAM_SIM):
                print("Invalid selection. Please choose a valid number from the list.")
                continue # Skips the rest of the loop and starts over
        except ValueError:
            print("Please enter a valid integer or 'q' to quit.")
            continue # Skips the rest of the loop and starts over

        # Handle Option 0: Delete All Cam Views
        if choice == 0:
            confirm = input("Are you sure you want to delete all camera views? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                mapping_changed = False
                
                # Delete cam-1.jpg, cam-2.jpg, cam-3.jpg
                for cam_id in ['cam-1', 'cam-2', 'cam-3']:
                    filename = f"{cam_id}.jpg"
                    file_to_remove = os.path.join(dest_dir, filename)
                    
                    if os.path.exists(file_to_remove):
                        try:
                            os.remove(file_to_remove)
                            print(f"Removed: {file_to_remove}")
                            
                            # Remove the mapping if it exists
                            if filename in mapping:
                                del mapping[filename]
                                mapping_changed = True
                                
                        except Exception as e:
                            print(f"Error removing {file_to_remove}: {e}")
                
                # Save mapping to file only if something was actually deleted
                if mapping_changed:
                    save_mapping(mapping_file, mapping)
                    print("-> SUCCESS: Mapping file updated.")

                print("-> SUCCESS: All camera views deleted.")
                print("-" * 50)
            else:
                print("-> Deletion cancelled.")
                print("-" * 50)
            
            continue # Go back to the main menu
            
        # 3. Setup file paths for the selected item (if choice 1-N)
        selected_sim = CAM_SIM[choice - 1]
        src_path = os.path.join(src_dir, selected_sim['filename'])
        
        # Extract the file extension (e.g., ".jpg") to create the new filename
        _, ext = os.path.splitext(selected_sim['filename'])
        new_filename = f"{selected_sim['id']}{ext}"
        
        dest_path = os.path.join(dest_dir, new_filename)

        # 4. Copy the file
        try:
            shutil.copy2(src_path, dest_path)
            print(f"-> SUCCESS: Copied '{selected_sim['filename']}' to '{dest_path}'")
            
            # 5. Update and save the mapping
            mapping[new_filename] = selected_sim['filename']
            save_mapping(mapping_file, mapping)
            print(f"-> MAPPED: '{new_filename}' -> '{selected_sim['filename']}'")
            
            print("-" * 50) # Just a visual separator for the next loop iteration
        except FileNotFoundError:
            print(f"-> ERROR: The source file '{src_path}' was not found. Please check your ./camconfig folder.")
            print("-" * 50)
        except Exception as e:
            print(f"-> ERROR: An unexpected error occurred while copying: {e}")
            print("-" * 50)

if __name__ == "__main__":
    main()