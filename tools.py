import os
import time
import yaml
import logging
from enum import Enum
from google.cloud import firestore
from dotenv import load_dotenv

# ==========================================
# Module-Level Setup
# ==========================================

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

load_dotenv(override=True)

# Define the Enum for device states
class DeviceState(str, Enum):
    ON = "on"
    OFF = "off"

class Tools:
    def __init__(self):
        """Initializes the Firestore connection and class state.""" 
        self._action_cache = {}
        
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        db_id = os.getenv("GOOGLE_CLOUD_FIRESTORE")
        
        if not project_id:
            logger.warning("⚠️ GOOGLE_CLOUD_PROJECT is missing from the environment!")
        if not db_id:
            logger.warning("⚠️ GOOGLE_CLOUD_FIRESTORE is missing from the environment!")

        try:
            self.db = firestore.client(
                project=project_id, 
                database_id=db_id
            )
            logger.info(f"✅ Connected to Firestore instance: {db_id}")
            
        except Exception as e:
            self.db = None
            logger.error(f"⚠️ Failed to connect to Firestore: {e}")

    def get_device_topology_yaml(self) -> str:
        """Fetches the live database and formats it into the strict YAML schema."""
        logger.info(f"[LOADING DEVICE CONFIG]")
        if not self.db:
            return yaml.dump({"error": "Database connection not initialized."})

        try:
            doc_ref = self.db.collection("home-users").document("default")
            doc = doc_ref.get()

            if not doc.exists:
                return "home_devices: []"

            devices_map = doc.to_dict().get("devices", {})
            yaml_list = []

            for device_id, data in devices_map.items():
                yaml_list.append({
                    "id": data.get("id", device_id),
                    "deviceLabel": data.get("deviceLabel", "Unknown"),
                    "room": data.get("room", "Unknown"),
                    "state": data.get("state", "off"),
                    "currentSettingValue": data.get("currentSettingValue", "")
                })

            logger.info(f"[@@@ DEVICE CONFIG] {yaml_list})")
            
            return yaml.dump({"home_devices": yaml_list}, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.error(f"Topology read failed: {str(e)}")
            return yaml.dump({"error": "Failed to fetch device topology."})

    def control_airconditioner(self, id: str, newState: bool, newSettingValue: str = None) -> str:
        """
        SILENT EXECUTION. Controls an Air Conditioner unit in the smart home.

        Args:
            id: The exact ID of the AC unit (mandatory).
            newState: The desired state as a boolean, True for ON, False for OFF (mandatory).
            newSettingValue: The target temperature as a string (optional).
        """
        if not self.db:
            return yaml.dump({"error": "Database connection not initialized."})

        # Evaluate the boolean against the Enum
        target_state_str = DeviceState.ON.value if newState else DeviceState.OFF.value

        logger.info(f"[🔧 TOOL EXECUTION] control_airconditioner(id={id}, newState={newState} -> '{target_state_str}', newSettingValue={newSettingValue})")

        # Cache check using the instance dictionary
        cache_key = f"{id}_{target_state_str}_{newSettingValue}"
        current_time = time.time()
        
        with self._lock: 
            if cache_key in self._action_cache and (current_time - self._action_cache[cache_key]) < 5:
                return yaml.dump({"status": "IGNORED_DUPLICATE_CALL"})
            self._action_cache[cache_key] = current_time

        try:
            doc_ref = self.db.collection("home-users").document("default")
            doc = doc_ref.get()
            
            if not doc.exists:
                return yaml.dump({"error": "User document not found."})
                
            devices = doc.to_dict().get("devices", {})

            if id not in devices:
                return yaml.dump({"error": f"Device {id} not found."})

            device = devices[id]
            oldState = device.get("state", "unknown")
            oldSettingValue = device.get("currentSettingValue", "unknown")
            
            # Use defaultSettingValue from DB if newSettingValue is None
            if newSettingValue is not None:
                finalSettingValue = newSettingValue
            else:
                finalSettingValue = device.get("defaultSettingValue", oldSettingValue) 

            # Update Firestore
            doc_ref.update({
                f"devices.{id}.state": target_state_str,
                f"devices.{id}.currentSettingValue": finalSettingValue
            })

            response_dict = {
                "id": id,
                "deviceLabel": device.get("deviceLabel"),
                "room": device.get("room"),
                "oldState": oldState,
                "newState": target_state_str,
                "oldSettingValue": oldSettingValue,
                "currentSettingValue": finalSettingValue
            }
            return yaml.dump(response_dict, default_flow_style=False, sort_keys=False)
            
        except Exception as e:
            logger.error(f"Failed to update AC {id}: {str(e)}")
            return yaml.dump({"error": f"Database update failed: {str(e)}"})