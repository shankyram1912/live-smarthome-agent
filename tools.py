import os
import time
import yaml
import json
import logging
import threading
import asyncio
from enum import Enum
from typing import Literal
from google.cloud import firestore
from dotenv import load_dotenv
from camfeed import analyze_camera_feed

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
        self._lock = threading.Lock()
        
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        db_id = os.getenv("GOOGLE_CLOUD_FIRESTORE")
        
        if not project_id:
            logger.error("⚠️ GOOGLE_CLOUD_PROJECT is missing from the environment!")
            raise ValueError("⚠️ GOOGLE_CLOUD_PROJECT is missing from the environment!")
        if not db_id:
            logger.error("⚠️ GOOGLE_CLOUD_FIRESTORE is missing from the environment!")
            raise ValueError("⚠️ GOOGLE_CLOUD_FIRESTORE is missing from the environment!")
            
        logger.info(f"Connecting to Firestore instance: {db_id} in project {project_id}")

        try:
            self.db = firestore.Client(
                project=project_id, 
                database=db_id
            )
            logger.info(f"✅ Connected to Firestore instance: {db_id} in project {project_id}")
            
        except Exception as e:
            self.db = None
            logger.error(f"⚠️ Failed to connect to Firestore: {e}")
            raise ValueError(f"⚠️ GOOGLE_CLOUD_FIRESTORE Failed to connect to Firestore: {e}")

    def get_smart_home_devices_info(self) -> str:
        """Fetches the live smart home devices"""
        if not self.db:
            logger.error("⚠️ Unable to retrieve smart home devices status, database connection not initialized!")
            return {"Information": "Unable to retrieve smart home devices status"}

        try:
            doc_ref = self.db.collection("home-users").document("default")
            doc = doc_ref.get()

            if not doc.exists:
                return "home_devices: []"

            devices_map = doc.to_dict().get("devices", {})
            smart_device_list = []

            for device_id, data in devices_map.items():
                
                current_state = data.get("state", "off")
                
                smart_device_list.append({
                    "id": data.get("id", device_id),
                    "deviceLabel": data.get("deviceLabel", "Unknown"),
                    "room": data.get("room", "Unknown"),
                    "state": data.get("state", "off"),
                    "currentSettingValue": "" if current_state == "off" else data.get("currentSettingValue", ""),
                    "defaultSettingValue": data.get("defaultSettingValue", "")
                })
            
            smart_device_list_yaml = yaml.dump(smart_device_list)
            
            # Log tool calls
            print("\n" + "="*50)
            print(f"🔧 TOOL EXECUTION] get_smart_home_devices_info \n {smart_device_list_yaml}")
            print("="*50 + "\n", flush=True)            
            logger.debug(f"Smart home devices YAML - \n {smart_device_list_yaml}")
            return smart_device_list_yaml

        except Exception as e:
            logger.error(f"Smart home devices read failed: {str(e)}")
            return {"error": "Failed to fetch smart device topology."}

    def control_airconditioner(self, id: str, newState: bool, newSettingValue: str = None, defaultSettingValue: str = None) -> str:
        """
        SILENT EXECUTION. Controls an Air Conditioner unit in the smart home.

        Args:
            id: The exact ID of the AC unit (mandatory).
            newState: The desired state as a boolean, True for ON, False for OFF (mandatory).
            newSettingValue: The target temperature as a string (optional).
            defaultSettingValue: The default temperature as a string (optional).
        """
        if not self.db:
            return {"error": "Database connection not initialized."}

        # Evaluate the boolean against the Enum
        target_state_str = DeviceState.ON.value if newState else DeviceState.OFF.value

        # Log tool calls
        print("\n" + "="*50)
        print(f"[🔧 TOOL EXECUTION] control_airconditioner(id={id}, newState={newState} -> '{target_state_str}', newSettingValue={newSettingValue})")
        print("="*50 + "\n", flush=True)   
        logger.info(f"[🔧 TOOL EXECUTION] control_airconditioner(id={id}, newState={newState} -> '{target_state_str}', newSettingValue={newSettingValue})")

        # Cache check using the instance dictionary
        cache_key = f"{id}_{target_state_str}_{newSettingValue}"
        current_time = time.time()
        
        # with self._lock: 
        #     if cache_key in self._action_cache and (current_time - self._action_cache[cache_key]) < 5:
        #         return {"status": "IGNORED_DUPLICATE_CALL"}
        #     self._action_cache[cache_key] = current_time

        try:
            doc_ref = self.db.collection("home-users").document("default")
            doc = doc_ref.get()
            
            if not doc.exists:
                return {"error": "User document not found."}
                
            devices = doc.to_dict().get("devices", {})

            if id not in devices:
                return {"error": f"Device {id} not found."}

            device = devices[id]
            oldState = device.get("state", "unknown")
            oldSettingValue = device.get("currentSettingValue", "unknown")
            
            # Use defaultSettingValue from DB if newSettingValue is None and defaultSettingValue is None
            if newSettingValue is not None:
                finalSettingValue = newSettingValue
            else:
                if(defaultSettingValue is not None):
                    finalSettingValue = defaultSettingValue
                else:
                    finalSettingValue = device.get("defaultSettingValue", oldSettingValue) 

            # Update Firestore
            if(defaultSettingValue is None):
                doc_ref.update({
                    f"devices.{id}.state": target_state_str,
                    f"devices.{id}.currentSettingValue": finalSettingValue
                })
            else:
                doc_ref.update({
                    f"devices.{id}.state": target_state_str,
                    f"devices.{id}.currentSettingValue": finalSettingValue,
                    f"devices.{id}.defaultSettingValue": defaultSettingValue
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
            return response_dict
            
        except Exception as e:
            logger.error(f"Failed to update AC {id}: {str(e)}")
            return {"error": f"Database update failed: {str(e)}"}
        
    def control_camera(
        self, id: str, 
        newState: bool, 
        newSettingValue: Literal["Online", "Private", "Protect"] = None,
        defaultSettingValue: Literal["Online", "Private", "Protect"] = None) -> str:
        """
        SILENT EXECUTION. Controls an Smart Camera in the smart home.

        Args:
            id: The exact ID of the Smart camera unit (mandatory).
            newState: The desired state as a boolean, True for ON, False for OFF (mandatory).
            newSettingValue: The camera's security mode as a string. Must be exactly "Online", "Private", or "Protect" (optional).
            defaultSettingValue: The camera's default security mode as a string. Must be exactly "Online", "Private", or "Protect" (optional).
        """
        if not self.db:
            return {"error": "Database connection not initialized."}

        # Evaluate the boolean against the Enum
        target_state_str = DeviceState.ON.value if newState else DeviceState.OFF.value

        # Log tool calls
        print("\n" + "="*50)
        print(f"[🔧 TOOL EXECUTION] control_camera(id={id}, newState={newState} -> '{target_state_str}', newSettingValue={newSettingValue})")
        print("="*50 + "\n", flush=True)   
        logger.info(f"[🔧 TOOL EXECUTION] control_camera(id={id}, newState={newState} -> '{target_state_str}', newSettingValue={newSettingValue})")

        # Cache check using the instance dictionary
        cache_key = f"{id}_{target_state_str}_{newSettingValue}"
        current_time = time.time()
        
        # with self._lock: 
        #     if cache_key in self._action_cache and (current_time - self._action_cache[cache_key]) < 5:
        #         return {"status": "IGNORED_DUPLICATE_CALL"}
        #     self._action_cache[cache_key] = current_time

        try:
            doc_ref = self.db.collection("home-users").document("default")
            doc = doc_ref.get()
            
            if not doc.exists:
                return {"error": "User document not found."}
                
            devices = doc.to_dict().get("devices", {})

            if id not in devices:
                return {"error": f"Device {id} not found."}

            device = devices[id]
            oldState = device.get("state", "unknown")
            oldSettingValue = device.get("currentSettingValue", "unknown")
            
            # Use defaultSettingValue from DB if newSettingValue is None and defaultSettingValue is None
            if newSettingValue is not None:
                finalSettingValue = newSettingValue
            else:
                if(defaultSettingValue is not None):
                    finalSettingValue = defaultSettingValue
                else:
                    finalSettingValue = device.get("defaultSettingValue", oldSettingValue) 

            # Update Firestore
            if(defaultSettingValue is None):
                doc_ref.update({
                    f"devices.{id}.state": target_state_str,
                    f"devices.{id}.currentSettingValue": finalSettingValue
                })
            else:
                doc_ref.update({
                    f"devices.{id}.state": target_state_str,
                    f"devices.{id}.currentSettingValue": finalSettingValue,
                    f"devices.{id}.defaultSettingValue": defaultSettingValue
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
            return response_dict
            
        except Exception as e:
            logger.error(f"Failed to update camera {id}: {str(e)}")
            return {"error": f"Database update failed: {str(e)}"}
        
    def control_light(
        self, id: str, 
        newState: bool, 
        newSettingValue: Literal["Cool", "Movie", "Bright"] = None,
        defaultSettingValue: Literal["Cool", "Movie", "Bright"] = None) -> str:
        """
        SILENT EXECUTION. Controls an Smart Lights in the smart home.

        Args:
            id: The exact ID of the Smart Light unit (mandatory).
            newState: The desired state as a boolean, True for ON, False for OFF (mandatory).
            newSettingValue: The lighting mode as a string. Must be exactly "Cool", "Movie", or "Bright" (optional).
            defaultSettingValue: The lighting default mode as a string. Must be exactly "Cool", "Movie", or "Bright" (optional).
        """
        if not self.db:
            return {"error": "Database connection not initialized."}

        # Evaluate the boolean against the Enum
        target_state_str = DeviceState.ON.value if newState else DeviceState.OFF.value

        # Log tool calls
        print("\n" + "="*50)
        print(f"[🔧 TOOL EXECUTION] control_light(id={id}, newState={newState} -> '{target_state_str}', newSettingValue={newSettingValue})")
        print("="*50 + "\n", flush=True)   
        logger.info(f"[🔧 TOOL EXECUTION] control_light(id={id}, newState={newState} -> '{target_state_str}', newSettingValue={newSettingValue})")

        # Cache check using the instance dictionary
        cache_key = f"{id}_{target_state_str}_{newSettingValue}"
        current_time = time.time()
        
        # with self._lock: 
        #     if cache_key in self._action_cache and (current_time - self._action_cache[cache_key]) < 5:
        #         return {"status": "IGNORED_DUPLICATE_CALL"}
        #     self._action_cache[cache_key] = current_time

        try:
            doc_ref = self.db.collection("home-users").document("default")
            doc = doc_ref.get()
            
            if not doc.exists:
                return {"error": "User document not found."}
                
            devices = doc.to_dict().get("devices", {})

            if id not in devices:
                return {"error": f"Device {id} not found."}

            device = devices[id]
            oldState = device.get("state", "unknown")
            oldSettingValue = device.get("currentSettingValue", "unknown")
            
            # Use defaultSettingValue from DB if newSettingValue is None and defaultSettingValue is None
            if newSettingValue is not None:
                finalSettingValue = newSettingValue
            else:
                if(defaultSettingValue is not None):
                    finalSettingValue = defaultSettingValue
                else:
                    finalSettingValue = device.get("defaultSettingValue", oldSettingValue) 

            # Update Firestore
            if(defaultSettingValue is None):
                doc_ref.update({
                    f"devices.{id}.state": target_state_str,
                    f"devices.{id}.currentSettingValue": finalSettingValue
                })
            else:
                doc_ref.update({
                    f"devices.{id}.state": target_state_str,
                    f"devices.{id}.currentSettingValue": finalSettingValue,
                    f"devices.{id}.defaultSettingValue": defaultSettingValue
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
            return response_dict
            
        except Exception as e:
            logger.error(f"Failed to update light {id}: {str(e)}")
            return {"error": f"Database update failed: {str(e)}"}        
        
    def control_lock(
        self, id: str, 
        newState: bool, 
        newSettingValue: Literal["Guest", "Party", "DND"] = None,
        defaultSettingValue: Literal["Guest", "Party", "DND"] = None) -> str:
        """
        SILENT EXECUTION. Controls an Smart Lock in the smart home.

        Args:
            id: The exact ID of the Smart Lock unit (mandatory).
            newState: The desired state as a boolean, True for ON, False for OFF (mandatory).
            newSettingValue: The locking mode as a string. Must be exactly "Guest", "Party", or "DND" (optional).
            defaultSettingValue: The locking default mode as a string. Must be exactly "Guest", "Party", or "DND" (optional).
        """
        if not self.db:
            return {"error": "Database connection not initialized."}

        # Evaluate the boolean against the Enum
        target_state_str = DeviceState.ON.value if newState else DeviceState.OFF.value

        # Log tool calls
        print("\n" + "="*50)
        print(f"[🔧 TOOL EXECUTION] control_lock(id={id}, newState={newState} -> '{target_state_str}', newSettingValue={newSettingValue})")
        print("="*50 + "\n", flush=True)   
        logger.info(f"[🔧 TOOL EXECUTION] control_lock(id={id}, newState={newState} -> '{target_state_str}', newSettingValue={newSettingValue})")

        # Cache check using the instance dictionary
        cache_key = f"{id}_{target_state_str}_{newSettingValue}"
        current_time = time.time()
        
        # with self._lock: 
        #     if cache_key in self._action_cache and (current_time - self._action_cache[cache_key]) < 5:
        #         return {"status": "IGNORED_DUPLICATE_CALL"}
        #     self._action_cache[cache_key] = current_time

        try:
            doc_ref = self.db.collection("home-users").document("default")
            doc = doc_ref.get()
            
            if not doc.exists:
                return {"error": "User document not found."}
                
            devices = doc.to_dict().get("devices", {})

            if id not in devices:
                return {"error": f"Device {id} not found."}

            device = devices[id]
            oldState = device.get("state", "unknown")
            oldSettingValue = device.get("currentSettingValue", "unknown")
            
            # Use defaultSettingValue from DB if newSettingValue is None and defaultSettingValue is None
            if newSettingValue is not None:
                finalSettingValue = newSettingValue
            else:
                if(defaultSettingValue is not None):
                    finalSettingValue = defaultSettingValue
                else:
                    finalSettingValue = device.get("defaultSettingValue", oldSettingValue) 

            # Update Firestore
            if(defaultSettingValue is None):
                doc_ref.update({
                    f"devices.{id}.state": target_state_str,
                    f"devices.{id}.currentSettingValue": finalSettingValue
                })
            else:
                doc_ref.update({
                    f"devices.{id}.state": target_state_str,
                    f"devices.{id}.currentSettingValue": finalSettingValue,
                    f"devices.{id}.defaultSettingValue": defaultSettingValue
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
            return response_dict
            
        except Exception as e:
            logger.error(f"Failed to update lock {id}: {str(e)}")
            return {"error": f"Database update failed: {str(e)}"}         
        
    async def check_camera(self, userQuery: str, camera_ids: list[str]) -> str:
        """
        Analyzes live camera feeds to answer a user's query about their smart home environment.

        Args:
            userQuery: The question the user is asking about the camera feeds (mandatory).
            camera_ids: A list/array of exact camera IDs to check, e.g., ["cam-1", "cam-2"] (mandatory).
        """
        # Log tool calls
        print("\n" + "="*50)
        print(f"[🔧 TOOL EXECUTION] check_camera(camera_ids={camera_ids}, userQuery='{userQuery}')")
        print("="*50 + "\n", flush=True)   
        logger.info(f"[🔧 TOOL EXECUTION] check_camera(camera_ids={camera_ids}, userQuery='{userQuery}')")

        results =[]
        
        # Changing to parallel async firing
        # for camera_id in camera_ids:
        #     try:
        #         # analyze_camera_feed returns a JSON string
        #         response_str = analyze_camera_feed(camera_id, userQuery)
                
        #         # Parse it back to a dictionary so we don't end up with nested/escaped JSON strings
        #         response_dict = json.loads(response_str)
        #         results.append(response_dict)
                
        #     except Exception as e:
        #         logger.error(f"Failed to check camera {camera_id}: {str(e)}")
        #         # Append a fallback payload if a specific camera fails
        #         results.append({
        #             "id": camera_id,
        #             "error": str(e),
        #             "is_user_query_addressed": False
        #         })
        
        async def process_camera(camera_id):
            try:
                # Runs the blocking function in a separate thread
                response_str = await analyze_camera_feed(camera_id, userQuery)
                return response_str
            except Exception as e:
                # ... (error handling)
                return {"id": camera_id, "error": str(e)}
            
        tasks = [process_camera(cid) for cid in camera_ids]
        results = await asyncio.gather(*tasks)                 

        # Return the aggregated list as a clean, formatted JSON string
        return results