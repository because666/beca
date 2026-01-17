import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import supabase, but don't fail if not present (for local dev without requirements installed yet)
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    Client = None
    SUPABASE_AVAILABLE = False
    logger.warning("Supabase library not found. Running in local-only mode.")

class StorageManager:
    def __init__(self):
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_KEY")
        self.client: Optional[Client] = None
        self.is_connected = False
        
        if SUPABASE_AVAILABLE and self.supabase_url and self.supabase_key:
            try:
                # Set a timeout for client creation if possible, though mostly it's lazy
                self.client = create_client(self.supabase_url, self.supabase_key)
                logger.info("Supabase client initialized successfully")
                self.is_connected = True
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                self.is_connected = False
        elif SUPABASE_AVAILABLE and (not self.supabase_url or not self.supabase_key):
             logger.info("Supabase credentials not found. Using local storage only.")

    def load_json(self, filepath: Path, key: str, default: Dict = None) -> Dict:
        """
        Load JSON data from Supabase (if available) or local file.
        """
        if default is None:
            default = {}
            
        # 1. Try Supabase first if configured
        if self.is_connected and self.client:
            try:
                # Add a timeout mechanism via options if supported, or just rely on global socket timeout
                # Supabase-py uses httpx, defaults are usually fine but we catch generic Exception
                response = self.client.table("app_storage").select("value").eq("key", key).execute()
                if response.data and len(response.data) > 0:
                    logger.info(f"Loaded {key} from Supabase")
                    return response.data[0]['value']
            except Exception as e:
                logger.warning(f"Failed to load {key} from Supabase (Network Issue?): {e}")
                # Don't disable connection permanently, just fail this request

        # 2. Fallback to local file
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load local file {filepath}: {e}")
        
        return default

    def save_json(self, filepath: Path, key: str, data: Dict) -> bool:
        """
        Save JSON data to Supabase (if available) and local file.
        """
        success = False
        
        # 1. Try Supabase
        if self.is_connected and self.client:
            try:
                # Upsert
                self.client.table("app_storage").upsert({
                    "key": key,
                    "value": data
                }).execute()
                success = True
                logger.info(f"Saved {key} to Supabase")
            except Exception as e:
                logger.error(f"Failed to save {key} to Supabase: {e}")

        # 2. Always save to local as backup/cache
        try:
            # Ensure dir exists
            if filepath.parent != Path('.'):
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            if not self.is_connected or not self.client:
                success = True
        except Exception as e:
            logger.error(f"Failed to save local file {filepath}: {e}")
            
        return success

    def load_predictions(self, date_str: str) -> Optional[Dict]:
        """
        Load historical predictions for a specific date.
        """
        key = f"predictions_{date_str}"
        filepath = Path(f"data/predictions/predictions_{date_str}.json")
        return self.load_json(filepath, key, None)

    def save_predictions(self, date_str: str, data: Dict) -> bool:
        """
        Save predictions for a specific date.
        """
        key = f"predictions_{date_str}"
        filepath = Path(f"data/predictions/predictions_{date_str}.json")
        return self.save_json(filepath, key, data)

# Global instance
storage = StorageManager()
