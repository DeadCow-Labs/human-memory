import os
from dataclasses import dataclass

@dataclass
class Config:
    openai_api_key: str
    supabase_url: str
    supabase_key: str

def load_config() -> Config:
    """
    Load configuration from environment variables.
    
    Required environment variables:
    - OPENAI_API_KEY
    - SUPABASE_URL
    - SUPABASE_KEY
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    # Validate required environment variables
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    if not supabase_url:
        raise ValueError("SUPABASE_URL environment variable is required")
    if not supabase_key:
        raise ValueError("SUPABASE_KEY environment variable is required")
    
    return Config(
        openai_api_key=openai_api_key,
        supabase_url=supabase_url,
        supabase_key=supabase_key
    )