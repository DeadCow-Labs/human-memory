import json
from typing import Dict, List, Any

from openai import OpenAI
from memory_sdk.models import Memory, Location

def structure_memory(client: OpenAI, raw_message: str, user_id: str) -> Dict[str, Any]:
    """
    Use OpenAI's API to structure a raw message into memory components.
    
    Args:
        client: OpenAI client
        raw_message: The raw user message
        user_id: The user's ID
        
    Returns:
        A dictionary with structured memory data
    """
    system_prompt = """
    You are a system that turns user messages into structured memory data. 
    Analyze the message and extract the following information:
    
    1. content: A short summary of the message (1-2 sentences)
    2. reflection: An insight about the user based on this message (1 sentence)
    3. emotional_tone: The emotional tone of the message (e.g., "hopeful", "anxious", "excited")
    4. location:
       - type: "mental" (thoughts/ideas), "physical" (real-world location), "digital" (online) or "described" (mentioned location)
       - name: Name or description of the location
    5. tags: 1-5 relevant topics, themes, or concepts mentioned in the message
    
    Return JSON with these fields and nothing else.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_message}
        ],
        response_format={"type": "json_object"}
    )
    
    # Parse the response
    result = json.loads(response.choices[0].message.content)
    
    # Ensure the result contains all required fields
    required_fields = ["content", "reflection", "emotional_tone", "location", "tags"]
    for field in required_fields:
        if field not in result:
            raise ValueError(f"OpenAI response missing required field: {field}")
    
    # Ensure location has type and name
    if not isinstance(result["location"], dict) or "type" not in result["location"] or "name" not in result["location"]:
        raise ValueError("Location must be an object with 'type' and 'name' fields")
    
    result["user_id"] = user_id
    
    return result

def get_embedding(client: OpenAI, text: str) -> List[float]:
    """
    Get embedding vector for text using OpenAI's API.
    
    Args:
        client: OpenAI client
        text: The text to embed
        
    Returns:
        A list of floats representing the embedding vector
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    
    # Return the embedding from the first (and only) result
    return response.data[0].embedding
