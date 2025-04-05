from openai import OpenAI
from memory_sdk.config import load_config
from memory_sdk.models import Memory, Location
from memory_sdk.openai_utils import structure_memory, get_embedding
from supabase import create_client
import json
from typing import List, Optional
from datetime import datetime

class MemorySDK:
    """
    SDK for storing structured memory objects in Supabase.
    """
    
    def __init__(self, user_id: str):
        """
        Initialize the SDK with user ID and load configuration.
        
        Args:
            user_id: The user's ID
        """
        self.user_id = user_id
        self.config = load_config()
        self.openai_client = OpenAI(api_key=self.config.openai_api_key)
        self.supabase = create_client(self.config.supabase_url, self.config.supabase_key)
    
    def save(self, raw_message: str) -> Memory:
        """
        Process a raw message into a structured memory and save it to Supabase.
        
        Args:
            raw_message: The raw user message
            
        Returns:
            The created Memory object
        """
        # Structure the raw message using OpenAI
        structured_data = structure_memory(
            client=self.openai_client,
            raw_message=raw_message,
            user_id=self.user_id
        )
        
        # Get embedding for the content
        embedding = get_embedding(
            client=self.openai_client,
            text=structured_data["content"]
        )
        
        # Create location object
        location = Location(
            type=structured_data["location"]["type"],
            name=structured_data["location"]["name"]
        )
        
        # Create memory object
        memory = Memory(
            user_id=self.user_id,
            content=structured_data["content"],
            reflection=structured_data["reflection"],
            embedding=embedding,
            emotional_tone=structured_data["emotional_tone"],
            location=location,
            tags=structured_data["tags"]
        )
        
        # Save memory to Supabase using the SDK's supabase client
        data = {
            "user_id": memory.user_id,
            "content": memory.content,
            "reflection": memory.reflection,
            "embedding": memory.embedding,
            "emotional_tone": memory.emotional_tone,
            "location": json.dumps(memory.location.__dict__),
            "tags": memory.tags
        }
        
        self.supabase.table("memories").insert(data).execute()
        
        return memory

    def get_memory(self, memory_id: str) -> Memory:
        """
        Retrieve a specific memory by ID.
        
        Args:
            memory_id: The UUID of the memory to retrieve
            
        Returns:
            The Memory object
        """
        response = self.supabase.table("memories").select("*").eq("id", memory_id).execute()
        
        if not response.data:
            raise ValueError(f"Memory with ID {memory_id} not found")
        
        memory_data = response.data[0]
        
        # Convert location from JSON string to Location object
        location_data = json.loads(memory_data["location"]) 
        location = Location(type=location_data["type"], name=location_data["name"])
        
        # Create Memory object
        return Memory(
            id=memory_data["id"],
            user_id=memory_data["user_id"],
            created_at=memory_data["created_at"],
            content=memory_data["content"],
            reflection=memory_data["reflection"],
            embedding=memory_data["embedding"],
            emotional_tone=memory_data["emotional_tone"],
            location=location,
            tags=memory_data["tags"]
        )

    def search_similar(self, query: str, limit: int = 5) -> List[Memory]:
        """
        Search for memories that are semantically similar to the query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of Memory objects ordered by similarity
        """
        # Generate embedding for the query
        query_embedding = get_embedding(
            client=self.openai_client,
            text=query
        )
        
        # Search using vector similarity
        response = self.supabase.rpc(
            "match_memories", 
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.3,
                "match_count": limit,
                "user_id_filter": self.user_id
            }
        ).execute()
        
        # Convert to Memory objects
        memories = []
        for item in response.data:
            location_data = json.loads(item["location"])
            location = Location(type=location_data["type"], name=location_data["name"])
            
            memory = Memory(
                id=item["id"],
                user_id=item["user_id"],
                created_at=item["created_at"],
                content=item["content"],
                reflection=item["reflection"],
                embedding=item["embedding"],
                emotional_tone=item["emotional_tone"],
                location=location,
                tags=item["tags"]
            )
            memories.append(memory)
        
        return memories

    def filter_memories(self, 
                        emotional_tone: str = None,
                        tags: List[str] = None,
                        location_type: str = None,
                        start_date: str = None,
                        end_date: str = None,
                        limit: int = 20) -> List[Memory]:
        """
        Filter memories by various properties.
        
        Args:
            emotional_tone: Filter by emotional tone
            tags: Filter by tags (any match)
            location_type: Filter by location type
            start_date: Filter by date range start (ISO format)
            end_date: Filter by date range end (ISO format)
            limit: Maximum number of results
            
        Returns:
            List of matching Memory objects
        """
        query = self.supabase.table("memories").select("*").eq("user_id", self.user_id)
        
        # Apply filters
        if emotional_tone:
            query = query.eq("emotional_tone", emotional_tone)
        
        if tags:
            query = query.contains("tags", tags)
        
        if location_type:
            # This is a bit more complex since location is stored as JSON
            # We'd ideally need a database function for this
            # For simplicity, we'll filter client-side
            pass
        
        if start_date:
            query = query.gte("created_at", start_date)
        
        if end_date:
            query = query.lte("created_at", end_date)
        
        # Execute query with limit
        response = query.limit(limit).order("created_at", desc=True).execute()
        
        # Filter by location_type if needed (client-side)
        results = response.data
        if location_type:
            results = [r for r in results if json.loads(r["location"])["type"] == location_type]
        
        # Convert to Memory objects
        memories = []
        for item in results:
            location_data = json.loads(item["location"])
            location = Location(type=location_data["type"], name=location_data["name"])
            
            memory = Memory(
                id=item["id"],
                user_id=item["user_id"],
                created_at=item["created_at"],
                content=item["content"],
                reflection=item["reflection"],
                embedding=item["embedding"],
                emotional_tone=item["emotional_tone"],
                location=location,
                tags=item["tags"]
            )
            memories.append(memory)
        
        return memories

    def recall(self, context: str, limit: int = 3) -> List[Memory]:
        """
        Retrieve memories that are relevant to the current context.
        Uses dynamic thresholding to ensure enough results.
        
        Args:
            context: The current conversation or context to find relevant memories for
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant Memory objects
        """
        # First, check if the match_memories function exists
        try:
            # This will either succeed or raise an exception
            self.supabase.rpc(
                "match_memories", 
                {
                    "query_embedding": [0.0] * 1536,  # Dummy embedding for testing
                    "match_threshold": 0.5,
                    "match_count": 1,
                    "user_id_filter": self.user_id
                }
            ).execute()
        except Exception:
            # Function doesn't exist, show the user how to create it
            print("Setting up vector search in your Supabase project...")
            print("Please run this SQL in your Supabase SQL Editor:")
            print("""
CREATE OR REPLACE FUNCTION match_memories(
  query_embedding vector(1536),
  match_threshold float,
  match_count int,
  user_id_filter text
)
RETURNS TABLE (
  id uuid,
  user_id text,
  created_at timestamptz,
  content text,
  reflection text,
  embedding vector(1536),
  emotional_tone text,
  location jsonb,
  tags text[],
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    memories.id,
    memories.user_id,
    memories.created_at,
    memories.content,
    memories.reflection,
    memories.embedding,
    memories.emotional_tone,
    memories.location,
    memories.tags,
    1 - (memories.embedding <=> query_embedding) as similarity
  FROM memories
  WHERE 
    memories.user_id = user_id_filter
    AND 1 - (memories.embedding <=> query_embedding) > match_threshold
  ORDER BY similarity DESC
  LIMIT match_count;
END;
$$;
            """)
            raise ValueError("Please set up vector search function in Supabase first")
        
        # Try with progressively lower thresholds until we get enough results
        thresholds = [0.7, 0.5, 0.3, 0.2]
        
        for threshold in thresholds:
            memories = self._search_memories(context, threshold=threshold, limit=limit)
            if len(memories) > 0:
                return memories
        
        # If we still don't have results, try one last time with very low threshold
        return self._search_memories(context, threshold=0.1, limit=limit)
    
    def remember(self, context: str, limit: int = 5) -> List[Memory]:
        """
        Intelligent human-like memory recall that prioritizes emotionally significant memories.
        
        Args:
            context: Natural language query or context
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant Memory objects, prioritizing emotional significance
        """
        # 1. Extract key entities, concepts, dates, emotions from the context
        extraction_prompt = f"""
        Extract key search parameters from this query: "{context}"
        Return as JSON with these possible fields (include only if mentioned):
        - locations: list of locations mentioned
        - emotions: list of emotions mentioned
        - time_period: any time references (e.g., "last summer", "yesterday")
        - topics: key topics or themes mentioned
        - is_question: whether this is a question requiring memory search
        """
        
        extraction = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract search parameters from natural language queries."},
                {"role": "user", "content": extraction_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        params = json.loads(extraction.choices[0].message.content)
        
        # 2. Determine and execute the best search strategy
        candidate_memories = []
        
        # Vector search is always our base (most human-like retrieval method)
        vector_results = self.recall(context, limit=limit*2)
        
        # Add base scores + emotional intensity scores
        for memory in vector_results:
            # Base score for semantic relevance
            base_score = 1.0
            
            # Add emotional intensity score (0.3-1.0)
            emotional_intensity = self._get_emotional_intensity(memory.emotional_tone)
            emotional_score = emotional_intensity * 0.5  # Scale the impact
            
            # Combined score
            total_score = base_score + emotional_score
            candidate_memories.append((memory, total_score))
        
        # Boost scores for any direct tag matches
        if 'topics' in params and params['topics']:
            for i, (memory, score) in enumerate(candidate_memories):
                for topic in params['topics']:
                    if any(topic.lower() in tag.lower() for tag in memory.tags):
                        candidate_memories[i] = (memory, score + 0.3)
        
        # Boost scores for emotional tone matches
        if 'emotions' in params and params['emotions']:
            for i, (memory, score) in enumerate(candidate_memories):
                for emotion in params['emotions']:
                    if emotion.lower() in memory.emotional_tone.lower():
                        # Boost more for matching emotions with high intensity
                        intensity_boost = self._get_emotional_intensity(memory.emotional_tone) * 0.4
                        candidate_memories[i] = (memory, score + 0.3 + intensity_boost)
        
        # Boost scores for location matches
        if 'locations' in params and params['locations']:
            for i, (memory, score) in enumerate(candidate_memories):
                for location in params['locations']:
                    if location.lower() in memory.location.name.lower():
                        candidate_memories[i] = (memory, score + 0.5)
        
        # Add memories with relevant time periods if mentioned
        if 'time_period' in params and params['time_period']:
            for i, (memory, score) in enumerate(candidate_memories):
                if any(period.lower() in memory.content.lower() for period in params['time_period']):
                    candidate_memories[i] = (memory, score + 0.2)
        
        # Remove duplicates and sort by relevance score
        unique_memories = {}
        for memory, score in candidate_memories:
            if memory.id not in unique_memories or score > unique_memories[memory.id][1]:
                unique_memories[memory.id] = (memory, score)
        
        # Sort by score (highest first)
        sorted_results = sorted(unique_memories.values(), key=lambda x: x[1], reverse=True)
        
        # Return just the memories, not the scores
        return [memory for memory, _ in sorted_results[:limit]]

    def _search_memories(self, context: str, threshold: float = 0.5, limit: int = 3) -> List[Memory]:
        """
        Helper method that performs vector similarity search with a configurable threshold.
        
        Args:
            context: The text to search for
            threshold: Similarity threshold (0-1, higher is more similar)
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of Memory objects sorted by relevance
        """
        # Get embedding for the context
        context_embedding = get_embedding(
            client=self.openai_client,
            text=context
        )
        
        # Search for similar memories with the specified threshold
        response = self.supabase.rpc(
            "match_memories", 
            {
                "query_embedding": context_embedding,
                "match_threshold": threshold,
                "match_count": limit,
                "user_id_filter": self.user_id
            }
        ).execute()
        
        # Convert to Memory objects
        memories = []
        for item in response.data:
            location_data = json.loads(item["location"])
            location = Location(type=location_data["type"], name=location_data["name"])
            
            memory = Memory(
                id=item["id"],
                user_id=item["user_id"],
                created_at=item["created_at"],
                content=item["content"],
                reflection=item["reflection"],
                embedding=item["embedding"],
                emotional_tone=item["emotional_tone"],
                location=location,
                tags=item["tags"]
            )
            memories.append(memory)
        
        return memories

    def _get_emotional_intensity(self, emotional_tone: str) -> float:
        """
        Calculate emotional intensity score based on the emotional tone.
        
        Args:
            emotional_tone: The emotional tone string
            
        Returns:
            Intensity score between 0.0-1.0
        """
        # High intensity emotions (very memorable)
        high_intensity = [
            "ecstatic", "thrilled", "overjoyed", "elated", "excited", "exhilarated",
            "devastated", "heartbroken", "terrified", "furious", "enraged", "shocked",
            "amazed", "astonished", "euphoric", "anguished", "traumatized"
        ]
        
        # Medium intensity emotions
        medium_intensity = [
            "happy", "sad", "angry", "scared", "nervous", "anxious", "proud",
            "disappointed", "frustrated", "worried", "annoyed", "hopeful", 
            "grateful", "relieved", "surprised", "content", "nostalgic"
        ]
        
        # Low intensity emotions (less memorable)
        low_intensity = [
            "calm", "relaxed", "neutral", "indifferent", "bored", "tired",
            "comfortable", "mild", "casual", "ordinary", "okay", "fine",
            "standard", "typical", "usual", "routine", "common"
        ]
        
        # Check if the emotion contains any of these intensity words
        emotion_lower = emotional_tone.lower()
        
        # Use a combination of exact match and substring matching
        if any(emotion == emotion_lower for emotion in high_intensity) or \
           any(word in emotion_lower for word in ["very ", "extremely ", "intensely ", "deeply "]):
            return 1.0  # High intensity
        elif any(emotion == emotion_lower for emotion in medium_intensity):
            return 0.6  # Medium intensity
        elif any(emotion == emotion_lower for emotion in low_intensity):
            return 0.3  # Low intensity
        else:
            return 0.5  # Default to medium-low if unknown

    def filtered_remember(self, context: str, limit: int = 5) -> List[dict]:
        """
        Human-like memory recall that returns partial, filtered memories
        rather than complete memory objects.
        
        Args:
            context: Natural language query or context
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of filtered memory dictionaries with only the most relevant details
        """
        # First get the full memories
        full_memories = self.remember(context, limit=limit)
        
        # Extract query focus to determine what aspects to highlight
        focus_prompt = f"""
        Analyze this memory query and determine what aspect the person is most interested in.
        Query: "{context}"
        Return one of: "emotional", "factual", "location", "time", "general"
        """
        
        focus_response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You determine what aspect of memory someone is asking about."},
                {"role": "user", "content": focus_prompt}
            ]
        )
        
        query_focus = focus_response.choices[0].message.content.strip().lower()
        
        # Process each memory to create a filtered version
        filtered_memories = []
        
        for memory in full_memories:
            # Start with essential elements that are always included
            filtered_memory = {
                "content": memory.content
            }
            
            # Calculate emotional intensity
            intensity = self._get_emotional_intensity(memory.emotional_tone)
            
            # Add elements based on intensity and query focus
            
            # Emotional tone is more likely to be recalled for high-intensity memories
            if intensity > 0.5 or query_focus == "emotional":
                filtered_memory["emotional_tone"] = memory.emotional_tone
            
            # Location is included based on query focus and sometimes randomly
            if query_focus == "location" or intensity > 0.7:
                filtered_memory["location"] = f"{memory.location.type}: {memory.location.name}"
            
            # Tags are included more for factual queries
            if query_focus == "factual" or query_focus == "general":
                # Sometimes include only the most relevant tags, not all
                if len(memory.tags) > 2 and intensity < 0.8:
                    filtered_memory["tags"] = memory.tags[:2]  # Just the first couple tags
                else:
                    filtered_memory["tags"] = memory.tags
            
            # Time information is fuzzy unless specifically asked about
            if query_focus == "time":
                filtered_memory["created_at"] = memory.created_at
            else:
                # Simply use a generic time indicator instead of doing datetime calculations
                # This avoids all the timezone issues
                filtered_memory["when"] = "some time ago"
            
            # Only include reflection for high intensity memories or when focus is emotional
            if intensity > 0.6 or query_focus == "emotional":
                filtered_memory["reflection"] = memory.reflection
                
            # Add an "importance" indicator based on emotional intensity
            if intensity > 0.8:
                filtered_memory["importance"] = "high"
            elif intensity > 0.5:
                filtered_memory["importance"] = "medium"
            else:
                filtered_memory["importance"] = "low"
                
            # Add a "clarity" field that indicates how clearly the memory is recalled
            filtered_memory["clarity"] = "vivid" if intensity > 0.7 else "hazy"
            
            filtered_memories.append(filtered_memory)
        
        return filtered_memories
