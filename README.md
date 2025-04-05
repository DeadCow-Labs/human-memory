# Memory SDK

A Python SDK for storing structured memory objects in Supabase using OpenAI's API.

## Installation

```bash
# Clone the repository
git clone https://github.com/deadcow-labs/human-memory.git
cd human-memory

# Install dependencies
pip install -r requirements.txt
```

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
openai>=1.0.0
supabase>=2.0.0
python-dotenv>=1.0.0
```

## Configuration

The SDK requires the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase API key

You can set these in your environment or use a `.env` file with python-dotenv.

## Supabase Setup

Create a `memories` table in your Supabase project with the following schema:

```sql
create table
  public.memories (
    id uuid primary key,
    user_id text not null,
    created_at timestamp with time zone not null,
    content text not null,
    reflection text not null,
    embedding vector(1536) not null,
    emotional_tone text not null,
    location jsonb not null,
    tags text[] not null
  );
```

Enable RLS (Row Level Security) on your table and create appropriate policies.

## Usage

```python
from memory_sdk import MemorySDK

# Initialize the SDK with a user ID
user_id = "user_123"
sdk = MemorySDK(user_id)

# Save a memory from a raw message
raw_message = "Today I made significant progress on my project and feel optimistic about finishing it soon."
memory = sdk.save(raw_message)

# The memory object contains all structured data:
print(f"Content: {memory.content}")
print(f"Reflection: {memory.reflection}")
print(f"Emotional tone: {memory.emotional_tone}")
```

See `example.py` for a complete usage example.

## Memory Structure

Each memory includes:

- `id`: UUID
- `user_id`: provided by the SDK user
- `created_at`: UTC timestamp
- `content`: short summary of the message
- `reflection`: insight about the user from the message
- `embedding`: vector from OpenAI embedding API
- `emotional_tone`: e.g. "hopeful", "anxious"
- `location`: dict with type and name
- `tags`: list of string topics or themes

## License

MIT 