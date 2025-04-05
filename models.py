from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional
from datetime import datetime
import uuid

LocationType = Literal["mental", "physical", "digital", "described"]

@dataclass
class Location:
    type: LocationType
    name: str

@dataclass
class Memory:
    user_id: str
    content: str
    reflection: str
    embedding: List[float]
    emotional_tone: str
    location: Location
    tags: List[str]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())
