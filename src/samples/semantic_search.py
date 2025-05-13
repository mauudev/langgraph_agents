import uuid
from operator import add
from typing import Annotated, TypedDict

from langchain.embeddings import init_embeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict


class UserPreferences(TypedDict):
    context: Annotated[list[str], add]
    food_preferences: Annotated[list[str], add]
    drink_preferences: Annotated[list[str], add]


user_id = "1"
namespace_for_memory = (user_id, "memories")

in_memory_store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
        "fields": ["food_preferences", "drink_preferences", "$"],
    }
)
checkpointer = InMemorySaver()


in_memory_store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {"food_preferences": ["I love Italian cuisine"], "context": ["Discussing dinner plans"]},
    index=["food_preferences", "drink_preferences"],  # Only embed "food_preferences" field
)

in_memory_store.put(namespace_for_memory, str(uuid.uuid4()), {"system_info": "Last updated: 2024-01-01"}, index=False)
in_memory_store.put(namespace_for_memory, str(uuid.uuid4()), {"food_preferences": ["pizza"]})
in_memory_store.put(namespace_for_memory, str(uuid.uuid4()), {"food_preferences": ["burgers"]})
in_memory_store.put(namespace_for_memory, str(uuid.uuid4()), {"drink_preferences": ["coffee"]})
in_memory_store.put(namespace_for_memory, str(uuid.uuid4()), {"drink_preferences": ["whisky"]})

# memories = in_memory_store.search(namespace_for_memory, query="What does the user like to eat?", limit=3)
# memories = in_memory_store.search(namespace_for_memory, query="What does the user like to drink?", limit=3)
memories = in_memory_store.search(
    namespace_for_memory, query="User's favorite drinks, beverages, or things they like to drink", limit=3
)

# memories = [m for m in memories if m.score > 0.4]  # This is a threshold for relevance, define it as you see fit

for memory in memories:
    print(memory.dict())

for memory in memories:
    in_memory_store.delete(namespace_for_memory, memory.key)
