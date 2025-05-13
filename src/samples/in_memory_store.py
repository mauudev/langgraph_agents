import uuid

from langgraph.store.memory import InMemoryStore

in_memory_store = InMemoryStore()
user_id = "1"
namespace_for_memory = (user_id, "memories")
memory_id = str(uuid.uuid4())
memory = {"food_preference": "I like pizza"}
in_memory_store.put(namespace_for_memory, memory_id, memory)
memories = in_memory_store.search(namespace_for_memory)
memory = memories[-1].dict()

print(memory)
