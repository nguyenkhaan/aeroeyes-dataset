from src.core.config import JSON_PATH
import json 
def loading_dataset(): 
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    first_key = next(iter(data))

    print("Sample Key:")
    print(first_key)

    print("\nSample Content:")
    print(json.dumps(data[first_key], indent=4))
    return data 