from src.core.config import JSON_PATH
import json 
def loading_dataset(json_path: str = JSON_PATH): 
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(
            "Expected the dataset JSON to be an object keyed by image id."
        )

    print(f"Total samples: {len(data)}")

    first_key = next(iter(data), None)

    if first_key is None:
        print("Dataset is empty.")
        return data

    print("Sample Key:")
    print(first_key)

    print("\nSample Content:")
    print(json.dumps(data[first_key], indent=4, ensure_ascii=False))
    return data 
