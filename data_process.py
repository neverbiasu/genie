import json

with open('data/pokemon.json', 'r', encoding='utf-8') as f:
    raw_data = []
    for line in f:
            raw_data.append(json.loads(line))

formatted_data = []
for item in raw_data:
    formatted_item = {
        "system": item.get("instruction", ""),
        "query": item.get("input", ""),
        "response": item.get("response", "")
    }
    formatted_data.append(formatted_item)

with open('data/pokemon_train.json', 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=4)

print("数据已成功保存到 data/pokemon_train.json")
