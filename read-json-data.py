import json

with open("deploy_real/deploy_real/twist2_demonstration/20260113_1515/episode_0000/data.json", "r") as f:
    data = json.load(f)

print("Keys in the loaded data:", data['data'][0].keys())