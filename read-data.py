import pickle

with open('/mnt/2eb9e109-0bb6-41db-a49a-483d3806fe10/xy-ws/unitree-g1-ws/TWIST2/assets/example_motions/0807_yanjie_walk_001.pkl', 'rb') as f:
    data = pickle.load(f)

print("Keys in the loaded data:", data)
