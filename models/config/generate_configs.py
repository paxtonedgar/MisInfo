import os, json

with open('lstm/lstm_200d_1x32x1.json') as fp_in:
    cfg = json.load(fp_in)

for num_layers in [1,2,3]:
    for hidden_size in [32,64]:
        for num_directions in [False, True]:
            cfg['cuda'] = True
            cfg['network']['hidden_size'] = hidden_size
            cfg['network']['num_layers'] = num_layers
            cfg['network']['bidirectional'] = num_directions
            cfg['name'] = f'lstm_200d_{num_layers}x{hidden_size}x{int(num_directions)+1}'
            if not os.path.exists(f'{cfg["name"]}.json'):
                with open(f'lstm/{cfg["name"]}.json', 'w') as fp_out:
                    json.dump(cfg, fp_out, indent=2)
