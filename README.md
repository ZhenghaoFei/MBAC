
# MBAC

modified from https://github.com/miyosuda/async_deep_reinforce

run
```bash
python a3c.py
```

config agent at constants.py
config imagination at network_parameter.py
MBAC network at game_ac_network.py

# forward port in gcloud
gcloud compute ssh fei_holly@tensorflow
gcloud compute ssh --ssh-flag="-N -f -L localhost:12345:localhost:12345" fei_holly@tensorflow
