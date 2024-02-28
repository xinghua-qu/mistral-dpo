import yaml
import wandb
from datetime import datetime

def load_config(config_path="config.yaml"):
    with open(config_path) as file:
        return yaml.safe_load(file)
    
def init_wandb(config, cfg_file):
    wandb.login(key=config['tokens']['wb'])
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg_file.replace('config/', '').replace('.yaml', '')}_{current_time}"
    wandb.init(project=config['tokens']['wb_project'], name=run_name)
