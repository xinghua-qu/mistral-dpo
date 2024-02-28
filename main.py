import argparse
from typing import NoReturn
from modules import DataManager, load_config, ModelManager, CustomDPOTrainer, init_wandb
from peft import LoraConfig
from transformers import BitsAndBytesConfig
import wandb
from datetime import datetime


def main(cfg_file: str) -> NoReturn:
    """Main function to initialize model, load data, and start the training process.
    
    Args:
        cfg_file (str): Path to the configuration file.
    """
    config = load_config(cfg_file)
    init_wandb(config, cfg_file)
    
    # Initialize model with configuration
    quant_config = BitsAndBytesConfig(quantize=True, bits=4)
    model_manager = ModelManager(config)
    model, tokenizer, peft_config = model_manager.prepare_model(quant_config)
    
    # Load and format dataset
    data_loader = DataManager(tokenizer, config)
    dataset = data_loader.load_format_data()
    
    # Setup custom training and execute
    custom_trainer = CustomDPOTrainer(model, config, dataset, tokenizer, peft_config)
    custom_trainer.train()
    custom_trainer.save_flush()
    
    model_manager.save_uploader(config['tokens']['hf'])
    
    while True:
        question, answer = model_manager.inference()
        print(f"Q: {question}\n A: {answer}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run the training process.")
    parser.add_argument("--cfg", type=str, default= 'config/mistral-7b-dpo.yaml', help="Path to the configuration file")
    args = parser.parse_args()
    main(args.cfg)