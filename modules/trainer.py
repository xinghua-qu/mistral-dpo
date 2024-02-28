from transformers import TrainingArguments, Trainer
from trl import DPOTrainer
from transformers import BitsAndBytesConfig
import gc
import torch

class CustomDPOTrainer:
    def __init__(self, model, config, dataset, tokenizer, peft_config):
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.peft_config = peft_config
        self.config = config
        self.training_cfg = config['training']
        self.quant_config = BitsAndBytesConfig(quantize=True, bits=4)
        # Training arguments
        self.training_args = TrainingArguments(
            per_device_train_batch_size=int(self.training_cfg['per_device_train_batch_size']),
            gradient_accumulation_steps=int(self.training_cfg['gradient_accumulation_steps']),
            gradient_checkpointing=self.training_cfg['gradient_checkpointing'],
            learning_rate=self.training_cfg['learning_rate'],
            lr_scheduler_type=self.training_cfg['lr_scheduler_type'],
            max_steps=self.training_cfg['max_steps'],
            save_strategy=self.training_cfg['save_strategy'],
            logging_steps=self.training_cfg['logging_steps'],
            output_dir=self.training_cfg['output_dir'],
            optim=self.training_cfg['optim'],
            warmup_steps=int(self.training_cfg['warmup_steps']),
            bf16=self.training_cfg['bf16'],
            report_to=self.training_cfg['report_to'],
            remove_unused_columns=self.training_cfg['remove_unused_columns'],
        )

    def train(self):
        self.dpo_trainer = DPOTrainer(
            self.model,
            args=self.training_args,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
            peft_config=self.peft_config,
            beta=self.training_cfg['beta'],
            max_prompt_length=self.training_cfg['max_prompt_length'],
            max_length=self.training_cfg['max_length'],
        )
        self.dpo_trainer.train()
        return self.dpo_trainer
    
    def save_flush(self):
        # Save artifacts
        self.dpo_trainer.model.save_pretrained("final_checkpoint")
        self.tokenizer.save_pretrained("final_checkpoint")

        # Flush memory
        del self.dpo_trainer, self.model
        gc.collect()
        torch.cuda.empty_cache()