from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
import torch
import transformers

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.model_config = config['model']
        self.model = None
        self.tokenizer = None
        if self.model_config['torch_dtype'] == 'float16':
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
        self.new_model = self.model_config['new_name'] 
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant chatbot."}
        ]

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config['name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = self.model_config['padding_side']
        return self.tokenizer

    def load_model(self, quant_config):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config['name'],
            torch_dtype=torch.float16,
            quantization_config=quant_config
        )
        self.model.config.use_cache = False
        return self.model

    def prepare_peft_model(self):
        self.peft_config = LoraConfig(**self.model_config['peft'])
        return self.peft_config
    
    def prepare_model(self, quant_config):
        self.load_tokenizer()
        self.load_model(quant_config)
        self.prepare_peft_model()
        return self.model, self.tokenizer, self.peft_config
    
    def save_uploader(self, hf_token):
        # Save artifacts
        model_name = self.model_config['name']

        # Reload model in FP16 (instead of NF4)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            return_dict=True,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Merge base model with the adapter
        model = PeftModel.from_pretrained(base_model, "final_checkpoint")
        model = model.merge_and_unload()

        # Save model and tokenizer
        model.save_pretrained(self.new_model)
        tokenizer.save_pretrained(self.new_model)

        # Push them to the HF Hub
        model.push_to_hub(self.new_model, use_temp_dir=False, token=hf_token)
        tokenizer.push_to_hub(self.new_model, use_temp_dir=False, token=hf_token)
        
        
    def inference(self):
        # Format prompt
        user_current_input = input("Enter your message: ")
        self.messages.append({"role": "user", "content": user_current_input})
        tokenizer = AutoTokenizer.from_pretrained(self.new_model)
        prompt = tokenizer.apply_chat_template(self.messages, add_generation_prompt=True, tokenize=False)

        # Create pipeline
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.new_model,
            tokenizer=tokenizer
        )

        # Generate text
        sequences = pipeline(
            prompt,
            do_sample=self.config['inference']['do_sample'],
            temperature=self.config['inference']['temperature'],
            top_p=self.config['inference']['top_p'],
            num_return_sequences=self.config['inference']['num_return_sequences'],
            max_length=self.config['inference']['max_length'],
        )
        self.messages.append({"role": "assistant", "content": sequences[0]['generated_text']})
        return user_current_input, sequences[0]['generated_text']
        