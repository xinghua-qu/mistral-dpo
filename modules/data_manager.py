from datasets import load_dataset
from transformers import AutoTokenizer# Classes for modularization

class DataManager:
    def __init__(self, tokenizer, config):
        self.dataset_name = config['dataset']['name']
        self.dataset = None
        self.tokenizer = tokenizer

    def load_format_data(self):
        self.dataset = load_dataset(self.dataset_name)['train']
        self.dataset = self.format_dataset()
        return self.dataset

    def format_dataset(self):
        original_columns = self.dataset.column_names
        self.dataset = self.dataset.map(
            self.chatml_format,
            remove_columns=original_columns,
            fn_kwargs={'tokenizer': self.tokenizer}
        )
        return self.dataset
    
    @staticmethod
    def chatml_format(example, tokenizer):
        # Format system
        if len(example['system']) > 0:
            message = {"role": "system", "content": example['system']}
            system = tokenizer.apply_chat_template([message], tokenize=False)
        else:
            system = ""

        # Format instruction
        message = {"role": "user", "content": example['question']}
        prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

        # Format chosen answer
        chosen = example['chosen'] + "<|im_end|>\n"

        # Format rejected answer
        rejected = example['rejected'] + "<|im_end|>\n"

        return {
            "prompt": system + prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
