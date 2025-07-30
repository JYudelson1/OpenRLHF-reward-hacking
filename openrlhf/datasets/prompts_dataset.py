from torch.utils.data import Dataset
from tqdm import tqdm
import json


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None, multiturn=False) -> str:
    if not multiturn:
        if apply_chat_template:
            chat = data[input_key]
            if isinstance(chat, str):
                chat = [{"role": "user", "content": chat}]
            if isinstance(chat, dict):
                chat = [chat]
            prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        else:
            prompt = data[input_key]
            if input_template:
                prompt = input_template.format(prompt)
    else:
        prompt = ""

    if multiturn:
        full_data = data
    else:
        full_data = None

    return prompt, full_data, data.get("solution", None)


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        multiturn = vars(self.strategy.args).get("env_makers", False)
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, full_data, solution = preprocess_data(
                data, input_template, input_key, apply_chat_template, multiturn
            )
            data_entry = {
                "prompts": prompt,
            }
            if full_data is not None:
                data_entry["full_data"] = full_data
                if full_data.get("input_output", None) is not None:
                    unit_tests_json = json.loads(full_data["input_output"])
                    unit_tests_json = {
                        "inputs": unit_tests_json["inputs"][:128],
                        "outputs": unit_tests_json["outputs"][:128],
                    }
                    full_data["input_output"] = json.dumps(unit_tests_json)
                    del full_data["solutions"]
            if solution is not None:
                data_entry["solution"] = solution
            self.prompts.append(data_entry)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
