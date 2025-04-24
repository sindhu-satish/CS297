from transformers import AutoTokenizer
from datasets import load_dataset
import ChatmlSpecialTokens

sys_prompt = """
        Respond in the following format:
        <think>
        {}
        </think>
        <answer>
        {}
        </answer>
        """

assistant_prompt = """
    <think>
    {}
    </think>
    <answer>
    {}
    </answer>
        """

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        pad_token=ChatmlSpecialTokens.pad_token.value,
        bos_token=ChatmlSpecialTokens.bos_token.value,
        eos_token=ChatmlSpecialTokens.eos_token.value,
        additional_special_tokens=ChatmlSpecialTokens.list(),
        trust_remote_code=True
    )
    tokenizer.chat_template = """{% for message in messages %}
    {{'
    \n <｜' + message['role'] + '｜>' + '\n' + message['content'] + '
    \n'}}
    {% endfor %}"""
    return tokenizer


def create_chatml_prompt(samples, tokenizer):
    batch = []
    for prompt, cot, answer in zip(samples["prompt"],samples["cot"],samples["answer"]):
        content = [
        {"role": "System", "content": sys_prompt},
        {"role": "User", "content": prompt},
        {"role": "Assistant", "content": assistant_prompt.format(cot, answer)}
        ]
        batch.append(tokenizer.apply_chat_template(content, tokenize=False))
    return {"text": batch}


def preprocess_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    train_dataset = dataset["train"].map(
        create_chatml_prompt,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    test_dataset = dataset["test"].map(
        create_chatml_prompt,
        batched=True,
        remove_columns=dataset["test"].column_names
    )

    return train_dataset, test_dataset


def load_dataset_and_tokenizer(dataset_name, model_name):
    train_dataset, test_dataset = preprocess_dataset(dataset_name)
    tokenizer = load_tokenizer(model_name)
    return train_dataset, test_dataset, tokenizer