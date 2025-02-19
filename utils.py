from transformers import pipeline , AutoTokenizer
import torch


def tokenize_hfmodel_inputs(input_text,tokeniser_model,device,is_chat=False):
    # sample_text="How can I respond to stupid texts without"
    tokeniser=AutoTokenizer.from_pretrained(tokeniser_model)
    # if(is_chat):
    #     messages = [
    #         {"role": "system", "content": "You are a helpful AI assistant."},
    #         {"role": "user", "content": input_text}
    #     ]
    #     prompt = tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    #     input_ids = tokeniser(prompt, return_tensors="pt")
    tokens=torch.tensor(tokeniser.encode(input_text))
    tokens=torch.tensor(tokens).view(1,tokens.shape[0])
    tokens=tokens.to(device)
    return tokens,tokeniser