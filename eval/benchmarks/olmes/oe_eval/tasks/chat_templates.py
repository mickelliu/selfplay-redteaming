# Functions copied from https://github.com/allenai/open-instruct/blob/main/eval/templates.py


def create_prompt_with_tulu_chat_format(messages, tokenizer, bos="<s>", eos="</s>", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                    message["role"]
                )
            )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text


# weirdness with olmo tokenizer means IP_ADDR is the eos token.
def create_prompt_with_olmo_chat_format(
    messages, tokenizer, bos="|||IP_ADDRESS|||", eos="|||IP_ADDRESS|||", add_bos=True
):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
        else:
            raise ValueError(
                "Olmo chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                    message["role"]
                )
            )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text  # forcibly add bos
    return formatted_text


def create_prompt_with_llama2_chat_format(messages, tokenizer, bos="<s>", eos="</s>", add_bos=True):
    """
    This function is adapted from the official llama2 chat completion script:
    https://github.com/facebookresearch/llama/blob/7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e/llama/generation.py#L274
    """
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    B_INST, E_INST = "[INST]", "[/INST]"
    formatted_text = ""
    # If you want to include system prompt, see this discussion for the template: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/discussions/4
    # However, see here that removing the system prompt actually reduce the false refusal rates: https://github.com/facebookresearch/llama/blob/main/UPDATES.md?utm_source=twitter&utm_medium=organic_social&utm_campaign=llama2&utm_content=text#observed-issue
    if messages[0]["role"] == "system":
        assert (
            len(messages) >= 2 and messages[1]["role"] == "user"
        ), "LLaMa2 chat cannot start with a single system message."
        messages = [
            {
                "role": "user",
                "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
            }
        ] + messages[2:]
    for message in messages:
        if message["role"] == "user":
            formatted_text += bos + f"{B_INST} {(message['content']).strip()} {E_INST}"
        elif message["role"] == "assistant":
            formatted_text += f" {(message['content'])} " + eos
        else:
            raise ValueError(
                "Llama2 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                    message["role"]
                )
            )
    # The llama2 chat template by default has a bos token at the start of each user message.
    # The next line removes the bos token if add_bos is False.
    formatted_text = formatted_text[len(bos) :] if not add_bos else formatted_text
    return formatted_text


def create_prompt_with_xwin_chat_format(messages, tokenizer, bos="<s>", eos="</s>", add_bos=True):
    """
    This function is adapted from the official xwin chat completion script:
    https://huggingface.co/Xwin-LM/Xwin-LM-70B-V0.1
    """
    formatted_text = "A chat between a curious user and an artificial intelligence assistant. "
    formatted_text += (
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    )
    for message in messages:
        if message["role"] == "user":
            formatted_text += "USER: " + message["content"] + " "
        elif message["role"] == "assistant":
            formatted_text += "ASSISTANT: " + message["content"] + eos
    formatted_text += "ASSISTANT:"
    return formatted_text


def create_prompt_with_zephyr_chat_format(messages, tokenizer, bos="<s>", eos="</s>", add_bos=True):
    """
    This function is adapted from the official zephyr chat completion script:
    https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
    """
    formatted_text = ""
    # if messages[0]["role"] != "system":
    #     messages = [{
    #         "role": "system",
    #         "content": ""
    #     }] + messages

    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + eos + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + eos + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"] + eos + "\n"
        else:
            raise ValueError(
                "Zephyr chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                    message["role"]
                )
            )
    formatted_text += "<|assistant|>\n"
    return formatted_text

def create_prompt_with_llama3_cot_chat_format(messages, tokenizer, bos="<s>", eos="</s>", add_bos=True):
    
    DEFENDER_INSTRUCTION_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.\n\nUser: {user_query}<|eot_id|>\n"""
    
    assert len(messages) == 1
    assert messages[0]["role"] == "user"

    cot_message = DEFENDER_INSTRUCTION_PROMPT.format(user_query=messages[-1]["content"])
    formatted_text = "<|start_header_id|>user<|end_header_id|>\n\n" + cot_message + "<|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step.\n<think> "

    return formatted_text


CHAT_TEMPLATES = {
    "tulu": create_prompt_with_tulu_chat_format,
    "olmo": create_prompt_with_olmo_chat_format,
    "llama2": create_prompt_with_llama2_chat_format,
    "xwin": create_prompt_with_xwin_chat_format,
    "zephyr": create_prompt_with_zephyr_chat_format,
    "llama3_cot": create_prompt_with_llama3_cot_chat_format,
}
