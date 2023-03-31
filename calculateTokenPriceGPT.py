
import tiktoken
import openai
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY =  os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# Read the data from file
with open(os.environ.get('FILENAME'), "r") as f:
  input_text = f.read()

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens






example_messages = [
    {
        "role": "system",
        "content": "You are a proposal designer, and your are giving a summary of a proposal to a client, who has a lot of questions about the proposal. You are trying to make the summary as short as possible, while still answering all of the client's questions.",
    },
    {
        "role": "user",
        "content": input_text,
    },
]

print(f"{'model':<20} {'tokens estimated':<20} {'cost estimated':<20}")
for model in ["gpt-3.5-turbo-0301", "gpt-4-0314"]:
    # example token count from the function defined above
    tokensEstimated = num_tokens_from_messages(example_messages, model) 
    
    pricesPerThousands = [0.002, 0.03]
    costPerThousandTokens = pricesPerThousands[0] if model == "gpt-3.5-turbo" else pricesPerThousands[1]
    costEstimated = (tokensEstimated * costPerThousandTokens) / 1000
    # print table to show the tokens and cost for each model
    print(f"{model:<20} {tokensEstimated:<20} {costEstimated:.2f} USD")

    # /* WARNING * /: The execution of this code implies a cost for the OpenAI API. Uncomment at your own risk.
    # example token count from the OpenAI API
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=example_messages,
    #     temperature=0,
    #     max_tokens=1  # we're only counting input tokens here, so let's not waste tokens on the output
    # )
    # print(f'{response["usage"]["prompt_tokens"]} prompt tokens counted by the OpenAI API.')
    # print()