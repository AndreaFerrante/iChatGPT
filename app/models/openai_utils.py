import os
import openai
import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForMaskedLM
from models.openaikeys import openai_main_key


##################################################################
client = OpenAI(api_key = openai_main_key)
##################################################################


def ask_gpt(user_query:str=''):

    if user_query == '':
        raise Exception('You must pass a text to be requested to OpenAI !')

    completion = client.completions.create(
        model       ="gpt-3.5-turbo-instruct",
        prompt      = user_query,
        max_tokens  = 2048,
        temperature = 0.8
    )

    return completion.choices[0].text.strip().replace('\n', ' ')


def get_embeddings_openai(text_to_embed, model="text-embedding-ada-002"):

    """
    Obtain embeddings for a given text using OpenAI's API and the specified model.

    Parameters:
    - text_to_embed (str): The input text for which embeddings are to be generated.
                           Newlines will be replaced with spaces for consistent formatting.
    - model (str, optional): The OpenAI model to be used for generating embeddings.
                             Default is "text-embedding-ada-002".

    Returns:
    - list: A list of numerical values representing the embeddings of the input text.

    Example:
    embeddings = get_embeddings_openai("Hello, world!")
    print(type(embeddings))
    <class 'list'>

    Note:
    - You must have the `openai` package installed and properly configured with your API key.
    - The function assumes that the `openai.Embedding.create` method exists and is compatible.
      If OpenAI's API changes, this function may break.
    - The `model` parameter must be a valid OpenAI model that supports text embedding.

    Raises:
    - Whatever exceptions the `openai.Embedding.create` method may raise, such as rate limit
      errors or model incompatibility errors.
    """

    text = text_to_embed.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


def get_embedding_shape_from_opeain(dummy_text_to_embed:str='dummy text to embed.'):

    embeddings = get_embeddings_openai( dummy_text_to_embed )
    return embeddings.shape[1]


def get_chatbot_response(user_request):

    user_request_embedded = get_embeddings_openai(text_to_embed = user_request)
    similarities          = get_similarities_given_all_pdf('', user_request_embedded)
    response              = ask_gpt4(question   = user_request,
                                     content    = similarities)

    return response