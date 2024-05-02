from openai import OpenAI


class OpenAIAssistant(object):

    def __init__(self, openai_api_key:str="") -> None:
        
        if openai_api_key == "":
            raise Exception("Attention: pass to OpenAIAssistant class the OpenAI key")
        
        super().__init__()
        self.openai_api_key = openai_api_key
        self.__get_openai_client()

    def __call__(self, openai_api_key:str="") -> None:
        
        if openai_key == "":
            raise Exception("Attention: pass to OpenAIAssistant class the OpenAI key")
        
        self.openai_api_key = openai_api_key
        self.__get_openai_client()
    
    def __get_openai_client(self) -> None:
        self.client = OpenAI(api_key = self.openai_api_key)

    def ask_gpt(self, 
                user_query:str       = "", 
                return_object:bool   = False,
                max_tokens:str       = 2048,
                temperature:float    = 0.85,
                model:str            = "gpt-3.5-turbo-0125",
                system_content:str   = "You are a very accurate AI assistant that answers like if its whole life depends on it."):

        """
        Sends a user query to the OpenAI API using a specified model and configuration, and returns the response.

        Parameters:
            user_query (str): The user's query to send to the OpenAI model. If this parameter is empty, an exception is raised.
            return_object (bool): If True, the full response object from the API is returned. Otherwise, only the text of the response is returned.
            max_tokens (int): The maximum number of tokens to generate in the response. Defaults to 2048.
            temperature (float): The creativity temperature of the response. Defaults to 0.85.
            model (str): The identifier of the model to use for generating the response. Defaults to 'gpt-3.5-turbo-0125'.
            system_content (str): Contextual message to inform the AI model's response style and content. If empty, an exception is raised.

        Returns:
            str or dict: Depending on the `return_object` flag, returns either the plain text response or the complete response object.

        Raises:
            Exception: If `user_query`, `system_content`, or `model` is empty, raises an exception indicating the missing required parameter.

        Notes:
            This function interacts with the OpenAI API's chat completion endpoint and requires appropriate API access and client setup.
        """

        if user_query == "":
            raise Exception('Attention: you must pass a query to be asked to OpenAI')
        
        if system_content == "":
            raise Exception('Attention: you must pass a system content to be used with OpenAI')

        if model == "":
            raise Exception('Attention: you must pass a model to used for OpenAI')

        completion = self.client.chat.completions.create(
                    model    = model,
                    messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_query}
                    ],
                    max_tokens  = max_tokens,
                    temperature = temperature,
                    stream      = False
        )

        if return_object:
            return completion

        return completion.choices[0].message.content

    def get_embeddings_from_openai(self, 
                                   return_object:bool = False,
                                   text_to_embed:str  = "", 
                                   embedding_model    = "text-embedding-3-small"):

        """
        Retrieves embeddings for a specified text using a specific OpenAI embedding model.

        Parameters:
            return_object (bool): If True, the full response object is returned. Otherwise, the embedding data is returned.
            text_to_embed (str): The text for which embeddings are to be generated. If empty, an exception is raised.
            embedding_model (str): The model to use for generating embeddings. Defaults to "text-embedding-3-small". If empty, an exception is raised.

        Returns:
            list or dict: Depending on `return_object`, returns either the embedding vector as a list or the full response object.

        Raises:
            Exception: If `text_to_embed` or `embedding_model` is empty, raises an exception indicating that the necessary parameters must be provided.

        Notes:
            This function interacts with the OpenAI API's embeddings endpoint and formats the text by replacing newline and tab characters with spaces to ensure consistent input formatting.
        """
        
        if text_to_embed == "":
            raise Exception("Attention: pass a text to be embedded using OpenAI.")
        
        if embedding_model == "":
            raise Exception("Attention: pass an embedding model to OpenAI.")

        response = self.client.embeddings.create(
            input = text_to_embed.replace("\n", " ").replace("\t", " "),
            model = embedding_model
        )

        if return_object:
            return response

        return response.data[0].embedding



