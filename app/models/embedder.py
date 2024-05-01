import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.openai import get_embeddings_openai


def __vector_direction(vector) -> float:

    '''
    Compute the angle between a given vector and the unit vector along the first coordinate axis.

    Parameters: vector (numpy.ndarray): An n-dimensional numpy vector.
    Returns: float: The angle (in radians) between the input vector and the unit vector along the first coordinate axis.
    '''

    reference_vector     = np.zeros_like(vector)
    reference_vector[0]  = 1
    dot_product          = np.dot(vector, reference_vector)
    magnitude_vector     = np.linalg.norm(vector)
    magnitude_ref_vector = np.linalg.norm(reference_vector)

    return np.arccos(dot_product / (magnitude_vector * magnitude_ref_vector))


def __cosine_similarity(vec_1:np.array, vec_2:np.array) -> float:

    '''
    This function performs cosine similarity in numpy.

    :param vec_1: first vector to assess
    :param vec_2: first vector to assess
    :return: a float between 0 and 1 to that is the cosine between two vectors
    '''

    den  = np.dot(vec_1,vec_2)
    norm = np.linalg.norm(vec_1) * np.linalg.norm(vec_2)

    return float(den / norm)


def __normalize_vectors(vectors):

    """
       Normalize a set of vectors.

       Parameters:
       - vectors (numpy.ndarray): A 2D NumPy array where each row is a vector that you want to normalize.

       Returns:
       - numpy.ndarray: A 2D NumPy array of the same shape as the input, where each row is the normalized vector of the corresponding row in the input array.

       Example:
       import numpy as np
       vectors = np.array([[1, 2], [3, 4], [5, 6]])
       normalized_vectors = normalize_vectors(vectors)
       print(normalized_vectors)
       [[0.4472136  0.89442719]
        [0.6        0.8       ]
        [0.6401844  0.76822128]]

       Dependencies:
       - This function requires NumPy to be installed (`pip install numpy`).

       Notes:
       - The function uses the L2 norm (Euclidean norm) for normalization.
       - If a vector has a norm of zero, the function will return nan values for that vector due to division by zero.

       Raises:
       - ValueError: If the input is not a 2D NumPy array or if the array is empty.
       - numpy.linalg.LinAlgError: If the computation of the norm fails for numerical reasons.
    """

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def __initialize_bert(bert_model:str="bert-base-uncased"):

    tokenizer = BertTokenizer.from_pretrained( bert_model )
    model     = BertModel.from_pretrained( bert_model )

    return tokenizer, model


def __get_embeddings_using_bert(text:str):

    ######################################
    tokenizer, model = __initialize_bert()
    ######################################

    inputs  = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


def __initialize_bert_roberta(data_model_path):

    tokenizer = AutoTokenizer.from_pretrained(os.getcwd() + data_model_path)
    model     = AutoModelForMaskedLM.from_pretrained(os.getcwd() + data_model_path)

    return tokenizer, model


def __get_embeddings_using_xml_roberta(model, tokenizer, text, return_reshaped):

    encoded_input = tokenizer(str(text).lower(), return_tensors='pt')
    output        = model(**encoded_input)

    if return_reshaped:
        embeddings = output[0].mean(dim=1)[0]
    else:
        embeddings = output[0]

    return embeddings


def search_a_query_in_docs_with_faiss(normalized_page_embeddings = None,
                                      dataframe_pdfs             = None,
                                      use_openai                 = True,
                                      print_output               = True,
                                      query                      = '',
                                      k_closest                  = 1):

    if query == '':
        raise Exception('Pass a query to embed and search')

    if normalized_page_embeddings is None or dataframe_pdfs is None:
        raise Exception('Pass a dataframe of all PDF read and the normalized page embeddings !')

    if 'FilePageFullText' not in dataframe_pdfs.columns:
        raise Exception('Attention, column named FilePageFullText is not in the dataframe of all PDFs scraped ! Pass it.')

    # 1. Build FAISS index (use Inner Product Similarity to equate CosineSimilarity when vectors are normalized)
    index = faiss.IndexFlatIP(normalized_page_embeddings.shape[1])
    index.add(normalized_page_embeddings)

    # 2. Embed the query
    if use_openai:
        query_embedding            = np.array(get_embeddings_openai(query)).squeeze()
        normalized_query_embedding = __normalize_vectors(np.array([query_embedding]))
    else:
        query_embedding            = np.array(__get_embeddings_using_bert(query)).squeeze()
        normalized_query_embedding = __normalize_vectors(np.array([query_embedding]))

    ####################################################################################################################
    # 3. Perform search to find close page/pages ...
    D, I = index.search(normalized_query_embedding, k_closest)

    # 4. Output the sentence that is most similar to the query
    closest_pages     = list(dataframe_pdfs['FilePageFullText'])[I[0][0]]
    cosine_similarity = D[0][0]
    ####################################################################################################################

    if print_output:
        print(f"The matrix distance is: {D}")
        print(f"The index is: {I}")
        print(f"The page/pages most similar to the query is: '{closest_pages}'")
        print(f"The cosine similarity is: '{cosine_similarity}'")

    return closest_pages, cosine_similarity


def get_pdf_dataframe_embeddings(all_pdf_in_path:pd.DataFrame=None, path_to_embed:str='', use_openai:bool=True):

    # path_to_embed   = os.getcwd() + '/RoboChatter/pdfs/'
    # use_openai      = False
    # all_pdf_in_path = None

    #########################################################
    if all_pdf_in_path is None and path_to_embed != '':
        all_pdf_in_path = read_all_pdf_in_path(path_to_embed)
    else:
        raise Exception('Pass a path to read PDFs from !')
    #########################################################

    if 'FilePageFullText' not in all_pdf_in_path.columns:
        raise Exception('Attention, column named FilePageFullText is not present.')

    print('Processing embeddings for each and single page...')
    page_embeddings = list()

    if use_openai:
        for page in tqdm( all_pdf_in_path['FilePageFullText'] ):
            page_embeddings.append( get_embeddings_openai(page) )
    else:
        for page in tqdm( all_pdf_in_path['FilePageFullText'] ):
            page_embeddings.append( __get_embeddings_using_bert(page) )

    page_embeddings                   = np.array(page_embeddings).squeeze()
    normalized_page_embeddings        = __normalize_vectors(page_embeddings)
    all_pdf_in_path['PageEmbeddings'] = [list(x) for x in normalized_page_embeddings]

    return all_pdf_in_path, normalized_page_embeddings




