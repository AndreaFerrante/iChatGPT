import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.openaikeys import openai_key
from models.openaiassistant import OpenAIAssistant


#################################################################
openAIAssistant = OpenAIAssistant(openai_api_key=openai_key)
#################################################################


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


def get_pdf_dataframe_embeddings(pdfs_in_path:pd.DataFrame=None, return_norm_embeddings:bool=False):

    ##############################################################
    if pdfs_in_path is None:
        raise Exception('Pass a DataFrame with all the PDFs read.')
    ##############################################################

    if 'FilePageFullText' not in pdfs_in_path.columns:
        raise Exception('Attention, a column named FilePageFullText is not present.')

    try:

        page_embeddings = list()
        for page in tqdm( pdfs_in_path['FilePageFullText'] ):
            page_embeddings.append( openAIAssistant.get_embeddings_from_openai(text_to_embed=str(page)) )

        page_embeddings            = np.array(page_embeddings).squeeze()
        normalized_page_embeddings = __normalize_vectors(page_embeddings)
        pdfs_in_path['FilePageEmbeddings'] = [list(x) for x in normalized_page_embeddings]

        if return_norm_embeddings:
            return pdfs_in_path, normalized_page_embeddings

        return pdfs_in_path

    except Exception as ex:
        raise Exception(f'While performing embedding calculation on DataFrame, this error occured: {ex}')


def search_a_query_in_docs_with_faiss(norm_embs=None, query="", dataframe_pdfs=None, k_closest=None):

    if query == '':
        raise Exception('Pass a query to embed and search')

    if norm_embs is None or dataframe_pdfs is None:
        raise Exception('Pass a dataframe of all PDF read and the normalized page embeddings !')

    if 'FilePageFullText' not in dataframe_pdfs.columns:
        raise Exception('Attention, column named FilePageFullText is not in the dataframe of all PDFs scraped ! Pass it.')

    # 1. Build FAISS index (use Inner Product Similarity to equate CosineSimilarity when vectors are normalized)
    index = faiss.IndexFlatL2(norm_embs.shape[1])
    index.add(norm_embs)

    # 2. Embed the query
    query_embedding            = np.array(openAIAssistant.get_embeddings_from_openai(text_to_embed=query)).squeeze()
    normalized_query_embedding = __normalize_vectors(np.array([query_embedding]))

    ####################################################################################################################
    # 3. Perform search to find close page/pages ...
    if k_closest is None:
        k_closest = len(dataframe_pdfs)
    D, I = index.search(normalized_query_embedding, k_closest)

    return D, I
    # 4. Output the sentence that is most similar to the query
    closest_pages     = list(dataframe_pdfs['FilePageFullText'])[I[0][0]]
    cosine_similarity = D[0][0]
    ####################################################################################################################

    print(f"The matrix distance is: {D[0]} \n")
    print(f"The index is: {I[0]} \n")
    print(f"The page/pages most similar to the query is: '{closest_pages}' \n")
    print(f"The cosine similarity for the most similar page is: '{cosine_similarity}'")

    return closest_pages, cosine_similarity
