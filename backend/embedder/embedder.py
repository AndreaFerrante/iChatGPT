import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from fastapi import HTTPException
from closeai.openaikeys import openai_main_key
from closeai.openaiassistant import OpenAIAssistant


#################################################################
openAIAssistant = OpenAIAssistant(openai_api_key=openai_main_key)
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


def get_pdf_dataframe_embeddings(pdfs_in_path:pd.DataFrame=None, return_norm_embeddings:bool=True):

    ############################################################################################
    if pdfs_in_path is None:
        return HTTPException(status_code=500, detail='Pass a DataFrame with all the PDFs read.')
    ############################################################################################

    if 'FilePageFullText' not in pdfs_in_path.columns:
        return HTTPException(status_code=500, detail=f"An error occurred: no 'FilePageFullText' column present.")

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
        return HTTPException(status_code=500, detail=f"While performing embedding calculation on DataFrame, this error occured: {ex}")


def search_a_query_in_docs_with_faiss(norm_embs:np.array          = None,
                                      query:str                   = "",
                                      dataframe_pdfs:pd.DataFrame = None,
                                      k_closest:int               = None,
                                      return_D_I:bool             = False):

    if query == '':
        raise Exception('Pass a query to embed and search')

    if norm_embs is None or dataframe_pdfs is None:
        raise Exception('Pass a dataframe of all PDF read and the normalized page embeddings !')

    if 'FilePageFullText' not in dataframe_pdfs.columns:
        raise Exception('Attention, column named FilePageFullText is not in the dataframe of all PDFs scraped.')

    # 1. Build FAISS index (use Inner Product Similarity to equate CosineSimilarity when vectors are normalized)
    index = faiss.IndexFlatIP(norm_embs.shape[1])
    index.add(norm_embs)

    # 2. Embed the query
    query_embedding            = np.array(openAIAssistant.get_embeddings_from_openai(text_to_embed=query)).squeeze()
    normalized_query_embedding = __normalize_vectors(np.array([query_embedding]))

    ####################################################################################################################
    # 3. Perform search to find close page/pages ...
    if k_closest is None:
        k_closest = len(dataframe_pdfs)
    D, I = index.search(normalized_query_embedding, k_closest)

    if return_D_I:
        return D, I

    # 4. Find the text, the pages and the file names where the answer lies more likely ...
    text_   = [str(x) for x in dataframe_pdfs.iloc[I[0]]['FilePageFullText']]
    pages_  = [str(x) for x in dataframe_pdfs.iloc[I[0]]['FilePageNumber']]
    files_  = [str(x) for x in dataframe_pdfs.iloc[I[0]]['FileName']]

    text_   = ' '.join(text_)
    pages_  = ' '.join(pages_)
    files_  = ' '.join(files_)

    ####################################################################################################################

    return text_, pages_, files_


# pdf_path            = 'C:/Users/WKS/Downloads/'
# pdf_df              = pd.read_csv('C:/Users/WKS/Downloads/pdf_df.csv', sep=';')
# norm_embeds         = np.load('C:/Users/WKS/Downloads/norm_embeds.npy')
# # pdf_df              = get_dataframe_pdf_content(pdf_path = pdf_path, chunck_text=True)
# # pdf_df, norm_embeds = get_pdf_dataframe_embeddings(pdfs_in_path=pdf_df, return_norm_embeddings=True)
#
#
# query = "In the middle of a project, a new requirement was added to the scope. The business analyst must determine if" + \
#         (" any impacts, dependencies, or risks are associated with the addition to the scope. " + \
#          "What task should the business analyst perform in order to identify these impacts? " + \
#          "The answer options are: " + \
#          "A. Manage requirements traceability. " + \
#          "B. Manage assumptions and constraints. " + \
#          "C. Manage solution scope. " + \
#          "D. Manage requirements prioritization.")
# t, p, f = search_a_query_in_docs_with_faiss(norm_embs      = norm_embeds,
#                                           query          = query,
#                                           dataframe_pdfs = pdf_df,
#                                           k_closest      = 10,
#                                           return_D_I     = False)
#
# content = 'Answer this question: ' + query + '. To answer the question use only this text: ' + t
# final   = openAIAssistant.ask_gpt(user_query=content)


