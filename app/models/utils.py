import os
import re
import nltk
import PyPDF2
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


#################################################################
nltk.download('punkt')
nltk.download('stopwords')
#################################################################


def extract_file_extension(file_name:str = ""):

    if file_name == "":
        raise Exception("Attention: pass to function a valid file name.")

    pattern = r'\.([^.]*)$'
    match   = re.search(pattern, file_name)

    if match:
        return str(match.group(1))
    else:
        return None


def spot_url(url_address:str=""):

    if url_address == "":
        raise Exception("Attention: pass to the function a valid URL.")

    url_pattern = re.compile(
    r'^(https?:\/\/)?'  # optional http or https scheme
    r'([\da-z\.-]+)'    # domain name
    r'(\.[a-z\.]{2,6})' # extension
    r'([\/\w \.-]*)*'   # path
    r'\/?'              # trailing slash (optional)
    r'(\?[\/\w \.-]*)?' # query string (optional)
    r'(#\w*)?$',        # anchor tag (optional)
    re.IGNORECASE)

    return bool(url_pattern.match(url_address))


def create_folder_if_not_exist(path_to_create:str=None) -> None:

    if path_to_create is None:
        raise Exception(f'No path passed as parameter "path_to_create" to the function.')

    if not os.path.exists(path_to_create):
        os.makedirs(path_to_create)

    return None


def is_folder_empty(folder_path:str=None):

    if folder_path is None:
        raise Exception(f'No path passed as parameter "folder_path" to the function.')

    # List the contents of the folder
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        return len(os.listdir(folder_path)) == 0
    else:
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist or is not a directory.")


def chunk_text(text:str='', chunk_size:int=100) -> list:


    """Chunks text into smaller sections based on sentence boundaries."""

    if text == '':
        raise Exception(f'Pass text to be chuncked first.')

    try:

        sentences     = sent_tokenize(text)
        chunks        = list()
        current_chunk = ''

        for sentence in sentences:

            if len(current_chunk) + len(sentence) <= chunk_size:

                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    except Exception as ex:
        raise Exception(f'While performing chunking, we saw this issue: {ex}')


def get_dataframe_pdf_content(pdf_path:str=None, chunck_text:bool=False, chunk_size:int=100) -> pd.DataFrame:


    """
        Extracts text content from each page of a PDF file and returns it in a pandas DataFrame.

        Parameters:
        -----------
        pdf_path : str, optional
            The full path to the PDF file. If not provided or an empty string, an exception is raised.
        pdf_name : str, optional
            The name of the PDF file to be included in the DataFrame. If not provided, the column will contain empty strings.

        Returns:
        --------
        pd.DataFrame
            A DataFrame with three columns:
            - 'FileName': The name of the PDF file (or the provided `pdf_name`).
            - 'FilePageFullText': The extracted text content from each page of the PDF.
            - 'FilePageNumber': The corresponding page numbers for each extracted text.

        Raises:
        -------
        Exception
            If `pdf_path` is not provided or if there is an error reading the PDF file.

        Example:
        --------
        >>> df = get_dataframe_pdf_content("example.pdf", "Example PDF")
        >>> print(df.head())
           FileName                                FilePageFullText  FilePageNumber
        0  Example PDF  This is the text content of page 1...                    1
        1  Example PDF  This is the text content of page 2...                    2

        Notes:
        ------
        This function uses the PyPDF2 library to extract text from PDF files. Make sure that the library is installed
        before using this function.
        """

    if pdf_path is None:
        raise Exception("Attention: pass to the function a valid path containing PDFs to embed.")

    all_files_in_path = os.listdir(pdf_path)
    pdf_in_path       = [file for file in all_files_in_path if file.endswith('.pdf')]
    page_text         = list()
    page_number       = list()
    pdf_name          = list()


    for pdf in tqdm(pdf_in_path):

        try:

            with open(pdf_path + pdf, 'rb') as file:

                pdf_reader  = PyPDF2.PdfReader(file)

                # For each page in the PDF extract its text...
                for num_page, page in enumerate(pdf_reader.pages):

                    pdf_page_text = str(page.extract_text())
                    pdf_page_text = pdf_page_text.replace('\n', ' ')

                    # If we want to chunck our page's text, we must "extend" to the number of chuncks our file df...
                    if chunck_text:

                        ###############################################################
                        chuncks = chunk_text(text=pdf_page_text, chunk_size=chunk_size)
                        ###############################################################

                        if len(chuncks):
                            for chunck in chuncks:
                                page_text.append(chunck)
                                page_number.append(int(num_page + 1))
                                pdf_name.append(str(pdf))
                        else:
                            continue

                    else:
                        page_text.append( pdf_page_text )
                        page_number.append( int(num_page + 1) )
                        pdf_name.append( str(pdf) )

                for num_page, page in enumerate(pdf_reader.pages):

                    page_text.append( str(page.extract_text()).replace('\n', ' ') )
                    page_number.append( int(num_page + 1) )
                    pdf_name.append( str(pdf) )


        except Exception as e:
            return str(e)

    return pd.DataFrame({'FileName':         pdf_name,
                         'FilePageChars':    [len(x) for x in page_text],
                         'FilePageFullText': page_text,
                         'FilePageNumber':   page_number})
