import os
import re
import PyPDF2
import pandas as pd
from tqdm import tqdm

def read_all_pdf_in_path(pdf_path:str) -> pd.DataFrame:

    pdf_filename    = list()
    pdf_page_number = list()
    pdf_page_text   = list()

    print('Reading all PDF at the given path...')

    for pdf in tqdm( os.listdir(pdf_path) ):
        if pdf.endswith('pdf'):
            with open(pdf_path + pdf, 'rb') as file:
                #######################################
                pdf_reader = PyPDF2.PdfFileReader(file)
                #######################################
                for page_num in range(pdf_reader.numPages):
                    page = pdf_reader.getPage(page_num)

                    text = page.extract_text()
                    text = text.replace('\n', ' ')
                    text = re.sub(' +', ' ', text) # Clean all double spaces...

                    pdf_page_text.append(text.strip())
                    pdf_page_number.append(page_num + 1)
                    pdf_filename.append(pdf)
        else:
            print(f'Skipping file named {pdf} because it is not a PDF !')

    return pd.DataFrame({'FileName':           pdf_filename,
                         'FileNamePageNumber': pdf_page_number,
                         'FilePageFullText':   pdf_page_text})


# all_pages = read_all_pdf_in_path(pdf_path=os.getcwd() + '/RoboChatter/pdfs/')
