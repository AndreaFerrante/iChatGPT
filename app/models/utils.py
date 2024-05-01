import re
import PyPDF2


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


def scrape_pdf_content(pdf_path:str = ""):

    if pdf_path == "":
        raise Exception("Attention: pass to the function a valid path tp the PDF.")

    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            all_text = ''
            
            for page in pdf_reader.pages:
                all_text += page.extract_text() + '\n'
            
            return all_text
        
    except Exception as e:
        return str(e)
    
