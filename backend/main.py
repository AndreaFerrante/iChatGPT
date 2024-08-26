import os
import uvicorn
from embedder.embedder import *
from backend.models.api_models import *
from fastapi.responses import JSONResponse
from openaiassistant import OpenAIAssistant
from closeai.openaikeys import openai_main_key
from fastapi import FastAPI, File, UploadFile, HTTPException
from backend.utils.utils import create_folder_if_not_exist, clear_all_files_in_folder
from backend.utils.utils import is_folder_empty,get_dataframe_pdf_content


#######################################################################
UPLOAD_DIR        = './uploads/'
PDFS              = None
NORM_EMBEDDINGS   = None
app               = FastAPI()
openAIBot         = OpenAIAssistant(openai_api_key=openai_main_key)
#######################################################################


@app.post("/api/upload")
async def upload_file(files: list[UploadFile] = File(...)):

    '''
    This is the method for the user to upload the files
    '''

    create_folder_if_not_exist( UPLOAD_DIR )
    clear_all_files_in_folder( UPLOAD_DIR )
    uploaded_files_info = list()

    for file in files:

        if file.content_type != 'application/pdf':
            return JSONResponse(
                status_code = 400,
                content     = {"response": f"Invalid file type: {file.filename}. Only PDFs are allowed."}
            )

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        try:

            with open(file_path, "wb") as f:
                contents = await file.read()
                f.write(contents)

            uploaded_files_info.append({"filename": file.filename, "status": "success"})

        except Exception as e:
            uploaded_files_info.append({"filename": file.filename, "status": f"failed: {str(e)}"})

    return {"uploaded_files": uploaded_files_info}


@app.post("/api/chat")
async def chat(request: ChatRequest):

    '''
    This is the API method to reply back to the user.
    If files are uploaded, we reply back only with the user files.
    '''


    try:

        # No file uploaded, return classic GPT style answer.
        if is_folder_empty('./uploads/'):
            query_answer = openAIBot.ask_gpt(user_query=str(request.query))
            return ChatResponse(response=f"{query_answer}")

        # In the other case, let's implement RAG.
        else:

            # PDFS                  = get_dataframe_pdf_content(pdf_path = UPLOAD_DIR, chunck_text = True)
            # PDFS, NORM_EMBEDDINGS = get_pdf_dataframe_embeddings(pdfs_in_path = PDFS, return_norm_embeddings = True)

            PDFS            = pd.read_csv('C:/Users/WKS/Downloads/pdf_df.csv', sep=';')
            NORM_EMBEDDINGS = np.load('C:/Users/WKS/Downloads/norm_embeds.npy')

            t, p, f  = search_a_query_in_docs_with_faiss(norm_embs      = NORM_EMBEDDINGS,
                                                         dataframe_pdfs = PDFS,
                                                         query          = request.query,
                                                         k_closest      = 10,
                                                         return_D_I     = False)

            content = 'Answer this question: ' + request.query + '.'      + \
                      'To answer the question use ONLY and nothing else than this text: ' + t   + \
                      'Justify your answer based on the text provided. ' + \
                      'Report orderly the file name and the pages listed here: ' + p + f
            final    = openAIAssistant.ask_gpt(user_query=content)
            final    = final.replace('\n',' ')

            return ChatResponse(response=f"{final}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.5", port=8000)
