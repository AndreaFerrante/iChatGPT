import os
import uvicorn
from pydantic import BaseModel
from embedder.embedder import *
from fastapi.responses import JSONResponse
from openaiassistant import OpenAIAssistant
from closeai.openaikeys import openai_main_key
from fastapi import FastAPI, File, UploadFile, HTTPException
from backend.utils.utils import create_folder_if_not_exist, clear_all_files_in_folder
from backend.utils.utils import is_folder_empty,get_dataframe_pdf_content


################################################################
norm_embeds           = None
pdfs                  = None
UPLOAD_DIR            = './uploads/'
app                   = FastAPI()
openAIBot             = OpenAIAssistant(openai_api_key=openai_main_key)
################################################################


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str


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

            if norm_embeds is None and pdfs is None:
                pdfs              = get_dataframe_pdf_content(pdf_path='./uploads/', chunck_text=True)
                pdfs, norm_embeds = get_pdf_dataframe_embeddings(pdfs_in_path='./uploads/', return_norm_embeddings=True)
            return ChatResponse(response='hello world')

            D, I  = search_a_query_in_docs_with_faiss(norm_embs = norm_embeds, query = query, dataframe_pdfs = pdf_df, k_closest = 5)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.5", port=8000)
