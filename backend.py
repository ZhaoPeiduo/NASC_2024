from fastapi import FastAPI, WebSocket, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from pydantic import BaseModel
from model import JapaneseLLM
from progress_manager import ProgressManager
import easyocr
import numpy as np
import cv2
import base64

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

connections = set()
manager = ProgressManager(connections)
japanese_model = JapaneseLLM(manager)
reader = easyocr.Reader(['ja'])

class Options(BaseModel):
    question: str
    option1: str
    option2: str
    option3: str
    option4: str

# Setup Jinja2 templates
templates = Jinja2Templates(directory=".")

# Serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("frontend.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    finally:
        connections.remove(websocket)

@app.post("/process-options")
async def process_options(options: Options):
    await manager.reset_progress_and_send()
    # Access the values
    question = options.question
    option1 = options.option1
    option2 = options.option2
    option3 = options.option3
    option4 = options.option4
    options = [option1, option2, option3, option4]
    options = [x for x in options if len(x) > 0]  # filter empty entries
    await manager.update_progress_and_send(10)

    answer = await japanese_model.generate_answer(question, options) 
    explanation = await japanese_model.generate_explanation(question, options, answer) 

    return explanation

@app.get("/progress")
async def get_progress():
    return {"progress": manager.progress}

def extract_question_and_options(cropped, num_options=4):
    result = reader.readtext(cropped)
    result = [x[1] for x in result]
    if len(result) < num_options + 1:
        suggested_question = result[0]
        suggested_options = result[1:]
    else:
        suggested_options = result[-num_options:]
        suggested_question = ' '.join(result[:-num_options])
    return suggested_question, suggested_options

@app.post("/extract-text/")
async def extract_text(image_data: str = Form(...), x1: int = Form(...), y1: int = Form(...), x2: int = Form(...), y2: int = Form(...)):
    try:
        _, encoded_data = image_data.split(',')
        decoded_data = base64.b64decode(encoded_data)
        image_array = np.frombuffer(decoded_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        cropped = image[y1:y2, x1:x2]
        suggested_question, suggested_options = extract_question_and_options(cropped)

        return {"suggested_question": suggested_question, "suggested_options": suggested_options}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})