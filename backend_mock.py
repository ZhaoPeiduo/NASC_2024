from fastapi import FastAPI, WebSocket, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from progress_manager import ProgressManager
from asyncio import sleep
import cv2
import numpy as np


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

class Options(BaseModel):
    question: str
    option1: str
    option2: str
    option3: str
    option4: str

with open('./frontend.html', 'r') as file:
    html_content = file.read()

@app.get("/")
async def get():
    return HTMLResponse(content=html_content, status_code=200)


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
    await manager.update_progress_and_send(50)
    await sleep(1)
    await manager.update_progress_and_send(50)

    return "test string"

@app.post("/extract-text/")
async def extract_text(image: UploadFile = File(...), threshold: bool = Form(False)):
    image_data = await image.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Process the image data as needed
    
    return None

@app.get("/progress")
async def get_progress():
    return {"progress": manager.progress}