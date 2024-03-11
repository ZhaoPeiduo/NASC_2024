from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import JapaneseLLM
from asyncio import sleep

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
progress = 0

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


async def reset_progress_and_send(): 
    global progress
    progress = 0
    for connection in connections:
        await connection.send_json({"progress": 0, "reset": 1})

async def update_progress_and_send(new_progress: int):
    global progress
    for i in range(new_progress):
        progress += 1
        await sleep(0.1)
        for connection in connections:
            await connection.send_json({"progress": progress})

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
    await reset_progress_and_send()
    # Access the values
    question = options.question
    option1 = options.option1
    option2 = options.option2
    option3 = options.option3
    option4 = options.option4
    options = [option1, option2, option3, option4]
    options = [x for x in options if len(x) > 0]  # filter empty entries
    await update_progress_and_send(25)
    print('25')
    await update_progress_and_send(50)
    print('75')
    await update_progress_and_send(25)
    print('100')

@app.get("/progress")
async def get_progress():
    return {"progress": progress}