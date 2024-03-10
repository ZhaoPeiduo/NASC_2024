from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import JapaneseLLM

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

japanese_model = JapaneseLLM()
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

async def echo(websocket: WebSocket):
    await websocket.accept()
    while True:
        message = await websocket.receive_text()
        await websocket.send_text(f"Received: {message}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await echo(websocket)


@app.post("/process-options")
async def process_options(options: Options):
    update_progress(0)
    # Access the values
    question = options.question
    option1 = options.option1
    option2 = options.option2
    option3 = options.option3
    option4 = options.option4
    options = [option1, option2, option3, option4]
    options = [x for x in options  if len(options) > 0] # filter empty entries
    update_progress(25)

    answer = japanese_model.generate_answer(question, options)
    update_progress(50)
    explanation = japanese_model.generate_explanation(question, options, answer)
    update_progress(100)
    return explanation

@app.get("/progress")
async def get_progress():
    return {"progress": progress}

@app.post("/progress")
async def update_progress(new_progress: int):
    global progress
    progress = new_progress
    return {"message": "Progress updated successfully"}