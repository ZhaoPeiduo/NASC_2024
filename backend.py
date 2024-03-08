from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi import WebSocket
from pydantic import BaseModel

app = FastAPI()

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
    # Access the values
    question = options.question
    option1 = options.option1
    option2 = options.option2
    option3 = options.option3
    option4 = options.option4

    # Process the options here
    # Return a response if needed
    print({"question": question, "options": [option1, option2, option3, option4]})
    