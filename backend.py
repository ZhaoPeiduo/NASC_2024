from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi import WebSocket

app = FastAPI()

html_content = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chatbot</title>
        <style>
            #messages {
                list-style-type: none;
                margin: 0;
                padding: 0;
            }
        </style>
    </head>
    <body>
        <ul id="messages"></ul>
        <input type="text" id="message" autocomplete="off" placeholder="Type your message here..."/>
        <button id="send">Send</button>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");

            ws.onmessage = function(event) {
                var messages = document.getElementById('messages');
                var message = document.createElement('li');
                var content = document.createTextNode(event.data);
                message.appendChild(content);
                messages.appendChild(message);
            };

            document.getElementById('send').onclick = function() {
                var input = document.getElementById('message');
                var message = input.value;
                ws.send(message);
                input.value = '';
            };
        </script>
    </body>
</html>
"""

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