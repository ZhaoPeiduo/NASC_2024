from asyncio import sleep

class ProgressManager:
    def __init__(self, connections):
        self.connections = connections
        self.progress = 0

    async def reset_progress_and_send(self):
        self.progress = 0
        for connection in self.connections:
            await connection.send_json({"progress": 0, "reset": 1})

    async def update_progress_and_send(self, new_progress):
        for _ in range(new_progress):
            self.progress += 1
            await sleep(0.1)
            for connection in self.connections:
                await connection.send_json({"progress": self.progress})
