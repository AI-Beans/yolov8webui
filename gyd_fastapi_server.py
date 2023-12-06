# Gradio YOLOv8 Det FastAPI Server
# 创建人：曾逸夫
# 创建时间：2023-11-16
# http://localhost:1024/gradio/

import gradio as gr
import uvicorn
from fastapi import FastAPI

from gradio_yolov8_det_v1 import main, parse_args

app = FastAPI()


@app.get("/")
def read_main():
    return {"message": "This is your main app"}


args = parse_args()
io = main(args)
app = gr.mount_gradio_app(app, io, path="/gradio")

if __name__ == "__main__":
    app_str = 'gyd_fastapi_server:app'
    uvicorn.run(app_str, host='localhost', port=1024, reload=True, workers=4)
