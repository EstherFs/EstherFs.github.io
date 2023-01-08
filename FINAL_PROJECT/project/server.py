import aiohttp
import asyncio
import uvicorn
from PIL import Image
import numpy as np
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse
import pathlib
from transfer import transfer
import os
import sys
import pybase64
from flask import make_response
path = pathlib.Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='/home/ubuntu/project/static'))
app.mount('/fullpage', StaticFiles(directory='/home/ubuntu/project/node_modules/fullpage.js/dist'))
app.mount('/fullpage_vendor', StaticFiles(directory='/home/ubuntu/project/node_modules/fullpage.js/vendors'))

@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())

    img = Image.open(BytesIO(img_bytes))

    prediction = await transfer(img)
    
    img_stream = ''
    img_local_path = "/home/ubuntu/project/tmp.jpeg"
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = str(pybase64.standard_b64encode(img_stream), encoding='utf-8')

    return JSONResponse({'result': img_stream})



@app.route('/image/{image_name}', methods=['GET'])
def get_image(request):
    image_name = request.path_params['image_name']
    img_local_path = "/home/ubuntu/project/view/" + image_name
    response = FileResponse(img_local_path)
    return response 

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=8000, log_level="info")