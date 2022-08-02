import typing
import os
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
# from pymongo import MongoClient
from decouple import config
from src.ts import gerate_graph_EVI_CHIRPS
from fastapi.responses import JSONResponse
import orjson


class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: typing.Any) -> bytes:
        return orjson.dumps(content)


app = FastAPI(default_response_class=ORJSONResponse)

# client = MongoClient(config('MONGO_HOST'), int(config('MONGO_PORT')))
# db = client[config('MONGO_DB')]

origins = [
    "*",
    "https://tvi.lapig.iesa.ufg.br",
    "http://127.0.0.1:8000",
    "http://localhost:3002",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


@app.get('/')
def read_root():
    return {'ok': True}


@app.get('/sentinel/evi')
def sentinel_evi(lon: float, lat: float, start_date: str, end_date: str):
#     series = db.sentinel_evi.find_one({"lon": lon, "lat": lat, "start_date": start_date, "end_date": end_date})
    series = None
    if series is not None:
        return series
    else:
        return gerate_graph_EVI_CHIRPS(lon, lat, start_date, end_date, 8, 'savgol')



@app.get('/sentinel/evi/chart', response_class=HTMLResponse)
def ndvi_chart(request: Request, lon: float, lat: float, start_date: str, end_date: str):
    return templates.TemplateResponse("sentinel.html",
                                      {"request": request, "lon": lon, "lat": lat, "start_date": start_date,
                                       "end_date": end_date, "server_url": config('SERVER_URL')})


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=config('HOST'),
        port=int(config('PORT')),
        workers=os.cpu_count() - 2,
        reload=False,
        timeout_keep_alive=15,
        log_config=config('LOG_CONFIG')
    )
