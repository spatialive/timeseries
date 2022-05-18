import typing

import ee
import os
import glob
import uvicorn
from random import randint
from datetime import date
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from decouple import config
from src.sentinel import get_series
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from typing import List, Union, Any
import orjson

def gee_multi_credentials(credentials_dir):
    def mpb_get_credentials_path():
        credentials_files = ee.oauth.credentials_files

        credential = credentials_files[randint(0, 5)]
        ee.oauth.current_credentials_idx += 1

        return credential

    ee.oauth.current_credentials_idx = 0
    ee.oauth.credentials_files = glob.glob(credentials_dir + '/*.json')

    ee.oauth.get_credentials_path = mpb_get_credentials_path


def getMODIS_Series(lon, lat):
    gee_multi_credentials(config('GEE_CREDENCIALS_DIR'))
    ee.Initialize()

    def mask_badPixels(img):
        mask = img.select('SummaryQA').eq(0)
        img = img.mask(mask)
        return img

    MODIS = (
        ee.ImageCollection('MODIS/006/MOD13Q1')
            .filterDate('2000', str(int(date.today().year) + 1))
            .map(mask_badPixels)
            .select('NDVI', 'EVI')
    )

    geo = ee.Geometry.Point([lon, lat])

    data = []

    pointSeries = MODIS.getRegion(
        geo, MODIS.first().projection().nominalScale()
    ).getInfo()

    for index, dt in enumerate(pointSeries):
        if index > 1 and dt[4] != None and dt[4] != None:
            _date = dt[0].split('_')
            data.append({
                "label": f"{_date[2]}/{_date[1]}/{_date[0]}",
                "ndvi": dt[4] / 10000,
                "evi": dt[5] / 10000
            })

    ee.Reset()
    return data

class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: typing.Any) -> bytes:
        return orjson.dumps(content)

app = FastAPI(default_response_class=ORJSONResponse)

client = MongoClient(config('MONGO_HOST'), int(config('MONGO_PORT')))
db = client[config('MONGO_DB')]

origins = [
    "https://tvi.lapig.iesa.ufg.br",
    "http://127.0.0.1:8000/",
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

class Data(BaseModel):
    evi_series: List[str]
    evi_dates: List[datetime]
    wtk_smooth: List[float]

@app.get('/')
def read_root():
    return {'ok': True}


@app.get('/modis')
def ndvi_data(lon: float, lat: float):
    series = db.evi_ndvi.find_one({"lon": lon, "lat": lat})
    if series is not None:
        return series['data']
    else:
        _data = getMODIS_Series(lon, lat)
        #db.evi_ndvi.insert_one({"lon": lon, "lat": lat, "data": _data})
        return _data

# @app.get('/sentinel/evi/{lon}/{lat}/{start_date}/{end_date}')
@app.get('/sentinel/evi')
def sentinel_evi(lon: float, lat: float, start_date: str, end_date: str):
    series = db.sentinel_evi.find_one({"lon": lon, "lat": lat, "start_date": start_date, "end_date": end_date})
    if series is not None:
        return series
    else:
        _data = get_series(lon, lat, start_date, end_date)
        # import web_pdb;
        # web_pdb.set_trace()
        #db.evi_ndvi.insert_one({"lon": lon, "lat": lat, "start_date": start_date, "end_date": end_date, "data": _data})
        json_compatible_item_data = jsonable_encoder(_data)
        return _data

@app.get('/modis/chart', response_class=HTMLResponse)
def ndvi_chart(request: Request, lon: float, lat: float):
    return templates.TemplateResponse("ndvi.html", {"request": request, "lon": lon, "lat": lat, "server_url": config('SERVER_URL')})

@app.get('/sentinel/evi/chart', response_class=HTMLResponse)
def ndvi_chart(request: Request, lon: float, lat: float, start_date: str, end_date: str):
    return templates.TemplateResponse("sentinel.html", {"request": request, "lon": lon, "lat": lat, "start_date": start_date, "end_date": end_date, "server_url": config('SERVER_URL')})


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=config('HOST'),
        port=int(config('PORT')),
        workers=os.cpu_count() - 2,
        reload=False,
        log_config=config('LOG_CONFIG')
    )
