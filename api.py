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


app = FastAPI()

client = MongoClient(config('MONGO_HOST'), int(config('MONGO_PORT')))
db = client[config('MONGO_DB')]

origins = [
    "https://tvi.lapig.iesa.ufg.br"
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


@app.get('/modis/{lon}/{lat}')
def ndvi_data(lon: float, lat: float):
    series = db.evi_ndvi.find_one({"lon": lon, "lat": lat})
    if series is not None:
        return series['data']
    else:
        _data = getMODIS_Series(lon, lat)
        db.evi_ndvi.insert_one({"lon": lon, "lat": lat, "data": _data})
        return _data

@app.get('/modis/chart/{lon}/{lat}', response_class=HTMLResponse)
def ndvi_chart(request: Request, lon: float, lat: float):
    return templates.TemplateResponse("ndvi.html", {"request": request, "lon": lon, "lat": lat, "server_url": config('SERVER_URL')})


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=config('HOST'),
        port=int(config('PORT')),
        workers=os.cpu_count() - 2,
        reload=False,
        log_config=config('LOG_CONFIG')
    )
