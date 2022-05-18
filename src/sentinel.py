import ee
import glob
from random import randint
from scipy import interpolate
import scipy.sparse as sparse
from scipy.sparse.linalg import splu
import pandas as pd
from decouple import config

import numpy as np

CLOUD_FILTER = 60
CLD_PRB_THRESH = 50
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50

def gee_multi_credentials(credentials_dir):
    def mpb_get_credentials_path():
        credentials_files = ee.oauth.credentials_files

        credential = credentials_files[randint(0, 5)]
        ee.oauth.current_credentials_idx += 1

        return credential

    ee.oauth.current_credentials_idx = 0
    ee.oauth.credentials_files = glob.glob(credentials_dir + '/*.json')

    ee.oauth.get_credentials_path = mpb_get_credentials_path

# Build a Sentinel-2 collection

def get_s2_sr_cld_col(aoi, start_date, end_date):
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
                 .filterBounds(aoi)
                 .filterDate(start_date, end_date)
                 .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                        .filterBounds(aoi)
                        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

# Define cloud mask component functions

# Cloud components
def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


def add_shadow_bands(img):
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH * SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
                .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
                .select('distance')
                .mask()
                .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

# Final cloud-shadow mask
def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focal_min(2).focal_max(BUFFER * 2 / 20)
                   .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
                   .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)


##Adding a NDVI band
def addNDVI(img):
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
    return img.addBands([ndvi])


# def scale_factor(img):
#     return image.multiply(0.0001).copyProperties(img, ['system:time_start'])# mapping function to multiply by the scale factor

# def maskS2clouds(img):
#     qa = img.select('QA60')
#     cloudBitMask = 1 << 10
#     cirrusBitMask = 1 << 11
#     mask = qa.bitwiseAnd(cloudBitMask).eq(0)and(qa.bitwiseAnd(cirrusBitMask).eq(0))
#     return img.updateMask(mask).divide(10000)


# img_collection_S2_SR = ee.ImageCollection("COPERNICUS/S2_SR").filterDate(start=start, opt_end=end).map(addNDVI).filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)).filterBounds(geometry)
# s2_cloudless_col = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filterDate(start=start, opt_end=end).filterBounds(geometry)
# img_collection_S2_SR_cm = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
#         'primary': img_collection_S2_SR,
#         'secondary': s2_cloudless_col,
#         'condition': ee.Filter.equals(**{
#             'leftField': 'system:index',
#             'rightField': 'system:index'
#         })
#     }))

def get_s2_sr_cld_col(aoi, start_date, end_date):
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
                 .filterBounds(aoi)
                 .filterDate(start_date, end_date)
                 .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                        .filterBounds(aoi)
                        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))


def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)


def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)


# Adding a NDVI band
def addNDVI(img):
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
    return img.addBands([ndvi])

# Adding a EVI band
def addEVI(img):
    evi = img.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
            'NIR': img.select('B8'),
            'RED': img.select('B4'),
            'BLUE': img.select('B2')
        }).rename('evi')
    return img.addBands([evi])

def speyediff(N, d, format='csc'):
    """
    (utility function)
    Construct a d-th order sparse difference matrix based on 
    an initial N x N identity matrix
    
    Final matrix (N-d) x N
    """

    assert not (d < 0), "d must be non negative"
    shape = (N - d, N)
    diagonals = np.zeros(2 * d + 1)
    diagonals[d] = 1.
    for i in range(d):
        diff = diagonals[:-1] - diagonals[1:]
        diagonals = diff
    offsets = np.arange(d + 1)
    spmat = sparse.diags(diagonals, offsets, shape, format=format)
    return spmat

"""
    WHITTAKER-EILERS SMOOTHER in Python 3 using numpy and scipy
    based on the work by Eilers [1].
        [1] P. H. C. Eilers, "A perfect smoother", 
            Anal. Chem. 2003, (75), 3631-3636
    coded by M. H. V. Werts (CNRS, France)
    tested on Anaconda 64-bit (Python 3.6.4, numpy 1.14.0, scipy 1.0.0)
    Read the license text at the end of this file before using this software.
    Warm thanks go to Simon Bordeyne who pioneered a first (non-sparse) version
    of the smoother in Python.
"""
def whittaker_smooth(y, lmbd, d=2):
    """
    Implementation of the Whittaker smoothing algorithm,
    based on the work by Eilers [1].
    [1] P. H. C. Eilers, "A perfect smoother", Anal. Chem. 2003, (75), 3631-3636
    
    The larger 'lmbd', the smoother the data.
    For smoothing of a complete data series, sampled at equal intervals
    This implementation uses sparse matrices enabling high-speed processing
    of large input vectors
    
    ---------
    
    Arguments :
    
    y       : vector containing raw data
    lmbd    : parameter for the smoothing algorithm (roughness penalty)
    d       : order of the smoothing 
    
    ---------
    Returns :
    
    z       : vector of the smoothed data.
    """

    m = len(y)
    E = sparse.eye(m, format='csc')
    D = speyediff(m, d, format='csc')
    coefmat = E + lmbd * D.conj().T.dot(D)
    z = splu(coefmat).solve(y)
    return z

"""
    Copyright M. H. V. Werts, 2017
    
    martinus point werts à ens-rennes point fr
    
    This software is a computer program whose purpose is to smooth noisy data.
    
    This software is governed by the CeCILL-B license under French law and
    abiding by the rules of distribution of free software.  You can  use, 
    modify and/ or redistribute the software under the terms of the CeCILL-B
    license as circulated by CEA, CNRS and INRIA at the following URL
    "http://www.cecill.info". 
    
    As a counterpart to the access to the source code and  rights to copy,
    modify and redistribute granted by the license, users are provided only
    with a limited warranty  and the software's author,  the holder of the
    economic rights,  and the successive licensors  have only  limited
    liability. 
    
    In this respect, the user's attention is drawn to the risks associated
    with loading,  using,  modifying and/or developing or reproducing the
    software by the user in light of its specific status of free software,
    that may mean  that it is complicated to manipulate,  and  that  also
    therefore means  that it is reserved for developers  and  experienced
    professionals having in-depth computer knowledge. Users are therefore
    encouraged to load and test the software's suitability as regards their
    requirements in conditions enabling the security of their systems and/or 
    data to be ensured and,  more generally, to use and operate it in the 
    same conditions as regards security. 
    
    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-B license and that you accept its terms.
"""
def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good], bounds_error=False)
    B = np.where(np.isfinite(A), A, f(inds))
    return B

def get_series(lon, lat, start_date, end_date):
    gee_multi_credentials(config('GEE_CREDENCIALS_DIR'))
    ee.Initialize()
    AOI = ee.Geometry.Point(lon, lat)

    # s2_sr_cld_col_eval = get_s2_sr_cld_col(AOI, start_date, end_date)

    s2_sr_cld_col = get_s2_sr_cld_col(AOI, start_date, end_date)

    # s2_sr_median = (s2_sr_cld_col.map(add_cld_shdw_mask)
    #                 .map(apply_cld_shdw_mask)
    #                 .median())

    s2_sr = (s2_sr_cld_col.map(add_cld_shdw_mask)
             .map(apply_cld_shdw_mask)
             .map(addNDVI)
             .map(addEVI))

    l = s2_sr.filterBounds(AOI).getRegion(AOI, 5).getInfo()
    out = [dict(zip(l[0], values)) for values in l[1:]]
    df = pd.DataFrame(out)

    df['id'] = df.id.str[0:8]
    df['id'] = pd.to_datetime(df['id'])

    evi_series = np.array(df['evi'])
    evi_dates = np.array(df['id'])

    # plt.figure(1, figsize=(20, 8), clear=True)
    # plt.scatter(evi_dates, evi_series, color="black", marker="+")
    # plt.show()

    ### Whittaker smoother
    wtk_smooth = whittaker_smooth(fill_nan(evi_series), 10, d=2)

    # plt.figure(1, figsize=(20, 8), clear=True)
    # plt.scatter(evi_dates, evi_series, color="black", marker="+")
    # plt.plot(evi_dates, wtk_smooth)
    # plt.show()
    eviDates = []

    for date in evi_dates:
        eviDates.append(np.datetime_as_string(date, unit='D'))


    data = {
        "evi_raw_series": evi_series.tolist(),
        "dates": eviDates,
        "evi_wtk_smooth_series": wtk_smooth.tolist()
    }

    return data

