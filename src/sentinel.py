import ee
import pandas as pd

#Build a Sentinel-2 collection

def ts_s2_collection(lon: float, lat: float, start_date: str, end_date: str) -> pd.DataFrame:

    """
        Extrai a séries temporal do sentinel 2 por meio do ponto informado
    """

    #Parametros do sentinel cloud less
    
    CLOUD_FILTER = 60
    CLD_PRB_THRESH = 50
    NIR_DRK_THRESH = 0.15
    CLD_PRJ_DIST = 1
    BUFFER = 50

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

    #Define cloud mask component functions

    #Cloud components
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
        dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

        # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
        shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

        # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
        cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
            .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
            .select('distance')
            .mask()
            .rename('cloud_transform'))

        # Identify the intersection of dark pixels with cloud shadow projection.
        shadows = cld_proj.multiply(dark_pixels).rename('shadows')

        # Add dark pixels, cloud projection, and identified shadows as image bands.
        return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

    #Final cloud-shadow mask
    def add_cld_shdw_mask(img):
        # Add cloud component bands.
        img_cloud = add_cloud_bands(img)

        # Add cloud shadow component bands.
        img_cloud_shadow = add_shadow_bands(img_cloud)

        # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
        is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

        # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
        # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
        is_cld_shdw = (is_cld_shdw.focal_min(2).focal_max(BUFFER*2/20)
            .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
            .rename('cloudmask'))

        # Add the final cloud-shadow mask to the image.
        return img_cloud_shadow.addBands(is_cld_shdw)

    def apply_cld_shdw_mask(img):
        # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
        not_cld_shdw = img.select('cloudmask').Not()

        # Subset reflectance bands and update their masks, return the result.
        return img.select('B.*').updateMask(not_cld_shdw)

    # Adding a NDVI band
    def addNDVI(img):
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return img.addBands([ndvi])

    # Adding a EVI band
    def addEVI(img):
        evi = img.normalizedDifference(['B8', 'B4']).rename('EVI')
        return img.addBands([evi])

    # Criação da feature Point no earth engine
    geometry = ee.Geometry.Point([lon, lat])

    # Geração da serie temporal de EVI do ponto e aplicação do sentinel cloud less
    s2_sr_cld_col = get_s2_sr_cld_col(geometry, start_date, end_date)
    s2_sr = (s2_sr_cld_col.map(add_cld_shdw_mask)
                        .map(apply_cld_shdw_mask)
                        .map(addEVI))

    # Extração da série temporal do EE para o Python
    l = s2_sr.filterBounds(geometry).getRegion(geometry, 5).getInfo() # for sentinel
    out = [dict(zip(l[0], values)) for values in l[1:]]
    
    
    # Transformaçẽos da série temporal de EVI para DataFrame
    df = pd.DataFrame(out)
    df['id'] = df.id.str[0:8] # sentinel é [0:8]
    df = df.loc[:, ['id', 'EVI']]
    df = df.groupby(by=['id'], as_index=False).max()
    df['id'] = pd.to_datetime(df['id'])
    df = df.sort_values(by=['id'])
    df = df.rename(columns={'id': 'Date'})

    return df