  ;20231024
  ; :Author: 尹子成
  ;修改版本
pro preMERSI2_250m
  compile_opt idl2, hidden
  tic
  DLM_LOAD, 'HDF5', 'XML', 'MAP_PE', 'NATIVE'
  e = ENVI(/h)
  

;  fnDat = 'F:\FY_vec\FY-3D\FY3D_MERSI_GBAL_L1_20210105_0940_0250M_MS.HDF'
;  fnSz  = 'F:\FY_vec\FY-3D\FY3D_MERSI_GBAL_L1_20210105_0940_GEO1K_MS.HDF'
  
  start_time = SYSTIME( /SECONDS )  



  ;数据路径
  data_folder_path = "I:\每年的1015-1119的0740左右的数据\HDF"
  geo_folder_path = "I:\每年的1015-1119的0740左右的数据\geo"
  
  data_file_list = FILE_SEARCH(data_folder_path + "\*.HDF", COUNT=file_count)
  geo_file_list = FILE_SEARCH(geo_folder_path + "\*.HDF", COUNT=file_count)
  
  
  
FOR i=0, file_count-1 DO BEGIN

 
  
  fnDat = data_file_list[i]
  print,fnDat

  fnSz  = geo_file_list[i]
  print,fnSz
 



  dirname = FILE_DIRNAME(fnDat)
  ;  print,dirname
  date = (FILE_BASENAME(fnDat)).Extract $
    ('20[1-4][0-9][0,1][0-9][0-3][0-9]_[0-9][0-9][0-9][0-9]')
    
  refb3fn = dirname + PATH_SEP() + 'MERSI2_' + date + '_B3TOA_zenith.dat'
  refb4fn = dirname + PATH_SEP() + 'MERSI2_' + date + '_B4TOA_zenith.dat'
  SolZfn = dirname + PATH_SEP() + 'MERSI2_' + date + '_SolZ.dat'
  SolAfn = dirname + PATH_SEP() + 'MERSI2_' + date + '_SolA.dat'
  SenZfn = dirname + PATH_SEP() + 'MERSI2_' + date + '_SenZ.dat'
  SenAfn = dirname + PATH_SEP() + 'MERSI2_' + date + '_SenA.dat'
 
  ;  refb2fn = dirname + PATH_SEP() + 'MERSI2_' + date + '_B2TOA_zenith.dat'

  fdid = H5F_OPEN(fnDat)
  rawRefb3 = h5GetData(fdid,'Data/EV_250_RefSB_b3')
  rawRefb4 = h5GetData(fdid,'Data/EV_250_RefSB_b4')
  calCoef = h5GetData(fdid, '/Calibration/VIS_Cal_Coeff')
  dst =(h5GetAttr(fdid, 'EarthSun Distance Ratio'))[0]
  ESUN = h5GetAttr(fdid, 'Solar_Irradiance')
  tbbA = h5GetAttr(fdid, 'TBB_Trans_Coefficient_A')
  tbbB = h5GetAttr(fdid, 'TBB_Trans_Coefficient_B')
  H5F_CLOSE, fdid



  fsid = H5F_OPEN(fnSz)
  SolZ = FLOAT(h5GetData(fsid, 'Geolocation/SolarZenith') * !pi / 180)
  SenZ = FLOAT(h5GetData(fsid, 'Geolocation/SensorZenith') * !pi / 180)
  SolA = FLOAT(h5GetData(fsid, 'Geolocation/SolarAzimuth') * !pi / 180)
  SenA = FLOAT(h5GetData(fsid, 'Geolocation/SensorAzimuth') * !pi / 180)
  
  H5F_CLOSE, fsid
  


  SolZ_TempFn = e.GetTemporaryFilename()
  SolZ_raster = e.CreateRaster(SolZ_TempFn,SolZ,INHERITS_FROM = raster)
  SolZ_raster.Save
  SolZ = !null


  SenZ_TempFn = e.GetTemporaryFilename()
  SenZ_raster = e.CreateRaster(SenZ_TempFn,SenZ,INHERITS_FROM = raster)
  SenZ_raster.Save
  SenZ = !null
  
  SolA_TempFn = e.GetTemporaryFilename()
  SolA_raster = e.CreateRaster(SolA_TempFn,SolA,INHERITS_FROM = raster)
  SolA_raster.Save
  SolA = !null
  
  SenA_TempFn = e.GetTemporaryFilename()
  SenA_raster = e.CreateRaster(SenA_TempFn,SenA,INHERITS_FROM = raster)
  SenA_raster.Save
  SenA = !null
  

  
  rw = N_elements(rawRefb4[0,*])
  cl = N_elements(rawRefb4[*,0])

  SolZ_r = ENVIResampleRaster(SolZ_raster, DIMENSIONS=[cl,rw], METHOD='Bilinear')
  SolZ_r = SolZ_r.GetData(bands=0)
  
  SolA_r = ENVIResampleRaster(SolA_raster, DIMENSIONS=[cl,rw], METHOD='Bilinear')
  SolA_r = SolA_r.GetData(bands=0)

  SenZ_r = ENVIResampleRaster(SenZ_raster, DIMENSIONS=[cl,rw], METHOD='Bilinear')
  SenZ_r = SenZ_r.GetData(bands=0)
  
  SenA_r = ENVIResampleRaster(SenA_raster, DIMENSIONS=[cl,rw], METHOD='Bilinear')
  SenA_r = SenA_r.GetData(bands=0)


  refb3 = MAKE_ARRAY([cl,rw], type = 5)

  refb3= (rawRefb3 * calCoef[1,2] + calCoef[0,2]) * (dst^2) / cos(SolZ_r)*0.01
  
  refb4 = MAKE_ARRAY([cl,rw], type = 5)

  refb4= (rawRefb4 * calCoef[1,3] + calCoef[0,3]) * (dst^2) / cos(SolZ_r)*0.01

  ;  refb3= (rawRefb3 * calCoef[1,2] + calCoef[0,2])/100


  refb3_Tempfn =  e.GetTemporaryFilename()
  refb3_raster = e.CreateRaster(refb3_Tempfn,refb3,INHERITS_FROM = raster)
  refb3_raster.Save
  refb3_id = ENVIRasterToFID(refb3_raster)
  refb3 = !null

  refb4_Tempfn =  e.GetTemporaryFilename()
  refb4_raster = e.CreateRaster(refb4_Tempfn,refb4,INHERITS_FROM = raster)
  refb4_raster.Save
  refb4_id = ENVIRasterToFID(refb4_raster)
  refb4 = !null
  
  
  SolZ_Tempfn2 =  e.GetTemporaryFilename()
  SolZ_raster2 = e.CreateRaster(SolZ_Tempfn2,SolZ_r,INHERITS_FROM = raster)
  SolZ_raster2.Save
  SolZ_id = ENVIRasterToFID(SolZ_raster2)
  
  SolA_Tempfn2 =  e.GetTemporaryFilename()
  SolA_raster2 = e.CreateRaster(SolA_Tempfn2,SolA_r,INHERITS_FROM = raster)
  SolA_raster2.Save
  SolA_id = ENVIRasterToFID(SolA_raster2)
  
  SenZ_Tempfn2 =  e.GetTemporaryFilename()
  SenZ_raster2 = e.CreateRaster(SenZ_Tempfn2,SenZ_r,INHERITS_FROM = raster)
  SenZ_raster2.Save
  SenZ_id = ENVIRasterToFID(SenZ_raster2)
  
  SenA_Tempfn2 =  e.GetTemporaryFilename()
  SenA_raster2 = e.CreateRaster(SenA_Tempfn2,SenA_r,INHERITS_FROM = raster)
  SenA_raster2.Save
  SenA_id = ENVIRasterToFID(SenA_raster2)
  




;  gltTempFn = dirname + PATH_SEP() + 'MERSI2_' + date + '_GLT.dat'
;  gltTempFn = dirname + PATH_SEP() + 'MERSI2_' + date + '_GLT'
;  gltTempFn = e.GetTemporaryFilename()
  ; 尹子成20231009：GLT文件需要存储在C盘，否则会大大增加处理时间

  
  glt_file='I:\每年的1015-1119的0740左右的数据\glt\'+ date + '_GLT.dat'
  glt_r=e.OpenRaster(glt_file)
  GLT_id=ENVIRasterToFID(glt_r)
  
  ENVI_DOIT,'ENVI_GEOREF_FROM_GLT_DOIT', $
    BACKGROUND = 0, $
    FID = refb3_id, $
    GLT_FID = GLT_id, $
    OUT_NAME = refb3fn, $
    pos = [0:0]
;  
;  ENVI_DOIT,'ENVI_GEOREF_FROM_GLT_DOIT', $
;    BACKGROUND = 0, $
;    FID = refb4_id, $
;    GLT_FID = GLT_id, $
;    OUT_NAME = refb4fn, $
;    pos = [0:0]
;    
;    
;  ENVI_DOIT,'ENVI_GEOREF_FROM_GLT_DOIT', $
;    BACKGROUND = 0, $
;    FID = SolZ_id, $
;    GLT_FID = GLT_id, $
;    OUT_NAME = SolZfn, $
;    pos = [0:0]
;    
;  ENVI_DOIT,'ENVI_GEOREF_FROM_GLT_DOIT', $
;    BACKGROUND = 0, $
;    FID = SolA_id, $
;    GLT_FID = GLT_id, $
;    OUT_NAME = SolAfn, $
;    pos = [0:0]
;    
;  ENVI_DOIT,'ENVI_GEOREF_FROM_GLT_DOIT', $
;    BACKGROUND = 0, $
;    FID = SenZ_id, $
;    GLT_FID = GLT_id, $
;    OUT_NAME = SenZfn, $
;    pos = [0:0]
  
;  ENVI_DOIT,'ENVI_GEOREF_FROM_GLT_DOIT', $
;    BACKGROUND = 0, $
;    FID = SenA_id, $
;    GLT_FID = GLT_id, $
;    OUT_NAME = SenAfn, $
;    pos = [0:0]



    rawRef4 = !null
    SolZ_r = !null
    SolA_r = !null
    SenZ_r = !null
    SenA_r = !null

    SolZ_raster.Close
    SolA_raster.Close
    SenZ_raster.Close
    SenA_raster.Close
;    refb3_raster.Close
    refb4_raster.Close
    SolZ_raster2.Close
    SolA_raster2.Close
    SenZ_raster2.Close
    SenA_raster2.Close
  ;TemporaryFilename 的路径 C:\Users\LENOVO\AppData\Local\Temp
  ;  CD, FILE_DIRNAME(e.GetTemporaryFilename())
  ;  TempFiles = FILE_SEARCH('[envitempfile]*')
  ;  n = N_elements(TempFiles)
  ;  for i = 0,n-1 do begin
  ;    FILE_DELETE, TempFiles[i]
  ;  endfor



  elapsed_time = SYSTIME( /SECONDS ) - start_time ; 
  print, "第",i+1,"个文件已经处理完毕","目前用时", elapsed_time, "秒"



ENDFOR
  

  toc
end

function h5GetData, fid, str
  compile_opt idl2, hidden

  str_id = H5D_OPEN(fid, str)
  slope = FLOAT(h5GetAttr(str_id, 'Slope'))
  intercept = FLOAT(h5GetAttr(str_id, 'Intercept'))
  data = FLOAT(H5D_READ(str_id))
  foreach _slope, slope, index do $
    data[*, *, index] = _slope * data[*, *, index] + intercept[index]
  H5D_CLOSE, str_id

  RETURN, data
end

function h5GetAttr, fid, str
  compile_opt idl2, hidden

  str_id = H5A_OPEN_NAME(fid, str)
  attr = H5A_READ(str_id)
  H5A_CLOSE, str_id

  RETURN, attr
end
