pro preMERSI2_250m
  compile_opt idl2, hidden
  tic
  DLM_LOAD, 'HDF5', 'XML', 'MAP_PE', 'NATIVE'
  e = ENVI(/h)
  

;  fnDat = 'F:\FY_vec\FY-3D\FY3D_MERSI_GBAL_L1_20210105_0940_0250M_MS.HDF'
;  fnSz  = 'F:\FY_vec\FY-3D\FY3D_MERSI_GBAL_L1_20210105_0940_GEO1K_MS.HDF'
  
  start_time = SYSTIME( /SECONDS )  



  ;数据路径
  data_folder_path = "J:\DomeC_data\20221120_20230228"
  geo_folder_path = "J:\DomeC_geo\20221120_20230228"
  
  data_file_list = FILE_SEARCH(data_folder_path + "\*.HDF", COUNT=file_count)
  geo_file_list = FILE_SEARCH(geo_folder_path + "\*.HDF", COUNT=file_count)
  
  
  
FOR i=0, file_count-1 DO BEGIN

 
  
  fnDat = data_file_list[i]
  print,fnDat

  fnSz  = geo_file_list[i]
  print,fnSz
 


  ;文件名
  dirname = FILE_DIRNAME(fnDat)
  ;  print,dirname
  ;提取HDF文件中的时间信息
  date = (FILE_BASENAME(fnDat)).Extract $
    ('20[1-4][0-9][0,1][0-9][0-3][0-9]_[0-9][0-9][0-9][0-9]')
    
  ;最后输出的第三个波段的名字，本程序这里只输出了第三个波段的信息
  refb3fn = dirname + PATH_SEP() + 'MERSI2_' + date + '_B3TOA_zenith.dat'
  ;  refb2fn = dirname + PATH_SEP() + 'MERSI2_' + date + '_B2TOA_zenith.dat'

  ;从data的hdf文件中读取相关数据的数组，这里只有单纯的数组，没有坐标系信息等，比如band3的数组，日地距离信息等
  fdid = H5F_OPEN(fnDat)
  rawRefb3 = h5GetData(fdid,'Data/EV_250_RefSB_b3')
  rawRefb2 = h5GetData(fdid,'Data/EV_250_RefSB_b2')
  calCoef = h5GetData(fdid, '/Calibration/VIS_Cal_Coeff')
  dst =(h5GetAttr(fdid, 'EarthSun Distance Ratio'))[0]
  ESUN = h5GetAttr(fdid, 'Solar_Irradiance')
  tbbA = h5GetAttr(fdid, 'TBB_Trans_Coefficient_A')
  tbbB = h5GetAttr(fdid, 'TBB_Trans_Coefficient_B')
  H5F_CLOSE, fdid


  ;从geo的hdf文件中读取相关数据的数组，比如经纬度，太阳天顶角等
  fsid = H5F_OPEN(fnSz)
  SolZ = FLOAT(h5GetData(fsid, 'Geolocation/SolarZenith') * !pi / 180  )
  lat = FLOAT(h5GetData(fsid, 'Geolocation/Latitude'))
  lon = FLOAT(h5GetData(fsid, 'Geolocation/Longitude'))
  H5F_CLOSE, fsid
  
  ;这一步是把经纬度数组中的经纬度数值转换成ps极地投影坐标系的数值，不做这一步后续利用经纬度数据直接生成GLT文件要花很长的时间
  in_proj = ENVI_PROJ_CREATE(/GEOGRAPHIC)
  out_proj= ENVI_PROJ_CREATE(PE_COORD_SYS_CODE=3031,TYPE=42)

  envi_convert_projection_coordinates,lon,lat,in_proj,$
    xmap,ymap,out_proj

  ;print,xmap,ymap

;  lat_TempFn = dirname + PATH_SEP() + 'MERSI2_' + date + '_lat.dat'
;  lon_TempFn = dirname + PATH_SEP() + 'MERSI2_' + date + '_lon.dat'
  
  ;生成经纬度和太阳日天顶角的栅格数据，因为太阳天顶角和经纬度数据都只有1000m分辨率的，这一步是为了后续重采样到250m分辨率做准备
  SolZ_TempFn = e.GetTemporaryFilename()
  Solz_raster = e.CreateRaster(SolZ_TempFn,SolZ,INHERITS_FROM = raster)
  Solz_raster.Save
  SolZ = !null

  lat_TempFn = e.GetTemporaryFilename()
  lat_raster = e.CreateRaster(lat_TempFn,ymap,INHERITS_FROM = raster)
  lat_raster.Save
  lat = !null

  lon_TempFn = e.GetTemporaryFilename()
  lon_raster = e.CreateRaster(lon_TempFn,xmap,INHERITS_FROM = raster)
  lon_raster.Save
  lon = !null
  
  
  
  

  rw = N_elements(rawRefb3[0,*])
  cl = N_elements(rawRefb3[*,0])
  
  ;对太阳天顶角和经纬度数据进行重采样
  SolZ_r = ENVIResampleRaster(Solz_raster, DIMENSIONS=[cl,rw], METHOD='Bilinear')
  SolZ_r = SolZ_r.GetData(bands=0)

  lat_r = ENVIResampleRaster(lat_raster, DIMENSIONS=[cl,rw], METHOD='Bilinear')
  lat_r = lat_r.GetData(bands=0)


;  lat_TempFn2 = dirname + PATH_SEP() + 'MERSI2_' + date + '_lat_resize_idl.dat'
  lat_TempFn2 = e.GetTemporaryFilename()
  lat_raster2 = e.CreateRaster(lat_TempFn2,lat_r,INHERITS_FROM = raster)
  lat_raster2.Save
  lat_id = ENVIRasterToFID(lat_raster2)



  lon_r = ENVIResampleRaster(lon_raster, DIMENSIONS=[cl,rw], METHOD='Bilinear')
  lon_r = lon_r.GetData(bands=0)

;  lon_TempFn2 = dirname + PATH_SEP() + 'MERSI2_' + date + '_lon_resize_idl.dat'
  lon_TempFn2 = e.GetTemporaryFilename()
  lon_raster2 = e.CreateRaster(lon_TempFn2,lon_r,INHERITS_FROM = raster)
  lon_raster2.Save
  lon_id = ENVIRasterToFID(lon_raster2)


  ;这一步是把原始的RawBand3数据从DN值定标到TOA
  refb3 = MAKE_ARRAY([cl,rw], type = 5)
  ;需要注意这里要除以100，但是没有查找到相关文献
  refb3= (rawRefb3 * calCoef[1,2] + calCoef[0,2])/100 * (dst^2) / cos(SolZ_r)

  ;  refb3= (rawRefb3 * calCoef[1,2] + calCoef[0,2])/100
  rawRefb3 = !null
  SolZ_r = !null
  lon_r = !null
  lat_r = !null

  Solz_raster.Close


  ;生成定标后band3_TOA的栅格数据
  refb3_Tempfn =  e.GetTemporaryFilename()
  refb3_raster = e.CreateRaster(refb3_Tempfn,refb3,INHERITS_FROM = raster)
  refb3_raster.Save
  refb3_id = ENVIRasterToFID(refb3_raster)
  ENVI_FILE_QUERY,refb3_id, DIMS = ref_dims
  refb3 = !null




;  GLTfn = ENVI_PICKFILE(title = 'select GLT file')
;  ENVI_OPEN_FILE,GLTfn,R_FID = GLT_id
  
  ;生成GLT的准备工作，设置输入的输出的坐标系，这里均为ps极地投影坐标系
  in_proj2 = ENVI_PROJ_CREATE(PE_COORD_SYS_CODE=3031,TYPE=42)
  out_proj2= ENVI_PROJ_CREATE(PE_COORD_SYS_CODE=3031,TYPE=42)
;  gltTempFn = dirname + PATH_SEP() + 'MERSI2_' + date + '_GLT.dat'
;  gltTempFn = dirname + PATH_SEP() + 'MERSI2_' + date + '_GLT'
;  gltTempFn = e.GetTemporaryFilename()
  ; 尹子成20231009：GLT文件需要存储在C盘，否则会大大增加处理时间
  gltTempFn = 'C:\glt\'+date+'_GLT.dat'
  print,gltTempFn
  ;利用重采样后的经纬度数据生成GLT文件，
  ENVI_DOIT, 'ENVI_GLT_DOIT', DIMS = ref_dims, $
      I_PROJ = in_proj2,  O_PROJ = out_proj2, $
      OUT_NAME = gltTempFn, $
      PIXEL_SIZE = 250.0, $
      R_FID = GLT_id, ROTATION = 0 , $
      X_FID = lon_id, X_POS = [0], $
      Y_FID = lat_id, Y_POS = [0]
  


  ;利用生成的GLT文件对Band3_TOA文件进行几何校正，也就是赋予坐标系信息
  ENVI_DOIT,'ENVI_GEOREF_FROM_GLT_DOIT', $
    BACKGROUND = 0, $
    FID = refb3_id, $
    GLT_FID = GLT_id, $
    OUT_NAME = refb3fn, $
    pos = [0:0]



  refb3_raster.Close
  lat_raster.Close
  lon_raster.Close
  lat_raster2.Close
  lon_raster2.Close
  ;TemporaryFilename 的路径 C:\Users\LENOVO\AppData\Local\Temp
  ;  CD, FILE_DIRNAME(e.GetTemporaryFilename())
  ;  TempFiles = FILE_SEARCH('[envitempfile]*')
  ;  n = N_elements(TempFiles)
  ;  for i = 0,n-1 do begin
  ;    FILE_DELETE, TempFiles[i]
  ;  endfor


  ;计算每一个文件的处理时间，大概是15-20min
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
