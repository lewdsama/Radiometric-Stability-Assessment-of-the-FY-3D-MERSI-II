1. The data preprocessing in this paper is mainly carried out by IDL and Arcmap, and the code is placed in the DataPreprocessing folder.

   The DataPreprocessing folder contains three files, FY-3D_MEISI-II_Geometric&Radiometric_Corrections.pro and SZA_B3B4_Extract.pro, and arcpy_clip_batch.txt.
         FY-3D_MEISI-II_Geometric&Radiometric_Corrections.pro： Contains code that extracts the band data from the FY-3D HDF file and performs the geometric and radiometric correction. This step                                                                      will generates the GLT file.
         SZA_B3B4_Extract.pro：Use the GLT file to extract other attribute files in need, such as solar zenith angle, sensor zenith angle, azimuth angle et al.
         arcpy_clip_batch.txt: This file is ARCPY code, need to be used in arcmap. The role of code is AOI clip.


2. The BRDF correction and graphing in this paper is done using python.

   The BRDF_Correction&Pic_export folder contains two files: The code for BRDF correction using the original and simplified models, respectively, which also includes Raw TOA time-series fitting, 
    BRDF model parameter generation, TOA time-series fitting after BRDF correction, and radiometric degraduation  calculation etc., and also includes the figure for each section.

3. Data.zip：The data in Data.zip is the data of the aoi region obtained by clipping in the step 1, and is also the data needed for BRDF correction in the step 2, it should be noted that due to the limitation of the file size, we do not upload the original data here.

4. SRF_MERSI-II&MDDIS: This section is the python code for plotting the MODIS and FY-3D MERSI-II spectral response functions
 
