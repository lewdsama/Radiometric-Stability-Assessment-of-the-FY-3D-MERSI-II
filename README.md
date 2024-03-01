1. The data preprocessing in this paper is mainly carried out by IDL and Arcmap, and the code is placed in the DataPreprocessing folder.

   The DataPreprocessing folder contains three files, FY-3D_MEISI-II_Geometric&Radiometric_Corrections.pro and SZA_B3B4_Extract.pro, and arcpy_clip_batch.txt.
         FY-3D_MEISI-II_Geometric&Radiometric_Corrections.pro： Contains code that extracts the band data from the FY-3D HDF file and performs the geometric and radiometric correction. This step                                                                      will generates the GLT file.
         SZA_B3B4_Extract.pro：Use the GLT file to extract other attribute files in need, such as solar zenith angle, sensor zenith angle, azimuth angle et al.
         arcpy_clip_batch.txt: This file is ARCPY code, need to be used in arcmap. The role of code is AOI clip.


2. The BRDF correction and graphing in this paper is done using python.

   
