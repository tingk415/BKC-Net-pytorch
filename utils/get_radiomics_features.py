import os
import SimpleITK as sitk
import six
from radiomics import featureextractor
import numpy as np

file_path = os.path.abspath(os.path.join('utils','RadiomicsFeaturesOutput', 'normalized_radiomics_features.npy'))

params = os.path.abspath(os.path.join('utils','RadiomicsFeaturesSettings','radiomicsFeatures.yaml'))

keys_to_remove = [
	'diagnostics_Versions_PyRadiomics',
	'diagnostics_Versions_Numpy',
	'diagnostics_Versions_SimpleITK',
	'diagnostics_Versions_PyWavelet',
	'diagnostics_Versions_Python',
	'diagnostics_Configuration_Settings',
	'diagnostics_Configuration_EnabledImageTypes',
	'diagnostics_Image-original_Hash',
	'diagnostics_Image-original_Dimensionality',
    'diagnostics_Image-original_Spacing',
    'diagnostics_Image-original_Size',
    'diagnostics_Mask-original_Spacing',
    'diagnostics_Mask-original_Size',
    'diagnostics_Mask-original_BoundingBox',
    'diagnostics_Mask-original_CenterOfMassIndex',
    'diagnostics_Mask-original_CenterOfMass',
    'diagnostics_Image-interpolated_Spacing',
    'diagnostics_Image-interpolated_Size',
    'diagnostics_Mask-interpolated_Spacing',
    'diagnostics_Mask-interpolated_Size',
    'diagnostics_Mask-interpolated_BoundingBox',
    'diagnostics_Mask-interpolated_CenterOfMassIndex',
    'diagnostics_Mask-interpolated_CenterOfMass',
    'diagnostics_Mask-original_Hash',
]


data = []

for i in range(210):
    case_number = str(i).zfill(5)
    print(case_number)
    case_path = f'G:\\kits\\kits19\\case_{case_number}'
    ct_img_path = os.path.join(case_path, 'imaging.nii.gz')
    segment_path = os.path.join(case_path, 'segmentation.nii.gz')
    
    ct_img = sitk.ReadImage(ct_img_path)
    segment = sitk.ReadImage(segment_path)
    
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    result = extractor.execute(ct_img, segment)
    
    for key in keys_to_remove:
        del result[key]
    
    values = list(result.values())
    data.append(values)

# Normalize the data
normalized_data = np.array(data)
normalized_data = (normalized_data - np.mean(normalized_data, axis=0)) / np.std(normalized_data, axis=0)

np.save(file_path,normalized_data)

print("Normalized data saved to:", file_path)