from ._guiBase import GUIBase
from magicgui import magicgui

import pathlib
import numpy as np
from natsort import natsorted
import glob
import napari
from napari_hippo import napari_get_hylite_reader
from napari_hippo._base import *
import os
import hylite
from hylite import io
import copy
import struct

class LibraryWidget(GUIBase):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self.test_widget = magicgui(construct, input={'mode': 'd'}, output={'mode': 'd'},
                                    fingerprints={"filter": "*.json"}, call_button='Build', auto_call=False)
        self._add([self.test_widget], 'Construct Library')

def construct( input : pathlib.Path = pathlib.Path(''), 
               output : pathlib.Path = None,
               fingerprints : pathlib.Path = None ):
    if output is None:
        print('No output provided')
        output = input
    if fingerprints is None:
        print('No fingerprints provided')
        fingerprints = pathlib.Path('')
    print( input, output, fingerprints )

    # Create masked image
    spectral_lib = None
    for root, dirs, _ in os.walk(input):
        for dir_name in dirs:
            sample_name = dir_name + "_"
            full_dir_path = os.path.join(root, dir_name)
            create_masked_image(full_dir_path)
            if spectral_lib is None:
                spectral_lib = create_masked_spec(full_dir_path)
            else:
                spectral_lib = merge_array(spectral_lib, create_masked_spec(full_dir_path))

def merge_array(arr1, arr2):
    import numpy as np

    # Determine the maximum size along axis=0 to pad arrays
    max_rows = max(arr1.shape[0], arr2.shape[0])

    # Pad both arrays along axis=0 to match the maximum size
    padded_array1 = np.pad(arr1, ((0, max_rows - arr1.shape[0]), (0, 0), (0, 0)), constant_values=np.nan)
    padded_array2 = np.pad(arr2, ((0, max_rows - arr2.shape[0]), (0, 0), (0, 0)), constant_values=np.nan)
    # Concatenate along axis=1
    result = np.concatenate((padded_array1, padded_array2), axis=1)

    print("Resulting shape:", result.shape)

def create_masked_image(folder_path):
    import cv2
    '''
    This function will create a new RGB_masked.png image.
    The image size is similar to the sensors
    '''
    rgb_path = os.path.join(folder_path, 'RGB.png')
    mask_path = os.path.join(folder_path, 'mask.hdr')
    image = cv2.imread(rgb_path)
    mask = io.load(mask_path)
    # Resizing to sensor frame size
    new_width = image.shape[0] // 6
    new_height = image.shape[1] // 6
    output_image = cv2.resize(image, (new_height, new_width))

    # Add mask on image
    mask_data = mask.data.astype(np.uint8)
    mask_data = mask_data.T[0]
    contours, _ = cv2.findContours(mask_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        cv2.drawContours(output_image, [contour], -1, (0, 0, 255), 2)
    folder_location = os.path.dirname(rgb_path)
    cv2.imwrite(os.path.join(folder_location,'RGB_masked.png'), output_image)

def create_masked_spec(folder_path):

    mask_path = os.path.join(folder_path, 'mask.hdr')
    # Assert that mask.hdr is present
    assert os.path.exists(mask_path), f"Error: {mask_path} does not exist."
    mask = io.load(mask_path)

    # Load all header files except mask.hdr
    sensor_files = glob.glob(os.path.join(folder_path, '*.hdr'))
    sensor_files.remove(mask_path)
    sensors = {}
    for sensor_file in sensor_files:
        sensor_name = os.path.splitext(os.path.basename(sensor_file))[0]
        try:
            sensor_data = io.load(sensor_file)
            sensor_data.data = sensor_data.data / float(io.loadHeader(sensor_file)['reflectance scale factor'])
            sensors[sensor_name] = sensor_data
        except:
            continue
        
    mask_labels = np.unique(mask.data)
    spectra_dict = {}
    for s_key, s_value in sensors.items():
        spectra_dict[s_key] = {}
        s_value[:,:,:][mask[:,:,0]==0] = np.nan
        for label in mask_labels:
            if label:
                spectra_dict[s_key][label] = {}
                # if the path not exist then create header file
                # otherwise just load the file
                sample = copy.deepcopy(s_value)
                sample[:,:,:][mask[:,:,0]!=label] = np.nan
                sample.data = sample.data.reshape(-1,sample.data.shape[-1])
                sample_spec = sample.data[~np.isnan(sample.data).all(axis=1)]
                # Save non_nan_values as sample_sensor.lib in a folder
                lib = hylite.HyLibrary(sample_spec, wav=s_value.get_wavelengths())
                spectra_dict[s_key][int(label)] = lib

    return spectra_dict




        



