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


class LibraryWidget(GUIBase):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self.test_widget = magicgui(construct, input={'mode': 'd'}, output_folder={"widget_type": "LineEdit"},
                                    fingerprints={"filter": "*.json"}, call_button='Build', auto_call=False)
        self._add([self.test_widget], 'Construct Library')

def construct( input : pathlib.Path = pathlib.Path(''), 
               output_folder : str = "annotated_lib",
               fingerprints : pathlib.Path = None ):
    
    if fingerprints is None:
        print('No fingerprints provided')
        fingerprints = pathlib.Path('')
    print( input, output_folder, fingerprints )

    # Create masked image
    spectral_lib = None
    for root, dirs, _ in os.walk(input):
        for dir_name in dirs:
            full_dir_path = os.path.join(root, dir_name)
            create_masked_image(full_dir_path)
            if spectral_lib is None:
                spectral_lib = create_masked_spec(full_dir_path)
            else:
                spectral_lib = merge_spectra(spectral_lib, create_masked_spec(full_dir_path))
    
    # Save the merged spectra to a file
    for sensor_name, sensor_data in spectral_lib.items():
        # Save each sensor's data to a separate file
        output_path = os.path.join(input.parent, output_folder)
        lib_file = os.path.join(output_path, f"{sensor_name}.lib")
        io.save(lib_file, sensor_data)
        viz_image_data(sensor_name, sensor_data, output_path)
        print(f"Saved {sensor_name} to {output_path}")

def viz_image_data(sensor_name, sensor_lib, output_path):
    import plotly.graph_objects as go
    import plotly.express as px
    ### For colour
    # define colors as a list 
    colors = px.colors.qualitative.Plotly
    # convert plotly hex colors to rgba to enable transparency adjustments
    def hex_rgba(hex, transparency):
        col_hex = hex.lstrip('#')
        col_rgb = list(int(col_hex[i:i+2], 16) for i in (0, 2, 4))
        col_rgb.extend([transparency])
        areacol = tuple(col_rgb)
        return areacol
    rgba = [hex_rgba(c, transparency=0.2) for c in colors]
    colCycle = ['rgba'+str(elem) for elem in rgba]
    # Make sure the colors run in cycles if there are more lines than colors
    def next_col(cols):
        while True:
            for col in cols:
                yield col
    line_color=next_col(cols=colCycle)
    ########
    fig = go.Figure()
    # for hylib_key, df in hylib_dict.items():
    for i in range(sensor_lib.data.shape[0]):
        color = next(line_color)
        # x = [float(c) for c in df.columns]
        x = sensor_lib.get_wavelengths()
        y_upper = []
        y_mean = []
        y_lower = []
        mean = np.nanmean(sensor_lib.data[i], axis=0)
        std = np.nanstd(sensor_lib.data[i], axis=0)
        y_upper = mean+std
        y_lower = mean-std
        
        # Add the shaded region
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),  # x, then x reversed
            y=np.concatenate([y_upper, y_lower[::-1]]),  # upper, then lower reversed
            fill='tozeroy',  # or 'tozerox' if you have y=constant
            fillcolor=color,  # Colour with some transparency
            line=dict(color='rgba(255,255,255,0)'),  # No line
            name=str(sensor_lib[i].get_sample_names()[0]),  # Use the first sample name as the legend label
            hoverinfo='none', # Remove hover info for shaded area
            showlegend=True, # Make sure this is True initially
            legendgroup=str(sensor_lib[i].get_sample_names()[0])
        ))

        # line trace
        fig.add_traces(go.Scatter(x=x,
                                y=y_mean,
                                line=dict(color=color, width=2.5),
                                mode='lines',
                                showlegend=False,
                                legendgroup=str(sensor_lib[i].get_sample_names()[0])
                                )
                                    )
    
    # text = batch_name + '_' + sensor
    fig.update_layout(
        # yaxis_range=[0, 1],
        xaxis_title='Wavelength',
        # yaxis_title='Reflectance',
        title={
            'text': sensor_name,
            'font': {
                'size': 24,
                'color': 'black',
                'family': 'Arial',
                'weight': 'bold'
            },
            'x': 0.5,
            'xanchor': 'center'
        },
    )
    fig.update_layout(hovermode="x unified")
    fig.update_traces(textposition='top center')
    fig.write_html(os.path.join(output_path, sensor_name + ".html"))
     
    fig.show()
    pass

def merge_spectra(spectra1, spectra2):
    """
    Merges two spectra dictionaries by concatenating the arrays for each sensor.
    """
    merged_spectra = {}

    # Iterate through the keys (sensors) in the first spectra dictionary
    for sensor in spectra1.keys():
        if sensor in spectra2.keys():
            # Concatenate the arrays for the same sensor
            merged = merge_array(spectra1[sensor].data, spectra2[sensor].data)
            merged_spectra[sensor] = hylite.HyLibrary(merged, wav=spectra1[sensor].get_wavelengths()
                                                      , lab=np.concatenate((spectra1[sensor].get_sample_names(), 
                                                                            spectra2[sensor].get_sample_names())))
        else:
            # If the sensor is not in the second dictionary, keep it as is
            merged_spectra[sensor] = spectra1[sensor]

    # Add any sensors from the second dictionary that are not in the first
    for sensor in spectra2.keys():
        if sensor not in merged_spectra:
            merged_spectra[sensor] = spectra2[sensor]

    return merged_spectra

def merge_array(arr1, arr2):
    import numpy as np

    # Determine the maximum size along axis=1 to pad arrays
    max_rows = max(arr1.shape[1], arr2.shape[1])

    # Pad both arrays along axis=1 to match the maximum size
    padded_array1 = np.pad(arr1, ((0, 0), (0, max_rows - arr1.shape[1]), (0, 0)), constant_values=np.nan)
    padded_array2 = np.pad(arr2, ((0, 0), (0, max_rows - arr2.shape[1]), (0, 0)), constant_values=np.nan)
    # Concatenate along axis=0
    result = np.concatenate((padded_array1, padded_array2), axis=0)

    return result


def create_masked_image(folder_path):
    import cv2
    '''
    This function will create a new RGB_masked.png image.
    The image size is similar to the sensors
    '''
    rgb_path = os.path.join(folder_path, 'RGB.png') # RGB.hdr
    # convert RGB.hdr to png image
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
    sample_name = os.path.splitext(os.path.basename(folder_path))[0]
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
        sample_spec = None
        measurements = []
        for label in mask_labels:
            if label:
                measurements.append(sample_name+'_M' + str(int(label)))
                sample = copy.deepcopy(s_value)
                sample[:,:,:][mask[:,:,0]!=label] = np.nan
                sample.data = sample.data.reshape(-1,sample.data.shape[-1])
                measurement_spec = sample.data[~np.isnan(sample.data).all(axis=1)]
                if sample_spec is None:
                    sample_spec = measurement_spec[np.newaxis, :, :]
                else:
                    sample_spec = merge_array(sample_spec, measurement_spec[np.newaxis, :, :])
                # Save non_nan_values as sample_sensor.lib in a folder
        if sample_spec is not None:
            lib = hylite.HyLibrary(sample_spec, wav=s_value.get_wavelengths(),
                                lab=measurements)
            spectra_dict[s_key] = lib

    return spectra_dict




        



