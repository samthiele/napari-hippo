# Standard library imports
import os
import glob
import pathlib
import copy

# Third-party imports
import numpy as np
from magicgui import magicgui
from natsort import natsorted
import hylite
from hylite import io

# Local imports
from ._guiBase import GUIBase
from napari_hippo import napari_get_hylite_reader
from napari_hippo._base import *

class LibraryWidget(GUIBase):
    """
    Widget for constructing a spectral library in the napari viewer.
    """
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)
        # Setup the magicgui widget for library construction
        self.test_widget = magicgui(
            construct,
            input={'mode': 'd'},
            output_folder={"widget_type": "LineEdit"},
            fingerprints={"filter": "*.json"},
            call_button='Build',
            auto_call=False
        )
        self._add([self.test_widget], 'Construct Library')

def construct(input: pathlib.Path = pathlib.Path(''), 
              output_folder: str = "annotated_lib",
              fingerprints: pathlib.Path = None):
    """
    Main function to construct a spectral library from a directory of samples.
    Args:
        input (pathlib.Path): Input directory containing sample folders.
        output_folder (str): Name of the output folder for results.
        fingerprints (pathlib.Path): Optional path to fingerprint file.
    """
    try:
        import plotly  # Check if plotly is installed for visualization
    except ImportError:
        print('Plotly is not installed. Please install it to visualize the data.')
        return

    if fingerprints is None:
        print('No fingerprints provided')
        fingerprints = pathlib.Path('')

    # Iterate through all subdirectories and build the spectral library
    spectral_lib = None
    for root, dirs, _ in os.walk(input):
        for dir_name in dirs:
            full_dir_path = os.path.join(root, dir_name)
            create_masked_image(full_dir_path)
            if spectral_lib is None:
                spectral_lib = create_masked_spec(full_dir_path)
            else:
                spectral_lib = merge_spectra(spectral_lib, create_masked_spec(full_dir_path))
    
    # Save the merged spectra to files and visualize
    for sensor_name, sensor_data in spectral_lib.items():
        output_path = os.path.join(input.parent, output_folder)
        os.makedirs(output_path, exist_ok=True)
        lib_file = os.path.join(output_path, f"{sensor_name}.lib")
        io.save(lib_file, sensor_data)
        viz_image_data(sensor_name, sensor_data, output_path)
        print(f"Saved {sensor_name} to {output_path}")

def merge_array(arr1, arr2):
    """
    Concatenate two 3D numpy arrays along axis=0, padding as needed along axis=1.
    Args:
        arr1 (np.ndarray): First array.
        arr2 (np.ndarray): Second array.
    Returns:
        np.ndarray: Concatenated array.
    """
    max_rows = max(arr1.shape[1], arr2.shape[1])
    padded_array1 = np.pad(arr1, ((0, 0), (0, max_rows - arr1.shape[1]), (0, 0)), constant_values=np.nan)
    padded_array2 = np.pad(arr2, ((0, 0), (0, max_rows - arr2.shape[1]), (0, 0)), constant_values=np.nan)
    result = np.concatenate((padded_array1, padded_array2), axis=0)
    return result

def merge_spectra(spectra1, spectra2):
    """
    Merge two spectral library dictionaries by concatenating arrays for each sensor.
    Args:
        spectra1 (dict): First spectral library dict.
        spectra2 (dict): Second spectral library dict.
    Returns:
        dict: Merged spectral library dict.
    """
    merged_spectra = {}
    for sensor in spectra1.keys():
        if sensor in spectra2.keys():
            merged = merge_array(spectra1[sensor].data, spectra2[sensor].data)
            merged_spectra[sensor] = hylite.HyLibrary(
                merged,
                wav=spectra1[sensor].get_wavelengths(),
                lab=np.concatenate((spectra1[sensor].get_sample_names(), spectra2[sensor].get_sample_names()))
            )
        else:
            merged_spectra[sensor] = spectra1[sensor]
    for sensor in spectra2.keys():
        if sensor not in merged_spectra:
            merged_spectra[sensor] = spectra2[sensor]
    return merged_spectra

def create_masked_image(folder_path):
    """
    Create a new RGB_masked.png image with mask overlay for a given folder.
    Args:
        folder_path (str): Path to the folder containing RGB and mask files.
    """
    try:
        import cv2
    except ImportError:
        print('cv2 (OpenCV) is not installed. RGB_masked.png will not be created.')
        return

    rgb_path = os.path.join(folder_path, 'RGB.png')
    mask_path = os.path.join(folder_path, 'mask.hdr')
    # Check if files exist
    if not os.path.exists(rgb_path):
        print(f'RGB image not found: {rgb_path}')
        return
    if not os.path.exists(mask_path):
        print(f'Mask file not found: {mask_path}')
        return
    image = cv2.imread(rgb_path)
    mask = io.load(mask_path)
    ratio = int(image.shape[0] / mask.data.shape[1])
    new_width = image.shape[0] // ratio
    new_height = image.shape[1] // ratio 
    output_image = cv2.resize(image, (new_height, new_width))
    mask_data = mask.data.astype(np.uint8)
    mask_data = mask_data.T[0]
    contours, _ = cv2.findContours(mask_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv2.drawContours(output_image, [contour], -1, (0, 0, 255), 2)
    folder_location = os.path.dirname(rgb_path)
    cv2.imwrite(os.path.join(folder_location, 'RGB_masked.png'), output_image)

def create_masked_spec(folder_path):
    """
    Create a masked spectral library for all sensors in a folder.
    Args:
        folder_path (str): Path to the sample folder.
    Returns:
        dict: Dictionary of sensor_name -> HyLibrary
    """
    sample_name = os.path.splitext(os.path.basename(folder_path))[0]
    mask_path = os.path.join(folder_path, 'mask.hdr')
    assert os.path.exists(mask_path), f"Error: {mask_path} does not exist."
    mask = io.load(mask_path)
    sensor_files = glob.glob(os.path.join(folder_path, '*.hdr'))
    sensor_files.remove(mask_path)
    sensors = {}
    for sensor_file in sensor_files:
        sensor_name = os.path.splitext(os.path.basename(sensor_file))[0]
        try:
            sensor_data = io.load(sensor_file)
            sensor_data.data = sensor_data.data / float(io.loadHeader(sensor_file)['reflectance scale factor']) #TODO: check if this is correct
            sensors[sensor_name] = sensor_data
        except Exception as e:
            print(f"Failed to load {sensor_file}: {e}")
            continue
    mask_labels = np.unique(mask.data)
    spectra_dict = {}
    for s_key, s_value in sensors.items():
        if s_value.data.shape[0] == mask.data.shape[0] and s_value.data.shape[1] == mask.data.shape[1]:
            spectra_dict[s_key] = {}
            s_value[:, :, :][mask[:, :, 0] == 0] = np.nan
            sample_spec = None
            measurements = []
            for label in mask_labels:
                if label:
                    measurements.append(sample_name + '_M' + str(int(label)))
                    sample = copy.deepcopy(s_value)
                    sample[:, :, :][mask[:, :, 0] != label] = np.nan
                    sample.data = sample.data.reshape(-1, sample.data.shape[-1])
                    measurement_spec = sample.data[~np.isnan(sample.data).all(axis=1)]
                    if sample_spec is None:
                        sample_spec = measurement_spec[np.newaxis, :, :]
                    else:
                        sample_spec = merge_array(sample_spec, measurement_spec[np.newaxis, :, :])
        else:
            print(f"Error: The shape of {s_key} does not match the mask size.")
            continue
        if sample_spec is not None:
            lib = hylite.HyLibrary(sample_spec, wav=s_value.get_wavelengths(), lab=measurements)
            spectra_dict[s_key] = lib
    return spectra_dict

def viz_image_data(sensor_name, sensor_lib, output_path):
    """
    Visualize the spectral library for a given sensor and save as HTML.
    Args:
        sensor_name (str): Name of the sensor.
        sensor_lib (HyLibrary): Spectral library object.
        output_path (str): Directory to save the visualization.
    """
    import plotly.graph_objects as go
    import plotly.express as px
    # Define colors for plotting
    colors = px.colors.qualitative.Plotly
    def hex_rgba(hex, transparency):
        col_hex = hex.lstrip('#')
        col_rgb = [int(col_hex[i:i+2], 16) for i in (0, 2, 4)]
        col_rgb.append(transparency)
        return tuple(col_rgb)
    rgba = [hex_rgba(c, transparency=0.2) for c in colors]
    colCycle = ['rgba'+str(elem) for elem in rgba]
    def next_col(cols):
        while True:
            for col in cols:
                yield col
    line_color = next_col(cols=colCycle)
    fig = go.Figure()
    # Plot each sample in the library
    for i in range(sensor_lib.data.shape[0]):
        color = next(line_color)
        x = sensor_lib.get_wavelengths()
        mean = np.nanmean(sensor_lib.data[i], axis=0)
        std = np.nanstd(sensor_lib.data[i], axis=0)
        y_upper = mean + std
        y_lower = mean - std
        # Add shaded region for std
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill='tozeroy',
            fillcolor=color,
            line=dict(color='rgba(255,255,255,0)'),
            name=str(sensor_lib[i].get_sample_names()[0]),
            hoverinfo='none',
            showlegend=True,
            legendgroup=str(sensor_lib[i].get_sample_names()[0])
        ))
        # Add mean line
        fig.add_traces(go.Scatter(
            x=x,
            y=mean,
            line=dict(color=color, width=2.5),
            mode='lines',
            showlegend=False,
            legendgroup=str(sensor_lib[i].get_sample_names()[0])
        ))
    fig.update_layout(
        xaxis_title='Wavelength',
        title={
            'text': sensor_name,
            'font': {'size': 24, 'color': 'black', 'family': 'Arial', 'weight': 'bold'},
            'x': 0.5,
            'xanchor': 'center'
        },
        hovermode="x unified"
    )
    fig.update_traces(textposition='top center')
    fig.write_html(os.path.join(output_path, sensor_name + ".html"))
    fig.show()





