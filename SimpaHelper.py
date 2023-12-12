from simpa import Tags
import simpa as sp
from src.utils.cyberdyne_led_array_system import CyberdyneLEDArraySystem
import numpy as np
from simpa.io_handling import load_hdf5
import matplotlib.pyplot as plt
import h5py
import re
import os
import logging
import shutil

logging.disable(logging.CRITICAL)


class SimpaHelper:
    """
    Simpa Helper is helper class to pour simpa methods into usable methods for
    dealing and analyzing simulated images on a larger scale.
    """
    def __init__(self, hdf5_file_path):
        self.reconstructed_data = None
        self.sinogram = None
        self.hdf5_file_path = hdf5_file_path
        self.data_field = Tags.DATA_FIELD_RECONSTRUCTED_DATA
        self.wavelength = 850
        self.path_manager = sp.PathManager()
        self.dict_path = self._generate_dict_path()
        self.save_path = self.hdf5_file_path.rsplit('/', 1)[0]

    def _generate_dict_path(self):
        """
        Generates a path within an hdf5 file in the SIMPA convention

        :param data_field: Data field that is supposed to be stored in an hdf5 file.
        :param wavelength: Wavelength of the current simulation.
        :return: String which defines the path to the data_field.
        """
        if self.data_field in [Tags.SIMULATIONS, Tags.SETTINGS, Tags.DIGITAL_DEVICE, Tags.SIMULATION_PIPELINE]:
            return "/" + self.data_field + "/"

        wavelength_dependent_properties = [Tags.DATA_FIELD_ABSORPTION_PER_CM,
                                           Tags.DATA_FIELD_SCATTERING_PER_CM,
                                           Tags.DATA_FIELD_ANISOTROPY]

        wavelength_independent_properties = [Tags.DATA_FIELD_OXYGENATION,
                                             Tags.DATA_FIELD_SEGMENTATION,
                                             Tags.DATA_FIELD_GRUNEISEN_PARAMETER,
                                             Tags.DATA_FIELD_SPEED_OF_SOUND,
                                             Tags.DATA_FIELD_DENSITY,
                                             Tags.DATA_FIELD_ALPHA_COEFF,
                                             Tags.KWAVE_PROPERTY_SENSOR_MASK,
                                             Tags.KWAVE_PROPERTY_DIRECTIVITY_ANGLE]

        simulation_output = [Tags.DATA_FIELD_FLUENCE,
                             Tags.DATA_FIELD_INITIAL_PRESSURE,
                             Tags.OPTICAL_MODEL_UNITS,
                             Tags.DATA_FIELD_TIME_SERIES_DATA,
                             Tags.DATA_FIELD_RECONSTRUCTED_DATA,
                             Tags.DATA_FIELD_DIFFUSE_REFLECTANCE,
                             Tags.DATA_FIELD_DIFFUSE_REFLECTANCE_POS,
                             Tags.DATA_FIELD_PHOTON_EXIT_POS,
                             Tags.DATA_FIELD_PHOTON_EXIT_DIR]

        simulation_output_fields = [Tags.OPTICAL_MODEL_OUTPUT_NAME,
                                    Tags.SIMULATION_PROPERTIES]

        wavelength_dependent_image_processing_output = [Tags.ITERATIVE_qPAI_RESULT]

        wavelength_independent_image_processing_output = [Tags.LINEAR_UNMIXING_RESULT]

        if self.wavelength is not None:
            wl = "/{}/".format(self.wavelength)

        if self.data_field in wavelength_dependent_properties:
            if self.wavelength is not None:
                dict_path = "/" + Tags.SIMULATIONS + "/" + Tags.SIMULATION_PROPERTIES + "/" + self.data_field + wl
            else:
                dict_path = "/" + Tags.SIMULATIONS + "/" + Tags.SIMULATION_PROPERTIES + "/" + self.data_field
        elif self.data_field in simulation_output:
            if self.data_field in [Tags.DATA_FIELD_FLUENCE, Tags.DATA_FIELD_INITIAL_PRESSURE, Tags.OPTICAL_MODEL_UNITS,
                              Tags.DATA_FIELD_DIFFUSE_REFLECTANCE, Tags.DATA_FIELD_DIFFUSE_REFLECTANCE_POS,
                              Tags.DATA_FIELD_PHOTON_EXIT_POS, Tags.DATA_FIELD_PHOTON_EXIT_DIR]:
                if self.wavelength is not None:
                    dict_path = "/" + Tags.SIMULATIONS + "/" + Tags.OPTICAL_MODEL_OUTPUT_NAME + "/" + self.data_field + wl
                else:
                    dict_path = "/" + Tags.SIMULATIONS + "/" + Tags.OPTICAL_MODEL_OUTPUT_NAME + "/" + self.data_field
            else:
                if self.wavelength is not None:
                    dict_path = "/" + Tags.SIMULATIONS + "/" + self.data_field + wl
                else:
                    dict_path = "/" + Tags.SIMULATIONS + "/" + self.data_field

        elif self.data_field in wavelength_independent_properties:
            dict_path = "/" + Tags.SIMULATIONS + "/" + Tags.SIMULATION_PROPERTIES + "/" + self.data_field + "/"
        elif self.data_field in simulation_output_fields:
            dict_path = "/" + Tags.SIMULATIONS + "/" + self.data_field + "/"
        elif self.data_field in wavelength_dependent_image_processing_output:
            if self.wavelength is not None:
                dict_path = "/" + Tags.IMAGE_PROCESSING + "/" + self.data_field + wl
            else:
                dict_path = "/" + Tags.IMAGE_PROCESSING + "/" + self.data_field

        elif self.data_field in wavelength_independent_image_processing_output:
            dict_path = "/" + Tags.IMAGE_PROCESSING + "/" + self.data_field + "/"
        else:
            raise ValueError(
                "The requested data_field is not a valid argument. Please specify a valid data_field using "
                "the Tags from simpa/utils/tags.py!")

        return dict_path

    def _get_simpa_output(self, simpa_output: dict):
        """
        Navigates through a dictionary in the standard simpa output format to a specific data field.

        :param simpa_output: Dictionary that is in the standard simpa output format.
        :param data_field: Data field that is contained in simpa_output.
        :param wavelength: Wavelength of the current simulation.
        :return: Queried data_field.
        """

        dict_path = self.dict_path
        keys_to_data_field = dict_path.split("/")
        current_dict = simpa_output
        for key in keys_to_data_field:
            if key == "":
                continue
            current_dict = current_dict[key]

        return current_dict

    # TODO: Add functionality to load sinogram from file to reconstruct
    def reconstruct(self, custom_sinogram_path=None, sinogram_data=None):
        settings = sp.load_data_field(self.hdf5_file_path, Tags.SETTINGS)
        settings[Tags.WAVELENGTH] = settings[Tags.WAVELENGTHS][0]
        settings[Tags.IGNORE_QA_ASSERTIONS] = True
        settings[Tags.SIMPA_OUTPUT_PATH] = self.hdf5_file_path

        settings.set_reconstruction_settings({
            Tags.ACOUSTIC_MODEL_BINARY_PATH: self.path_manager.get_matlab_binary_path(),
            Tags.ACOUSTIC_SIMULATION_3D: True,
            Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
            Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
            Tags.KWAVE_PROPERTY_PMLInside: False,
            Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
            Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
            Tags.KWAVE_PROPERTY_PlotPML: False,
            Tags.RECORDMOVIE: False,
            Tags.MOVIENAME: "visualization_log",
            Tags.ACOUSTIC_LOG_SCALE: True,
            Tags.DATA_FIELD_SPEED_OF_SOUND: 1480,
            Tags.DATA_FIELD_ALPHA_COEFF: 0.01,
            Tags.DATA_FIELD_DENSITY: 1000,
            Tags.SPACING_MM: settings[Tags.SPACING_MM],
            Tags.SENSOR_SAMPLING_RATE_MHZ: 40,
            # Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
            # Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: True,
        })

        dim_x_mm = settings[Tags.DIM_VOLUME_X_MM]
        dim_y_mm = settings[Tags.DIM_VOLUME_Y_MM]
        dim_z_mm = settings[Tags.DIM_VOLUME_Z_MM]
        spacing = settings[Tags.SPACING_MM]

        device = CyberdyneLEDArraySystem(device_position_mm=np.array([dim_x_mm / 2, dim_y_mm / 2, 0]),
                                         field_of_view_extent_mm=np.asarray([-10, 10, -2, 2, 2, 201]))

        sp.DelayAndSumAdapter(settings).run(device)

        path_to_hdf5_file = self.hdf5_file_path
        print('PATH', path_to_hdf5_file)

        file = load_hdf5(path_to_hdf5_file)
        data = self._get_simpa_output(file)
        self.reconstructed_data = np.rot90(data[:, :], -1)

    # TODO: Dynamic naming
    def extract_sinogram(self, save_to_file=False):
        with h5py.File(self.hdf5_file_path, 'r') as file:
            sinogram = file['simulations']['time_series_data']['850'][:]
            if save_to_file:
                pattern = re.compile(r'(\d+)_gt')
                match = pattern.search(self.save_path)
                if match:
                    extracted_number = match.group(1)
                    print("Extracted Number:", extracted_number)
                else:
                    print("Number not found in the file path, using 999 as number")
                    extracted_number = 999
                np.savez(os.path.join(self.save_path, f'{extracted_number}_sinogram.npz'), sinogram=sinogram)
            return sinogram
    def visualize_reconstruction(self):
        # TODO:Save figure
        # TODO: Log scale
        # TODO: oringal image
        if self.reconstructed_data is not None:
            plt.imshow(self.reconstructed_data, cmap='viridis')
            plt.colorbar()
        else:
            print("No reconstruction has been perfomred yet, use: 'reconstruct()'")


    def show_sinogram(self, plot_title='Sinogram', show=False, isNumpy=False):
        if self.sinogram is None:
            self.sinogram = self.extract_sinogram()

        Nx, Nt = np.shape(self.sinogram)
        xlim = Nt / 40e6

        plt.figure()
        plt.title(plot_title)
        if isNumpy:
            plt.imshow(self.sinogram, aspect=xlim / Nx, extent=[0, xlim, Nx, 0], cmap="coolwarm", vmin=-15, vmax=15)
        else:
            plt.imshow(self.sinogram, aspect=xlim / Nx, extent=[0, xlim, Nx, 0], cmap="coolwarm", vmin=-500, vmax=500)
        plt.xlabel("Time [s]")
        plt.ylabel("Detector index")
        plt.colorbar(label="Pressure [a.u.]")

        if show:
            plt.show()

    def save_sinogram(self, sinogram, output_filename):
        data_path = 'simulations/time_series_data/850'

        shutil.copy(self.hdf5_file_path, output_filename)

        # Open the copied file in read-write mode
        with h5py.File(output_filename, 'a') as file:
            # Modify the specific entry
            file[data_path][...] = sinogram
