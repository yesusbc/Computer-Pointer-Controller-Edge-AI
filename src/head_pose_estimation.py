"""
Base class for Head Pose detection Model
head-pose-estimation-adas-0001
name: "data" , shape: [1x3x60x60] - An input image in [1xCxHxW] format. Expected color order is BGR

Output layer names in Inference Engine format:
name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).
"""
import sys
import logging as log
from openvino.inference_engine import IECore
import cv2
import numpy as np

EXTENSIONS_PATH = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
FACE_MODEL_PATH = "../intel_models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001"


class HeadPoseDetectionModel:
    """
    Class for the Head Pose Detection Model.
    """
    def __init__(self, model_path=FACE_MODEL_PATH, device="CPU", extensions=None):
        """
        Set instance variables.
        """
        self.ie = IECore()
        self.net = None
        self.exec_net = None

        self.model_weights = model_path+".bin"
        self.model_structure = model_path+".xml"
        self.device = device
        self.extensions = extensions

        try:
            # Read the IR as an IENetwork
            self.net = self.ie.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you entered the correct model path? \
                            Check for {0}".format(e))

        self.input_name = next(iter(self.net.inputs))
        self.input_shape = self.net.inputs[self.input_name].shape
        self.yaw_output_name = "angle_y_fc"
        self.yaw_output_shape = self.net.outputs[self.yaw_output_name].shape
        self.pitch_output_name = "angle_p_fc"
        self.pitch_output_shape = self.net.outputs[self.pitch_output_name].shape
        self.roll_output_name = "angle_r_fc"
        self.roll_output_shape = self.net.outputs[self.roll_output_name].shape

    def load_model(self):
        """
        Check for supported layers and add extensions if necessary
        Initialize Inference Engine, to work with the plugin
        Load the IENetwork into the plugin
        """

        # Add any necessary extension
        if self.extensions and self.device == "CPU":
            self.ie.add_extension(self.extensions, self.device)
            self.ie.add_extension(extension_path=EXTENSIONS_PATH, device_name=self.device)

        # Get the supported layers of the network
        layers_map = self.ie.query_network(network=self.net, device_name=self.device)

        # Look for unsupported layers
        unsupported_layers = [layer for layer in self.net.layers.keys() if layer not in layers_map]

        # If there're unsupported layers, notify and exit
        if len(unsupported_layers):
            log.error("There were unsupported layers on the network, try checking if path \
                      on --cpu_extension is correct. The unsupported layers were: {0}\
                      ".format(unsupported_layers))
            sys.exit(1)

        # Load the IENetwork into the plugin
        self.exec_net = self.ie.load_network(self.net, self.device, num_requests=1)

    def predict(self, image):
        """
        Make inference over the exectutable network
        """
        p_frame = self.preprocess_input(image)
        outputs = self.exec_net.infer({self.input_name: p_frame})

        yaw = outputs[self.yaw_output_name][0][0]
        pitch = outputs[self.pitch_output_name][0][0]
        roll = outputs[self.roll_output_name][0][0]
        head_pose_angles = [yaw, pitch, roll]
        # TODO: Head Pose draw
        return head_pose_angles

    def preprocess_input(self, image):
        """
        Given an input image, height and width:
        - Resize to height and width
        - Transpose the final "channel" dimension to be first
        - Reshape the image to add a "batch" of 1 at the start
        """
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame


