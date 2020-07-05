"""
Base class for landmarks detection Model
landmarks-regression-retail-0009
Name: "data" , shape: [1x3x48x48] - An input image in the format [BxCxHxW]

The net outputs a blob with the shape: [1, 10], containing a row-vector of 10 floating point values for five landmarks
coordinates in the form (x0, y0, x1, y1, ..., x5, y5). All the coordinates are normalized to be in range [0,1].
"""
import sys
import logging as log
from openvino.inference_engine import IECore
import cv2
import time


class LandmarksDetectionModel:
    """
    Class for the landmarks Detection Model.
    """
    def __init__(self, model_path, device="CPU", extensions=None):
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
        self.output_name = next(iter(self.net.outputs))
        self.output_shape = self.net.outputs[self.output_name].shape
        self.w = None
        self.h = None

    def load_model(self):
        """
        Check for supported layers and add extensions if necessary
        Initialize Inference Engine, to work with the plugin
        Load the IENetwork into the plugin
        """

        # Add any necessary extension
        if self.extensions and self.device == "CPU":
            self.ie.add_extension(self.extensions, self.device)
            self.ie.add_extension(extension_path=self.extensions, device_name=self.device)

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
        self.w = image.shape[1]
        self.h = image.shape[0]
        p_frame = self.preprocess_input(image)

        start_time = time.time()
        outputs = self.exec_net.infer({self.input_name: p_frame})
        prediction_time = time.time() - start_time

        left_eye, right_eye = self.preprocess_outputs(outputs[self.output_name])

        # Left eye coords and cropped image
        y_left_eye = int(left_eye[1])
        x_left_eye = int(left_eye[0])
        cropped_left_eye = image[(y_left_eye-15):(y_left_eye+15), (x_left_eye-15):(x_left_eye+15)]

        # Right eye coords and cropped image
        y_right_eye = int(right_eye[1])
        x_right_eye = int(right_eye[0])
        cropped_right_eye = image[(y_right_eye-15):(y_right_eye+15), (x_right_eye-15):(x_right_eye+15)]

        # coords from every eye in the form [[eye1_coords], [eye2_coords]]
        eyes_coords = [[(x_left_eye-15, y_left_eye-15), (x_left_eye+15, y_left_eye+15)],
                       [(x_right_eye-15, y_right_eye-15), (x_right_eye+15, y_right_eye+15)]]

        cv2.rectangle(image, (eyes_coords[0][0][0], eyes_coords[0][0][1]),
                      (eyes_coords[0][1][0], eyes_coords[0][1][1]), (255, 0, 0), 2)
        cv2.rectangle(image, (eyes_coords[1][0][0], eyes_coords[1][0][1]),
                      (eyes_coords[1][1][0], eyes_coords[1][1][1]), (255, 0, 0), 2)

        return cropped_left_eye, cropped_right_eye, eyes_coords, prediction_time

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

    def preprocess_outputs(self, landmarks_vector):
        """
        The return will contain the related coordinates of the prediction, resized to the original image size
        """
        left_eye = (landmarks_vector[0][0][0][0]*self.w, landmarks_vector[0][1][0][0]*self.h)
        right_eye = (landmarks_vector[0][2][0][0]*self.w, landmarks_vector[0][3][0][0]*self.h)
        return left_eye, right_eye
