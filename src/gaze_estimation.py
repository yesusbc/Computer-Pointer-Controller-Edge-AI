"""
Base class for Gaze Estimation Model
gaze-estimation-adas-0002

The network takes three inputs: square crop of left eye image, square crop of right eye image, and three head pose
angles – (yaw, pitch, and roll). The network outputs 3-D vector corresponding to the direction of a person's gaze
in a Cartesian coordinate system in which z-axis is directed from person's eyes (mid-point between left and right eyes'
centers) to the camera center, y-axis is vertical, and x-axis is orthogonal to both z,y axes so that (x,y,z) constitute
a right-handed coordinate system.
"""
import sys
import logging as log
from openvino.inference_engine import IECore
import cv2
import time


class GazeEstimationModel:
    """
    Class for the Gaze EStimation Model.
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

        self.input_name = "left_eye_image"  # Same as Right Eye
        self.input_shape = self.net.inputs[self.input_name].shape
        self.output_name = next(iter(self.net.outputs))
        self.output_shape = self.net.outputs[self.output_name].shape

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

    def predict(self, left_eye, right_eye, head_pose_angles, cropped_face, eyes_coords):
        """
        Make inference over the exectutable network
        """
        p_left_eye = self.preprocess_input(left_eye)
        p_right_eye = self.preprocess_input(right_eye)

        start_time = time.time()
        outputs = self.exec_net.infer({"head_pose_angles": head_pose_angles,
                                       "left_eye_image": p_left_eye,
                                       "right_eye_image": p_right_eye
                                       })
        prediction_time = time.time() - start_time

        x = round(outputs[self.output_name][0][0], 4)
        y = round(outputs[self.output_name][0][1], 4)
        z = outputs[self.output_name][0][2]

        center_x_left_eye = int((eyes_coords[0][1][0] - eyes_coords[0][0][0])/2 + eyes_coords[0][0][0])
        center_y_left_eye = int((eyes_coords[0][1][1] - eyes_coords[0][0][1])/2 + eyes_coords[0][0][1])
        new_x_left_eye = int(center_x_left_eye + x*40)
        new_y_left_eye = int(center_y_left_eye + y*40*-1)
        cv2.line(cropped_face, (center_x_left_eye, center_y_left_eye), (new_x_left_eye, new_y_left_eye), (0, 255, 0), 2)

        center_x_right_eye = int((eyes_coords[1][1][0] - eyes_coords[1][0][0])/2 + eyes_coords[1][0][0])
        center_y_right_eye = int((eyes_coords[1][1][1] - eyes_coords[1][0][1])/2 + eyes_coords[1][0][1])
        new_x_right_eye = int(center_x_right_eye + x*40)
        new_y_right_eye = int(center_y_right_eye + y*40*-1)
        cv2.line(cropped_face, (center_x_right_eye, center_y_right_eye), (new_x_right_eye, new_y_right_eye), (0, 255, 0), 2)

        return x, y, z, prediction_time

    def preprocess_input(self, image):
        """
        Given an input image, height and width:
        - Resize to height and width
        - Transpose the final "channel" dimension to be first
        - Reshape the image to add a "batch" of 1 at the start
        """
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame
