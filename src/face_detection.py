"""
Base class for face detection Model
face-detection-adas-binary-0001
input shape [1x3x384x672] - An input image in the format [BxCxHxW]
The net outputs blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes. For each detection,
the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
"""
import sys
import logging as log
from openvino.inference_engine import IECore
import cv2
import time


class FaceDetectionModel:
    """
    Class for the Face Detection Model.
    """
    def __init__(self, model_path, device="CPU", extensions=None, threshold=0.60):
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
        self.threshold = threshold

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
            self.ie.add_extension(extension_path=self.extensions, device_name=self.device)    # self.extensions

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

        coords = self.preprocess_outputs(outputs[self.output_name])
        coords = coords[0]
        # cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (255,0,0), 2)
        # cv2.imshow("img",image)
        # cv2.waitKey(0)
        cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]
        return cropped_face, coords, prediction_time

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

    def preprocess_outputs(self, outputs):
        """
        This function applies a probability threshold to the output data.
        The return will contain the related coordinates of the prediction, resized to the original image size
        """
        coords = []
        probs = outputs[0, 0, :, 2]

        for idx, prob in enumerate(probs):
            if prob >= self.threshold:
                box = outputs[0, 0, idx, 3:]
                xmin = int(box[0] * self.w)
                ymin = int(box[1] * self.h)
                xmax = int(box[2] * self.w)
                ymax = int(box[3] * self.h)
                coords.append((xmin, ymin, xmax, ymax))
        return coords
