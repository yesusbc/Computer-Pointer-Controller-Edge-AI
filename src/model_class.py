import sys
import logging as log
from openvino.inference_engine import IECore
import cv2


class Model:
    """
    Superclass Model
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
        pass

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
        pass
