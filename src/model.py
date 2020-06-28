"""
Base class for face detection Model
face-detection-adas-binary-0001
input shape [1x3x384x672] - An input image in the format [BxCxHxW]
The net outputs blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes. For each detection,
the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
"""
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

class FaceDetectionModel:
    """
    Class for the Face Detection Model.
    """
    def __init__(self, model_name, device="CPU", extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.network = None
        self.plugin = None
        self.exec_network = None
        self.input_blob = None
        self.output_blob = None
        self.model_name = model_name
        self.device = device
        self.extensions = extensions


    def load_model(self):
        """
        Load Model given IR Files to the device specified by the user.
        Check for supported layers and add extensions if necessary
        """

        # Initialize the plugin
        self.plugin = IECore()

        # Read the IR as an IENetwork
        self.network = IENetwork(model=self.model_name+".xml", weights=self.model_name+".bin")

        # Add any necessary extension
        if self.extensions  and self.device=="CPU":
            self.pĺugin.add_extension(self.extensions, self.device)

        # Get the supported layers of the network
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)


        # Look for unsupported layers
        unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]

        # If there're unsupported layers, notify and exit
        if len(supported_layers):
            log.error("There were unsupported layers on the network, try checking if path \
                      on --cpu_extension is correct. The unsupported layers were: {0}\
                      ".format(unsupported_layers))
            sys.exit(1)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, self.device)


    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        input_shape = self.exec_network.get_input_shape()
        input_name = self.exec_network.get_input_name()
        raise NotImplementedError

    def preprocess_output(self, outputs):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        raise NotImplementedError
