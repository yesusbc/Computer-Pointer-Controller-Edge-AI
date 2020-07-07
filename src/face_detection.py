"""
Base class for face detection Model
face-detection-adas-binary-0001
input shape [1x3x384x672] - An input image in the format [BxCxHxW]
The net outputs blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes. For each detection,
the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
"""
from model_class import Model
import time


class FaceDetectionModel(Model):
    """
    Class for the Face Detection Model.
    """
    def __init__(self, model_path, device="CPU", extensions=None, threshold=0.60):
        super().__init__(model_path, device, extensions)
        self.threshold = threshold

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
