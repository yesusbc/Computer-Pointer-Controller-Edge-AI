"""
Base class for landmarks detection Model
landmarks-regression-retail-0009
Name: "data" , shape: [1x3x48x48] - An input image in the format [BxCxHxW]

The net outputs a blob with the shape: [1, 10], containing a row-vector of 10 floating point values for five landmarks
coordinates in the form (x0, y0, x1, y1, ..., x5, y5). All the coordinates are normalized to be in range [0,1].
"""
from model_class import Model
import cv2
import time


class LandmarksDetectionModel(Model):
    """
    Class for the landmarks Detection Model.
    """
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

    def preprocess_outputs(self, landmarks_vector):
        """
        The return will contain the related coordinates of the prediction, resized to the original image size
        """
        left_eye = (landmarks_vector[0][0][0][0]*self.w, landmarks_vector[0][1][0][0]*self.h)
        right_eye = (landmarks_vector[0][2][0][0]*self.w, landmarks_vector[0][3][0][0]*self.h)
        return left_eye, right_eye
