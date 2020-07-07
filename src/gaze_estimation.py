"""
Base class for Gaze Estimation Model
gaze-estimation-adas-0002

The network takes three inputs: square crop of left eye image, square crop of right eye image, and three head pose
angles – (yaw, pitch, and roll). The network outputs 3-D vector corresponding to the direction of a person's gaze
in a Cartesian coordinate system in which z-axis is directed from person's eyes (mid-point between left and right eyes'
centers) to the camera center, y-axis is vertical, and x-axis is orthogonal to both z,y axes so that (x,y,z) constitute
a right-handed coordinate system.
"""
from model_class import Model
import cv2
import time


class GazeEstimationModel(Model):
    """
    Class for the Gaze EStimation Model.
    """
    def __init__(self, model_path, device="CPU", extensions=None):
        super().__init__(model_path, device, extensions)

        self.input_name = "left_eye_image"  # Same as Right Eye
        self.input_shape = self.net.inputs[self.input_name].shape
        self.output_name = next(iter(self.net.outputs))
        self.output_shape = self.net.outputs[self.output_name].shape

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

