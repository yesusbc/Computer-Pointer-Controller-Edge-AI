"""
Base class for Head Pose detection Model
head-pose-estimation-adas-0001
name: "data" , shape: [1x3x60x60] - An input image in [1xCxHxW] format. Expected color order is BGR

Output layer names in Inference Engine format:
name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).
"""
from model_class import Model
import time


class HeadPoseDetectionModel(Model):
    """
    Class for the Head Pose Detection Model.
    """
    def __init__(self, model_path, device="CPU", extensions=None):
        super().__init__(model_path, device, extensions)

        self.input_name = next(iter(self.net.inputs))
        self.input_shape = self.net.inputs[self.input_name].shape
        self.yaw_output_name = "angle_y_fc"
        self.yaw_output_shape = self.net.outputs[self.yaw_output_name].shape
        self.pitch_output_name = "angle_p_fc"
        self.pitch_output_shape = self.net.outputs[self.pitch_output_name].shape
        self.roll_output_name = "angle_r_fc"
        self.roll_output_shape = self.net.outputs[self.roll_output_name].shape

    def predict(self, image):
        """
        Make inference over the exectutable network
        """
        p_frame = self.preprocess_input(image)

        start_time = time.time()
        outputs = self.exec_net.infer({self.input_name: p_frame})
        prediction_time = time.time() - start_time

        yaw = outputs[self.yaw_output_name][0][0]
        pitch = outputs[self.pitch_output_name][0][0]
        roll = outputs[self.roll_output_name][0][0]
        head_pose_angles = [yaw, pitch, roll]
        # TODO: Head Pose drawing
        return head_pose_angles, prediction_time



