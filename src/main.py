"""
Main File
"""

from face_detection import FaceDetectionModel
from facial_landmarks_detection import LandmarksDetectionModel
from head_pose_estimation import HeadPoseDetectionModel
from input_feeder import InputFeeder
from mouse_controller import MouseController
import argparse
import cv2

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Basic Edge App with Inference Engine")

    c_desc = "CPU extension file location, if applicable"
    d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input file"
    mface_desc = "The location of the face model path"
    p_desc = "Probability threshold"
    it_desc = "Input Type, video, cam, or image"

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument("-i", help=i_desc, required=True)
    required.add_argument("-m", help=mface_desc, required=True)
    required.add_argument("-it", help=it_desc, required=True)
    optional.add_argument("-c", help=c_desc, default=None)
    optional.add_argument("-d", help=d_desc, default="CPU")
    optional.add_argument("-p", help=p_desc, default="0.6")
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    feed = InputFeeder(input_type=args.it, input_file=args.i)

    face_model = FaceDetectionModel()    # FaceDetectionModel(args.m, args.d, args.c,args.p)
    face_model.load_model()

    landmarks_model = LandmarksDetectionModel()
    landmarks_model.load_model()

    headpose_model = HeadPoseDetectionModel()
    headpose_model.load_model()

    feed.load_data()
    for batch in feed.next_batch():
        cropped_face, coords = face_model.predict(batch)
        landmarks_model.predict(cropped_face)
        yaw, pitch, roll = headpose_model.predict(cropped_face)
        break
    feed.close()


if __name__ == "__main__":
    main()