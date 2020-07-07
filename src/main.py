"""
Main File
"""

from face_detection import FaceDetectionModel
from facial_landmarks_detection import LandmarksDetectionModel
from head_pose_estimation import HeadPoseDetectionModel
from gaze_estimation import GazeEstimationModel
from input_feeder import InputFeeder
from mouse_controller import MouseController
import matplotlib.pyplot as plt
import logging as log
import argparse
import cv2
import sys
import time

HEAD_POSE_MODEL = "../intel_models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001"
GAZE_ESTIMATION_MODEL = "../intel_models/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002"
LANDMARKS_MODEL_PATH = "../intel_models/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009"
FACE_MODEL_PATH = "../intel_models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001"


def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Mouse Controller Edge App with Inference Engine")

    c_desc = "CPU extension file location, if applicable"
    d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input file, if cam just write None"
    p_desc = "Probability threshold"
    it_desc = "Input Type, video, cam, or image"
    hpm_desc = "Head Pose Model Path"
    gem_desc = "Gaze Estimation Model Path"
    lm_desc = "Landmarks Model Path"
    fm_desc = "Face Detection Model Path"
    bm_desc = "Benchmark for specified FP (32, 16, 16-INT8)"

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument("--i", help=i_desc, required=True)
    required.add_argument("--it", help=it_desc, required=True)
    optional.add_argument("--c", help=c_desc, default=None)
    optional.add_argument("--d", help=d_desc, default="CPU")
    optional.add_argument("--p", help=p_desc, default="0.6")
    optional.add_argument("--hpm", help=hpm_desc, default=HEAD_POSE_MODEL)
    optional.add_argument("--gem", help=gem_desc, default=GAZE_ESTIMATION_MODEL)
    optional.add_argument("--lm", help=lm_desc, default=LANDMARKS_MODEL_PATH)
    optional.add_argument("--fm", help=fm_desc, default=FACE_MODEL_PATH)
    optional.add_argument("--bm", help=bm_desc, default=None)
    args = parser.parse_args()

    return args


def main(args):
    feed = InputFeeder(input_type=args.it, input_file=args.i)

    face_model = FaceDetectionModel(args.fm, args.d, args.c, float(args.p))
    face_model.load_model()

    landmarks_model = LandmarksDetectionModel(args.lm, args.d, args.c)
    landmarks_model.load_model()

    headpose_model = HeadPoseDetectionModel(args.hpm, args.d, args.c)
    headpose_model.load_model()

    gaze_model = GazeEstimationModel(args.gem, args.d, args.c)
    gaze_model.load_model()

    mouse = MouseController("medium", "fast")

    feed.load_data()
    for batch in feed.next_batch():
        # try:
            cropped_face, coords, _ = face_model.predict(batch)
            cv2.rectangle(batch, (coords[0], coords[1]), (coords[2], coords[3]), (255,0,0), 2)

            left_eye, right_eye, eyes_coords, _ = landmarks_model.predict(cropped_face)

            head_pose_angles, _ = headpose_model.predict(cropped_face)
            x, y, z, _ = gaze_model.predict(left_eye, right_eye, head_pose_angles, cropped_face, eyes_coords)

            mouse.move(x, y)

            cv2.imshow("img", batch)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # except:
        #     print("Frame without prediction. Error: ", sys.exc_info()[0])
        #     log.error(sys.exc_info()[0])
    feed.close()


def main_benchmark(args):
    feed = InputFeeder(input_type=args.it, input_file=args.i)

    face_model = FaceDetectionModel(args.fm, args.d, args.c, float(args.p))
    start_time = time.time()
    face_model.load_model()
    face_load_model_time = time.time() - start_time

    landmarks_model = LandmarksDetectionModel(args.lm, args.d, args.c)
    start_time = time.time()
    landmarks_model.load_model()
    landmarks_model_time = time.time() - start_time

    headpose_model = HeadPoseDetectionModel(args.hpm, args.d, args.c)
    start_time = time.time()
    headpose_model.load_model()
    headpose_model_time = time.time() - start_time

    gaze_model = GazeEstimationModel(args.gem, args.d, args.c)
    start_time = time.time()
    gaze_model.load_model()
    gaze_model_time = time.time() - start_time

    feed.load_data()
    for batch in feed.next_batch():
        try:
            start_time = time.time()
            cropped_face, coords, face_time_prediction = face_model.predict(batch)
            cv2.rectangle(batch, (coords[0], coords[1]), (coords[2], coords[3]), (255,0,0), 2)
            io_face_model_time = time.time() - start_time

            start_time = time.time()
            left_eye, right_eye, eyes_coords, landmarks_time_prediction = landmarks_model.predict(cropped_face)
            io_landmarks_model_time = time.time() - start_time

            start_time = time.time()
            head_pose_angles, headpose_time_prediction = headpose_model.predict(cropped_face)
            io_head_pose_model_time = time.time() - start_time

            start_time = time.time()
            x, y, z, gaze_time_prediction = gaze_model.predict(left_eye, right_eye, head_pose_angles, cropped_face,
                                                               eyes_coords)
            io_gaze_model_time = time.time() - start_time

            print("Graphing loading time...")
            graph_loading_time(face_load_model_time, landmarks_model_time, headpose_model_time, gaze_model_time,
                               args.bm)
            print("Graphing io processing time...")
            graph_io_processing_time(io_face_model_time, io_landmarks_model_time, io_head_pose_model_time,
                                     io_gaze_model_time,  args.bm)
            print("Graphing inference time...")
            graph_model_inference_time(face_time_prediction, landmarks_time_prediction, headpose_time_prediction,
                                       gaze_time_prediction,  args.bm)
            print("Done")

            break

        except:
            print("Frame without prediction. Error: ", sys.exc_info()[0])
            log.error(sys.exc_info()[0])
    feed.close()


def graph_loading_time(face_time, landmarks_time, headpose_time, gaze_time, bm):
    model_list = ["Face Det.", "Landmarks Det.", "Headpose Est.", "Gaze Est."]
    model_load_time = [face_time, landmarks_time, headpose_time, gaze_time]
    plt.bar(model_list, model_load_time)
    plt.xlabel("Models with FP "+bm)
    plt.ylabel("Model Loading Time in Seconds")
    plt.savefig("../graphs/"+bm+"/loading_time.png")
    plt.close()


def graph_io_processing_time(face_time, landmarks_time, headpose_time, gaze_time,  bm):
    model_list = ["Face Det.", "Landmarks Det.", "Headpose Est.", "Gaze Est."]
    model_io_processing_time = [face_time, landmarks_time, headpose_time, gaze_time]
    plt.bar(model_list, model_io_processing_time)
    plt.xlabel("Models with FP "+bm)
    plt.ylabel("Model I/O Processing Time in Seconds")
    plt.savefig("../graphs/"+bm+"/io_processing_time.png")
    plt.close()

def graph_model_inference_time(face_time, landmarks_time, headpose_time, gaze_time,  bm):
    model_list = ["Face Det.", "Landmarks Det.", "Headpose Est.", "Gaze Est."]
    model_inference_time = [face_time, landmarks_time, headpose_time, gaze_time]
    plt.bar(model_list, model_inference_time)
    plt.xlabel("Models with FP "+bm)
    plt.ylabel("Model Inference Time in Seconds")
    plt.savefig("../graphs/"+bm+"/inference_time.png")
    plt.close()


if __name__ == "__main__":
    args = get_args()
    if args.bm:
        main_benchmark(args)
    else:
        main(args)