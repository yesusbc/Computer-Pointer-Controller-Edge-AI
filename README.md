# Computer Pointer Controller

<img src="https://github.com/yesusbc/Computer-Pointer-Controller-Edge-AI/blob/master/images/computerpointer.jpg" alt="ComputerPointer" width="800" height="400">

Third project of the *Udacity's Intel edge AI for IoT developers Nanodegree*, this project will consist of a deep learning pipeline, with the objective of creating an application to control the mouse pointer with the gaze of the user's eyes.

The flow of data looks like this:

<img src="https://github.com/yesusbc/Computer-Pointer-Controller-Edge-AI/blob/master/images/pipeline.png" alt="Pipeline" width="400" height="350">

## Project Set Up and Installation
### Prerequisites
* Intel® OpenVINO™ Toolkit 2020 or Above. [Installation Guide](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)
* Python 3.6
* Pyautogui


### Setup
* Clone this repository
* Source OpenVINO Environment 'source /opt/intel/openvino/bin/setupvars.sh'
* Download models
  * [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
'python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"'

  * [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
'python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"'

  * [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
'python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"'

  * [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
'python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"'

*Save models under the 'intel_models' folder.*

#### Project Structure
.
├── bin
│   └── demo.mp4
├── graphs
│   ├── 16
│   │   ├── inference_time.png
│   │   ├── io_processing_time.png
│   │   └── loading_time.png
│   ├── 16-INT8
│   │   ├── inference_time.png
│   │   ├── io_processing_time.png
│   │   └── loading_time.png
│   └── 32
│       ├── inference_time.png
│       ├── io_processing_time.png
│       └── loading_time.png
├── intel_models
│   ├── face-detection-adas-binary-0001
│   │   └── FP32-INT1
│   │       ├── face-detection-adas-binary-0001.bin
│   │       └── face-detection-adas-binary-0001.xml
│   ├── gaze-estimation-adas-0002
│   │   ├── FP16
│   │   │   ├── gaze-estimation-adas-0002.bin
│   │   │   └── gaze-estimation-adas-0002.xml
│   │   ├── FP16-INT8
│   │   │   ├── gaze-estimation-adas-0002.bin
│   │   │   └── gaze-estimation-adas-0002.xml
│   │   └── FP32
│   │       ├── gaze-estimation-adas-0002.bin
│   │       └── gaze-estimation-adas-0002.xml
│   ├── head-pose-estimation-adas-0001
│   │   ├── FP16
│   │   │   ├── head-pose-estimation-adas-0001.bin
│   │   │   └── head-pose-estimation-adas-0001.xml
│   │   ├── FP16-INT8
│   │   │   ├── head-pose-estimation-adas-0001.bin
│   │   │   └── head-pose-estimation-adas-0001.xml
│   │   └── FP32
│   │       ├── head-pose-estimation-adas-0001.bin
│   │       └── head-pose-estimation-adas-0001.xml
│   └── landmarks-regression-retail-0009
│       ├── FP16
│       │   ├── landmarks-regression-retail-0009.bin
│       │   └── landmarks-regression-retail-0009.xml
│       ├── FP16-INT8
│       │   ├── landmarks-regression-retail-0009.bin
│       │   └── landmarks-regression-retail-0009.xml
│       └── FP32
│           ├── landmarks-regression-retail-0009.bin
│           └── landmarks-regression-retail-0009.xml
├── LICENSE
├── README.md
├── requirements.txt
└── src
    ├── face_detection.py   
    ├── facial_landmarks_detection.py  
    ├── gaze_estimation.py
    ├── head_pose_estimation.py    
    ├── input_feeder.py    
    ├── main.py    
    ├── mouse_controller.py
    
## Basic Demo
```cd ComputerPointerController/src

python main.py --it "video" --i "../bin/demo.mp4"
```


Running on Webcam

`python main.py --it "cam" --i "None"`


Running on video

`python main.py --it "video" --i "../bin/demo.mp4"`


Running on Image

`python main.py --it "image" --i "../bin/demo.jpg"`


Models path are included on `main.py`, if you want to specify your own path, check next section "Command Line Arguments"


## Command Line Arguments
`python main.py -h`

usage: Mouse Controller Edge App with Inference Engine

required arguments:

  `--i` Input_file    The location of the input file, if cam just write None
  
  `--it` InputType  Input Type, video, cam, or image

optional arguments:

  `--c` CPUExtension    CPU extension file location, if applicable (ex. "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so")
  
  `--d` Device    Device, if not CPU (GPU, FPGA, MYRIAD)
  
  `--p` Probabilty    Probability threshold
  
  `--hpm` ModelPath Head Pose Model Path
  
  `--gem` ModelPath Gaze Estimation Model Path
  
  `--lm` ModelPath Landarmarks Model Path
  
  `--fm` ModelPath Face Detection Model Path
  
  `--bm` Benchmark Benchmark for specified FP (32, 16, 16-INT8)


## Benchmarks
Model Loading time, Input/Output processing time and Model inference time, can be found under the folder `graphs`.

## Results
Precisions are related to floating point values, less precision means less memory used by the model, and less compute resources. However, there are some trade-offs with accuracy when using lower precision.

Comparison...
* Loading time: There's a big difference in time (~10 ms), when loading headpose est. and gaze est. models with higher precission, but this doesn't represents a problem because this time will only happen 1 time.
* Inference time: Inference time is almost equal in all precisions, the difference is not higher than 3 ms, so if possible, FP32 should be used due to its better acuracy.
* I/O Processing time: As it's related to the inference time, in addition with the image and model processing, the results where almost similar, three differents precisions have almost the same values, max difference is about the 3 ms.


### Edge Cases

If more than one person in on the image, the detection and pointer control will run on the first person that the models detect.

An appropiate light should be provided for a better detection.
