# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

# import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync
from utils.augmentations import letterbox
class ScreenCrack(object):
    @torch.no_grad()
    def __init__(self):
        self.model = None
        weights='backend/screencrack.pt'  # model.pt path(s) ROOT / 'yolov5s.pt'
        source=None  # ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=[640]  # inference size (pixels)
        self.conf_thres=0.40  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        device="0"  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True  # show results
        save_txt=False  # save results to *.txt
        save_conf=False  # save confidences in --save-txt labels
        save_crop=False  # save cropped prediction boxes
        nosave=False  # do not save images/videos
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        augment=False  # augmented inference
        visualize=False  # visualize features
        update=False  # update all models
        project=ROOT / 'runs/detect'  # save results to project/name
        name='exp'  # save results to project/name
        exist_ok=False  # existing project/name ok, do not increment
        line_thickness=3  # bounding box thickness (pixels)
        hide_labels=False  # hide labels
        hide_conf=False  # hide confidences
        half=False  # use FP16 half-precision inference
        dnn=False  # use OpenCV DNN for ONNX inference

        imgsz *= 2 if len(imgsz) == 1 else 1  # expand
        print('imgsz value is ===>', imgsz)
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device=""
        
        device = select_device("")
        print("DEVICE: ", device)
        # half &= device.type != 'cpu'  # half precision only supported on CUDA
        self.half = True#half
        self.device = device
        # Load model
        w = str(weights[0] if isinstance(weights, list) else weights)
        classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
        # check_suffix(w, suffixes)  # check weights have acceptable suffix
        self.pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        # weights = "screencrack.pt"
        print("Weight: ", weights)
        self.model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())  # model stride
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16
        imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # Dataloader
        # if webcam:
        # view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        # bs = len(dataset)  # batch_size
        # else:
        #     dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        #     bs = 1  # batch_size
        # vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        if self.pt and device.type != 'cpu':
            self.model(torch.zeros(1, 3, *imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        dt, self.seen = [0.0, 0.0, 0.0], 0

    def predict(self, img_org):
        img = letterbox(img_org, 640, stride=self.stride, auto=self.pt)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0 # Max Min Scaler between 0-1
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = self.model(img, augment=False, visualize=False)[0]

        # NMS
        # print("CONF: ", self.conf_thres)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        crack_lst = []
        # Process predictions
        for i, det in enumerate(pred):  # per image
            self.seen += 1
            # im0= img.copy()
            # else:
            #     p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

           
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy()  # if save_crop else im0  # for save_crop bcoz save crop dosent save the annotations & labels
            
            # Initializing the fixed coordinates
            x1, y1 = 200, 50
            x2, y2 = x1 + 210, y1 + 370  # 410, 420
            focus_x, focus_y = (x2 - x1) / 2, (y2 - y1) / 2

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_org.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    crack_lst.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf])
        return crack_lst

# if __name__ == "__main__":
#     run(source=0)