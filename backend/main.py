# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

# import argparse
import os
import sys
sys.path.append("./ocr/")
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import screen_crack_detection
import pytesseract
import json
from playsound import playsound
# from pyzbar.pyzbar import decode
from ThreadCameraStream import CameraStream
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
import time
import threading
crack_detect = screen_crack_detection.ScreenCrack()

@torch.no_grad()

def isInRectangle(rect, p) :
   if (p[0] > rect[0] and p[0] < rect[2] and p[1] > rect[1] and p[1] < rect[3]) :
      return True
   else :
      return False
class vars:
    request = ""
    requestDict = {"perfect": r"backend\audio\perfect.wav", 
                    "crack_detected": r"backend\audio\crack_detected.wav",
                    "pls_focus": r"backend\audio\focus_phone_in_frame.wav",
                    "finger_out": r"backend\audio\move_finger_out.wav",
                    "crack_detect": r"backend\audio\crack_detected.wav"}
    prev_request = ""
    cnt_phone_out_box = 0
    cnt_finger_in_phone = 0
    cnt_phone_out_box = 0
    cnt_perfect = 0
    cnt_crack_detect = 0
def playAudio():
    duplicate_request_wait_time = 0
    duplicate_request_wait_thres = 300
    isPlaying = False
    while True:
        if vars.request != "" and isPlaying == False and vars.request in vars.requestDict.keys():
            if vars.request != vars.prev_request or duplicate_request_wait_time>duplicate_request_wait_thres:
                if vars.request == vars.prev_request and (vars.request == "perfect" or vars.request == "crack_detect"):
                    isPlaying = False
                    time.sleep(0.1)
                    continue
                playsound(vars.requestDict[vars.request])
                print("Playing: ", vars.requestDict[vars.request])
                time.sleep(2)
                isPlaying = False
                vars.prev_request = vars.request
                vars.request = ""
                duplicate_request_wait_time = 0
            else:
                duplicate_request_wait_time += 1
        else:
            time.sleep(0.1)


def run(weights=ROOT / 'phonehand.pt',  # model.pt path(s) ROOT / 'yolov5s.pt'
        source="0",#r"D:\Van\Downloads\IMG_2254.MOV",#0,  # ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=[640],  # inference size (pixels)
        conf_thres=0.3,  # confidence threshold
        iou_thres=0.25,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='results',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    f = open('ip.json')
    camera = json.load(f)
    ip_addr = camera["IP"]
    port = camera["port"]
    username = camera["username"]
    pw = camera["password"]
    source = "rtsp://{}:{}@{}:{}/cam/realmonitor?channel=1&subtype=0".format(username, pw, ip_addr, port)
    print("Camera info: ", source, flush=True)

    mode = 2    #0: ocr     1: phone detect

    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    print('imgsz value is ===>', imgsz)
    source = str(source)
    save_img = True#not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        print("CLASS NAME: ", names)
        if half:
            model.half()  # to FP16
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    print("IMGSZ: ", imgsz)
    voide_thread = threading.Thread(target=playAudio)
    voide_thread.daemon = True
    voide_thread.start()

    # Dataloader
    print("[R]|~|Finish init backend", flush=True)
    input("Click start")
    if webcam:
        view_img = True#check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset =  LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None], [None]# [None] * bs, [None] * bs
    now = datetime.now()
    now = now.strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = "results/"# increment_path(Path(project)/"", exist_ok=exist_ok)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # print("Save dir: ", save_dir)
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    save_path = save_dir +  now  # img.jpg
    print("save path: ", save_path)
    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    stable=0
    print("start stream")
    # Initializing the fixed coordinates
    x1, y1 = 200, 50
    x2, y2 = x1 + 210, y1 + 370  # 410, 420
    focus_x, focus_y = (x2 - x1) / 2, (y2 - y1) / 2
    imei = ""
    for path, img, im0s, vid_cap in dataset:
        # save_path = str(save_dir / path[0])  # img.jpg
        t1 = time_sync()
        if webcam:
            im0 = im0s[0].copy()  
        else:
            im0 = im0s.copy() 
        if webcam:
            # print("webcam: ", img[0].shape)
            h_img, w_img = 640,480# img[0].shape[1], img[0].shape[2] 
        else:
            print("img/video: ", im0s.shape)
            w_img,h_img,  _ = im0s.shape#, img[0].shape[2] 
        
        # h_focus = int(h_img*0.9)
        # w_focus = h_focus*16/9
        w_focus = int(w_img*0.6)
        h_focus = w_focus*16/9#, h_img*0.9)
        x_center_img, y_center_img = int(w_img/2), int(h_img/2)
        x1 = x_center_img-int(w_focus/2)
        x2 = x_center_img+int(w_focus/2)
        y1 = y_center_img-int(h_focus/2)
        y2 = y_center_img+int(h_focus/2)
        focus_x, focus_y = (x2 - x1) / 2, (y2 - y1) / 2
        # print("HW: ", h_img, w_img)
        # print("HW_focus: ", h_focus, w_focus)
        # print("XY12: ", x1, y1, x2, y2)
        p_x1, p_y1, p_x2, p_y2 = 0,0,0,0    #phone location
        if mode == 0:   #OCR
            cv2.rectangle(im0, (x1-10, y1+30), (x2+10, y1+100), (0, 0, 255), 2)
            cv2.putText(im0, 'OCR mode. Press "o" to get serial number', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)
            cv2.putText(im0, imei, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)
        elif mode == 1:
            cv2.rectangle(im0, (x1-10, y1+30), (x2+10, y1+100), (0, 0, 255), 2)
            ocr_roi = im0s[0][y1+30:y1+100,x1-10:x2+10]
            ocr_roi = cv2.cvtColor(ocr_roi, cv2.COLOR_RGB2GRAY)
            cv2.imwrite("bar.jpeg", ocr_roi)
            imei = pytesseract.image_to_string(ocr_roi)
            # detectedBarcodes = decode(ocr_roi)
            # if not detectedBarcodes:
            #     print("Barcode Not Detected or your barcode is blank/corrupted!")
            # else:
            #     for barcode in detectedBarcodes:
            #         print("barcode.data: ", barcode.data)
            #         imei = barcode.data
            #         break
            tmp = imei.split("\n")
            imei = tmp[0]
            print("IMEI: ", imei)
            cv2.putText(im0, imei, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)
            mode = 0
        elif mode == 2:
            
            
            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Drawing the rectangular box on the screen 16:9 ratio for phone
            if onnx:
                img = img.astype('float32')
            else:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0 # Max Min Scaler between 0-1
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            if pt:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(img, augment=augment, visualize=visualize)[0]

            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            phone_finger_offset = 10
            list_fingers = []
            # Process predictions
            
            for i, det in enumerate(pred):  # per image
                is_phone_focus = False
                storephonedata = []
                storephoneoriginalcords = []
                storefingerdata = []
                euclidean_dist = 500
                seen += 1
                if webcam:  # batch_size >= 1
                    p, s, frame = path[i], f'{i}: ', dataset.count
                else:
                    p, s, frame = path, '', getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy()  # if save_crop else im0  # for save_crop bcoz save crop dosent save the annotations & labels
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                

                if len(det):

                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # dictonary for storing the results 
                    cnt = dict()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        cnt[names[int(c)]] = n  # n is the total number of detections detected per class eg. 2 phones or 5 phones etc.

                    # Write results
                    is_nonFinger = True
                    no_crack = 0
                    for *xyxy, conf, cls in reversed(det):
                        # if save_txt:  # Write to file
                        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        #     with open(txt_path + '.txt', 'a') as f:
                        #         f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        # print(det, 'type is :', type(det))
                        # print('xyxy is ', xyxy)
                        
                        if True:#save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            # print('coordinates are ===> {}'.format(xyxy))
                            # print('names[c] are ===> {}'.format(names[c]))
                            # p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            centrw, centrh = (int(xyxy[2]) - int(xyxy[0])) / 2, (int(xyxy[3]) - int(xyxy[1])) / 2
                            centrx, centry = xyxy[0] + centrw, xyxy[1] + centrh
                            # cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Drawing the rectangular box on the screen 16:9 ratio for phone
                            # cv2.circle(im0, (int(x1 + focus_x), int(y1 + focus_y)), 4, (0, 0, 255),-1)  # fixed center point

                            if names[c] == 'phone':
                                if conf < 0.9:
                                    continue
                                storephonedata = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                                p_x1, p_y1, p_x2, p_y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                                storephoneoriginalcords = xyxy # Storing the latest phone coordinates in the global variable
                                # cv2.circle(im0, (int(centrx), int(centry)), 4, (255, 0, 0),-1)  # fixed center point of the bounding box

                                # euclidean_dist = np.sqrt(
                                #     np.square(int(centrx.cpu().numpy()) - int(x1 + focus_x)) + np.square(
                                #         int(centry.cpu().numpy()) - int(y1 + focus_y)))
                                # print("euclidean_dist: ", euclidean_dist)
                                # is_phone_focus = False
                                if int(xyxy[0])>x1 and int(xyxy[2])<x2 and int(xyxy[1])>y1 and int(xyxy[3])<y2:
                                # if euclidean_dist > 60:
                                    is_phone_focus = True
                                    cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0),3)  
                                    cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0),2) 
                                else:
                                    # cv2.putText(im0, 'Please move your phone in Focus box', (8, 20),
                                    #             cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 255), thickness=2)
                                    cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255),2) 
                                # else:
                                #     cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0),3)  
                            if names[c] == 'finger':
                                list_fingers.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

                            # annotator.box_label(xyxy, label, color=colors(c, True))
                            # if save_crop:
                            #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)



                # Print time (inference-only)
                # print(f'{s}Done. ({t3 - t2:.3f}s)') 
                # FPS custom code
                delta = float(t3-t2)
                if delta == 0:
                    delta = 0.01
                fps = int(1/delta) 
                cv2.putText(im0, "FPS : {}".format(str(fps)), (5, im0.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                            color=(0, 0, 255), thickness=1)

                # iterate through all the detections points then check if any point is lying or not
                if True:#len(storephonedata):  # If phone is detected then only perform these operations 
                    if len(list_fingers)>0:
                        for finger in list_fingers:
                            if len(storephonedata) > 0:
                                phone_rect = [p_x1+phone_finger_offset, p_y1+phone_finger_offset, p_x2-phone_finger_offset, p_y2-phone_finger_offset]
                                p1 = (finger[0], finger[1])
                                p2 = (finger[0], finger[3])
                                p3 = (finger[2], finger[1])
                                p4 = (finger[2], finger[3])
                                if isInRectangle(phone_rect, p1) or isInRectangle(phone_rect, p2) or isInRectangle(phone_rect, p3) or isInRectangle(phone_rect, p4):
                                    cv2.rectangle(im0, (finger[0], finger[1]), (finger[2], finger[3]), (0, 0, 255),1)
                                    is_nonFinger = False
                                else:
                                    cv2.rectangle(im0, (finger[0], finger[1]), (finger[2], finger[3]), (255, 0, 0),1)


                    if is_phone_focus and is_nonFinger:#not fingercheck and is_phone_focus:
                        # print("\n............now saving the file..........")
                        # cv2.putText(im0, 'ALL OKKKKKKKKKKKKKK', (8, im0.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=2)
                        # stable+=1 # For stability ... if all is ok till 10 continous frames then only save the last frame to the disk
                        # if stable == 10:
                        #     save_one_box(storephoneoriginalcords, imc, file=save_dir / 'crops' / 'phoneonly' / f'{p.stem}.jpg', BGR=True) # Saving the cropped image of phone
                            # return save_dir / 'crops' / 'phoneonly' / f'{p.stem}.jpg' # returning to the project_main.py file
                        print(y1,y2,x1,x2)
                        focus_roi = im0s[0][y1:y2,x1:x2]
                        # focus_roi = cv2.resize(cv2.imread(r"D:\Van\OneDrive\NeuralEngine\ScreenCrackDetection\v2\screen-crack-detection\yolov5\a.jpg"), (640,480))
                        # cv2.imshow("roi", focus_roi)
                        crack_list = crack_detect.predict(focus_roi)
                        # print("CRACK list:", crack_list)
                        no_crack = len(crack_list)
                        if len(crack_list)>0:
                            for crack in crack_list:
                                cv2.rectangle(im0, (crack[0]+x1, crack[1]+y1), (crack[2]+x1, crack[3]+y1), (0, 255, 255), 2)
                    if not is_phone_focus:
                        cv2.putText(im0, 'Please move your phone in Focus box', (8, 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 255), thickness=2)
                        
                        vars.cnt_phone_out_box = vars.cnt_phone_out_box+1#
                        print("[R]|~|Please move your phone in Focus box!!!", flush=True)
                        vars.cnt_perfect = max(0, vars.cnt_perfect-1)
                        vars.cnt_crack_detect = max(0, vars.cnt_crack_detect - 1)
                    elif is_nonFinger == False:
                        cv2.putText(im0, 'Please move finger out of phone area', (8, 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 255), thickness=2)
                        print("[R]|~|Please move finger out of phone area", flush=True)
                        vars.cnt_finger_in_phone = vars.cnt_finger_in_phone +1
                        vars.cnt_perfect = max(0, vars.cnt_perfect-1)
                        vars.cnt_crack_detect = max(0, vars.cnt_crack_detect - 1)
                        
                    else:
                        vars.cnt_finger_in_phone = max(0, vars.cnt_finger_in_phone -1)
                        vars.cnt_phone_out_box = max(0, vars.cnt_phone_out_box -1)
                        vars.cnt_perfect = vars.cnt_perfect+1
                        cv2.putText(im0, 'Crack checking. Detect {} crack(s).'.format(no_crack), (8, 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 255), thickness=2)
                        print("[R]|~|Crack checking. Detect {} crack(s)".format(no_crack), flush=True)
                        if no_crack>0:
                            vars.cnt_crack_detect = vars.cnt_crack_detect + 1
                        else:
                            vars.cnt_crack_detect = max(0, vars.cnt_crack_detect - 1)
                    play_thres = 30 # 3s
                    # requestDict = {"perfect": "backend\\audio\\perfect.mp3", 
                    #     "crack_detected": "backend\\audio\\crack_detected.mp3",
                    #     "pls_focus": "backend\\audio\\focus_phone_in_frame.mp3",
                    #     "finger_out": "backend\\audio\\move_finger_out.mp3"}
                    # print("CNT: {} {} {}".format(vars.cnt_phone_out_box, vars.cnt_finger_in_phone, vars.cnt_perfect))
                    if vars.cnt_crack_detect > (play_thres + 30):
                        print("Requested: crack_detect")
                        vars.request = "crack_detect"
                        vars.cnt_crack_detect = 0
                    elif vars.cnt_perfect > play_thres:
                        print("Requested: perfect")
                        vars.request = "perfect"
                        vars.cnt_perfect = 0
                    elif vars.cnt_phone_out_box > play_thres:
                        print("Requested: pls_focus")
                        vars.request = "pls_focus"
                        vars.cnt_phone_out_box = 0
                    elif vars.cnt_finger_in_phone > play_thres:
                        print("Requested: finger_out")
                        vars.request = "finger_out"
                        vars.cnt_finger_in_phone = 0
                    
            # Stream results
            # im0 = annotator.result()
        if True:#view_img:
            im0 = cv2.resize(im0, (480,640))
            cv2.imshow("ScreenCrackDetection", im0)
            k = cv2.waitKey(10)&0xFF
            if k == ord("q"):  # 1 millisecond
                mode = 5
            # elif k == ord("o"):
            #     mode = 1
            # elif k == ord("f"):
            #     mode = 2
            # elif k == ord("b"):
            #     if mode == 2:
            #         mode = 0
        # Save results (image with detections)
        if save_img:
            if False:#dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path[0] != save_path:  # new video
                    
                    vid_path[0] = save_path
                    print("Init video writer: ", save_path)
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    # save_path += '.mp4'
                    vid_writer[0] = cv2.VideoWriter(save_path + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[0].write(im0)
        if mode == 5:
            break
    # Print results
    # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    #if isinstance(vid_writer[0], cv2.VideoWriter):
    print("Release video {}".format(vid_path[0]), flush=True)
    vid_writer[0].release()  # release previous video writer

if __name__ == "__main__":
    #     # opt = parse_opt()
    #     # main(opt)
    file = run()
    print('finished...crop saved at ==> ', file)
