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


@torch.no_grad()
def run(weights=ROOT / 'phonehand.pt',  # model.pt path(s) ROOT / 'yolov5s.pt'
        source=0,  # ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=[640],  # inference size (pixels)
        conf_thres=0.40,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    print('imgsz value is ===>', imgsz)
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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
        if half:
            model.half()  # to FP16
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    stable=0
    for path, img, im0s, vid_cap in dataset:
        t1 = time_sync()
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

        # Process predictions
        for i, det in enumerate(pred):  # per image
            storephonedata = []
            storephoneoriginalcords = []
            storefingerdata = []
            euclidean_dist = 500
            seen += 1
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy()  # if save_crop else im0  # for save_crop bcoz save crop dosent save the annotations & labels
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # Initializing the fixed coordinates
            x1, y1 = 200, 50
            x2, y2 = x1 + 210, y1 + 370  # 410, 420
            focus_x, focus_y = (x2 - x1) / 2, (y2 - y1) / 2

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
                for *xyxy, conf, cls in reversed(det):
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # print(det, 'type is :', type(det))
                    # print('xyxy is ', xyxy)

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        print('coordinates are ===> {}'.format(xyxy))
                        print('names[c] are ===> {}'.format(names[c]))
                        # p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        centrw, centrh = (int(xyxy[2]) - int(xyxy[0])) / 2, (int(xyxy[3]) - int(xyxy[1])) / 2
                        centrx, centry = xyxy[0] + centrw, xyxy[1] + centrh
                        cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Drawing the rectangular box on the screen 16:9 ratio for phone
                        cv2.circle(im0, (int(x1 + focus_x), int(y1 + focus_y)), 4, (0, 0, 255),-1)  # fixed center point

                        if names[c] == 'phone':
                            storephonedata = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                            storephoneoriginalcords = xyxy # Storing the latest phone coordinates in the global variable
                            cv2.circle(im0, (int(centrx), int(centry)), 4, (255, 0, 0),-1)  # fixed center point of the bounding box

                            euclidean_dist = np.sqrt(
                                np.square(int(centrx.cpu().numpy()) - int(x1 + focus_x)) + np.square(
                                    int(centry.cpu().numpy()) - int(y1 + focus_y)))

                            if euclidean_dist > 20:
                                cv2.putText(im0, 'Please move your phone in Focus box', (10, im0.shape[0] - 450),
                                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 255), thickness=2)
                            else:
                                cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0),
                                              3)  
                        if names[c] == 'finger':
                            storefingerdata.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            print(f'{s}Done. ({t3 - t2:.3f}s)') 
            # FPS custom code
            fps = int(1/float(t3-t2)) 
            cv2.putText(im0, "FPS : {}".format(str(fps)), (5, im0.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                        color=(0, 0, 255), thickness=2)

            # iterate through all the detections points then check if any point is lying or not
            if len(storephonedata):  # If phone is detected then only perform these operations 
                fingercheck = []
                region_one = np.array([[storephonedata[0], storephonedata[1]],
                                       [storephonedata[0], storephonedata[2]],
                                       [storephonedata[2], storephonedata[-1]],
                                       [storephonedata[2], storephonedata[2]]], np.int32)
                region_one = region_one.reshape((-1, 1, 2))
                for cords in storefingerdata:
                    centrw, centrh = (int(cords[2]) - int(cords[0])) / 2, (int(cords[3]) - int(cords[1])) / 2
                    centrx, centry = cords[0] + centrw, cords[1] + centrh
                    cv2.circle(im0, (int(centrx + 0.60 * centrw), int(centry)), 4, (255, 0, 0), -1)
                    cv2.circle(im0, (int(centrx - 0.60 * centrw), int(centry)), 4, (255, 0, 0), -1)
                    inside_region_one = cv2.pointPolygonTest(region_one, (int(centrx - 0.60 * centrw), int(centry)),
                                                             False)
                    inside_region_two = cv2.pointPolygonTest(region_one, (int(centrx + 0.60 * centrw), int(centry)),
                                                             False)
                    fingercheck.append(inside_region_one)
                    fingercheck.append(inside_region_two)

                    # print("cords====", cords)
                if True in fingercheck:
                    stable = 0
                    cv2.putText(im0, 'Please move your finger out of phone!', (8, im0.shape[0] - 420), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 255), thickness=2)

                if True not in fingercheck and euclidean_dist and euclidean_dist < 20:
                    print("\n............now saving the file..........")
                    cv2.putText(im0, 'ALL OKKKKKKKKKKKKKK', (8, im0.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=2)
                    stable+=1 # For stability ... if all is ok till 10 continous frames then only save the last frame to the disk
                    if stable == 10:
                        save_one_box(storephoneoriginalcords, imc, file=save_dir / 'crops' / 'phoneonly' / f'{p.stem}.jpg', BGR=True) # Saving the cropped image of phone
                        return save_dir / 'crops' / 'phoneonly' / f'{p.stem}.jpg' # returning to the project_main.py file

            # Stream results
            im0 = annotator.result()
            if view_img:
                # im0 = cv2.resize(im0, (640, 640))
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


if __name__ == "__main__":
    #     # opt = parse_opt()
    #     # main(opt)
    file = run()
    print('finished...crop saved at ==> ', file)
