

def ocr_function(frame):        #ALREADY
    return text

def phone_fingerDetect(frame):
    return bbox, class_name

def crackDetect(frame):
    return bbox, class_name





cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    cv2.imshow("screen", frame)
    k = cv2.waitKey(1)
    if k == ord("o"):
        #crop ROI => roi_frame
        # call OCR function : input roi_frame
                #out put: text

        pass #ocr_mode       only process on lastest frame
        #ask to go forward or continue stay at OCR mode "f" (forward) vs "b"
    elif k == ord("b"):  # backward
        pass    # stay at OCR
    elif k == ord("f"):
        # start to check phone/finger detection mode
        #if phone inside RED box AND no hand inside RED box:    => run crack detection FUNCTION




