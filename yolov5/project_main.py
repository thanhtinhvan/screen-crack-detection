import cv2
import cv2
import easyocr
import hand_phone_detection
import screen_crack_detection

print("Loading Easy OCR....")
reader = easyocr.Reader(['en'])  # en for English
print("Loaded Successfully!!")


def start_camera(frame_name: str, text: str, imei_box, display_ocr, ocr_output, path: str):
    """Function which will start the whole process one by one"""
    x1, y1 = 80, 100
    x2, y2 = x1 + 500, y1 + 90
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (700, 500))

        if imei_box:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        if display_ocr:
            cv2.putText(frame, f"IMEI Number : {str(ocr_output)}", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                        color=(255, 0, 0), thickness=2)

        cv2.putText(frame, text, (5, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0),
                    thickness=2)
        cv2.imshow(frame_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('o'):
            print('Pressed O key')
            cropped = frame[y1:y2, x1:x2]
            cv2.imwrite(path, cropped)
            cap.release()
            cv2.destroyAllWindows()
            ocr_output = reader.readtext(path)
            if len(ocr_output):
                ocr_output = ocr_output[-1][-2]
                c = (0, 255, 0)  # setting the color to green
            else:
                ocr_output = 'Not Detected! Please try Again :('
                c = (255, 0, 0)  # setting the color to red
            cv2.putText(frame, f"IMEI Number : {str(ocr_output)}", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                        color=c, thickness=2)
            return start_camera(frame_name='OCR-BOX', text='Press "b" to OCR again or "f" to continue...',
                                imei_box=False, display_ocr=True, ocr_output=ocr_output, path=path)

        elif cv2.waitKey(1) & 0xFF == ord('b'):
            print('Pressed B key')
            cropped = frame[y1:y2, x1:x2]
            cv2.imwrite(path, cropped)
            cap.release()
            cv2.destroyAllWindows()
            return start_camera(frame_name='OCR-BOX Again', text='Press "b" to OCR again or "f" to continue...',
                                imei_box=True, display_ocr=False, ocr_output=ocr_output, path=path)

        elif cv2.waitKey(1) & 0xFF == ord('f'):
            print('Pressed F key')
            cap.release()
            cv2.destroyAllWindows()
            file = hand_phone_detection.run()
            return file  # break # going outside the loop 

        elif cv2.waitKey(1) & 0xFF == 27:
            print('Pressed ESC key')
            cap.release()
            cv2.destroyAllWindows()
            break


frame_name = 'First time OCR-BOX'
text = "Please place IMEI bin the box & press 'o' to run OCR!"
imei_box = True
display_ocr = False
ocr_output = None
path = 'cropped_images/captured.png' # constant path where the crop of the ocr will be saved will be saved!!
file_path = start_camera(frame_name, text, imei_box, display_ocr, ocr_output, path)
cv2.destroyAllWindows()
print('Cropped saved at ....', str(file_path))
screen_crack_detection.run(source=str(file_path))  # run the screen crack detector function
print("Process completed Successfully!!!")
