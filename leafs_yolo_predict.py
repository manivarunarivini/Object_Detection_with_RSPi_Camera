import cv2
from picamera2 import Picamera2
import time
from custom_fun import YOLOv8, draw_text
import sys

stream = "--stream" in sys.argv

# Initialize Picamera2
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (512, 512)})
picam2.configure(camera_config)
picam2.start()

# Load the ONNX model
onnx_model_path = "/home/rspi4/rspi-scripts/models/Yolov8s_Object_Detection_1.onnx"

# Class labels corresponding to your leaves dataset
class_labels = [
    "Fern",  # Replace with your actual class names
    "Ginkgo",
    "Ivy",
    "Kummerowia striata",
    "Laciniata",
    "Macrolobium acaciifolium",
    "Micranthes odontoloma",
    "Murraya",
    "No Leaf",
    "Robinia pseudoacacia",
    "Selaginella davidi franch",
]

yolo = YOLOv8(onnx_model_path, class_labels)

update_image = True
title_suffix = "stream" if stream else "click"
title = f"Leafs YOLO ({title_suffix})"
cv2.namedWindow(title, cv2.WINDOW_NORMAL)

if not stream:
    def on_mouse_click(event, x, y, flags, param):
        global update_image
        if event == cv2.EVENT_LBUTTONDOWN:
            update_image = True
    cv2.setMouseCallback(title, on_mouse_click)

time.sleep(1)  # Allow camera to warm up

while True:
    try:
        if stream or update_image:
            # Capture an image using Picamera2
            frame = picam2.capture_array()
            frame = cv2.flip(frame, -1)

            # Perform object detection and obtain the output image
            output_image, pred_dict = yolo.main(frame)
            if len(pred_dict):
                for k, v in pred_dict.items():
                    print(f"{k}: {v:.2f}")
            else:
                text = "No Leafs Detected"
                draw_text(output_image, text, text_color=(255, 0, 0), font_scale=3)
                print(text)
            cv2.imshow(title, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            update_image = False  # Wait for the next click to update

        time.sleep(0.5)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break


    except KeyboardInterrupt:
        print("Interrupted by user.")
        break

    except Exception as e:
        print(f"An error occurred: {e}")
        break

cv2.destroyAllWindows()
picam2.stop()
