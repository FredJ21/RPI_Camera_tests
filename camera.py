# -----------------------------------------------------------------------------
#                                   TEST
#                                                           Fred J. 02/2025
# -----------------------------------------------------------------------------
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import onnxruntime 

import argparse
import sys
from time import sleep

from lib import cmd_lib
from lib import cv2_util


cmd = cmd_lib()
cvu = cv2_util()


# -------------------------------------
def main():

    # ---------------------------------
    # Options ligne de cmd
    parser = argparse.ArgumentParser(description="Test de la camera RPI")
    parser.add_argument('-show_cam',    action='store_true', help="Liste les cameras")
    parser.add_argument('-show_hailo',  action='store_true', help="Verifie la presence le module AI HAILO")
    parser.add_argument('-show_cv',     action='store_true', help="OpenCV version")
    parser.add_argument('-camera', type=int,   choices=[0, 1],      help="ID de la camera utiliser (defaut: 0)")
    parser.add_argument('-size', type=int,   choices=[320, 640480, 640640, 800, 1536, 2304],
                                            help="Video mode : 320x240, 640x480(defaut), 640x640, 800x600, 1536x864, 2304x1296")

    parser.add_argument('-yolo', type=str, metavar="file_name",  help="Chemin vers le model Yolo")
    parser.add_argument('-onnx', type=str, metavar="file_name",  help="Chemin vers le model ONNX")

    if len(sys.argv) == 1:
        print()
        parser.print_help()
        print()
        exit(1)

    # ---------------------------------
    # Analyser les arguments 
    #
    args = parser.parse_args()

    if args.show_cam :      cmd.show_cam()
    if args.show_hailo :    cmd.show_hailo()
    if args.show_cv :       cmd.show_cv()

    if args.camera == 1:    ID_CAM = 1
    else :                  ID_CAM = 0

    if   args.size == 320:      SIZE = (320, 240)
    elif args.size == 800:      SIZE = (800, 600)
    elif args.size == 1536:     SIZE = (1536, 864)
    elif args.size == 2304:     SIZE = (2304, 1296)
    elif args.size == 640640:   SIZE = (640, 640)
    else:                       SIZE = (640, 480)


    if args.yolo:
        print("Chargement du modele: ", args.yolo)
        model = YOLO(args.yolo)

    elif args.onnx:
        print("Chargement du modele: ", args.onnx)
        onnx_session = onnxruntime.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

        onnx_input_name = onnx_session.get_inputs()[0].name
        onnx_output_name = onnx_session.get_outputs()[0].name


    cmd.show_line()

    # ---------------------------------
    # Lancement de la camera
    #
    picam2 = Picamera2(ID_CAM)
    picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": SIZE}))
    picam2.start()

    im = picam2.capture_array()
    cv2.imshow("Camera", im)
    cvu.center("Camera", SIZE)


    a = 0

    while True:

        print(a)


        frame = picam2.capture_array()
        frame = cv2.flip(frame,1)


        if args.yolo:

            # Inference YOLO
            results = model(frame)

            annotated_frame = results[0].plot()

            cv2.imshow("Camera", annotated_frame)

        elif args.onnx:

            # Prétraitement
            input_tensor, resized_image = cvu.onnx_preprocess(frame)

            # Inference via ONNX Runtime
            outputs = onnx_session.run([onnx_output_name], {onnx_input_name: input_tensor})

            # Post-traitement (à adapter selon votre modèle)
            #detections = cvu.onnx_postprocess(outputs, conf_threshold=0.5, iou_threshold=0.4)


            # Affichage des détections sur l'image redimensionnée (640x640)
            #annotated_image = cvu.onnx_draw_detections(resized_image.copy(), detections)


            # Affichage 
            #cv2.imshow("Camera", annotated_image)
            cv2.imshow("Camera", resized_image)




        else :

            cv2.imshow("Camera", frame)

        a += 1
  
        # ---------------------------------------
        # Gestion du clavier
        key = cv2.waitKey(1) & 0xFF
        if key == 27:                               exit()



# -------------------------------------
if __name__ == '__main__':
    main()

