# -----------------------------------------------------------------------------
#                                   TEST
#                                                           Fred J. 02/2025
# -----------------------------------------------------------------------------
import cv2
from picamera2 import Picamera2
from libcamera import controls

from time import sleep

from lib import cmd_lib
from lib import cv2_util
from lib import my_argparse


cmd = cmd_lib()
cvu = cv2_util()


# -------------------------------------
def main():

    # ---------------------------------
    parser = my_argparse()


    # ---------------------------------
    # Analyser les arguments  et chargement des lib
    #
    args = parser.parse_args()

    if args.show_cam :      cmd.show_cam()
    if args.show_hailo :    cmd.show_hailo()
    if args.show_cv :       cmd.show_cv()

    cmd.show_line()

    if args.cam == 1:       ID_CAM = 1
    else :                  ID_CAM = 0

    if   args.size == 320:      SIZE = (320, 240)
    elif args.size == 800:      SIZE = (800, 600)
    elif args.size == 1536:     SIZE = (1536, 864)
    elif args.size == 2304:     SIZE = (2304, 1296)
    elif args.size == 640640:   SIZE = (640, 640)
    else:                       SIZE = (640, 480)


    if args.yolo:
        
        print("Chargement du modele: ", args.yolo)

        from ultralytics import YOLO

        model = YOLO(args.yolo)

    elif args.onnx:

        print("onnx de fonctionne pas encore")

        print("Chargement du modele: ", args.onnx)
        
        import onnxruntime 

        options = onnxruntime.SessionOptions()
        options.enable_profiling=True

        onnx_session = onnxruntime.InferenceSession(
                args.onnx, 
                sess_options=options,
                providers=["CPUExecutionProvider"] )

        onnx_input_name = onnx_session.get_inputs()[0].name
        onnx_output_name = onnx_session.get_outputs()[0].name

    elif not args.go :     exit()


    # ---------------------------------
    # Lancement de la camera
    #
    picam2 = Picamera2(ID_CAM)
    picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": SIZE}))
    picam2.set_controls({"AeMeteringMode": controls.AeMeteringModeEnum.Spot})
    #picam2.set_controls({"ExposureTime": 10000, "AnalogueGain": 1.0})
    picam2.set_controls({"AnalogueGain": 1.0})


    picam2.start()

    im = picam2.capture_array()
    cv2.imshow("Camera", im)
    cvu.center("Camera", SIZE)

    a = 0

    while True:



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


            # Affichage simple : on affiche ici la forme du tenseur de sortie
            print("Output shape:", outputs[0].shape)

            # Post-traitement (à adapter selon votre modèle)
            #detections = cvu.onnx_postprocess(outputs, conf_threshold=0.5, iou_threshold=0.4)


            # Affichage des détections sur l'image redimensionnée (640x640)
            #annotated_image = cvu.onnx_draw_detections(resized_image.copy(), detections)


            # Affichage 
            #cv2.imshow("Camera", annotated_image)
            cv2.imshow("Camera", resized_image)




        else :
        
            #print(a)


            cv2.imshow("Camera", frame)


        a += 1
  
        # ---------------------------------------
        # Gestion du clavier
        key = cv2.waitKey(1) & 0xFF
        if key == 27:                               exit()



# -------------------------------------
if __name__ == '__main__':
    main()

