# -----------------------------------------------------------------------------
#                                   TEST
#                                                           Fred J. 02/2025
# -----------------------------------------------------------------------------
import cv2

from time import sleep

from lib import cmd_lib
from lib import cv2_util
from lib import my_argparse


cmd = cmd_lib()
cvu = cv2_util()


# -------------------------------------
def main():

    # ---------------------------------
    FLAG_FLIP       = True
    FLAG_FULLSCREEN = False
    FLAG_PRINT_HELP = False



    # ---------------------------------
    # Analyser les arguments  et chargement des lib
    # ---------------------------------
    parser = my_argparse()
    args = parser.parse_args()

    if args.show_cam :      cmd.show_cam()
    if args.show_hailo :    cmd.show_hailo()
    if args.show_cv :       cmd.show_cv()

    cmd.show_line()

    if args.cam == 0 or args.cam == 1:

        from picamera2 import Picamera2
        from libcamera import controls

        ID_CAM = args.cam

    elif args.video :

        VIDEO = args.video

    else :  
        exit ()


    if   args.size == 320:      SIZE = (320, 240)
    elif args.size == 800:      SIZE = (800, 600)
    elif args.size == 1536:     SIZE = (1536, 864)
    elif args.size == 2304:     SIZE = (2304, 1296)
    elif args.size == 4608:     SIZE = (4608, 2592)
    elif args.size == 640640:   SIZE = (640, 640)
    else:                       SIZE = (640, 480)

    if   args.resize:           RESIZE = (1152, 648)
    else :                      RESIZE = None

    if args.yolo:
        
        print("Chargement du modele: ", args.yolo)

        from ultralytics import YOLO

        model = YOLO(args.yolo)

    elif args.onnx:

        print("onnx de fonctionne pas encore")
        exit(1)

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



    # ---------------------------------
    # Lancement de la camera
    # ---------------------------------
    if args.cam == 0 or args.cam == 1:
    
        picam2 = Picamera2(ID_CAM)
        picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": SIZE}))
        picam2.set_controls({"AeMeteringMode": controls.AeMeteringModeEnum.Spot})
        #picam2.set_controls({"ExposureTime": 10000, "AnalogueGain": 1.0})
        #picam2.set_controls({"AnalogueGain": 1.0})

        picam2.start()

        # Premiere capture pour positionner la fenettre
        im = picam2.capture_array()

        cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)

        if RESIZE : 
            im = cv2.resize(im, RESIZE )
        
        loop_delay = 1

    # ---------------------------------
    # ou lecture de la video
    # ---------------------------------
    elif args.video :

        cap = cv2.VideoCapture(VIDEO)

        # Obtenir le framerate de la vidéo
        fps = cap.get(cv2.CAP_PROP_FPS)  # Récupérer le nombre d'images par seconde
        loop_delay = int(1000 / fps)  # Calcul du délai entre chaque image en millisecondes

        if args.yolo : loop_delay = 1

    # ------------------------------------------------------------------
    # ---------------------------------
    #           MAIN LOOP
    # ---------------------------------
    frame_nb = 0
    while True:


        # ------------------------------
        # Selection de la source vidéo
        # ------------------------------

        # ----------[ Camera
        if args.cam == 0 or args.cam == 1:

            frame = picam2.capture_array()

            if FLAG_FLIP == True:  
                frame = cv2.flip(frame,1)
    
            #cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            #cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        # ----------[ Video file
        elif args.video :

            ret, frame = cap.read()

            if not ret:  break
    
            if args.size :
                frame = cv2.resize(frame, SIZE)




        # ------------------------------
        # Selection du traitement
        # ------------------------------

        # ----------[ YOLO inference
        if args.yolo:

            cv2.imshow("1", frame)

            if frame_nb > 0 :

                results = model(frame)
                
                annotated_frame = results[0].plot()
                cv2.imshow("Camera", annotated_frame)
                '''
                keypoint = results[0].keypoints.xy

                if len(keypoint[0]) == 2: 
                    p1, p2 = keypoint[0]

                    x = int(p1[0])
                    y = int(p1[1])

                    cv2.line(frame, (0,y), (SIZE[0], y), (0, 255, 0), 1)
                    cv2.line(frame, (x,0), (x, SIZE[1]), (0, 255, 0), 1)

                cv2.imshow("3", frame)

                '''
            #elif a > 10 : a = 0

        # ----------[ ONNX  --> TODO
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


        # ----------[ Affichage de la CAM
        else :

            if frame_nb == 1 : cvu.center("Camera") 
       
            # Resize uniquement pour l'affichage , pas pour le traitement
            if FLAG_FULLSCREEN :
                frame = cvu.resize_fullscreen(frame)
            elif RESIZE :
                frame = cv2.resize(frame, RESIZE )

            # Aide
            if FLAG_PRINT_HELP :
                frame = cvu.print_help(frame)

            cv2.imshow("Camera", frame)

        frame_nb += 1
  
        # ---------------------------------------
        # Gestion du clavier
        key = cv2.waitKey(loop_delay) & 0xFF
        if key == 27:               exit()
        elif key == ord('f'):       FLAG_FULLSCREEN = cvu.switch_fullscreen("Camera")
        elif key == ord('c'):       cvu.center("Camera")
        elif key == ord('g'):       FLAG_FLIP ^= 1
        elif key == ord('h'):       FLAG_PRINT_HELP ^= 1


    if args.video : cap.release()
    cv2.destroyAllWindows()

# -------------------------------------
if __name__ == '__main__':
    main()

