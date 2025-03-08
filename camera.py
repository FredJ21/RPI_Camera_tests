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
    FLAG_PRINT_INFO = True

    img_size = (0,0)    # w,h



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


    if   args.size == 320:      CapSIZE = (320, 240)
    elif args.size == 800:      CapSIZE = (800, 600)
    elif args.size == 1536:     CapSIZE = (1536, 864)
    elif args.size == 2304:     CapSIZE = (2304, 1296)
    elif args.size == 4608:     CapSIZE = (4608, 2592)
    elif args.size == 640640:   CapSIZE = (640, 640)
    else:                       CapSIZE = (640, 480)

    cvu.setCaptureSize(CapSIZE)

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
        picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": CapSIZE}))
        picam2.set_controls({"AeMeteringMode": controls.AeMeteringModeEnum.Spot})
        #picam2.set_controls({"ExposureTime": 10000, "AnalogueGain": 1.0})
        #picam2.set_controls({"AnalogueGain": 1.0})

        picam2.start()

        cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)

        img_size = CapSIZE

        loop_delay = 1

    # ---------------------------------
    # ou lecture de la video
    # ---------------------------------
    elif args.video :

        cap = cv2.VideoCapture(VIDEO)

        if not cap.isOpened():
            print("Erreur : Impossible d'ouvrir la vidéo.")
            exit(1)

        # Obtenir le framerate de la vidéo
        fps = cap.get(cv2.CAP_PROP_FPS)  # Récupérer le nombre d'images par seconde
        loop_delay = int(1000 / fps)  # Calcul du délai entre chaque image en millisecondes

        # Obtenir la largeur et la hauteur des frames
        img_size = ( int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) )

        cvu.setCaptureSize(img_size)
        cvu.setCurrentSize(img_size)


        if args.yolo : loop_delay = 1

    # ------------------------------------------------------------------
    # ---------------------------------
    #           MAIN LOOP
    # ---------------------------------
    frame_nb = 0
    while True:


        # ------------------------------
        # 1. Selection de la source vidéo
        # ------------------------------

        # ----------[ Camera
        if args.cam == 0 or args.cam == 1:

            frame = picam2.capture_array()

            if FLAG_FLIP == True:  
                frame = cv2.flip(frame,1)
    
        # ----------[ Video file
        elif args.video :

            ret, frame = cap.read()
            if not ret:  break
    

        # ------------------------------
        # 2. Selection du traitement
        # ------------------------------

        # ----------[ YOLO inference
        if args.yolo:


            if frame_nb > 0 :

                results = model(frame)
                
                frame = results[0].plot()
        
                # Si un point est détecté
                if results[0].keypoints and hasattr(results[0].keypoints, "xy"):

                    keypoint = results[0].keypoints.xy
                    
                    if len(keypoint[0]) == 2: 

                        p1, p2 = keypoint[0]

                        x = int(p1[0])
                        y = int(p1[1])

                        cv2.line(frame, (0,y), (img_size[0], y), (0, 255, 0), 1)
                        cv2.line(frame, (x,0), (x, img_size[1]), (0, 255, 0), 1)



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



        # ------------------------------
        # 3. Affichage
        # ------------------------------
        if True :

            if frame_nb == 1 : cvu.center("Camera") 
       
            # Resize uniquement pour l'affichage , pas pour le traitement
            if FLAG_FULLSCREEN :
                frame = cvu.resize_fullscreen(frame)
            elif RESIZE :
                frame = cvu.resize(frame, RESIZE )

            # Aide
            if FLAG_PRINT_HELP :        frame = cvu.print_help(frame)
            elif FLAG_PRINT_INFO :      frame = cvu.print_info(frame)

            cv2.imshow("Camera", frame)


            # FPS
            cvu.update_fps()

        frame_nb += 1
  
        # ------------------------------
        # 4. Gestion du clavier
        # ------------------------------
        key = cv2.waitKey(loop_delay) & 0xFF
        if key == 27:               exit()
        elif key == ord('f'):       FLAG_FULLSCREEN = cvu.switch_fullscreen("Camera")
        elif key == ord('c'):       cvu.center("Camera")
        elif key == ord('g'):       FLAG_FLIP ^= 1
        elif key == ord('h'):       FLAG_PRINT_HELP ^= 1
        elif key == ord('i'):       FLAG_PRINT_INFO ^= 1


    if args.video : cap.release()
    cv2.destroyAllWindows()

# -------------------------------------
if __name__ == '__main__':
    main()

