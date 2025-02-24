# -----------------------------------------------------------------------------
#                  MY LIB
#                                               Frederic JELMONI - 02/2025
# -----------------------------------------------------------------------------
import os
import cv2
import numpy as np

from screeninfo import get_monitors

import argparse
import sys

# -------------------------------------
def my_argparse():

    # ---------------------------------
    # Options ligne de cmd
    parser = argparse.ArgumentParser(
    description='''
    *******************************************
    *   Test de la camera du Raspberry PI 5   *
    *   ou lecture d'une video mp4            *
    *******************************************

    Exemples d'utilisation :

        python camera.py -show_cam
        python camera.py -video exemples/video_1.mp4
        python camera.py -cam 0
        python camera.py -cam 0 -yolo exemples/best.pt
        python camera.py -cam 1 -onnx exemples/best.onnx
    ''',
    formatter_class=argparse.RawTextHelpFormatter )

    parser.add_argument('-show_cam',    action='store_true', help="Liste les cameras")
    parser.add_argument('-show_hailo',  action='store_true', help="Verifie la presence le module AI HAILO")
    parser.add_argument('-show_cv',     action='store_true', help="OpenCV version")
    parser.add_argument('-video', type=str, metavar="file_name",  help="Chemin vers un fichier video")
    parser.add_argument('-cam',  type=int,   choices=[0, 1],      help="ID de la camera utiliser (defaut: 0)")
    parser.add_argument('-size', type=int,   choices=[320, 640480, 640640, 800, 1536, 2304],
                                            help="Video mode : 320x240, 640x480(defaut), 640x640, 800x600, 1536x864, 2304x1296")

    parser.add_argument('-yolo', type=str, metavar="file_name",  help="Chemin vers le model YOLO")
    parser.add_argument('-onnx', type=str, metavar="file_name",  help="Chemin vers le model ONNX")
    parser.add_argument('-hef',  type=str, metavar="file_name",  help="Chemin vers le model HEF pour le module HAILO")

    if len(sys.argv) == 1:
        print()
        parser.print_help()
        exit(1)


    return parser




# -------------------------------------
class cmd_lib:

    line = "-------------------------------------------------------------------------------"

    def show_line(self):
        print(self.line)

    def launch_cmd(self, cmd):

        print(self.line)
        os.system(cmd)

    def show_cv(self):

        print(self.line)
        print("cv2 version : ", cv2.__version__)

    def show_cam(self):
        
        cmd = "rpicam-hello --list-camera"
        self.launch_cmd(cmd)

    def show_hailo(self):

        cmd = "hailortcli fw-control identify"
        self.launch_cmd(cmd)


# -------------------------------------
class cv2_util:

    def __init__(self):
        self.screen_x = get_monitors()[0].width
        self.screen_y = get_monitors()[0].height

    def center(self, win_name, win_size):

        cv2.moveWindow(win_name, 
                       int((self.screen_x - win_size[0])/2), 
                       int((self.screen_y - win_size[1])/2))

    def onnx_preprocess(self, image):

        # Adapter la taille en fonction de votre modèle (ici 640x640 par exemple)
        img_resized = cv2.resize(image, (640, 640))
    
        # Convertir en float32 et normaliser
        img = img_resized.astype(np.float32) / 255.0
    
        # Réorganiser les axes (HWC -> CHW)
        img = np.transpose(img, (2, 0, 1))
    
        # Ajouter la dimension batch
        img = np.expand_dims(img, axis=0)

        return img, img_resized  
    

    def onnx_postprocess(self, outputs, conf_threshold=0.5, iou_threshold=0.4):
   

        return False




    def onnx_draw_detections(self, image, detections):
        """
        Affiche sur l'image les boîtes et scores pour chaque détection.
        """
        for (box, score, cls_id) in detections:
            x, y, w, h = box
            # Dessiner la boîte
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Afficher le score et l'identifiant de la classe
            label = f"ID {cls_id}: {score:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        return image
