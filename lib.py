# -----------------------------------------------------------------------------
#                  MY LIB
#                                               Frederic JELMONI - 02/2025
# -----------------------------------------------------------------------------
import os
import cv2
import numpy as np

from screeninfo import get_monitors
from time import time

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
        python camera.py -cam 0 -yolo exemples/form_detect.pt
        python camera.py -cam 1 -onnx exemples/best.onnx
        python camera.py -video exemples/video_form.mp4 -yolo exemples/form_detect.pt
        python camera.py -video exemples/video_captcha_60s.mp4 -yolo exemples/captcha_detect.pt

        python camera.py -video video/NMC2023_Fred_1280x720.mp4 -hef /usr/share/hailo-models/yolov8s_pose_h8.hef
    ''',
    formatter_class=argparse.RawTextHelpFormatter )

    parser.add_argument('-show_cam',    action='store_true', help="Liste les cameras")
    parser.add_argument('-show_hailo',  action='store_true', help="Verifie la presence le module AI HAILO")
    parser.add_argument('-show_cv',     action='store_true', help="OpenCV version")
    parser.add_argument('-show_hef',     action='store_true', help="Liste des modèles HAILO")
    parser.add_argument('-video', type=str, metavar="file_name",  help="Chemin vers un fichier video")
    parser.add_argument('-cam',  type=int,   choices=[0, 1],      help="ID de la camera utiliser (defaut: 0)")
    parser.add_argument('-size', type=int,   choices=[320, 640480, 640640, 800, 1536, 2304, 4608],
                                            help="Video mode : 320x240, 640x480(defaut), 640x640, 800x600, 1536x864, 2304x1296, 4608x2592")

    parser.add_argument('-resize',     action='store_true', help="Resize en 1152x648 pour l'affichage ")

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

    def show_hef(self):
        
        cmd = "ls -al /usr/share/hailo-models/*hef"
        self.launch_cmd(cmd)


# -------------------------------------
class cv2_util:


    # ------------------------------
    def __init__(self):

        self.fps         = 0
        self.fpsLastTime = 0
        self.fpsCallCount = 0


        self.FONT    = cv2.FONT_HERSHEY_PLAIN
        self.helpPos_X = 30
        self.helpPos_Y = 50
        self.helpText = [
            "----------[ Help ]----------",
            "",
            "c --> center screen",
            "f --> full screen",
            "g --> flip img",
            "i --> info"
            ]

        try: 
            self.screen_x = get_monitors()[0].width
            self.screen_y = get_monitors()[0].height
        except:
            self.screen_x = None
            self.screen_y = None


        self.fullscreen = False
        self.WinImgRect = None              # x, y, w, h
        self.WinCaptureSize = None          # w, h
        self.WinCurrentSize = None          # w, h

    # ---------------------
    def isScreenAvailable(self):

        if self.screen_x and self.screen_y :    return True
        else:                                   return False

    # ---------------------
    def setCaptureSize(self, size):

        self.WinCaptureSize = size
        if not self.WinCurrentSize  :
            self.WinCurrentSize = size
    
    # ---------------------
    def setCurrentSize(self, size):
        self.WinCurrentSize = size

    # ---------------------
    def update_fps(self):

        #self.fpsCallCount += 1
        #if self.fpsCallCount < 5 : return

        currentTime = time()

        self.fps = round(1/(currentTime - self.fpsLastTime), 2)
        self.fpsLastTime = currentTime

        #self.fpsCallCount = 0



    # ------------------------------
    def center(self, win_name, SIZE=None):

        if SIZE :

            cv2.moveWindow(win_name, 
                       int((self.screen_x - SIZE[0])/2), 
                       int((self.screen_y - SIZE[1])/2))

        else :
            self.WinImgRect = cv2.getWindowImageRect(win_name)
        
            cv2.moveWindow(win_name, 
                       int((self.screen_x - self.WinImgRect[2])/2), 
                       int((self.screen_y - self.WinImgRect[3])/2))

    # ------------------------------
    def switch_fullscreen(self, win_name):

        if self.fullscreen :

            print("Fullscreen OFF")
            self.fullscreen = False

            #cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.destroyWindow(win_name)
            cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
            
            cv2.moveWindow(win_name,   self.WinImgRect[0], self.WinImgRect[1])
            cv2.resizeWindow(win_name, self.WinImgRect[2], self.WinImgRect[3])

            return False
        
        else :

            print("Fullscreen ON")
            self.WinImgRect = cv2.getWindowImageRect(win_name)
            self.fullscreen = True

            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            return True

    # ------------------------------
    def resize(self, frame, newSize):

        self.WinCurrentSize = newSize
        return cv2.resize(frame, newSize ) 

    # ------------------------------
    def resize_fullscreen(self, frame):

        self.WinCurrentSize = (self.screen_x, self.screen_y)
        return cv2.resize(frame, (self.screen_x, self.screen_y) )


    # ------------------------------
    def print_help(self, img):

        y_offset = 0

        for line in self.helpText :

            cv2.putText(img, line, (self.helpPos_X, self.helpPos_Y+y_offset), self.FONT,1.5 ,(255,255,0),1)
            cv2.putText(img, line, (self.helpPos_X+1, self.helpPos_Y+y_offset+1), self.FONT,1.5 ,(0,0,0),1)

            y_offset += 20
            
        return img

    # ------------------------------
    def print_info(self, img):

        X = self.helpPos_X
        Y = self.helpPos_Y
       
        self.print_txt(img, X, Y,    "Screen : "+ str(self.screen_x) +"x"+ str(self.screen_y))
        self.print_txt(img, X, Y+20, "Cap size : "+ str(self.WinCaptureSize[0]) +"x"+ str(self.WinCaptureSize[1]))
        self.print_txt(img, X, Y+40, "Img size : "+ str(self.WinCurrentSize[0]) +"x"+ str(self.WinCurrentSize[1]))
        self.print_txt(img, X, Y+60, "FPS : "+ str(self.fps) )
            
        return img

    # ------------------------------
    def print_txt(self, img, X, Y, txt):

        cv2.putText(img, txt, (X, Y), self.FONT,1.5 ,(255,255,0),1)
        cv2.putText(img, txt, (X+1, Y+1), self.FONT,1.5 ,(0,0,0),1)


    # ------------------------------
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
