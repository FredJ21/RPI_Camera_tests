# -----------------------------------------------------------------------------
#                  MY LIB
#                                               Frederic JELMONI - 02/2025
# -----------------------------------------------------------------------------
import os
import cv2
import numpy as np

from screeninfo import get_monitors


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
        """
        Post-traitement pour extraire les détections :
          - Filtrage selon un seuil de confiance
          - Application de NMS pour supprimer les détections redondantes
        """
        # Supposons que outputs[0] est un tableau de forme (N, 6)
        detections = outputs[0]
        boxes = []
        scores = []
        class_ids = []
    
        # Extraction des détections valides
        for detection in detections:
            print("detection:", len(detection))
            x1, y1, x2, y2, conf, cls_id = detection
            if conf > conf_threshold:
                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                scores.append(float(conf))
                class_ids.append(int(cls_id))
    
        # Application de la suppression non maximale (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
        final_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_detections.append((boxes[i], scores[i], class_ids[i]))
        return final_detections
    
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
