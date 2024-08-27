from ultralytics import YOLO
import streamlit as st
import cv2
 
import os
from datetime import datetime
from pathlib import Path
import sys
import torchvision.transforms as T
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np
import app
from settings import VIDEOS_DICT  # Assurez-vous que VIDEOS_DICT est importé
import settings
from torchvision import models
import os
from datetime import datetime
  
import tempfile

def load_model(model_path):
    model = YOLO(model_path)
    return model

# Charger le modèle
@st.cache_resource
def load_modelFASTER():
    # Charger le modèle pré-entraîné sans les poids
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # Adapter le modèle au nombre de classes de votre modèle sauvegardé
    model = modify_model(model, num_classes=9)  # Nombre de classes dans votre modèle sauvegardé
    model_path = 'weights/FRCNN-V0-5epochs.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# Fonction pour adapter le modèle au nombre de classes
def modify_model(model, num_classes):
    # Obtenir le nombre de caractéristiques d'entrée pour la tête de classification
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Remplacer la tête de classification
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, num_classes)
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, num_classes * 4)

    return model

 
 
id_to_class = {0: 'drone', 1: 'plongeur', 2: 'roche', 3: 'bateau', 4: 'poisson', 5: 'corail', 6: 'tortue', 7: 'asterie', 8: 'avion'}
def process_and_display_frame(conf, model, st_frame, image, label_map=id_to_class):
    """
    Traite et affiche les cadres vidéo avec les objets détectés en utilisant Faster R-CNN.

    :param conf: Confiance minimale du modèle pour afficher les objets.
    :param model: Instance du modèle Faster R-CNN.
    :param st_frame: Zone de l'interface Streamlit pour afficher l'image.
    :param image: Image du cadre vidéo à traiter.
    :param label_map: Dictionnaire de mappage des labels (ID -> Nom).
    """
    # Redimensionner l'image à une taille standard
    image_resized = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Convertir l'image en tensor
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image_resized).unsqueeze(0)  # Ajouter la dimension du batch

    # Effectuer la détection d'objets
    with torch.no_grad():
        outputs = model(image_tensor)

    # Extraire les objets détectés
    boxes = outputs[0]['boxes'].cpu().numpy()  # Boîtes de délimitation
    labels = outputs[0]['labels'].cpu().numpy()  # Labels
    scores = outputs[0]['scores'].cpu().numpy()  # Scores

    # Filtrer les objets en fonction du score de confiance
    indices = scores >= conf
    boxes = boxes[indices]
    labels = labels[indices]
    scores = scores[indices]

    # Compter les objets détectés
    number_of_objects = len(boxes)
    #st.write(f'Nombre d\'objets détectés avec une confiance >= {conf} : {number_of_objects}')

    # Dessiner les boîtes de délimitation et les labels sur l'image
    for box, label_id, score in zip(boxes, labels, scores):
        # Dessiner la boîte de délimitation
        cv2.rectangle(image_resized, 
                      (int(box[0]), int(box[1])), 
                      (int(box[2]), int(box[3])), 
                      (0, 255, 0), 
                      2)
        
        # Préparer le texte du label et du score
        label_text = f'{label_map.get(label_id, "Unknown")} ({score:.2f})'
        
        # Afficher le texte au-dessus de la boîte de délimitation
        cv2.putText(image_resized, 
                    label_text, 
                    (int(box[0]), int(box[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2)

    st_frame.image(image_resized, 
                   caption='Vidéo avec objets détectés', 
                   channels="BGR", 
                   use_column_width=True)

def display_tracker_options():
    display_tracker = st.radio("Suivi d'affichage", ('Oui', 'Non'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Traqueur", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    image = cv2.resize(image, (720, int(720*(9/16))))

    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)

    detected_objects = res[0].boxes.cls
    number_of_objects = len(detected_objects)

    # Display the number of detected objects
    #st.write(f'Nombre d\'objets détectés : {number_of_objects}')
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, caption='Video détectée', channels="BGR", use_column_width=True)

    return res_plotted

 

def play_stored_video(conf, model, typeModel):

 
    uploaded_file = st.file_uploader("Téléversez votre vidéo", type=["mp4", "mov", "avi", "mkv", "MOV"])
    
    if uploaded_file is not None:
        # Sauvegarder le fichier téléversé dans un emplacement temporaire
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
        temp_file.close()
        
        st.video(temp_file_path)

        if st.button('Détecter les objets'):
            try:
                st.info(f"Ouverture du fichier : {temp_file_path}")

                if not os.path.exists(temp_file_path):
                    st.error("Le fichier temporaire n'existe pas. Veuillez réessayer.")
                    return

                vid_cap = cv2.VideoCapture(temp_file_path)

                if not vid_cap.isOpened():
                    st.error("Échec de l'ouverture du fichier, veuillez essayer une autre vidéo.")
                    return

                st.success("Fichier vidéo ouvert avec succès!")

                output_directory = './processed_videos'
                os.makedirs(output_directory, exist_ok=True)

                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                test_output_path = os.path.join(output_directory, f'test_output_{current_time}.mp4')
                
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                frame_width = 720
                frame_height = 405
                out = cv2.VideoWriter(test_output_path, fourcc, 20.0, (frame_width, frame_height))

                if not out.isOpened():
                    st.error("ERREUR D'ECRITURE.")
                    vid_cap.release()
                    return

                st_frame = st.empty()
                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    if success:
                        if typeModel == "YOLOv8" or typeModel == "YOLOv9":
                            res_plotted = _display_detected_frames(conf, model, st_frame, image)
                        elif typeModel == "Faster R-CNN":
                            res_plotted = process_and_display_frame(conf, model, st_frame, image)

                        if res_plotted is None:
                            continue

                        if isinstance(res_plotted, np.ndarray):
                            if res_plotted.shape[0] != frame_height or res_plotted.shape[1] != frame_width:
                                continue
                            out.write(res_plotted)
                    else:
                        break

                vid_cap.release()
                out.release()
                st.success("La vidéo a été traitée et enregistrée avec succès.")

            except Exception as e:
                st.error(f"Une erreur s'est produite: {str(e)}")
                if 'vid_cap' in locals():
                    vid_cap.release()
                if 'out' in locals():
                    out.release()
    else:
        st.warning("Veuillez téléverser une vidéo pour commencer la détection.")


 

def play_stored_video1(conf, model, typeModel):
    # Créer une liste déroulante pour sélectionner la vidéo
    video_options = list(VIDEOS_DICT.keys())
    selected_video = st.selectbox("Choisissez une vidéo", video_options)

    if selected_video:
        video_path = str(VIDEOS_DICT[selected_video])

        # Vérifier que le fichier vidéo existe
        if not os.path.isfile(video_path):
            st.error(f"Le fichier vidéo n'existe pas à l'emplacement : {video_path}")
            return

        # Afficher la vidéo sélectionnée
        try:
            st.video(video_path)

            # Ajouter un bouton pour lancer la détection des objets
            if st.button('Détecter les objets'):
                if not video_path:
                    st.error("SVP, Sélectionnez une vidéo")
                    return

                try:
                    st.info(f"Ouverture du fichier : {video_path}")
                    vid_cap = cv2.VideoCapture(video_path)

                    if not vid_cap.isOpened():
                        st.error("Échec de l'ouverture du fichier, veuillez essayer une autre vidéo.")
                        return

                    st.success("Fichier vidéo ouvert avec succès!")

                    # Créer le répertoire de sortie si nécessaire
                    output_directory = './processed_videos'
                    os.makedirs(output_directory, exist_ok=True)

                    # Créer un nom de fichier unique avec horodatage
                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    test_output_path = os.path.join(output_directory, f'test_output_{current_time}.mp4')
                    
                    # Définir les paramètres du writer vidéo
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    frame_width = 720
                    frame_height = 405
                    out = cv2.VideoWriter(test_output_path, fourcc, 20.0, (frame_width, frame_height))

                    if not out.isOpened():
                        st.error("ERREUR D'ECRITURE.")
                        vid_cap.release()
                        return

                    st_frame = st.empty()
                    while vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if success:
                            if typeModel == "YOLOv8" or typeModel == "YOLOv9":
                                res_plotted = _display_detected_frames(conf, model, st_frame, image)
                            elif typeModel == "Faster R-CNN":
                                res_plotted = process_and_display_frame(conf, model, st_frame, image)

                            if res_plotted is None:
                                continue

                            if isinstance(res_plotted, np.ndarray):
                                if res_plotted.shape[0] != frame_height or res_plotted.shape[1] != frame_width:
                                    continue
                                out.write(res_plotted)
                        else:
                            break

                    vid_cap.release()
                    out.release()
                    st.success("La vidéo a été traitée et enregistrée avec succès.")



                except Exception as e:
                    st.error(f"Une erreur s'est produite: {str(e)}")
                    if 'vid_cap' in locals():
                        vid_cap.release()
                    if 'out' in locals():
                        out.release()

        except Exception as e:
            st.error(f"Une erreur s'est produite lors du chargement de la vidéo : {str(e)}")
