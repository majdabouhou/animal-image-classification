# **************************************************************************
# Nom : Majda Bouhou
# ===========================================================================
# NOTE: Ce script évalue le modèle entraîné dans 1_Modele.py sur Kaggle.
# Nécessite Model.keras généré par 1_Modele.py dans /kaggle/working/
# ===========================================================================


# ===========================================================================
# OBJECTIF DU SCRIPT
# ===========================================================================
# Ce script évalue les performances du classificateur CNN sur 1200 images de test
# (200 par classe). Il génère:
#
# 1. Métriques de performance:
#    - Test loss et test accuracy globale
#    - Nombre d'images bien/mal classées
#
# 2. Matrice de confusion (6×6):
#    - Visualise les confusions entre classes
#    - Permet d'identifier les paires d'espèces difficiles à distinguer
#
# 3. Grille de visualisation d'erreurs (6×6):
#    - Diagonale: nombre de classifications correctes (en vert)
#    - Hors diagonale: une image d'exemple pour chaque type d'erreur
#    - Permet d'analyser VISUELLEMENT pourquoi le modèle se trompe
#
# RÉSULTAT CLÉ: 90.08% accuracy (1081/1200 correctes)
# INSIGHT PRINCIPAL: 42% des erreurs = confusion éléphant ↔ rhinocéros
# ===========================================================================

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des données dans la mémoire
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Affichage des graphes
import matplotlib.pyplot as plt

# La librairie numpy 
import numpy as np

# Configuration du GPU
import tensorflow as tf
from keras import backend as K

# Utilisé pour le calcul des métriques de validation
from sklearn.metrics import confusion_matrix, roc_curve , auc, classification_report

# Utlilisé pour charger le modèle
from keras.models import load_model
from keras import Model

# Pour charger les images
from PIL import Image


# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ==========================================
# ==================MODÈLE==================
# ==========================================

#Chargement du modéle sauvegardé dans la section 1 via modele.py
model_path = "/kaggle/working/Model.keras"
Classifier: Model = load_model(model_path)

# ==========================================
# ================VARIABLES=================
# ==========================================

# L'emplacement des images de test
mainDataPath = "/kaggle/input/datasets/majdabouhou/donnee/donnees/"
testPath = mainDataPath + "test"

# Le nombre des images de test à évaluer
number_images = 1200
number_images_per_class = 200

# Nombre de classes
num_classes = 6

# La taille des images à classer
image_scale = 224

# La couleur des images à classer
images_color_mode = "rgb"  # grayscale or rgb

# Noms des classes (ordre alphabétique des dossiers)
class_names = ['elephant', 'girafe', 'leopard', 'rhino', 'tigre', 'zebre']

# ==========================================
# =========CHARGEMENT DES IMAGES============
# ==========================================

# Chargement des images de test
test_data_generator = ImageDataGenerator(rescale=1. / 255)

test_itr = test_data_generator.flow_from_directory(
    testPath,# place des images
    target_size=(image_scale, image_scale), # taille des images
    class_mode="categorical",# Type de classification
    shuffle=False,# pas besoin de les boulverser
    batch_size=1,# on classe les images une à la fois
    color_mode=images_color_mode)# couleur des images

(x, y_true) = test_itr.__next__()

# ==========================================
# ===============ÉVALUATION=================
# ==========================================

# Les classes correctes des images (1000 pour chaque classe) -- the ground truth
y_true = np.array([0] * number_images_per_class +
                  [1] * number_images_per_class +
                  [2] * number_images_per_class +
                  [3] * number_images_per_class +
                  [4] * number_images_per_class +
                  [5] * number_images_per_class)

# evaluation du modele
#test_eval = Classifier.evaluate_generator(test_itr, verbose=1)
test_eval = Classifier.evaluate(test_itr, verbose=1)

# Affichage des valeurs de perte et de precision
print('>Test loss (Erreur):', test_eval[0])
print('>Test précision:', test_eval[1])

# Prédiction des classes des images de test
predicted_classes = Classifier.predict(test_itr, verbose=1)

predicted_classes_perc = np.round(predicted_classes.copy(), 4)
predicted_classes = np.argmax(predicted_classes, axis=1)


# Cette liste contient les images bien classées
correct = []
for i in range(0, len(predicted_classes) ):
    if predicted_classes[i] == y_true[i]:
        correct.append(i)

# Nombre d'images bien classées
print("> %d  Étiquettes bien classées" % len(correct))

# Cette list contient les images mal classées
incorrect = []
for i in range(0, len(predicted_classes) ):
    if predicted_classes[i] != y_true[i]:
        incorrect.append(i)

# Nombre d'images mal classées
print("> %d Étiquettes mal classées" % len(incorrect))

# ==========================================
# ===========MATRICE DE CONFUSION===========
# ==========================================

cm = confusion_matrix(y_true, predicted_classes)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap='Greens')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()


# =================================================
# ======GRILLE DE VISUALISATION D'ERREURS===========
# ==================================================

filenames = test_itr.filenames

fig, axes = plt.subplots(num_classes, num_classes, figsize=(18, 18))

for i in range(num_classes):
    axes[i, 0].set_ylabel(class_names[i], fontsize=10, rotation=0, labelpad=60, va='center')
    axes[0, i].set_title(class_names[i], fontsize=10)

for true_class in range(num_classes):
    for pred_class in range(num_classes):
        ax = axes[true_class][pred_class]
        ax.set_xticks([])
        ax.set_yticks([])

        if true_class == pred_class:
            ax.text(0.5, 0.5, f'{cm[true_class][pred_class]}',
                    ha='center', va='center', fontsize=12, color='green',
                    transform=ax.transAxes)
            ax.set_facecolor('#f0f0f0')
            continue

        found = False
        for idx in incorrect:
            if y_true[idx] == true_class and predicted_classes[idx] == pred_class:
                img_path = testPath + "/" + filenames[idx]
                img = Image.open(img_path)
                ax.imshow(img)
                found = True
                break

        if not found:
            ax.text(0.5, 0.5, '0', ha='center', va='center',
                    fontsize=12, color='gray', transform=ax.transAxes)
            ax.set_facecolor('#f0f0f0')

plt.tight_layout()
plt.show()