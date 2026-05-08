# **************************************************************************
# Nom : Majda Bouhou
# ===========================================================================
# NOTE: Ce code a été développé et exécuté sur Kaggle (GPU Tesla T4).
# Les chemins de données (/kaggle/input/...) sont spécifiques à Kaggle.
# Pour reproduire: importer ce notebook sur Kaggle avec le dataset approprié.
# ===========================================================================

# ===========================================================================
# OBJECTIF DU PROJET
# ===========================================================================
# Développer un CNN capable de classifier 6 espèces d'animaux sauvages:
# éléphant, girafe, léopard, rhinocéros, tigre, zèbre
#
# ARCHITECTURE:
# - 4 blocs convolutifs progressifs (32→64→128→256 filtres)
# - GlobalAveragePooling + 2 couches denses (256→128→6)
# - BatchNormalization et Dropout pour régularisation
#
# STRATÉGIES D'OPTIMISATION:
# - Data augmentation (rotation, zoom, cisaillement, flip)
# - Class weights pour compenser déséquilibre (640 à 1600 images/classe)
# - Label smoothing (0.1) pour meilleure généralisation
# - Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#
# DONNÉES:
# - Train: 6160 images | Validation: 1540 images | Test: 1200 images
# - Images RGB 224×224 pixels
# - Classes déséquilibrées (ex: 640 girafes vs 1600 tigres)
#
# RÉSULTAT: 96.54% val_accuracy (meilleure époque: 64/100)
# TEMPS D'ENTRAÎNEMENT: ~48 minutes sur GPU Tesla T4 (Kaggle)
# ===========================================================================

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des données dans la mémoire

# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Le Type de notre modéle (séquentiel)

from keras.models import Model
from keras.models import Sequential

# Le type d'optimisateur utilisé dans notre modèle (RMSprop, adam, sgd, ...)
# L'optimisateur ajuste les poids de notre modèle par descente du gradient
# Chaque optimisateur a ses propres paramètres
# Note: Il faut tester plusieurs et ajuster les paramètres afin d'avoir les meilleurs résultats

from keras.optimizers import Adam

# Les types des couches utlilisées dans notre modèle
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D, Activation, Dropout, Flatten, \
    Dense, GlobalAveragePooling2D

# Des outils pour suivre et gérer l'entrainement de notre modèle
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configuration du GPU
import tensorflow as tf
from keras import backend as K

# Sauvegarde du modèle
from keras.models import load_model

# Affichage des graphes
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
import time
import numpy as np

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ==========================================
# ================VARIABLES=================
# ==========================================

# Le dossier principal qui contient les données
mainDataPath = "/kaggle/input/datasets/majdabouhou/donnee/donnees/"

# Le dossier contenant les images d'entrainement
trainPath = mainDataPath + "entrainement"

# Le dossier contenant les images de validation
validationPath = mainDataPath + "validation"

# Le dossier contenant les images de test
testPath = mainDataPath + "test"

# Le nom du fichier du modèle à sauvegarder
modelsPath = "/kaggle/working/Model.keras"

training_batch_size = 6160
validation_batch_size = 1540

# Configuration des images
image_scale = 224  # la taille des images
image_channels = 3  # le nombre de canaux de couleurs
images_color_mode = "rgb"  # rgb pour les images en couleurs
image_shape = (image_scale, image_scale,
               image_channels)  # la forme des images d'entrées, ce qui correspond à la couche d'entrée du réseau

# Configuration des paramètres d'entrainement
fit_batch_size = 32  # le nombre d'images entrainées ensemble: un batch
fit_epochs = 100  # Le nombre d'époques

# Nombre de classes du problème
num_classes = 6

# ==========================================
# ==================MODÈLE==================
# ==========================================

# Couche d'entrée:
# Cette couche prend comme paramètre la forme des images (image_shape)
input_layer = Input(shape=image_shape)


# Partie feature extraction (ou cascade de couches d'extraction des caractéristiques)
def feature_extraction(input):
    # 1-couche de convolution avec nombre de filtre  (exp 32)  avec la taille de la fenetre de ballaiage exp : 3x3
    # 2-fonction d'activation : sigmoid
    # 3-couche d'echantillonage (pooling) pour reduire la taille avec la taille de la fenetre de ballaiage exp :2x2
    # --- Bloc 1 : 32 filtres ---
    x = Conv2D(32, (3, 3), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # --- Bloc 2 : 64 filtres ---
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # --- Bloc 3 : 128 filtres ---
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # --- Bloc 4 : 256 filtres ---
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    encoded = MaxPooling2D((2, 2))(x)
    encoded = Dropout(0.3)(encoded)
    return encoded


# Partie complètement connectée (Fully Connected Layer)
def fully_connected(encoded):
    # Flatten: pour convertir les matrices en vecteurs pour la couche MLP
    # Dense: une couche neuronale simple avec le nombre de neurone (exemple 64)
    # fonction d'activation exp: sigmoid, relu, tanh ...
    x = GlobalAveragePooling2D()(encoded)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(num_classes)(x)
    sortie = Activation('softmax')(x)
    return sortie


# Déclaration du modèle:
# La sortie de l'extracteur des features sert comme entrée à la couche complétement connectée
model = Model(input_layer, fully_connected(feature_extraction(input_layer)))

# Affichage des paramétres du modèle
# Cette commande affiche un tableau avec les détails du modèle
# (nombre de couches et de paramétrer ...)
model.summary()

# Compilation du modèle
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    optimizer=Adam(learning_rate=0.0003),
    metrics=['accuracy'])

# ==========================================
# ==========CHARGEMENT DES IMAGES===========
# ==========================================

# training_data_generator: charge les données d'entrainement en mémoire
training_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=25,
    zoom_range=0.2,
    shear_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

# validation_data_generator: charge les données de validation en memoire
validation_data_generator = ImageDataGenerator(rescale=1. / 255)

# training_generator: indique la méthode de chargement des données d'entrainement
training_generator = training_data_generator.flow_from_directory(
    trainPath,  # Place des images d'entrainement
    color_mode=images_color_mode,  # couleur des images
    target_size=(image_scale, image_scale),  # taille des images
    batch_size=training_batch_size,  # nombre d'images à entrainer (batch size)
    class_mode="categorical",  # classement categorical
    shuffle=True)  # on "brasse" (shuffle) les données -> pour prévenir le surapprentissage

# validation_generator: indique la méthode de chargement des données de validation
validation_generator = validation_data_generator.flow_from_directory(
    validationPath,  # Place des images de validation
    color_mode=images_color_mode,  # couleur des images
    target_size=(image_scale, image_scale),  # taille des images
    batch_size=validation_batch_size,  # nombre d'images à valider
    class_mode="categorical",  # classement categorical
    shuffle=True)  # on "brasse" (shuffle) les données -> pour prévenir le surapprentissage

# On imprime l'indice de chaque classe (Keras numerote les classes selon l'ordre des dossiers des classes)
print(training_generator.class_indices)
print(validation_generator.class_indices)

# On charge les données d'entrainement et de validation
# x_train: Les données d'entrainement
# y_train: Les Ètiquettes des données d'entrainement
# x_val: Les données de validation
# y_val: Les Ètiquettes des données de validation
(x_train, y_train) = training_generator.__next__()
(x_val, y_val) = validation_generator.__next__()


# ==========================================
# ==============ENTRAINEMENT================
# ==========================================

# Calcul des poids de classes pour compenser le déséquilibre
y_train_labels = np.argmax(y_train, axis=1)
class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weight_dict = dict(enumerate(class_weights_array))

# Savegarder le modèle avec la meilleure validation accuracy ('val_acc')
# Note: on sauvegarder le modèle seulement quand la précision de la validation s'améliore
modelcheckpoint = ModelCheckpoint(filepath=modelsPath,
                                  monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')

# EarlyStopping : arrête l'entrainement si val_accuracy ne s'améliore
earlystop = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# ReduceLROnPlateau : réduit le learning rate automatiquement quand la val_accuracy stagne
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    mode='max',
    verbose=1
)

# Enregistrer le temps du début de l'entrainement
start_time = time.time()

# entrainement du modèle
classifier = model.fit(x_train, y_train,
                       epochs=fit_epochs,  # nombre d'époques
                       batch_size=fit_batch_size,  # nombre d'images entrainées ensemble
                       validation_data=(x_val, y_val),  # données de validation
                       verbose=1,  # mets cette valeur ‡ 0, si vous voulez ne pas afficher les détails d'entrainement
                       callbacks=[modelcheckpoint, earlystop, reduce_lr],
                       # les fonctions à appeler à la fin de chaque époque (dans ce cas modelcheckpoint: qui sauvegarde le modèle)
                       class_weight=class_weight_dict,
                       shuffle=True)  # shuffle les images

# Enregistrer le temps de la fin de l'entrainement
end_time = time.time()

# ==========================================
# ========AFFICHAGE DES RESULTATS===========
# ==========================================

# Afficher le temps d'execution
total_time = end_time - start_time
print(f"\n>>> Temps d'exécution total : {total_time:.2f} secondes ({total_time / 60:.2f} minutes)")


# Afficher la courbe d’exactitude par époque (Training vs Validation) ainsi que la courbe de perte (loss)
print(f">>> Erreur minimale (loss) : {min(classifier.history['loss']):.4f}")
print(f">>> Exactitude maximale : {max(classifier.history['accuracy']):.4f}")

# Plot accuracy over epochs (precision par époque)
print(classifier.history.keys())
plt.plot(classifier.history['accuracy'])
plt.plot(classifier.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
fig = plt.gcf()
plt.show()

# Plot loss over epochs (precision par époque)
plt.plot(classifier.history['loss'])
plt.plot(classifier.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()