import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (

    Conv1D,

    MaxPooling1D,

    Flatten,

    Dense,

    Dropout,

    BatchNormalization

)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from preprocess import extract_features

# ---------------------------------------
# Dataset
# ---------------------------------------

DATASET_PATH = "dataset"

emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

X = []
y = []

print("Reading Audio Files...")

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if not file.endswith(".wav"):
            continue

        emotion = file.split("-")[2]

        features = extract_features(
             os.path.join(actor_path, file)
       ) 

        for feature in features:

         X.append(feature)

         y.append(emotion_dict[emotion])

X = np.array(X)

print("Total Samples :", len(X))

# ---------------------------------------
# Scale Features
# ---------------------------------------
os.makedirs("models", exist_ok=True)
scaler = StandardScaler()

X = scaler.fit_transform(X)
joblib.dump(
    scaler,
    "models/scaler.pkl"
)
X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)

# ---------------------------------------
# Labels
# ---------------------------------------

encoder = LabelEncoder()

y = encoder.fit_transform(y)

joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)
# ---------------------------------------
# Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

print("Training :", X_train.shape)

print("Testing :", X_test.shape)

# ---------------------------------------
# CNN (Dense Network)
# ---------------------------------------

model = Sequential([

    tf.keras.Input(shape=(182,1)),

    Conv1D(
        64,
        3,
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling1D(2),

    Conv1D(
        128,
        3,
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling1D(2),

    Conv1D(
        256,
        3,
        activation="relu"
    ),

    BatchNormalization(),

    Flatten(),

    Dense(
        256,
        activation="relu"
    ),

    Dropout(0.4),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        8,
        activation="softmax"
    )

])

model.summary()

# ---------------------------------------
# Compile
# ---------------------------------------

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

# ---------------------------------------
# Callbacks
# ---------------------------------------

os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(

    "models/emotion_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    verbose=1

)

# ---------------------------------------
# Train
# ---------------------------------------

history = model.fit(

    X_train,

    y_train,

    epochs=80,

    batch_size=32,

    validation_data=(X_test, y_test),

    callbacks=[

        checkpoint,

        early_stop,

        reduce_lr

    ]

)

# ---------------------------------------
# Evaluate
# ---------------------------------------

loss, accuracy = model.evaluate(

    X_test,

    y_test

)

print()

print("Final Accuracy : {:.2f}%".format(accuracy*100))

# ---------------------------------------
# Accuracy Graph
# ---------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"])

plt.plot(history.history["val_accuracy"])

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(["Train","Validation"])

plt.grid(True)

plt.savefig("accuracy.png")

# ---------------------------------------
# Loss Graph
# ---------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"])

plt.plot(history.history["val_loss"])

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(["Train","Validation"])

plt.grid(True)

plt.savefig("loss.png")

print()

print("Model Saved Successfully!")