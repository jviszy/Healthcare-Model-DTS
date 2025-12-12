# =========================================================
# ✅ Federated Learning + DP with Automatic Feature Detection
# =========================================================

!pip install -q --upgrade tensorflow-privacy

# -----------------------------
# 1. Imports
# -----------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

print("Packages loaded successfully.")
print("TensorFlow version:", tf.__version__)
print("Pandas version:", pd.__version__)
print("Numpy version:", np.__version__)

# -----------------------------
# 2. Generate synthetic balanced dataset
# -----------------------------
N = 6000  # total samples
H = 3     # number of hospital silos
POS = N // 2
NEG = N // 2

# Define synthetic features (you can add more)
feature_names = ["age", "bp", "chol", "diabetes", "bmi", "smoking"]
num_features = len(feature_names)

# Negative samples (label=0)
neg_data = np.random.randint(30, 100, size=(NEG, num_features))
neg_labels = np.zeros(NEG)

# Positive samples (label=1)
pos_data = np.random.randint(50, 200, size=(POS, num_features))
pos_labels = np.ones(POS)

# Combine into a DataFrame
data = np.vstack([neg_data, pos_data])
labels = np.hstack([neg_labels, pos_labels])
df = pd.DataFrame(data, columns=feature_names)
df['label'] = labels

df = shuffle(df)
print("Synthetic dataset sample:")
print(df.head())

# -----------------------------
# 3. Automatic feature detection
# -----------------------------
target_col = "label"
feature_cols = df.select_dtypes(include=np.number).columns.tolist()
feature_cols.remove(target_col)
print("\nAuto-selected features:", feature_cols)

# -----------------------------
# 4. Split dataset into balanced hospital silos
# -----------------------------
silo_datasets = []

pos_df = df[df[target_col] == 1]
neg_df = df[df[target_col] == 0]

pos_split = np.array_split(pos_df, H)
neg_split = np.array_split(neg_df, H)

for i in range(H):
    silo = pd.concat([pos_split[i], neg_split[i]], axis=0)
    silo = shuffle(silo)
    silo_datasets.append(silo)
    print(f"Hospital {i+1} samples: {silo.shape}")

# -----------------------------
# 5. Define model (input dim auto-detected)
# -----------------------------
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return model

input_dim = len(feature_cols)
global_model = build_model(input_dim)

# -----------------------------
# 6. Federated learning with DP
# -----------------------------
NUM_ROUNDS = 5
BATCH_SIZE = 64
NOISE_MULTIPLIER = 0.5
CLIP_NORM = 1.0
LR = 0.05

# Split test set
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)
X_test = test_df[feature_cols].values
y_test = test_df[target_col].values

for round in range(NUM_ROUNDS):
    print(f"\n=== Round {round+1}/{NUM_ROUNDS} ===")
    updated_weights = []

    for silo in silo_datasets:
        X = silo[feature_cols].values
        y = silo[target_col].values

        local_model = build_model(input_dim)
        local_model.set_weights(global_model.get_weights())

        optimizer = DPKerasSGDOptimizer(
            l2_norm_clip=CLIP_NORM,
            noise_multiplier=NOISE_MULTIPLIER,
            num_microbatches=BATCH_SIZE,
            learning_rate=LR
        )

        local_model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        local_model.fit(X, y, epochs=1, batch_size=BATCH_SIZE, verbose=0)
        updated_weights.append(local_model.get_weights())

    # FedAvg aggregation
    new_weights = [np.mean(weights, axis=0) for weights in zip(*updated_weights)]
    global_model.set_weights(new_weights)

    # Evaluate
    loss, acc = global_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.4f}")

# -----------------------------
# 7. Final evaluation
# -----------------------------
loss, acc = global_model.evaluate(X_test, y_test)
print("\nFinal Accuracy:", acc)
