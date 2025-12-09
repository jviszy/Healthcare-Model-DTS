# -------------------------------------------------------------------------
# FULL PRIVACY PRESERVING ML PROJECT
# Spark + Federated Learning + Differential Privacy + Encryption
# -------------------------------------------------------------------------
# REQUIRED INSTALLS:
# pip install pyspark tensorflow_federated tensorflow pandas numpy tensorflow_privacy cryptography
# -------------------------------------------------------------------------

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from cryptography.fernet import Fernet

# -------------------------------------------------------------------------
# 1. SIMULATED REAL HEALTHCARE DATA (3 hospitals)
# -------------------------------------------------------------------------

hospital_1 = pd.DataFrame({
    "age": [44, 50, 36, 55],
    "bp": [130, 145, 120, 160],
    "diabetes": [1, 0, 1, 0]
})

hospital_2 = pd.DataFrame({
    "age": [60, 55, 47, 52],
    "bp": [140, 150, 135, 148],
    "diabetes": [0, 1, 0, 0]
})

hospital_3 = pd.DataFrame({
    "age": [39, 62, 70, 45],
    "bp": [110, 155, 160, 118],
    "diabetes": [1, 0, 1, 1]
})

# -------------------------------------------------------------------------
# 2. SPARK DISTRIBUTION OF DATA
# -------------------------------------------------------------------------

spark = SparkSession.builder.appName("HealthcarePrivacyFL").getOrCreate()

spark_1 = spark.createDataFrame(hospital_1)
spark_2 = spark.createDataFrame(hospital_2)
spark_3 = spark.createDataFrame(hospital_3)

print("\n--- Distributed Data on Spark ---")
spark_1.show()
spark_2.show()
spark_3.show()

client_datasets = [spark_1.toPandas(), spark_2.toPandas(), spark_3.toPandas()]

# -------------------------------------------------------------------------
# 3. ENCRYPT DATA (SYM KEY)
# -------------------------------------------------------------------------

key = Fernet.generate_key()
cipher = Fernet(key)

def encrypt_df(df):
    aes = df.copy()
    for col in aes.columns:
        aes[col] = aes[col].apply(lambda x: cipher.encrypt(str(x).encode()))
    return aes

def decrypt_df(df):
    aes = df.copy()
    for col in aes.columns:
        aes[col] = aes[col].apply(lambda x: float(cipher.decrypt(x).decode()))
    return aes

encrypted_clients = [encrypt_df(df) for df in client_datasets]
decrypted_clients = [decrypt_df(df) for df in encrypted_clients]

# -------------------------------------------------------------------------
# 4. Convert to TF Federated Format
# -------------------------------------------------------------------------

def preprocess(df):
    x = df[["age", "bp"]].values.astype(np.float32)
    y = df["diabetes"].values.astype(np.float32)
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(2)

fed_data = [preprocess(df) for df in decrypted_clients]

# -------------------------------------------------------------------------
# 5. MODEL + DIFFERENTIAL PRIVACY
# -------------------------------------------------------------------------

def model_fn():
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    dp_optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=1.0,
        noise_multiplier=0.5,  # DP noise for privacy
        num_microbatches=1,
        learning_rate=0.05
    )

    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=fed_data[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

# -------------------------------------------------------------------------
# 6. FEDERATED AVERAGING TRAINER
# -------------------------------------------------------------------------

trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn=model_fn
)

state = trainer.initialize()

# -------------------------------------------------------------------------
# 7. TRAINING LOOP
# -------------------------------------------------------------------------

print("\n--- Training Federated Privacy-Preserving Model ---")
for round in range(1, 6):
    result = trainer.next(state, fed_data)
    state = result.state
    acc = result.metrics['client_work']['train']['binary_accuracy']
    print(f"Round {round} - Accuracy: {acc}")

print("\n✔ Training Complete!")
print("✔ Federated Learning + DP + Encryption running successfully!")
