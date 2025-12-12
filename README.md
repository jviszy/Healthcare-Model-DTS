# Privacy-Preserving Federated Learning with Spark — Prototype

This repository contains a small prototype to simulate privacy-preserving federated learning across distributed data silos (e.g., hospitals) using Spark to parallelize local training and aggregate model updates.

Goals
- Simulate N independent silos, each with its own local dataset.
- Each silo trains a local Keras model and produces a weight update (delta).
- Use Apache Spark to parallelize local training and aggregate updates centrally.
- Demonstrate simple Differential Privacy (DP) via clipping + Gaussian noise.
- Sketch a simple additive-mask secure aggregation approach (simulation only).

Not intended for production. Use this scaffold as a starting point for:
- integrating TensorFlow Federated (TFF) for federated API-level semantics,
- integrating TensorFlow Privacy for rigorous DP mechanisms,
- integrating cryptographic secure aggregation (e.g. Prio, Bonawitz et al.) or PySyft for MPC.

Quickstart (local single-machine)
1. Create a Python virtualenv and install dependencies:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Run a local simulation that uses Spark to parallelize training across simulated silos:
   bash run_local.sh

   The script will:
   - generate synthetic siloed data,
   - train a small Keras model locally on each silo,
   - aggregate updates via Spark,
   - apply optional DP noise and optionally run a naive secure aggregation sketch.

Files
- requirements.txt — Python packages required.
- run_local.sh — convenience script to run demo.
- src/model.py — Keras model builders and weight vector helpers.
- src/simulate_silos.py — functions to generate synthetic silo datasets and local training logic.
- src/aggregator_spark.py — Spark-based aggregator that parallelizes local training, collects updates, performs aggregation and DP.
- src/secure_agg.py — simple sketch of additive-mask secure aggregation (simulation).

Next steps
- Replace synthetic data with a real dataset suited to your domain (note: MIMIC and other hospital datasets require credentialed access).
- Replace naive DP implementation with TensorFlow Privacy for stronger guarantees and accountant utilities.
- Replace the secure-aggregation sketch with an audited library or MPC protocol (Bonawitz et al. approach).
- Add evaluation scripts, logging, privacy accounting, and unit/integration tests for distributed execution.

Safety & Data
- This scaffold uses synthetic data. If you bring in real medical or financial datasets, follow all institutional, legal and ethical requirements for data governance and privacy.
