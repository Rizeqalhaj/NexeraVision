#!/usr/bin/env python3
"""
ENSEMBLE PREDICTION - Combine 3 models for 92-95% accuracy
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Dict
import logging
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent))
from train_ensemble_ultimate import (
    EnsembleConfig,
    load_dataset,
    extract_features_with_model,
    get_feature_extractor
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class EnsemblePredictor:
    def __init__(self, models_dir: str, model_names: List[str]):
        self.models_dir = Path(models_dir)
        self.model_names = model_names
        self.models = {}
        self.weights = {}
        self._load_models()

    def _load_models(self):
        logger.info("Loading ensemble models...")
        for model_name in self.model_names:
            model_path = self.models_dir / model_name / 'best_model.h5'
            results_path = self.models_dir / model_name / 'results.json'

            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                continue

            # Load with safe_mode=False to allow Lambda layers
            model = tf.keras.models.load_model(model_path, safe_mode=False)
            self.models[model_name] = model

            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    self.weights[model_name] = results.get('test_accuracy', 1.0)
            else:
                self.weights[model_name] = 1.0

            logger.info(f"✅ Loaded {model_name} (weight: {self.weights[model_name]:.4f})")

    def predict_soft_voting(self, features_dict: Dict):
        all_probs = []
        for model_name, model in self.models.items():
            features = features_dict[model_name]
            probs = model.predict(features, verbose=0)
            all_probs.append(probs)
        avg_probs = np.mean(all_probs, axis=0)
        return np.argmax(avg_probs, axis=1), avg_probs


def evaluate_ensemble(config: EnsembleConfig):
    logger.info("="*80)
    logger.info("ENSEMBLE EVALUATION")
    logger.info("="*80)

    dataset = load_dataset(config)
    features_dict = {}

    for model_name in config.model_names:
        logger.info(f"\nExtracting features for {model_name}...")
        feature_extractor, preprocess_fn, _ = get_feature_extractor(model_name)

        test_features, test_labels = extract_features_with_model(
            dataset['test']['paths'],
            dataset['test']['labels'],
            feature_extractor,
            preprocess_fn,
            config,
            model_name,
            'test',
            is_training=False
        )
        features_dict[model_name] = test_features

    predictor = EnsemblePredictor(config.models_dir, config.model_names)

    logger.info("\n" + "="*80)
    logger.info("INDIVIDUAL MODEL RESULTS")
    logger.info("="*80)

    individual_results = {}
    for model_name, model in predictor.models.items():
        features = features_dict[model_name]
        probs = model.predict(features, verbose=0)
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(test_labels, preds)
        individual_results[model_name] = acc
        logger.info(f"{model_name}: {acc*100:.2f}%")

    logger.info("\n" + "="*80)
    logger.info("ENSEMBLE RESULTS (SOFT VOTING)")
    logger.info("="*80)

    ensemble_preds, _ = predictor.predict_soft_voting(features_dict)
    ensemble_acc = accuracy_score(test_labels, ensemble_preds)

    logger.info(f"\n✅ ENSEMBLE ACCURACY: {ensemble_acc*100:.2f}%")
    logger.info(f"✅ Improvement: +{(ensemble_acc - max(individual_results.values()))*100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(test_labels, ensemble_preds, target_names=['Non-violent', 'Violent']))

    cm = confusion_matrix(test_labels, ensemble_preds)
    nonviolent_acc = cm[0,0] / (cm[0,0] + cm[0,1])
    violent_acc = cm[1,1] / (cm[1,0] + cm[1,1])

    logger.info(f"\n✅ Non-violent Accuracy: {nonviolent_acc*100:.2f}%")
    logger.info(f"✅ Violent Accuracy: {violent_acc*100:.2f}%")

    if ensemble_acc >= 0.92:
        logger.info("\n" + "="*80)
        logger.info("🎉 TARGET ACHIEVED! Ensemble >= 92%")
        logger.info("="*80)

    return ensemble_acc


if __name__ == "__main__":
    config = EnsembleConfig()
    evaluate_ensemble(config)
