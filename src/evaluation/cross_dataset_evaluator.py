"""
Cross-Dataset Evaluator
========================

Evaluates model performance across multiple datasets to measure generalization.
Quantifies domain shift and cross-dataset transfer learning effectiveness.

Part of ARC Phase E Week 4: Cross-Dataset Curriculum Learning
Dev 2 implementation

Usage:
    >>> evaluator = CrossDatasetEvaluator(model, datasets_dict)
    >>> results = evaluator.evaluate_all()
    >>> print(f"Avg AUC: {results['overall']['mean_auc']:.3f}")
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


class CrossDatasetEvaluator:
    """
    Evaluates model performance across multiple datasets.

    Provides per-dataset metrics and cross-dataset generalization analysis.
    """

    def __init__(
        self,
        model: nn.Module,
        datasets: Dict[str, DataLoader],
        device: str = "cpu"
    ):
        """
        Initialize cross-dataset evaluator.

        Args:
            model: Trained model to evaluate
            datasets: {dataset_name: DataLoader} mapping
            device: Device for evaluation
        """
        self.model = model
        self.datasets = datasets
        self.device = device
        self.model.to(device)
        self.model.eval()

    def evaluate_dataset(
        self,
        dataset_name: str,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on a single dataset.

        Args:
            dataset_name: Name of dataset
            dataloader: DataLoader for dataset

        Returns:
            Dict with metrics (auc, sensitivity, specificity, etc.)
        """
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    if len(batch) >= 2:
                        images, labels = batch[0], batch[1]
                    else:
                        continue
                elif isinstance(batch, dict):
                    images = batch['image']
                    labels = batch['label']
                else:
                    continue

                images = images.to(self.device)
                labels = labels.cpu().numpy()

                # Forward pass
                logits = self.model(images)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_preds.extend(probs.flatten())
                all_labels.extend(labels.flatten())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        try:
            auc_score = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc_score = 0.0

        # Compute precision-recall AUC
        try:
            precision, recall, _ = precision_recall_curve(all_labels, all_preds)
            pr_auc = auc(recall, precision)
        except ValueError:
            pr_auc = 0.0

        # Threshold at 0.5 for binary metrics
        preds_binary = (all_preds >= 0.5).astype(int)
        tp = np.sum((preds_binary == 1) & (all_labels == 1))
        tn = np.sum((preds_binary == 0) & (all_labels == 0))
        fp = np.sum((preds_binary == 1) & (all_labels == 0))
        fn = np.sum((preds_binary == 0) & (all_labels == 1))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        return {
            'dataset': dataset_name,
            'auc': auc_score,
            'pr_auc': pr_auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'num_samples': len(all_labels),
            'num_positive': int(np.sum(all_labels))
        }

    def evaluate_all(self) -> Dict[str, Any]:
        """
        Evaluate model on all datasets.

        Returns:
            Dict with per-dataset results and overall statistics
        """
        per_dataset_results = {}

        for dataset_name, dataloader in self.datasets.items():
            metrics = self.evaluate_dataset(dataset_name, dataloader)
            per_dataset_results[dataset_name] = metrics

        # Compute overall statistics
        all_aucs = [r['auc'] for r in per_dataset_results.values()]
        all_sensitivities = [r['sensitivity'] for r in per_dataset_results.values()]
        all_specificities = [r['specificity'] for r in per_dataset_results.values()]

        overall_stats = {
            'mean_auc': float(np.mean(all_aucs)),
            'std_auc': float(np.std(all_aucs)),
            'min_auc': float(np.min(all_aucs)),
            'max_auc': float(np.max(all_aucs)),
            'mean_sensitivity': float(np.mean(all_sensitivities)),
            'mean_specificity': float(np.mean(all_specificities))
        }

        return {
            'per_dataset': per_dataset_results,
            'overall': overall_stats
        }

    def compute_domain_shift(
        self,
        source_dataset: str,
        target_dataset: str
    ) -> float:
        """
        Compute performance drop from source to target dataset.

        Measures domain shift as difference in AUC scores.

        Args:
            source_dataset: Source dataset name
            target_dataset: Target dataset name

        Returns:
            Performance drop (source_auc - target_auc)
        """
        results = self.evaluate_all()
        source_auc = results['per_dataset'][source_dataset]['auc']
        target_auc = results['per_dataset'][target_dataset]['auc']
        return source_auc - target_auc


def compare_curriculum_strategies(
    models: Dict[str, nn.Module],
    eval_datasets: Dict[str, DataLoader],
    device: str = "cpu"
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple curriculum strategies.

    Args:
        models: {strategy_name: trained_model} mapping
        eval_datasets: {dataset_name: DataLoader} mapping
        device: Device for evaluation

    Returns:
        {strategy_name: evaluation_results} mapping
    """
    strategy_results = {}

    for strategy_name, model in models.items():
        evaluator = CrossDatasetEvaluator(model, eval_datasets, device)
        results = evaluator.evaluate_all()
        strategy_results[strategy_name] = results

    return strategy_results


# Example usage
if __name__ == '__main__':
    """Demonstrate cross-dataset evaluator."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    print("=" * 80)
    print("Cross-Dataset Evaluator Demo")
    print("=" * 80)

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 224 * 224, 1)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = DummyModel()
    print("✓ Model created")

    # Create dummy datasets
    print("\n1. Creating dummy datasets...")

    # REFUGE (easier - higher AUC expected)
    refuge_data = TensorDataset(
        torch.randn(100, 3, 224, 224),
        torch.randint(0, 2, (100,))
    )
    refuge_loader = DataLoader(refuge_data, batch_size=16)

    # ORIGA (medium difficulty)
    origa_data = TensorDataset(
        torch.randn(80, 3, 224, 224),
        torch.randint(0, 2, (80,))
    )
    origa_loader = DataLoader(origa_data, batch_size=16)

    # Drishti (harder - lower AUC expected)
    drishti_data = TensorDataset(
        torch.randn(50, 3, 224, 224),
        torch.randint(0, 2, (50,))
    )
    drishti_loader = DataLoader(drishti_data, batch_size=16)

    print("✓ Datasets created")

    # Create evaluator
    print("\n2. Creating CrossDatasetEvaluator...")
    datasets = {
        "REFUGE": refuge_loader,
        "ORIGA": origa_loader,
        "Drishti": drishti_loader
    }
    evaluator = CrossDatasetEvaluator(model, datasets, device="cpu")
    print("✓ Evaluator created")

    # Evaluate all datasets
    print("\n3. Evaluating all datasets...")
    results = evaluator.evaluate_all()

    print("\nPer-Dataset Results:")
    for dataset_name, metrics in results['per_dataset'].items():
        print(f"\n  {dataset_name}:")
        print(f"    AUC: {metrics['auc']:.3f}")
        print(f"    PR-AUC: {metrics['pr_auc']:.3f}")
        print(f"    Sensitivity: {metrics['sensitivity']:.3f}")
        print(f"    Specificity: {metrics['specificity']:.3f}")
        print(f"    Accuracy: {metrics['accuracy']:.3f}")
        print(f"    Samples: {metrics['num_samples']}")

    print("\nOverall Statistics:")
    print(f"  Mean AUC: {results['overall']['mean_auc']:.3f}")
    print(f"  Std AUC: {results['overall']['std_auc']:.3f}")
    print(f"  Min AUC: {results['overall']['min_auc']:.3f}")
    print(f"  Max AUC: {results['overall']['max_auc']:.3f}")

    # Compute domain shift
    print("\n4. Domain Shift Analysis")
    shift_refuge_to_drishti = evaluator.compute_domain_shift("REFUGE", "Drishti")
    print(f"  REFUGE → Drishti shift: {shift_refuge_to_drishti:.3f}")

    shift_origa_to_drishti = evaluator.compute_domain_shift("ORIGA", "Drishti")
    print(f"  ORIGA → Drishti shift: {shift_origa_to_drishti:.3f}")

    print("\n" + "=" * 80)
    print("Cross-Dataset Evaluator Demo Complete!")
    print("=" * 80)
