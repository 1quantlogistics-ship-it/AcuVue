"""
Batch Processor
===============

Batch inference processor for fundus images.
Handles folder processing, DataLoader integration, and result aggregation.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Iterator
from dataclasses import dataclass
import json

from .predictor import GlaucomaPredictor, PredictionResult


# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


@dataclass
class BatchResult:
    """
    Aggregated results from batch processing.

    Attributes:
        results: List of individual prediction results
        summary: Summary statistics
        errors: List of failed images with error messages
    """
    results: List[PredictionResult]
    summary: Dict[str, Any]
    errors: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'results': [r.to_dict() for r in self.results],
            'summary': self.summary,
            'errors': self.errors,
        }

    def save_json(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class BatchProcessor:
    """
    Batch inference processor for fundus images.

    Provides methods for processing folders of images and
    aggregating results with summary statistics.

    Example:
        >>> predictor = GlaucomaPredictor.from_checkpoint("model.pt")
        >>> processor = BatchProcessor(predictor)
        >>> result = processor.process_folder("/path/to/images")
        >>> print(f"Processed {result.summary['total']} images")
        >>> print(f"Glaucoma detected: {result.summary['glaucoma_count']}")
    """

    def __init__(
        self,
        predictor: GlaucomaPredictor,
        batch_size: int = 32,
        extensions: Optional[Tuple[str, ...]] = None
    ):
        """
        Initialize batch processor.

        Args:
            predictor: Initialized GlaucomaPredictor
            batch_size: Number of images per batch
            extensions: Valid image extensions (default: common image formats)
        """
        self.predictor = predictor
        self.batch_size = batch_size
        self.extensions = set(extensions) if extensions else IMAGE_EXTENSIONS

    def find_images(self, folder: str) -> List[Path]:
        """
        Find all valid images in a folder.

        Args:
            folder: Path to folder

        Returns:
            Sorted list of image paths
        """
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        images = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in self.extensions
        ]

        return sorted(images)

    def process_folder(
        self,
        folder: str,
        recursive: bool = False,
        verbose: bool = False
    ) -> BatchResult:
        """
        Process all images in a folder.

        Args:
            folder: Path to folder containing images
            recursive: Search subdirectories (not implemented yet)
            verbose: Print progress

        Returns:
            BatchResult with predictions, summary, and errors
        """
        folder = Path(folder)
        images = self.find_images(folder)

        if verbose:
            print(f"Found {len(images)} images in {folder}")

        results = []
        errors = []

        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_paths = images[i:i + self.batch_size]

            for img_path in batch_paths:
                try:
                    result = self.predictor.predict(str(img_path))
                    results.append(result)

                    if verbose:
                        print(f"  {img_path.name}: {result.prediction} ({result.confidence:.3f})")

                except Exception as e:
                    errors.append({
                        'image_path': str(img_path),
                        'error': str(e)
                    })
                    if verbose:
                        print(f"  {img_path.name}: ERROR - {e}")

        # Compute summary
        summary = self._compute_summary(results, errors)

        return BatchResult(results=results, summary=summary, errors=errors)

    def process_paths(
        self,
        image_paths: List[str],
        verbose: bool = False
    ) -> BatchResult:
        """
        Process a list of image paths.

        Args:
            image_paths: List of paths to images
            verbose: Print progress

        Returns:
            BatchResult with predictions, summary, and errors
        """
        results = []
        errors = []

        for img_path in image_paths:
            try:
                result = self.predictor.predict(img_path)
                results.append(result)

                if verbose:
                    print(f"  {Path(img_path).name}: {result.prediction} ({result.confidence:.3f})")

            except Exception as e:
                errors.append({
                    'image_path': img_path,
                    'error': str(e)
                })
                if verbose:
                    print(f"  {Path(img_path).name}: ERROR - {e}")

        summary = self._compute_summary(results, errors)

        return BatchResult(results=results, summary=summary, errors=errors)

    def process_dataloader(
        self,
        dataloader,
        verbose: bool = False
    ) -> Iterator[List[PredictionResult]]:
        """
        Process images from a PyTorch DataLoader.

        Yields results batch by batch for memory efficiency.

        Args:
            dataloader: PyTorch DataLoader yielding (images, ...) tuples
            verbose: Print progress

        Yields:
            List of PredictionResult for each batch
        """
        import torch

        for batch_idx, batch in enumerate(dataloader):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                images = batch[0]  # Assume images are first
            else:
                images = batch

            # Ensure on correct device
            images = images.to(self.predictor.device)

            # Run inference
            with torch.no_grad():
                logits = self.predictor.model(images)
                probs = torch.softmax(logits, dim=1)
                pred_indices = torch.argmax(probs, dim=1)

            # Build results
            batch_results = []
            for i in range(len(images)):
                pred_idx = pred_indices[i].item()
                prediction = self.predictor.class_names[pred_idx]
                confidence = probs[i, pred_idx].item()
                probabilities = {
                    name: probs[i, j].item()
                    for j, name in enumerate(self.predictor.class_names)
                }

                batch_results.append(PredictionResult(
                    prediction=prediction,
                    confidence=confidence,
                    probabilities=probabilities,
                    image_path=None
                ))

            if verbose:
                print(f"Batch {batch_idx + 1}: {len(batch_results)} predictions")

            yield batch_results

    def _compute_summary(
        self,
        results: List[PredictionResult],
        errors: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Compute summary statistics from results."""
        if not results:
            return {
                'total': 0,
                'processed': 0,
                'errors': len(errors),
                'normal_count': 0,
                'glaucoma_count': 0,
            }

        normal_count = sum(1 for r in results if r.prediction == 'normal')
        glaucoma_count = sum(1 for r in results if r.prediction == 'glaucoma')

        # Compute confidence statistics
        confidences = [r.confidence for r in results]
        avg_confidence = sum(confidences) / len(confidences)

        return {
            'total': len(results) + len(errors),
            'processed': len(results),
            'errors': len(errors),
            'normal_count': normal_count,
            'glaucoma_count': glaucoma_count,
            'normal_rate': normal_count / len(results) if results else 0,
            'glaucoma_rate': glaucoma_count / len(results) if results else 0,
            'avg_confidence': avg_confidence,
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
        }

    def __repr__(self) -> str:
        return (
            f"BatchProcessor(batch_size={self.batch_size}, "
            f"extensions={sorted(self.extensions)})"
        )
