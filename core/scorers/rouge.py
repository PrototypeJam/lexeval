from typing import Dict, List
from rouge_score import rouge_scorer
import numpy as np

class RougeScorer:
    def __init__(self, metrics: List[str] = None):
        """
        Initialize the ROUGE scorer.
        
        Args:
            metrics: List of ROUGE metrics to use. Defaults to ['rouge1', 'rouge2', 'rougeL']
        """
        if metrics is None:
            metrics = ['rouge1', 'rouge2', 'rougeL']
        self.metrics = metrics
        self.scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)

    def compute_score(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Compute ROUGE scores between prediction and reference.
        
        Args:
            prediction: Model's output text
            reference: Reference text
            
        Returns:
            Dict containing scores for each metric
        """
        scores = self.scorer.score(reference, prediction)
        
        # Convert scores to float values
        return {
            metric: getattr(scores[metric], 'fmeasure', 0.0)
            for metric in self.metrics
        }

    def compute_batch_scores(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute average ROUGE scores for a batch of predictions and references.
        
        Args:
            predictions: List of model outputs
            references: List of reference texts
            
        Returns:
            Dict containing average scores for each metric
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions must match number of references")
            
        all_scores = []
        for pred, ref in zip(predictions, references):
            scores = self.compute_score(pred, ref)
            all_scores.append(scores)
            
        # Calculate averages
        return {
            metric: np.mean([s[metric] for s in all_scores])
            for metric in self.metrics
        } 