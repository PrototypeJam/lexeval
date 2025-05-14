from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
from .together_client import TogetherClient

@dataclass
class EvaluationTask:
    task_id: str
    prompt: str
    context: str
    expected_output: str
    reference: str
    metric: str
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationTask':
        """Create an EvaluationTask instance from a dictionary."""
        # Handle migration from old format
        if 'input' in data and 'prompt' not in data:
            data['prompt'] = data['input']
            data['context'] = ''
        
        return cls(
            task_id=data['task_id'],
            prompt=data['prompt'],
            context=data['context'],
            expected_output=data['expected_output'],
            reference=data['reference'],
            metric=data['metric'],
            metadata=data.get('metadata')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the task to a dictionary."""
        return {
            'task_id': self.task_id,
            'prompt': self.prompt,
            'context': self.context,
            'expected_output': self.expected_output,
            'reference': self.reference,
            'metric': self.metric,
            'metadata': self.metadata
        }

class TaskEvaluator:
    def __init__(self, client: TogetherClient):
        """Initialize the task evaluator with a Together.ai client."""
        self.client = client

    def evaluate_task(self, task: EvaluationTask, response: str, model: str = None) -> float:
        """
        Evaluate a task based on the model's response.
        If metric is 'llm_judge', use the LLM to judge the output.
        If the task has a 'judge_model' field, use that model for judging; otherwise, use the passed-in model.
        Args:
            task: The evaluation task
            response: The model's response text
            model: The model to use for LLM judging (if needed)
        Returns:
            float: Score between 0 and 1
        """
        try:
            print(f"\nEvaluating task {task.task_id}")
            print(f"Task metric: {task.metric}")
            print(f"Response: {response[:100]}...")
            print(f"Expected output: {task.expected_output[:100]}...")
            
            # Handle empty responses
            if not response or not isinstance(response, str):
                print("Empty or invalid response")
                return 0.0
            
            if task.metric == 'llm_judge':
                return self.llm_judge_score(task.expected_output, response, model=model)
            
            # Convert to lowercase for comparison
            expected = task.expected_output.lower()
            response = response.lower()
            
            # Split into words and remove punctuation
            import re
            def clean_text(text):
                # Remove punctuation and extra whitespace
                text = re.sub(r'[^\w\s]', ' ', text)
                return ' '.join(text.split())
                
            expected = clean_text(expected)
            response = clean_text(response)
            
            expected_words = set(expected.split())
            response_words = set(response.split())
            
            if task.metric == 'rouge':
                # Simple ROUGE-like scoring (word overlap)
                if not expected_words:
                    return 0.0
                word_overlap = len(expected_words.intersection(response_words))
                precision = word_overlap / len(response_words) if response_words else 0.0
                recall = word_overlap / len(expected_words)
                # F1 score
                return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
            elif task.metric == 'keyword_match':
                # Extract keywords (words longer than 3 characters)
                expected_keywords = {w for w in expected_words if len(w) > 3}
                response_keywords = {w for w in response_words if len(w) > 3}
                
                if not expected_keywords:
                    return 0.0
                keyword_overlap = len(expected_keywords.intersection(response_keywords))
                precision = keyword_overlap / len(response_keywords) if response_keywords else 0.0
                recall = keyword_overlap / len(expected_keywords)
                # F1 score
                return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
            else:
                print(f"Unknown metric: {task.metric}, defaulting to ROUGE")
                if not expected_words:
                    return 0.0
                word_overlap = len(expected_words.intersection(response_words))
                precision = word_overlap / len(response_words) if response_words else 0.0
                recall = word_overlap / len(expected_words)
                # F1 score
                return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
        except Exception as e:
            print(f"\nError evaluating task:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print(f"Task data: {task.__dict__}")
            print(f"Response data: {response}")
            return 0.0

    def llm_judge_score(self, expected: str, output: str, model: str = None) -> float:
        """
        Use the selected LLM to judge the output vs expected answer.
        Returns a score between 0 and 1.
        """
        prompt = (
            "You are an expert legal evaluator. Compare the following model output to the expected answer. "
            "Give a score from 0 (completely incorrect) to 1 (perfectly correct), and briefly explain your reasoning.\n\n"
            f"Expected answer:\n{expected}\n\nModel output:\n{output}\n\nScore (0-1):"
        )
        messages = [
            {"role": "system", "content": "You are a helpful and strict legal evaluator."},
            {"role": "user", "content": prompt}
        ]
        # Use the same model as selected for generation
        if model is None:
            model = "meta-llama/Llama-3.2-3B-Instruct-Turbo"  # fallback
        try:
            response = self.client.generate_chat(messages, model)
            # response is expected to be a dict with 'text' or similar
            if isinstance(response, dict):
                text = response.get('text') or response.get('response') or response.get('output')
            else:
                text = str(response)
            print(f"LLM judge raw response: {text}")
            score = self._extract_score_from_llm_judge(text)
            print(f"LLM judge extracted score: {score}")
            return score
        except Exception as e:
            print(f"Error in llm_judge_score: {e}")
            return 0.0

    def _extract_score_from_llm_judge(self, text: str) -> float:
        """
        Extract a score (0-1) from the LLM judge's response.
        """
        import re
        if not text:
            return 0.0
        # Look for a float between 0 and 1
        match = re.search(r"([01](?:\.\d+)?)", text)
        if match:
            try:
                score = float(match.group(1))
                if 0.0 <= score <= 1.0:
                    return score
            except Exception:
                pass
        return 0.0 