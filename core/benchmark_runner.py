from typing import Dict, List, Any, Optional
import json
import time
from pathlib import Path
from tqdm import tqdm
from .eval_task import EvaluationTask, TaskEvaluator
from .together_client import TogetherClient

class BenchmarkRunner:
    def __init__(self, client: TogetherClient):
        """Initialize the benchmark runner with a Together.ai client."""
        self.client = client
        self.evaluator = TaskEvaluator(client)

    def load_tasks(self, task_file: str) -> List[EvaluationTask]:
        """
        Load evaluation tasks from a JSON file.
        
        Args:
            task_file: Path to the JSON file containing tasks
            
        Returns:
            List of EvaluationTask objects
        """
        try:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            
            if isinstance(task_data, dict) and 'tasks' in task_data:
                tasks = task_data['tasks']
            else:
                tasks = task_data
                
            return [EvaluationTask.from_dict(task) for task in tasks]
        except Exception as e:
            print(f"Error loading tasks: {str(e)}")
            return []

    def run_benchmark(
        self,
        tasks: List[EvaluationTask],
        model: str,
        use_chat: bool = True,
        **kwargs
    ) -> Dict:
        """Run benchmark on tasks."""
        print(f"\nRunning {model} benchmark: {len(tasks)} tasks")
        
        results = []
        total_score = 0
        successful_runs = 0
        scored_runs = 0
        total_latency = 0
        
        for task in tasks:
            try:
                print(f"\nProcessing task {task.task_id}")
                
                # Concatenate prompt and context for the API request
                full_input = f"{task.prompt}\n\nContext:\n{task.context}" if task.context else task.prompt
                print(f"Input: {full_input[:100]}...")
                
                # Generate response
                start_time = time.time()
                if use_chat:
                    messages = [{"role": "user", "content": full_input}]
                    response = self.client.generate_chat(messages, model, **kwargs)
                else:
                    response = self.client.generate(model, full_input, **kwargs)
                
                # Calculate latency
                latency = time.time() - start_time
                total_latency += latency
                
                # Handle different response formats
                if isinstance(response, dict):
                    response_text = response.get('text', '') or response.get('response', '') or response.get('output', '')
                elif isinstance(response, str):
                    response_text = response
                else:
                    print(f"Invalid response format for task {task.task_id}: {response}")
                    continue
                
                if not response_text:
                    print(f"Empty response for task {task.task_id}")
                    continue
                    
                print(f"Response: {response_text[:100]}...")
                print(f"Expected output: {task.expected_output[:100]}...")
                
                # Count successful run if we got a valid response
                successful_runs += 1
                
                # Evaluate response
                score = None
                if task.metric != 'human_review':
                    score = self.evaluator.evaluate_task(task, response_text, model=model)
                    print(f"Score: {score:.3f}")
                    total_score += score
                    scored_runs += 1
                else:
                    print("Human review required - no automatic scoring")
                
                results.append({
                    'task_id': task.task_id,
                    'prompt': task.prompt,
                    'context': task.context,
                    'expected_output': task.expected_output,
                    'response': response_text,
                    'score': score,
                    'latency': latency,
                    'metric': task.metric
                })
                
            except Exception as e:
                print(f"\nError processing task {task.task_id}:")
                print(f"Error type: {type(e)}")
                print(f"Error message: {str(e)}")
                print(f"Task data: {task.__dict__}")
                continue
        
        # Calculate metrics
        metrics = {
            'total_tasks': len(tasks),
            'successful_runs': successful_runs,
            'avg_score': total_score / scored_runs if scored_runs > 0 else None,
            'avg_latency': total_latency / len(tasks) if tasks else 0
        }
        
        return {
            'model': model,
            'metrics': metrics,
            'results': results,
            'timestamp': int(time.time())
        }

    def _save_results(self, results: Dict[str, Any], results_dir: str) -> None:
        """
        Save benchmark results to a JSON file.
        
        Args:
            results: Benchmark results to save
            results_dir: Directory to save results
        """
        # Create results directory if it doesn't exist
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        filename = f"run_{results['timestamp']}.json"
        filepath = Path(results_dir) / filename
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2) 