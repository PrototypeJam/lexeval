import streamlit as st
import json
import pandas as pd
from pathlib import Path
import sys
import os
import re
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.together_client import TogetherClient
from core.benchmark_runner import BenchmarkRunner
from core.eval_task import EvaluationTask

# Set page config
st.set_page_config(
    page_title="LexEval",
    page_icon="⚖️",
    layout="wide"
)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'client' not in st.session_state:
    st.session_state.client = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'current_task' not in st.session_state:
    st.session_state.current_task = None
if 'available_models' not in st.session_state:
    st.session_state.available_models = {}
if 'available_tasks' not in st.session_state:
    st.session_state.available_tasks = []
if 'task_data' not in st.session_state:
    st.session_state.task_data = {}
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 1024

METRIC_DISPLAY_NAMES = {
    'human_review': 'Human Review',
    'rouge': 'ROUGE',
    'keyword_match': 'Keyword Match',
    'llm_judge': 'LLM Judge',
}
REVERSE_METRIC_DISPLAY_NAMES = {v: k for k, v in METRIC_DISPLAY_NAMES.items()}

# Basic styling
st.markdown(
    """
    <style>
    .stApp {
        background: #f5f5f5;
    }
    .stButton>button {
        background-color: #2b6777;
        color: white;
        border-radius: 4px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2b6777;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    '''
    <div style="background: #2b6777; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h1 style="color: white; margin: 0;">⚖️ LexEval</h1>
        <p style="color: #e0e7ef; margin: 0.5rem 0 0 0;">Legal model benchmarks</p>
    </div>
    ''',
    unsafe_allow_html=True
)

def slugify(text):
    text = text.strip().lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = re.sub(r'_+', '_', text)
    return text.strip('_')

def load_available_models():
    """Load available models from Together.ai API."""
    if not st.session_state.client:
        return {}
    
    # Return cached models if available
    if st.session_state.available_models:
        return st.session_state.available_models
    
    # Load models and cache them
    models = st.session_state.client.get_available_models()
    st.session_state.available_models = models
    return models

def load_available_tasks():
    """Load available evaluation tasks."""
    # Return cached tasks if available
    if st.session_state.available_tasks:
        return st.session_state.available_tasks
    
    tasks_dir = Path('tasks')
    if not tasks_dir.exists():
        return []
    
    tasks = []
    for task_file in tasks_dir.glob('*.json'):
        with open(task_file, 'r') as f:
            task_data = json.load(f)
            if isinstance(task_data, dict) and 'tasks' in task_data:
                # Use the top-level task_name for display
                tasks.append((task_file.stem, task_data['task_name']))
    
    # Cache the tasks
    st.session_state.available_tasks = tasks
    return tasks

def prettify_task_id(task_id):
    # Convert 'uk_legal_reasoning' -> 'UK Legal Reasoning'
    return task_id.replace('_', ' ').title()

def load_task(task_name: str) -> list:
    """Load a specific task file."""
    # Return cached task data if available
    if task_name in st.session_state.task_data:
        return st.session_state.task_data[task_name]
    
    task_file = Path('tasks') / f"{task_name}.json"
    if not task_file.exists():
        return []
    
    with open(task_file, 'r') as f:
        task_data = json.load(f)
        tasks = task_data.get('tasks', [])
        for task in tasks:
            if 'task_name' not in task or not task['task_name']:
                task['task_name'] = prettify_task_id(task.get('task_id', 'Task'))
        
        # Cache the task data
        st.session_state.task_data[task_name] = tasks
        return tasks

def save_task(task_name: str, tasks: list) -> bool:
    """Save tasks to a file."""
    try:
        tasks_dir = Path('tasks')
        tasks_dir.mkdir(exist_ok=True)
        task_file = tasks_dir / f"{task_name}.json"
        
        # Create task file structure with metadata
        task_data = {
            'schema_version': '1.0',
            'task_name': task_name,
            'description': f"Legal evaluation tasks for {task_name}",
            'created_at': int(time.time()),
            'tasks': []
        }
        
        # Process tasks
        for task in tasks:
            # Always generate task_id from task_name before saving
            task['task_id'] = slugify(task['task_name'])
            if 'judge_model' in task and not task['judge_model']:
                del task['judge_model']
            
            # Add task metadata
            task['metadata'] = {
                'created_at': int(time.time()),
                'last_modified': int(time.time())
            }
            
            task_data['tasks'].append(task)
        
        # Save with proper formatting
        with open(task_file, 'w') as f:
            json.dump(task_data, f, indent=2)
        
        # Update cache
        st.session_state.task_data[task_name] = tasks
        # Refresh available tasks list
        st.session_state.available_tasks = []
        load_available_tasks()
        
        return True
    except Exception as e:
        st.error(f"Error saving task: {str(e)}")
        return False

def validate_task(task: dict) -> tuple[bool, str]:
    """Validate a task dictionary."""
    required_fields = ['task_id', 'prompt', 'context', 'expected_output', 'reference', 'metric']
    for field in required_fields:
        if field not in task:
            return False, f"Missing required field: {field}"
    if not isinstance(task['task_id'], str):
        return False, "task_id must be a string"
    if not isinstance(task['prompt'], str):
        return False, "prompt must be a string"
    if not isinstance(task['context'], str):
        return False, "context must be a string"
    if not isinstance(task['expected_output'], str):
        return False, "expected_output must be a string"
    if not isinstance(task['reference'], str):
        return False, "reference must be a string"
    if task['metric'] not in ['human_review', 'rouge', 'keyword_match', 'llm_judge']:
        return False, "metric must be one of: human_review, rouge, keyword_match, llm_judge"
    return True, ""

def task_editor():
    """Task editor interface."""
    st.subheader("Task Editor")
    
    # Task selection
    tasks = load_available_tasks()
    selected_task = st.selectbox(
        "Select Task to Edit",
        options=["Create New Task"] + [t[1] for t in tasks],
        format_func=lambda x: x
    )
    
    if selected_task == "Create New Task":
        task_name = st.text_input("New Task Name (without .json extension)")
        tasks_data = []
    else:
        # Find the file name for the selected task ID
        task_file = next((t[0] for t in tasks if t[1] == selected_task), None)
        if task_file:
            task_name = task_file
            tasks_data = load_task(task_name)
        else:
            st.error("Task file not found")
            return
    
    # Task list
    st.write("Tasks:")
    for i, task in enumerate(tasks_data):
        with st.expander(f"Task {i+1}: {task.get('task_name', task.get('task_id', f'Task {i+1}'))}"):
            # Pretty name field
            task['task_name'] = st.text_input("Task Name", value=task.get('task_name', task.get('task_id', f'Task {i+1}')), key=f"name_{i}")
            # Auto-generate task_id from task_name
            task['task_id'] = slugify(task['task_name'])
            
            # Split input into prompt and context
            if 'input' in task:
                # Handle migration from old format
                task['prompt'] = task.get('prompt', task['input'])
                task['context'] = task.get('context', '')
                del task['input']
            
            task['prompt'] = st.text_area("Prompt", value=task.get('prompt', ''), key=f"prompt_{i}")
            task['context'] = st.text_area("Context", value=task.get('context', ''), key=f"context_{i}")
            task['expected_output'] = st.text_area("Expected Output", value=task['expected_output'], key=f"exp_{i}")
            task['reference'] = st.text_area("Reference", value=task['reference'], key=f"ref_{i}")
            
            metric_display_options = list(METRIC_DISPLAY_NAMES.values())
            current_metric_display = METRIC_DISPLAY_NAMES.get(task.get('metric', 'human_review'), 'Human Review')
            selected_metric_display = st.selectbox(
                "Metric",
                options=metric_display_options,
                index=metric_display_options.index(current_metric_display),
                key=f"metric_{i}"
            )
            task['metric'] = REVERSE_METRIC_DISPLAY_NAMES[selected_metric_display]
            
            # If LLM Judge, allow optional judge model selection
            if task['metric'] == 'llm_judge':
                models = load_available_models()
                judge_model_options = [''] + [models[m]['display_name'] for m in models]
                judge_model_ids = [''] + list(models.keys())
                current_judge_model = task.get('judge_model', '')
                if current_judge_model and current_judge_model in models:
                    current_judge_model_display = models[current_judge_model]['display_name']
                else:
                    current_judge_model_display = ''
                selected_judge_model_display = st.selectbox(
                    "LLM Judge Model (optional)",
                    options=judge_model_options,
                    index=judge_model_options.index(current_judge_model_display) if current_judge_model_display in judge_model_options else 0,
                    help="Leave blank to use the main model for judging.",
                    key=f"judge_model_{i}"
                )
                judge_model_id = judge_model_ids[judge_model_options.index(selected_judge_model_display)]
                if judge_model_id:
                    task['judge_model'] = judge_model_id
                elif 'judge_model' in task:
                    del task['judge_model']
            if st.button("Delete Task", key=f"del_{i}"):
                tasks_data.pop(i)
                st.rerun()
    
    # Add new task
    if st.button("Add New Task"):
        new_task = {
            'task_id': f"task_{len(tasks_data) + 1}",
            'prompt': "",
            'context': "",
            'expected_output': "",
            'reference': "",
            'metric': 'human_review'  # Set default metric to human review
        }
        tasks_data.append(new_task)
        st.rerun()
    
    # Save tasks
    if task_name and tasks_data:
        if st.button("Save Tasks"):
            # Validate all tasks
            valid = True
            for task in tasks_data:
                is_valid, error_msg = validate_task(task)
                if not is_valid:
                    st.error(f"Invalid task {task['task_id']}: {error_msg}")
                    valid = False
                    break
            
            if valid:
                if save_task(task_name, tasks_data):
                    st.success(f"Tasks saved to {task_name}.json")
                    st.rerun()

def main():
    # Create tabs
    api_tab, eval_tab, editor_tab = st.tabs(["API Configuration", "Run Evaluation", "Task Editor"])
    
    # API Configuration Tab
    with api_tab:
        st.header("Together.ai API Configuration")
        st.markdown("""
        To use LexEval, you'll need a Together.ai API key. You can get one by:
        1. Creating an account at [Together.ai](https://api.together.xyz/)
        2. Going to your [API Keys page](https://api.together.xyz/settings/api-keys)
        3. Creating a new API key
        """)
        
        api_key = st.text_input(
            "Enter your Together.ai API Key",
            type="password",
            value=st.session_state.api_key or ""
        )
        
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            if api_key:
                st.session_state.client = TogetherClient(api_key)
                # Clear caches when API key changes
                st.session_state.available_models = {}
                st.session_state.available_tasks = []
                st.session_state.task_data = {}
                st.success("API key configured successfully!")
                # Automatically switch to Run Evaluation tab
                st.session_state.active_tab = "Run Evaluation"
            else:
                st.session_state.client = None
                st.warning("Please enter a valid API key to continue.")

    # Run Evaluation Tab
    with eval_tab:
        if not st.session_state.client:
            st.warning("Please configure your Together.ai API key in the API Configuration tab to begin.")
        else:
            # Model and Task Selection
            st.subheader("Model and Task Selection")
            select_col1, select_col2 = st.columns(2)
            
            with select_col1:
                models = load_available_models()
                if not models:
                    st.error("No models available. Please check your configuration.")
                    return
                model_options = list(models.keys())
                model_display_names = {model_id: model_info['display_name'] for model_id, model_info in models.items()}
                model_options.sort(key=lambda x: model_display_names[x])
                selected_model = st.selectbox(
                    "Select Model",
                    options=model_options,
                    format_func=lambda x: model_display_names.get(x, x),
                    key="model_selector"  # Add a unique key
                )
            with select_col2:
                tasks = load_available_tasks()
                if not tasks:
                    st.error("No tasks available. Please add tasks first.")
                    return
                selected_task = st.selectbox(
                    "Select Task",
                    options=[t[0] for t in tasks],
                    format_func=lambda x: next((t[1] for t in tasks if t[0] == x), x)
                )

            # Model Info and Pricing in a new row
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                if selected_model in models:
                    model_info = models[selected_model]
                    st.subheader("Model Information")
                    st.markdown(f"**Organization:** {model_info['organization']}")
                    st.markdown(f"**Context Length:** {model_info['context_length']:,} tokens")
            with info_col2:
                if selected_model in models:
                    model_info = models[selected_model]
                    pricing = model_info.get('pricing', {})
                    if pricing:
                        st.subheader("Pricing")
                        st.markdown(f"<div style='display: flex; justify-content: space-between;'><span>Input:</span><span><b>${pricing.get('input', 0):.2f}/1M tokens</b></span></div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='display: flex; justify-content: space-between;'><span>Output:</span><span><b>${pricing.get('output', 0):.2f}/1M tokens</b></span></div>", unsafe_allow_html=True)

            # Generation Parameters under Advanced Settings
            with st.expander("Advanced Settings"):
                st.subheader("Generation Parameters")
                col1, col2, col3 = st.columns(3)
                
                # Get context length for selected model
                context_length = 4096
                if selected_model in models:
                    context_length = models[selected_model].get('context_length', 4096)
                
                with col1:
                    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
                    max_tokens = st.number_input(
                        "Max Tokens",
                        1,
                        context_length,
                        value=context_length // 3, # Default to 1/3 of context length
                        key=f"max_tokens_{selected_model}"  # Unique key per model
                    )

                with col2:
                    top_p = st.slider("Top P", 0.0, 1.0, 0.7, 0.1)
                    top_k = st.number_input("Top K", 1, 100, 50)
                
                with col3:
                    repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.0, 0.1)
                    use_chat = st.checkbox("Use Chat Mode", value=True)

            # Run Evaluation
            if st.button("Run Evaluation"):
                if not selected_task:
                    st.error("Please select a task to evaluate.")
                    return

                try:
                    # Initialize benchmark runner
                    runner = BenchmarkRunner(st.session_state.client)
                    
                    # Load tasks
                    task_file = f"tasks/{selected_task}.json"
                    tasks = runner.load_tasks(task_file)
                    
                    # Run benchmark
                    with st.spinner("Running evaluation..."):
                        results = runner.run_benchmark(
                            tasks=tasks,
                            model=selected_model,
                            use_chat=use_chat,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=repetition_penalty
                        )
                        
                        st.session_state.results = results
                        
                    st.success("Evaluation completed!")
                    
                except Exception as e:
                    st.error(f"Error running evaluation: {str(e)}")

            # Display Results
            if st.session_state.results:
                st.subheader("Evaluation Results")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Tasks", st.session_state.results['metrics']['total_tasks'])
                with col2:
                    st.metric("Successful Runs", st.session_state.results['metrics']['successful_runs'])
                with col3:
                    st.metric("Avg Response Time", f"{st.session_state.results['metrics']['avg_latency']:.2f}s")

                # Create DataFrame from results
                results_list = st.session_state.results['results']
                
                
                # Create DataFrame with explicit columns
                df = pd.DataFrame(results_list, columns=[
                    'task_id', 'prompt', 'context', 'expected_output', 
                    'response', 'score', 'latency', 'metric'
                ])
                
                # Rename columns to be more user-friendly
                column_mapping = {
                    'task_id': 'Task ID',
                    'prompt': 'Prompt',
                    'context': 'Context',
                    'expected_output': 'Expected Output',
                    'response': 'Model Response',
                    'score': 'Score',
                    'latency': 'Response Time (s)',
                    'metric': 'Evaluation Method'
                }
                
                # Rename columns
                df = df.rename(columns=column_mapping)
                
                # Format score column to show as percentage if not None
                df['Score'] = df.apply(lambda row: f"{row['Score']:.1%}" if pd.notnull(row['Score']) else "Pending Human Review", axis=1)
                
                # Format response time to 2 decimal places
                df['Response Time (s)'] = df['Response Time (s)'].map('{:.2f}'.format)
                
                # Display the DataFrame
                st.write("Results Table:")
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Add export options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download Results (CSV)",
                        df.to_csv(index=False).encode('utf-8'),
                        "evaluation_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
                with col2:
                    st.download_button(
                        "Download Results (JSON)",
                        json.dumps(st.session_state.results, indent=2).encode('utf-8'),
                        "evaluation_results.json",
                        "application/json",
                        key='download-json'
                    )

    # Task Editor Tab
    with editor_tab:
        task_editor()

    # Add footer
    st.markdown(
        '''
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #f5f5f5; padding: 1rem; text-align: center; border-top: 1px solid #e0e0e0;">
            Built by <a href="https://www.ryanmcdonough.co.uk/" target="_blank">Ryan McDonough</a>
        </div>
        ''',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 