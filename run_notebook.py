"""
Fix deprecated pandas API calls, optimize LSTM architecture for better performance, 
and execute the notebook robustly in a Windows environment.
"""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import re
import asyncio
import sys
import os

# Suppress TensorFlow logging for a cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Fix for asyncio.WindowsSelectorEventLoopPolicy on Windows (Python 3.8+)
if sys.platform == 'win32':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass

def fix_notebook_content(nb):
    """
    Apply robust regex-based fixes to notebook cells.
    Targets deprecations and optimizes hyperparameters to beat baseline models.
    """
    replacements = [
        # Fix deprecated fillna methods
        (r"\.fillna\(method=['\"]ffill['\"]", ".ffill(", 0),
        (r"\.fillna\(method=['\"]bfill['\"]", ".bfill(", 0),
        
        # Optimize LSTM Architecture: Simplified 1-layer LSTM with more units
        (r"model = Sequential\(\[.*?\]\)", 
         "model = Sequential([LSTM(units=128, input_shape=(seq_length, 1)), Dropout(0.1), Dense(units=1)])", 
         re.DOTALL),
        
        # Optimize Training: 100 epochs is sufficient for convergence
        (r"epochs=50", "epochs=100", 0),
        
        # Optimize Batch Size
        (r"batch_size=16", "batch_size=32", 0),
    ]
    
    fixed_count = 0
    for cell in nb.cells:
        if cell.cell_type == "code":
            original_source = cell.source
            for pattern, replacement, flags in replacements:
                cell.source = re.sub(pattern, replacement, cell.source, flags=flags)
            
            if cell.source != original_source:
                fixed_count += 1
    return fixed_count

def main():
    # Use absolute paths to avoid directory issues
    base_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(base_dir, "LSTM_Sales_Forecasting.ipynb")
    output_path = os.path.join(base_dir, "LSTM_Sales_Forecasting_Output.ipynb")
    
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} not found.")
        return

    print(f"Reading {notebook_path}...")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Apply performance and compatibility fixes
    fixed_cells = fix_notebook_content(nb)
    print(f"Applied optimizations and fixes to {fixed_cells} cells.")

    print("Executing notebook (Optimizing LSTM performance)...")
    # Timeout set to 20 minutes to allow for 100 epochs
    # allow_errors=True ensures we get partial results even if one plot fails
    ep = ExecutePreprocessor(timeout=1200, kernel_name="python3", allow_errors=True)
    
    try:
        ep.preprocess(nb, {"metadata": {"path": base_dir}})
        print("Notebook execution completed.")
    except Exception as e:
        print(f"\nExecution Warning/Error:\n{str(e)}")

    # Save the result
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    
    print(f"Final output saved to: {output_path}")

if __name__ == "__main__":
    main()
