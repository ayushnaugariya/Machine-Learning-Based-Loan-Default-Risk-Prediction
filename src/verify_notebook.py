import argparse
import json
import os
from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def execute_notebook(notebook_path):
    print(f"Loading notebook: {notebook_path}")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    print(f"Found {len(code_cells)} code cells to execute.\n")
    
    # Shared execution context. Use a non-interactive plotting backend so
    # verification never blocks on plt.show() in terminal or CI runs.
    os.environ.setdefault("MPLBACKEND", "Agg")
    context = {
        "__name__": "__main__",
    }
    
    for idx, cell in enumerate(code_cells, 1):
        source_lines = cell["source"]
        code = "".join(source_lines)
        
        # Skip empty cells
        if not code.strip():
            continue
            
        print(f"Executing cell {idx}/{len(code_cells)}...")
        print("-" * 40)
        # Print a preview of the code
        preview = "\n".join(source_lines[:3])
        if len(source_lines) > 3:
            preview += "\n..."
        print(preview)
        print("-" * 40)
        
        try:
            # Execute the cell's code within the shared context
            exec(code, context, context)
            if "plt" in context:
                context["plt"].close("all")
            print("Cell executed successfully!\n")
        except Exception as e:
            print(f"\nERROR in Cell {idx} execution!")
            print("=" * 60)
            print(code)
            print("=" * 60)
            print("Traceback details:")
            traceback.print_exc()
            sys.exit(1)
            
    print("All notebook cells executed successfully! No syntax or runtime errors found.")


def parse_args():
    parser = argparse.ArgumentParser(description="Execute code cells from a generated notebook.")
    parser.add_argument(
        "notebook",
        nargs="?",
        default=PROJECT_ROOT / "Loan_Default_Prediction.ipynb",
        type=Path,
        help="Notebook path to validate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    execute_notebook(args.notebook)
