import re
from pathlib import Path

def parse_convergence_from_output(output_file):
    """
    Parse Fatrop solver output file to extract convergence data.
    
    Expected to find lines with iteration information, objective values,
    and feasibility metrics.
    
    Args:
        output_file (str): Path to the solver output file
    
    Returns:
        dict: Dictionary with keys:
            - 'iteration': list of iteration numbers
            - 'obj': list of objective values
            - 'primal_feasibility': list of primal feasibility (constraint violation)
            - 'dual_feasibility': list of dual feasibility (gradient violation)
    """
    
    convergence_stats = {
        'iteration': [],
        'obj': [],
        'primal_feasibility': [],
        'dual_feasibility': [],
    }
    
    if not Path(output_file).exists():
        print(f"Warning: Output file '{output_file}' not found")
        return convergence_stats
    
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    # Parse lines looking for iteration data
    # Common Fatrop output patterns include:
    # "iter | obj | pr_feas | du_feas" or similar
    # We'll look for lines with numerical data
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and headers
        if not line or 'iter' in line.lower() or '---' in line or '|' in line and 'obj' in line.lower():
            continue
        
        # Try to extract numerical data
        # Look for patterns like: "0    1234.5   0.001   0.002"
        # or with pipes: "0 | 1234.5 | 0.001 | 0.002"
        
        # Remove leading/trailing whitespace and split by multiple delimiters
        parts = re.split(r'[\s|]+', line)
        parts = [p for p in parts if p]  # Remove empty strings
        
        # Try to parse as numbers
        if len(parts) >= 4:
            try:
                iter_num = int(float(parts[0]))
                obj_val = float(parts[1])
                primal_feas = float(parts[2])
                dual_feas = float(parts[3])
                
                convergence_stats['iteration'].append(iter_num)
                convergence_stats['obj'].append(obj_val)
                convergence_stats['primal_feasibility'].append(primal_feas)
                convergence_stats['dual_feasibility'].append(dual_feas)
            except (ValueError, IndexError):
                # Skip lines that don't match the expected format
                continue
    
    return convergence_stats


def save_convergence_stats(convergence_stats, output_pkl_file):
    """
    Save convergence statistics to a pickle file.
    
    Args:
        convergence_stats (dict): Convergence statistics dictionary
        output_pkl_file (str): Path to save the pickle file
    """
    with open(output_pkl_file, 'wb') as f:
        pkl.dump(convergence_stats, f)
    print(f"Convergence stats saved to: {output_pkl_file}")


def load_and_parse_convergence(solver_output_file, pkl_output_file):
    """
    Load solver output, parse convergence data, and save to pickle.
    
    Args:
        solver_output_file (str): Path to the solver output text file
        pkl_output_file (str): Path where to save the pickle file
    
    Returns:
        dict: Convergence statistics dictionary
    """
    convergence_stats = parse_convergence_from_output(solver_output_file)
    
    if convergence_stats['iteration']:
        print(f"Parsed {len(convergence_stats['iteration'])} iterations from output")
        save_convergence_stats(convergence_stats, pkl_output_file)
    else:
        print(f"Warning: No convergence data found in {solver_output_file}")
    
    return convergence_stats

# ==================== End of convergence parsing ====================