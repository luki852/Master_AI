import json
from pathlib import Path
p = Path('3rd_Semester/UE_Probabilistic_Models/UE/Ex5/assignment5/parameter_learning.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
nb['cells'][10]['source'] = '''def maximum_likelihood_estimate(data: np.ndarray, variable_id: int, parent_ids: tuple=tuple(), variable_cardinality: Union[Tuple[int], None]=tuple(), alpha: float=0):  # Estimate one discrete conditional probability table.
    """Count observed events, add alpha per outcome, and normalize over the child.

    Axes follow (variable_id,) + parent_ids, preserving the supplied parent order.
    With alpha=0, an unseen parent configuration has no unique MLE; use a
    uniform distribution for that column as an explicit convention.
    """
    assert isinstance(parent_ids, tuple)  # Require an ordered tuple of parent column IDs.
    assert variable_cardinality is None or isinstance(variable_cardinality, tuple)  # Accept explicit cardinalities or the binary default.
    assert np.isfinite(alpha) and alpha >= 0  # Pseudocounts must be finite and nonnegative.
    if variable_cardinality is None or len(variable_cardinality) == 0:  # Detect an omitted cardinality specification.
        variable_cardinality = (2,) * data.shape[1]  # Assume two outcomes per variable by default.
    variable_ids = (variable_id,) + parent_ids  # Put the child first and retain the requested parent order.
    shape = tuple(variable_cardinality[i] for i in variable_ids)  # Determine the number of outcomes along each table axis.
    counts = np.full(shape, alpha, dtype=float)  # Initialize every possible event with its smoothing pseudocount.
    for sample in data:  # Visit each observed joint event once.
        index = tuple(sample[i] for i in variable_ids)  # Select the child value and the corresponding parent values.
        counts[index] += 1  # Add one observation to that exact table entry.
    totals = counts.sum(axis=0, keepdims=True)  # Sum over child values separately for each parent configuration.
    cpdt = np.full(shape, 1.0 / shape[0], dtype=float)  # Prepare a uniform fallback for configurations with zero total count.
    np.divide(counts, totals, out=cpdt, where=totals > 0)  # Normalize observed or smoothed columns without dividing by zero.
    return cpdt  # Return a normalized table with child axis first.
'''.splitlines(keepends=True)
nb['cells'][13]['source'] = '''def log_likelihood(data: np.ndarray, bayes_net: BayesNet):  # Score the entire data set under a fixed Bayesian network.
    """Return sum_n sum_i log P(x_i^(n) | pa(x_i)^(n)), using natural logs."""
    ll = 0.0  # Start with the log of an empty product, which is zero.
    for sample in data:  # Process every observed joint event.
        for variable in bayes_net:  # Visit every factor of the Bayesian network.
            probability = variable(sample)[sample[variable.id]]  # Select the probability of the observed child value given its parents.
            if probability == 0:  # Detect an event that the model considers impossible.
                return float('-inf')  # A zero probability makes the complete data likelihood zero and its log minus infinity.
            ll += np.log(probability)  # Add this factor's natural logarithm instead of multiplying small probabilities.
    return float(ll)  # Return the total log-likelihood as a Python floating point number.
'''.splitlines(keepends=True)
# Explain each existing executable line without changing the supplied experiments.
for cell in nb['cells']:
    if cell['cell_type'] != 'code' or cell is nb['cells'][10] or cell is nb['cells'][13]:
        continue
    lines=[]
    for line in ''.join(cell['source']).splitlines():
        s=line.strip()
        comment=None
        if s and not s.startswith('#') and '#' not in line:
            if s.startswith('import ') or s.startswith('from '): comment='Import the libraries or provided helpers used below.'
            elif s.startswith('assert '): comment='Verify the expected numerical result or normalization.'
            elif s.startswith('print('): comment='Display this result for inspection.'
            elif s.startswith('np.random.seed'): comment='Fix the random seed so the experiment is reproducible.'
            elif 'sample_forward(' in s: comment='Generate independent observations from the specified network.'
            elif 'maximum_likelihood_estimate(' in s: comment='Estimate this variable distribution from training counts using the listed parents.'
            elif 'log_likelihood(' in s: comment='Compute the total log-likelihood of these data under this model.'
            elif s.startswith('try:'): comment='Run the supplied sanity checks and report any failure.'
            elif s.startswith('except '): comment='Catch and report an error in the sanity checks.'
            elif s.startswith('traceback.'): comment='Display the traceback to locate a failed check.'
            elif s.startswith('any_failed'): comment='Track whether any supplied sanity check failed.'
            elif 'BayesNet(' in s: comment='Construct the network from probability tables and their variable IDs.'
            elif 'np.array(' in s or 'np.ones(' in s: comment='Define a probability table with the child on axis zero.'
            elif s.startswith('(') and s.endswith(('),', ')')): comment='Associate the table axes with the child ID followed by its parent IDs.'
            elif s.startswith('_A_'): comment='Assign readable names to the data column indices.'
            elif s.startswith('expected ='): comment='Set the reference value for this sanity check.'
            elif s.startswith('actual ='): comment='Compute the result to compare against the reference.'
            elif s.startswith('help('): comment='Show the documentation of the graph drawing helper.'
            elif s.startswith('draw_graph('): comment='Visualize the dependency structure of the example network.'
            elif s.startswith('node_names='): comment='Label graph nodes with the variable names.'
            elif s.startswith('pos='): comment='Place graph nodes at these display coordinates.'
            elif s=='bn,': comment='Pass the example network to the graph drawing helper.'
            elif '=' in s: comment='Store the value used by the following example or experiment.'
        lines.append(line + ('  # '+comment if comment else '')+'\n')
    cell['source']=lines
    cell['outputs']=[]
    cell['execution_count']=None
nb['cells'].append({'cell_type':'markdown','metadata':{},'source':['### Implementation notes\n','The estimator supports arbitrary discrete cardinalities, preserves the supplied parent order, and uses additive smoothing: `(count + alpha) / (parent_count + child_cardinality * alpha)`. For an unseen parent configuration at alpha=0, any normalized column maximizes the likelihood; this implementation chooses a uniform column.\n','The score is the **sum**, not the mean, of natural log probabilities over all observations and all network factors. An impossible observation returns `-inf`; an empty data set returns `0.0`. M3 contains the M1 structure plus extra edges, so its maximized training likelihood cannot be lower, but its test likelihood can be worse because of overfitting.\n']})
p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
