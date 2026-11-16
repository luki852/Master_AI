from pathlib import Path
import nbformat
from nbclient import NotebookClient
import os

p = Path('3rd_Semester/UE_Advanced_Machine_Learning/A3/A3_exercise.ipynb').resolve()
os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
n = nbformat.read(p, as_version=4)
client = NotebookClient(n, timeout=1200, kernel_name='python3', resources={'metadata': {'path': str(p.parent)}})
try:
    client.execute()
finally:
    nbformat.write(n, p)
for i in [34, 55, 58]:
    print(f'CELL {i}')
    for output in n.cells[i].get('outputs', []):
        if output.output_type == 'stream':
            print(output.text)
print('Notebook executed successfully.')
