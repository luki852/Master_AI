from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
import sys, tempfile, json
p=Path('3rd_Semester/UE_Probabilistic_Models/UE/Ex5/assignment5/parameter_learning.ipynb').resolve()
ks=Path(tempfile.mkdtemp())/'assignment5-python'
ks.mkdir()
(ks/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Python (assignment 5)','language':'python'}))
KernelSpecManager().install_kernel_spec(str(ks),kernel_name='assignment5-python',user=True)
nb=nbformat.read(p,as_version=4)
NotebookClient(nb,timeout=180,kernel_name='assignment5-python',resources={'metadata':{'path':str(p.parent)}}).execute()
nbformat.write(nb,p)
for c in nb.cells:
 for o in c.get('outputs',[]):
  if o.output_type=='stream' and 'logP(' in o.text: print(o.text)
# Independently check a nonbinary child, reversed parents, smoothing, and unseen columns.
sys.path.insert(0,str(p.parent))
ns={}
exec(nb.cells[0].source,ns)
exec(nb.cells[10].source,ns)
exec(nb.cells[13].source,ns)
import numpy as np
mle=ns['maximum_likelihood_estimate']; ll=ns['log_likelihood']
d=np.array([[0,0,0],[2,1,0],[2,1,0],[1,0,1]])
c=mle(d,0,(2,1),(3,2,2),alpha=1)
expected=np.array([[[2,1],[1,1]],[[1,1],[2,1]],[[1,3],[1,1]]],dtype=float)
expected/=expected.sum(axis=0,keepdims=True)
np.testing.assert_allclose(c,expected)
u=mle(d,0,(2,1),(3,2,2))
np.testing.assert_allclose(u[:,1,1],np.ones(3)/3)
np.testing.assert_allclose(mle(d,0,variable_cardinality=(3,2,2)),[.25,.25,.5])
np.testing.assert_allclose(mle(np.empty((0,1),dtype=int),0),[.5,.5])
from bayesian_network import BayesNet
bn=BayesNet((np.array([1.,0.]),(0,)))
assert ll(np.array([[1]]),bn)==float('-inf')
assert ll(np.empty((0,1),dtype=int),bn)==0.
bn=BayesNet((np.array([.25,.75]),(0,)))
np.testing.assert_allclose(ll(np.array([[0],[1],[1]]),bn),np.log(.25)+2*np.log(.75))
print('Notebook executed successfully; all supplied and independent edge-case checks passed.')
KernelSpecManager().remove_kernel_spec('assignment5-python')
