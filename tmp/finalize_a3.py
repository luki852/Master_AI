from pathlib import Path
import nbformat

p = Path('3rd_Semester/UE_Advanced_Machine_Learning/A3/A3/A3_exercise.ipynb')
n = nbformat.read(p, as_version=4)
n.cells[35].source = n.cells[35].source.replace('The estimates of', 'In this run, the estimated mean is approximately $(1.01547,2.02239,2.99595)^\\top$, the noise variance is $0.062234$ (true value $0.0625$), and the noise standard deviation is $0.249467$ (true value $0.25$). The estimated loading vector is approximately $(-0.78470,-0.47980,0.30067)^\\top$, which is close to the negative of the generating vector.\n\nThe estimates of', 1)
n.cells[67].source = 'The first 25 color-coded encodings form a curved, approximately U-shaped trajectory, with several frames clustered near the left end and other frames returning along similar parts of the curve. This is consistent with gradual changes during inhalation and exhalation and with similar images at recurring breathing phases. The trajectory is not a perfect closed circle: breathing varies, and a linear two-dimensional model cannot represent every detail of the motion.'
n.cells[73].source = 'The generated images retain the broad anatomical layout, including the dark lung region and the bright lower boundary, but they are blurry and show grain and some doubled or smeared boundaries. A two-dimensional linear representation cannot capture all fine details or nonlinear breathing motion, and independent isotropic Gaussian observation noise adds grain; latent samples far from the observed encodings may also produce implausible patterns. Quality could improve by increasing the latent dimension (selected on held-out data), using a richer noise model, or learning a nonlinear generative model; displaying the noiseless decoder mean removes grain but is no longer a full sample from the fitted observation distribution.'
nbformat.validate(n)
assert len(n.cells) == 74
assert all(c.execution_count is not None for c in n.cells if c.cell_type == 'code')
assert not any(o.output_type == 'error' for c in n.cells if c.cell_type == 'code' for o in c.outputs)
nbformat.write(n, p)
print('Restoration complete; all 74 cells preserved and all code cells executed without errors.')
