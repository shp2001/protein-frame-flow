
import sys
sys.path.append('/home/psh/project_v2')
from local_ImmuneBuilder.local_abodybuilder.refine import refine

input_file = "/home/psh/protein-frame-flow/inference_outputs/fm_hybrid/2025-05-28_17-17-52/epoch=105-step=150096_copy/run_2025-05-31_09-40-37/7df1_F_J_C/sample_0/sample_1_copy.pdb"
output_file = "/home/psh/protein-frame-flow/inference_outputs/fm_hybrid/2025-05-28_17-17-52/epoch=105-step=150096_copy/run_2025-05-31_09-40-37/7df1_F_J_C/sample_0/sample_1_relaxed.pdb"

refine(input_file, output_file)
