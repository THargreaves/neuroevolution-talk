import os
import platform
import subprocess

if not os.path.exists('presentation.ipynb'):
    raise OSError("could not find presentation notebook")

system = platform.system()
if system == 'Windows':
    process = subprocess.Popen([
        'jupyter', 'nbconvert',
        'presentation.ipynb',
        '--to', 'slides',
        '--TagRemovePreprocessor.enabled=True',
        "--TagRemovePreprocessor.remove_cell_tags=['remove_cell']",
        "--TagRemovePreprocessor.remove_input_tags=['remove_input']",
        "--TagRemovePreprocessor.remove_single_output_tags=['remove_single_output']",
        "--TagRemovePreprocessor.remove_all_outputs_tags=['remove_all_output']",
    ], shell=True)
elif system == 'Linux':
    process = subprocess.Popen(' '.join([
        'jupyter nbconvert',
        'presentation.ipynb',
        '--to slides',
        '--TagRemovePreprocessor.enabled=True',
        "--TagRemovePreprocessor.remove_cell_tags=\"['remove_cell']\"",
        "--TagRemovePreprocessor.remove_input_tags=\"['remove_input']\"",
        "--TagRemovePreprocessor.remove_single_output_tags=\"['remove_single_output']\"",
        "--TagRemovePreprocessor.remove_all_outputs_tags=\"['remove_all_output']\"",
    ]), shell=True)
else:
    raise NotImplementedError("build does currently not support system")
process.wait()

