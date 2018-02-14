# Singing Voice Separation via Recurrent Inference and Skip-Filtering connections

Support material and source code for the method described in : S.I. Mimilakis, K. Drossos, J.F. Santos, G. Schuller, T. Virtanen, Y. Bengio, "Monaural Singing Voice Separation with Skip-Filtering Connections and Recurrent Inference of Time-Frequency Mask", in arXiv:1711.01437 [cs.SD], Nov. 2017.
This work has been accepted for poster presentation at ICASSP 2018.

Please use the above citation if you find any of the code useful.

Listening Examples :  https://js-mim.github.io/mss_pytorch/

### Extensions     :
- An improvement of this work, which includes a novel regularization technique, can be found here: https://github.com/dr-costas/mad-twinnet .



### Requirements   :
- Numpy            :  numpy==1.13.1
- SciPy            :  scipy==0.19.1
- PyTorch          :  pytorch==0.2.0_2  (For inference and model testing pytorch==0.3.0 is supported. Training needs to be checked.)
- TorchVision      :  torchvision==0.1.9
- Other            :  wave(used for wav file reading), pyglet(used only for audio playback), pickle(for storing some results)
- Trained Models   :  https://doi.org/10.5281/zenodo.1064805 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1064805.svg)](https://doi.org/10.5281/zenodo.1064805)
					  Download and place them under "results/results_inference/"
- MIR_Eval         :  mir_eval=='0.4'  (This is used only for unofficial cross-validation. For the reported evaluation please refer to: https://github.com/faroit/dsdtools)

### Usage          :
- Clone the repository.
- Add the base directory to your Python path.
- While "mss_pytorch" is your current directory simply execute the "processes_scripts/main_script.py" file.
- Arguments for training and testing are given to the main function of the "processes_scripts/main_script.py" file.

### Acknowledgements :
The research leading to these results has received funding from the European Union's H2020 Framework Programme (H2020-MSCA-ITN-2014) under grant agreement no 642685 MacSeNet.
