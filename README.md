# pollcomm
Code for the manuscript under revision "Adaptive Foraging of Pollinators Fosters Gradual Tipping under Resource Competition and Rapid Environmental Change" authored by Sjoerd Terpstra, Flávia M.D. Marquitti & Vítor V. Vasconcelos.

## Installation
The Python environment is specified in ```environment.yml```.

Install the virtual environment using ```conda env create```.

Then, activate the environment using ```conda activate pollcomm```.

The environment can be deactivated using ```conda deactivate```.

## Usage
The graphs of the manuscript can be reproduced by the functions defined in ```ms.py```.

The graphs of the sensitivity analysis in the Supporting Information can be reproduced by the functions defined in ```sa.py```.

The model is implemented in the ```pollcomm``` module. The two models are ```adaptive_model.py``` (with adaptive foraging)
and ```base_model.py``` (no adaptive foraging) which are both described in the manuscript.
