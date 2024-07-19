# Memory-Efficient Pseudo-Labeling for Online Source-Free Universal Domain Adaptation using a Gaussian Mixture Model

This is the official repository to the paper "Memory-Efficient Pseudo-Labeling for Online Source-Free Universal Domain Adaptation using a Gaussian Mixture Model".

## Usage
### Preparation
- Clone this repository
- Install the requirements by running `pip install -r requirements.txt`
- Download datasets into the folder [data](data).

### Source training
We uploaded the checkpoints of our pre-trained source models into the folder [source_models](source_models). To still do the source training yourself, edit the corresponding config file [source_training.yaml](configs/source_training.yaml) accordingly and run the following command: `python main.py fit --config configs/source_training.yaml`

### Source-only testing
To test without adaptation, i.e. to get the source-only results, edit the corresponding config file [source_only_testing.yaml](configs/source_only_testing.yaml) to select the desired scenario and run the following command: `python main.py test --config configs/source_only_testing.yaml`

### Perform the online source-free universal domain adaptation to adapt to the target domain
To perform the online test-time adaptation, edit the corresponding config file [target_adaptation.yaml](configs/target_adaptation.yaml) to select the desired scenario and run the following command: `python main.py fit --config configs/target_adaptation.yaml`
