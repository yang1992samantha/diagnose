# Directory structure
- diagnoise
  - code
    - github
  - data
    - electronic-medical-record-50
    - mimic-3
  - logs
    - checkpoints
    - logs
    - result

# Run
## Environment
python 3.9
torch 1.11.0

`sh install_pyg.sh`

`pip install requirements.txt`

## Run

parser.py : HOME_DIR = '/home/***/diagnoise/'

nohup python main.py > DILH-mimic-longformer.out &
