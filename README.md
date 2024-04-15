# LLM-footprint


## Usage
```
# Clone the repo and submodules
git clone --recurse-submodules https://github.com/chenghuzi/LLM-footprint.git

cd LLM-footprint

# create virtual environment. Here python == 3.10 or 3.9
python -m venv .venv
# activate the env
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# enter the submodule and install its dependencies 
cd detectors/Binoculars
pip install -e .

# go back to the project root dir
cd ..
cd ..

# create a dir for data
mkdir data
# Now move the full_text.json data to the data dir and run

python run_score.py data/full_text.json --bs 16 --chunksize 256 # here 16 is the batch size, you can change it to maximize efficency.

# Since the script will modify the data, you can stop it anytime and rerun with the exact same command.

```

## GPU assignment notice

Go to  `detectors/Binoculars/binoculars/detector.py` to change  `DEVICE` id if the default GPU is not enough.

