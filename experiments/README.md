```bash
tmux new -s mohsen / tmux a -t mohsen
conda create -n henv --clone pt_gpu
conda activate henv
pip install -r requirements.txt
cd globenc_extension
python ./huggingface-finetune/training.py ./args/sst2.json
python ./huggingface-finetune/training.py ./args/mnli.json
python ./huggingface-finetune/training.py ./args/cola2.json
python ./huggingface-finetune/training.py ./args/mrpc2.json
python ./huggingface-finetune/training.py ./args/qnli2.json
# Jupyter
tmux a -t jupyter
conda activate globenc-venv
jupyter notebook
```