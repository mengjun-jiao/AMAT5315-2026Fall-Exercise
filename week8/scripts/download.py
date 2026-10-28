from pathlib import Path
from urllib.request import urlretrieve
root=Path(__file__).resolve().parent.parent; data=root/'data';data.mkdir(exist_ok=True)
urls={
'TinyStoriesV2-GPT4-valid.txt':'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt',
'tok512.model':'https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da/stories260K/tok512.model',
'model.safetensors':'https://huggingface.co/minishlab/potion-base-8M/resolve/bf8b056/model.safetensors',
'tokenizer.json':'https://huggingface.co/minishlab/potion-base-8M/resolve/bf8b056/tokenizer.json',
'chinchilla-runs.csv':'https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/92258837425e1b5f2851d624287f0120583a3d0e/data/svg_extracted_data.csv'}
for name,url in urls.items():
 path=data/name
 if path.exists() and path.stat().st_size: print('exists',path); continue
 print('download',url,flush=True);urlretrieve(url,path)
