# cats_and_dogs
A neural network to distinguish between pictures of cats and dogs

## Initial configuration
**For usage:**  

install pyproject.toml  
install README.md  
```
pip install miniconda
python -m venv cats_and_dogs
poetry install
pre-commit install
pre-commit run -a
```

**For development:**  

install dvc  
install pre-commit  
run poetry install  
run pre-commint install  
run dvc pull to get all datasets and models  

## Train model
install train.py  
install make_dataloader.py  
install configs  
install dataset  
unzip the dataset files  
```
python train.py
```

## Predict on input
install img.png  
install infer.py  
```
python infer.py
```
