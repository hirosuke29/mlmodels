# Development guide

## Environment Setup
### requirements
- Python >=3.8 and <3.10

### install dependency
```shell
$ python3 -m venv .venv
$ pip install -r requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html
$ source ./.venv/bin/activate
```

## Development
Testing
```shell
$ pytest
``` 

mypy
```shell
$ mypy .
```

flake8
```
$ flake8
```

black
```
$ black .
```
