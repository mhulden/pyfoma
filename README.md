# PyFoma

[![PyPI version](https://badge.fury.io/py/pyfoma.svg)](https://badge.fury.io/py/pyfoma)

## Quickstart

```bash
pip install pyfoma
```

```python
from pyfoma import FST
myfst = FST.re("(cl|n|s)?e(ve|a)r")
myfst.view()
```

<img src="./docs/images/quickstart1.png" width="722">

Next, check out the documentation!

## Contributing

<!-- TODO: Include details on how to open PRs -->

### Rebuilding the documentation

```bash
# Update automatically-scraped docs
sphinx-apidoc -o docs src/pyfoma/

# Rebuild docs
cd docs
make html
```

## Citation

```
@inproceedings{hulden-etal-2024-pyfoma,
    title = "{P}y{F}oma: a Python finite-state compiler module",
    author = "Hulden, Mans and Ginn, Michael and Silfverberg, Miikka and Hammond, Michael",
    editor = "Cao, Yixin and Feng, Yang and Xiong, Deyi",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-demos.24",
    doi = "10.18653/v1/2024.acl-demos.24",
    pages = "258--265"
}
```
