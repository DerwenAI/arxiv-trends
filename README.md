# arxiv-trends

Analyze trends among articles published on [arXiv](https://arxiv.org/help/api)


## Install

```
python -m venv venv
source venv/bin/activate

python -m pip install -U pip
python -m pip install -r requirements.txt 
python -m spacy download en_core_web_sm

echo "{}" > arxiv.json
```


## Usage

```
python arxiv.py cmd-query --min-date=2021-01-01 "knowledge graph"
```

```
python arxiv.py cmd-extract
```


## License and Copyright

Source code for **arxiv-trends** plus its logo, documentation, and
examples have an [MIT license](https://spdx.org/licenses/MIT.html)
which is succinct and simplifies use in commercial applications.


## Kudos

Thank you to arXiv for use of its open access interoperability.
