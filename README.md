# arxiv-trends

Analyze trends among articles published on [arXiv](https://arxiv.org/help/api)


## Install

```bash
python3 -m venv venv
source venv/bin/activate

python3 -m pip install -U pip
python3 -m pip install -r requirements.txt 
python3 -m spacy download en_core_web_sm
cp base.ttl arxiv.ttl
```


## Usage

```bash
python3 arxiv.py cmd-query --min-date=2021-01-01 "knowledge graph"
```

```bash
python3 arxiv.py cmd-analyze
python3 arxiv.py cmd-visualize
```

```bash
python3 arxiv.py cmd-extract
```


## License and Copyright

Source code for **arxiv-trends** plus its logo, documentation, and
examples have an [MIT license](https://spdx.org/licenses/MIT.html)
which is succinct and simplifies use in commercial applications.


## Kudos

Kudos to arXiv for use of its open access interoperability;
to Jürgen Müller @ BASF for the original idea;
plus general support from [Derwen, Inc.](https://derwen.ai/);
the [Knowledge Graph Conference](https://www.knowledgegraph.tech/)
and [Connected Data World](https://connected-data.world/).
