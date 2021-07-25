#!/usr/bin/env bash -x

python arxiv.py cmd-query --min-date=2020-01-01 "graph algorithms"
python arxiv.py cmd-query --min-date=2020-01-01 "graph neural networks"
python arxiv.py cmd-query --min-date=2020-01-01 "knowledge graph"
