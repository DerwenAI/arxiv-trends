#!/usr/bin/env bash -x

TTL_FILE=./arxiv.ttl
LOOKBACK=`date -v-2w +%Y-%m-%d`

PHRASES=( "graph algorithms" "graph neural networks" "knowledge graph" )

for QUERY in "${PHRASES[@]}"
do
    python arxiv.py cmd-query --min-date=$LOOKBACK --kg-path $TTL_FILE "$QUERY"
done

CSV_FILE=/tmp/arxiv.csv
PNG_FILE=/tmp/arxiv.png

python arxiv.py cmd-analyze --kg-path $TTL_FILE --csv-file $CSV_FILE
python arxiv.py cmd-visualize --csv-file $CSV_FILE --png-file $PNG_FILE

