#!/usr/bin/env bash
set -eux

LOOKBACK=`date -v-2w +%Y-%m-%d`

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -l|--lookback)
      LOOKBACK="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done

# query arXiv

PHRASES=( "graph algorithms" "graph neural networks" "knowledge graph" )
TTL_FILE=/opt/derwen/chwedl/trends/arxiv.ttl

for QUERY in "${PHRASES[@]}"
do
    python3 arxiv.py cmd-query --min-date=$LOOKBACK --kg-path $TTL_FILE "$QUERY"
done

# analyze trends

CSV_FILE=/opt/derwen/chwedl/trends/arxiv.csv
PNG_FILE=/opt/derwen/chwedl/trends/arxiv.png

python3 arxiv.py cmd-analyze --kg-path $TTL_FILE --csv-file $CSV_FILE
python3 arxiv.py cmd-visualize --csv-file $CSV_FILE --png-file $PNG_FILE

# extract phrases

TODAY=`date +%Y%m%d`
KPA_FILE=/opt/derwen/chwedl/trends/phrases.$TODAY.csv

python3 arxiv.py cmd-extract --min-date=$LOOKBACK --kg-path $TTL_FILE --kpa-file $KPA_FILE
