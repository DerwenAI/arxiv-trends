#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import itertools
import json
import sys
import urllib
import urllib.request
import xml.etree.ElementTree as ET

from icecream import ic
import pytextrank
import spacy
import typer

APP = typer.Typer()
META = {}

DEFAULT_META_PATH = "arxiv.json"


def format_param (key, value):
    return key + "=" + value


def format_query (query, start, max_results):
    return "&".join([
        format_param("search_query", "all:" + query),
        format_param("start", str(start)),
        format_param("max_results", str(max_results)),
        format_param("sortBy", "submittedDate"),
        format_param("sortOrder", "descending"),
    ])


def parse_entry (entry, ns):
    global META

    href = entry.find("atom:link[@title='pdf']", ns).attrib["href"]

    if href not in META:
        META[href] = {
            "title": entry.find("atom:title", ns).text,
            "date": entry.find("atom:published", ns).text,
            "summary": entry.find("atom:summary", ns).text.replace("\n", " ").strip(),
            "authors": [
                author.text
                for author in entry.findall("atom:author/atom:name", ns)
            ],
        }

    return href, META[href]


def arxiv_api (
    query,
    min_date,
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    },
    api_base = "http://export.arxiv.org/api/query?",
    max_items = 1,
    page_items = 1,
    ):
    """
Access the arXiv API and parse the results.
    """
    start_index = 0
    max_index = max_items

    while (start_index < max_index):
        # prepare the API query
        url = api_base + format_query(query, start_index, page_items)
        handle = urllib.request.urlopen(url)
        xml = handle.read().decode("utf-8")
        #print(xml)

        # track the API results paging
        root = ET.fromstring(xml)
        total_results = int(root.findall("opensearch:totalResults", ns)[0].text)
        start_index = int(root.findall("opensearch:startIndex", ns)[0].text)
        page_items = int(root.findall("opensearch:itemsPerPage", ns)[0].text)

        print("---")
        ic(total_results)
        ic(start_index)
        ic(page_items)

        # parse each entry
        for entry in root.findall("atom:entry", ns):
            href, results = parse_entry(entry, ns)
            date = results["date"][:10]

            if date < min_date:
                return

            yield date, href

        # iterate to the next page of results
        max_index = min(max_items, total_results)
        start_index += page_items

    return


@APP.command()
def cmd_query (
    query,
    min_date = "2021-06-15",
    meta_path = DEFAULT_META_PATH,
    ):
    """
Query the arXiv API for the given search.
    """
    global META

    # load the metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        META = json.load(f)

    # search parameters
    max_items = 1000
    page_items = 100

    # get metadata for the matching articles
    hit_iter = arxiv_api(
        "+AND+".join(query.split(" ")),
        min_date,
        max_items=max_items,
        page_items=page_items,
    )

    for date, href in hit_iter:
        print(query, date, href)

    # persist the metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(META, f, ensure_ascii=False, indent=2)


@APP.command()
def cmd_extract (
    meta_path = DEFAULT_META_PATH,
    max_phrase = 10,
    ):
    """
Extract the entities fron each article.
    """
    global META

    # load the metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        META = json.load(f)

    # prepare the NLP pipeline
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")

    # run the pipeline for each article
    for href, meta in META.items():
        text = meta["title"] + ".  " + meta["summary"]
        doc = nlp(text)

        for phrase in itertools.islice(doc._.phrases, max_phrase):
            entity_label = " ".join(phrase.text.replace("\n", " ").strip().split()).lower()
            print(href, round(phrase.rank, 3), phrase.count, entity_label)


if __name__ == "__main__":
    APP()

    # simply reminders...
    query_list = [
        "knowledge graph",
        "graph database",
        "graph algorithm",
        "graph neural networks",
        "graph embedding",
        ]
