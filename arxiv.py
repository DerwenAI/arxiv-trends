#!/usr/bin/env python
# -*- coding: utf-8 -*-
# see license https://github.com/DerwenAI/arxiv-trends#license-and-copyright

"""
arxiv-trends:

Analyze trends in articles published on *arXiv* using NLP, knowledge graph, and time-series.
"""

from collections import defaultdict
import itertools
import pathlib
import re
import sys
import typing
import unicodedata
import urllib
import urllib.parse
import urllib.request
import xml.etree.ElementTree as et

from icecream import ic  # type: ignore  # pylint: disable=E0401
import dateutil.tz
import kglab  # type: ignore  # pylint: disable=E0401
import numpy as np
import matplotlib.pyplot as plt  # type: ignore  # pylint: disable=E0401
import pandas as pd  # type: ignore  # pylint: disable=E0401
import pytextrank  # type: ignore  # pylint: disable=E0401
import rdflib  # type: ignore  # pylint: disable=E0401
import spacy
import typer


APP = typer.Typer()


######################################################################
## utility functions

def strip_accents (
    text: str,
    ) -> str:
    """
Strip accents from the input string.

See <https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string>

    text:
the input string

    returns:
the processed string
    """
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore")  # type: ignore
    text = text.decode("utf-8")  # type: ignore

    return str(text)


def text_to_id (
    text: str,
    ) -> str:
    """
Convert input text to an identifier, suitable for a URI

    text:
raw text for the label of a node in the graph

    returns:
a string usable as a unique symbol in RDF
    """
    text = strip_accents(text.lower())
    text = re.sub("[ ]+", "_", text)
    text = re.sub("[^0-9a-zA-Z_-]", "", text)

    return text


######################################################################
## class definitions

class Trends:
    """
Analyze trends among papers published on arXiv.
    """
    NS = {
        "atom":       "http://www.w3.org/2005/Atom",
        "bibo":       "http://purl.org/ontology/bibo/",
        "cito":       "http://purl.org/spar/cito/",
        "dct":        "http://purl.org/dc/terms/",
        "derw":       "https://derwen.ai/ns/v1#",
        "foaf":       "http://xmlns.com/foaf/0.1/",
        "lcsh":       "http://id.loc.gov/authorities/subjects/",
        "madsrdf":    "http://www.loc.gov/mads/rdf/v1#",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        "owl":        "http://www.w3.org/2002/07/owl#",
        "prov":       "http://www.w3.org/ns/prov#",
        "rdf":        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs":       "http://www.w3.org/2000/01/rdf-schema#",
        "schema":     "http://schema.org/",
        "sh":         "http://www.w3.org/ns/shacl#",
        "skos":       "http://www.w3.org/2004/02/skos/core#",
        "wd":         "http://www.wikidata.org/entity/",
        "xsd":        "http://www.w3.org/2001/XMLSchema#",
    }

    API_BASE = "http://export.arxiv.org/api/query?"


    def __init__ (
        self,
        *,
        kg_path: str = "arxiv.ttl",
        ):
        """
Constructor.

    kg_path:
optional - path to the TTL file for kglab to load/save
        """
        self.kg = kglab.KnowledgeGraph(namespaces=self.NS)
        self.kg_path = pathlib.Path(kg_path)
        self.topics: typing.Dict[str, rdflib.Node] = {}
        self.load_kg()


    def load_kg (
        self,
        ) -> None:
        """
Load the previous definitions from a serialized KG and initialize the
topics lookup.
        """
        self.topics = {}
        self.kg.load_rdf(self.kg_path)

        sparql = """
SELECT ?entry ?label
WHERE {
 ?entry a derw:Topic .
 ?entry skos:prefLabel ?label
}
        """

        for node, topic in self.kg.query(sparql):
            self.topics[topic.toPython()] = node


    def save_kg (
        self,
        ) -> None:
        """
Serialize the updated KG to a file.
        """
        self.kg.save_rdf(self.kg_path)


    def lookup_author (
        self,
        name: str,
        ) -> rdflib.URIRef:
        """
Lookup an author by name, creating a node in the KG if it doesn't
already exist.

    name:
raw text for the name of the author

    returns:
author node
        """
        uri = self.kg.get_ns("derw") + "author_" + text_to_id(name)
        node = rdflib.URIRef(uri)
        p = self.kg.get_ns("rdf").type
        o = self.kg.get_ns("derw").Author

        if (node, p, o) not in self.kg.rdf_graph():
            self.kg.add(node, p, o)
            self.kg.add(node, self.kg.get_ns("foaf").name, rdflib.Literal(name, lang=self.kg.language))

        return node


    def parse_entry (
        self,
        entry: et.Element,
        ) -> typing.Tuple[rdflib.URIRef, str]:
        """
Parse the XML from one entry in an Atom feed, and add it to the KG.

    entry:
XML object for one Atom feed entry

    returns:
node and date of the parsed results
        """
        href = entry.find("atom:link[@title='pdf']", self.NS).attrib["href"]  # type: ignore
        date = entry.find("atom:published", self.NS).text[:10]  # type: ignore
        title = entry.find("atom:title", self.NS).text  # type: ignore
        abstract = entry.find("atom:summary", self.NS).text.replace("\n", " ").strip()  # type: ignore

        # lookup the specified article in the KG, and create a node if
        # it doesn't already exist
        node = rdflib.URIRef(href)
        p = self.kg.get_ns("rdf").type
        o = self.kg.get_ns("bibo").Article

        if (node, p, o) not in self.kg.rdf_graph():
            self.kg.add(node, p, o)
            self.kg.add(node, self.kg.get_ns("dct").Date, self.kg.encode_date(date, [dateutil.tz.gettz("UTC")]))
            self.kg.add(node, self.kg.get_ns("dct").title, rdflib.Literal(title, lang=self.kg.language, normalize=False))
            self.kg.add(node, self.kg.get_ns("dct").abstract, rdflib.Literal(abstract, lang=self.kg.language))

            # add author list
            for author in entry.findall("atom:author/atom:name", self.NS):
                self.kg.add(node, self.kg.get_ns("bibo").authorList, self.lookup_author(author.text))  # type: ignore

        return node, date


    @classmethod
    def format_query (
        cls,
        query: str,
        start: int,
        max_results: int,
        ) -> str:
        """
Format a URL to search arXiv via its API, based on the given search 
criteria.

    query:
query string

    start:
start index within the results

    max_results:
maximum results to return per API call

    returns:
query URL
        """
        params: dict = {
            "search_query": "all:" + query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        return urllib.parse.urlencode(params, safe=":")


    def arxiv_api (
        self,
        query: str,
        min_date: str,
        *,
        max_items: int = 1,
        page_items: int = 1,
        ) -> typing.Iterable:
        """
Access the arXiv API based on the given search criteria, parse the XML
results (Atom feed), then update the KG to represent any new entries.

    query:
query string

    min_date:
minimum date to include in the results

    max_items:
optional - maximum items requested per API call

    page_items:
optional - maximum items requested per page

    yields:
`(date, href)` tuple for each search hit within the criteria
        """
        start_index = 0
        max_index = max_items

        while (start_index < max_index):
            # prepare the API query
            url = self.API_BASE + self.format_query(query, start_index, page_items)
            handle = urllib.request.urlopen(url)

            xml = handle.read().decode("utf-8")
            #print(xml)

            # track the API results paging
            root = et.fromstring(xml)
            total_results = int(root.findall("opensearch:totalResults", self.NS)[0].text)  # type: ignore
            start_index = int(root.findall("opensearch:startIndex", self.NS)[0].text)  # type: ignore
            page_items = int(root.findall("opensearch:itemsPerPage", self.NS)[0].text)  # type: ignore

            print("---")
            ic(total_results)
            ic(start_index)
            ic(page_items)

            # parse each entry
            for entry in root.findall("atom:entry", self.NS):
                node, date = self.parse_entry(entry)
                yield date, node

                if date < min_date:
                    return

            # iterate to the next page of results
            max_index = min(max_items, total_results)
            start_index += page_items

        return


######################################################################
## commands

@APP.command()
def cmd_query (
    query: str,
    *,
    kg_path: str = "arxiv.ttl",
    min_date: str = "2021-06-15",
    max_items: int = 5000,
    ) -> None:
    """
Query the arXiv API for the given search, then update the KG.

    kg_path:
optional - path to the TTL file for kglab to load/save

    min_date:
optional - minimum date to include in the results

    max_items:
optional - maximum items requested per API call
    """
    trends = Trends(kg_path=kg_path)

    # search parameters
    page_items = 100

    # get metadata for the matching articles
    hit_iter = trends.arxiv_api(
        " AND ".join(query.split(" ")),
        min_date,
        max_items=max_items,
        page_items=page_items,
    )

    for date, node in hit_iter:
        trends.kg.add(node, trends.kg.get_ns("derw").fromQuery, trends.topics[query])
        # TODO: what if query new?
        print(query, date, node)

    # persist the metadata
    trends.save_kg()


@APP.command()
def cmd_extract (
    kg_path: str = "arxiv.ttl",
    min_date: str = "2021-06-15",
    kpa_file: str = "phrases.csv",
    max_phrase: int = 10,
    ) -> None:
    """
Extract the entities from each article.

    kg_path:
optional - path to the TTL file for kglab to load/save

    min_date:
optional - minimum date to include in the results

    kpa_file:
optional - path to the CSV file for extracted phrases

    max_phrase:
optional - maximum number of extracted phrases to represent
    """
    trends = Trends(kg_path=kg_path)

    # prepare the NLP pipeline
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")

    sparql = f"""
SELECT ?article ?title ?abstract
WHERE {{
 ?article a bibo:Article .
 ?article dct:title ?title .
 ?article dct:abstract ?abstract .
 ?article dct:Date ?date .
 FILTER (?date > "{min_date}T00:00:00"^^xsd:dateTime)
}}
    """

    # run the pipeline for each article
    for node, title, abstract in trends.kg.query(sparql):
        text = title.toPython() + ".  " + abstract.toPython()
        doc = nlp(text)
        df_list = []

        for phrase in itertools.islice(doc._.phrases, max_phrase):
            entity_label = " ".join(phrase.text.replace("\n", " ").strip().split()).lower()
            entity_id = text_to_id(entity_label)

            df_list.append({
                "url": str(node),
                "rank": round(phrase.rank, 3),
                "count": phrase.count,
                "id": entity_id,
                "label": entity_label,
                })

        # serialize extracted phrases to a CSV file
        path = pathlib.Path(kpa_file)
        df = pd.DataFrame(df_list)
        df.to_csv(path, index=False)


@APP.command()
def cmd_analyze (
    kg_path: str = "arxiv.ttl",
    csv_file: str = "arxiv.csv",
    ) -> None:
    """
Analyze the article trends.

    kg_path:
optional - path to the TTL file for kglab to load/save

    csv_file:
optional - path to the CSV file for trend data
    """
    trends = Trends(kg_path=kg_path)

    sparql = """
SELECT ?article ?date ?label
WHERE {
 ?article a bibo:Article .
 ?article dct:Date ?date .
 ?article derw:fromQuery ?topic .
 ?topic skos:prefLabel ?label
}
    """
    # run the pipeline for each article
    df = pd.DataFrame([
        {
            "topic": topic.toPython(),
            "date": date.toPython(),
            "counts": 0,
        }
        for article, date, topic in trends.kg.query(sparql)
    ]).groupby(["topic", "date"]).count()

    # serialize trend data to a CSV file
    path = pathlib.Path(csv_file)
    df.to_csv(path)


@APP.command()
def cmd_visualize (
    csv_file: str = "arxiv.csv",
    png_file: str = "arxiv.png",
    ) -> None:
    """
Visualize the article trends.

    csv_file:
optional - path to the CSV file fo trend data

    png_file:
optional - path to the PNG file for the rendered diagram
    """
    df = pd.read_csv(csv_file, parse_dates=True, index_col="date")
    df_list = []

    for query in sorted(set(df["topic"])):
        df_sub = df[df["topic"] == query]
        df_samp = df_sub.resample("M").sum()
        df_list.append(df_samp.rename(columns={ "counts": query }))

    df_full = pd.concat(df_list, axis=1, join="inner").reindex(df_samp.index).fillna(0)

    # delete the min value as an outlier
    df_full = df_full.iloc[1: , :]

    # drop the last row – to let arXiv settle
    df_full.drop(df_full.tail(1).index, inplace=True)

    # set up the plot
    plot = df_full.plot(
        subplots=True,
        legend=False,
        figsize=(11, 7),
        xlabel="date submitted"
    )

    plot[0].set(ylabel="monthly counts")

    summary = list(df.groupby("topic").sum().to_dict()["counts"].items())
    y_max = round(max(df_full.max(axis=1)) + 10.0)

    for index, ax in enumerate(plot):
        query, count = summary[index]
        ax.set(ylim=(0, y_max), title=f"{query}, total = {count}")

    fig = plot[0].get_figure()
    fig.tight_layout()
    fig.savefig(png_file)


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
