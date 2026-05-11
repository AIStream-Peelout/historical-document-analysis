# Neo4j Cairo Genizah Knowledge Graph

The goal of the Neo4j Cairo Genizah Knowledge Graph is to create a knowledge graph linking people, places, fragments, and scholars to each other.

## Basic Schema 


## Creating Knowledge Graph

There are two main files to create the knowledge graph: 
1. `biblio_import.py` Imports the bibliographic data that we've collected into the knowledge graph.
2. `knowlege_graph_poc.py` Imports the publicly available data from PGP into the knowledge graph.

The order which the files are run in doesn't matter. 



## Example Queries 

### Top 20 Places by Fragment Count
```
MATCH (f:Fragment)-[:MENTIONS_PLACE]->(pl:Place)
RETURN pl.name                          AS place,
       count(DISTINCT f)                AS fragment_count,
       count(DISTINCT f.type)           AS doc_type_count,
       collect(DISTINCT f.type)[..5]    AS doc_types
ORDER BY fragment_count DESC
LIMIT 20
```
###  Fragments by number of times cited
```aiignore
MATCH (b:BookArticle)-[r:REFERENCES]->(f:Fragment)
RETURN f.shelfmark, f.canonical_shelfmark,
       count(DISTINCT b) AS times_cited,
       sum(CASE WHEN r.has_discussion    THEN 1 ELSE 0 END) AS discussed_in,
       sum(CASE WHEN r.has_transcription THEN 1 ELSE 0 END) AS transcribed_in,
       sum(CASE WHEN r.has_translation   THEN 1 ELSE 0 END) AS translated_in
ORDER BY times_cited DESC
LIMIT 20

```


### Most referenced sources by number of fragments in ES
```aiignore
MATCH (b:BookArticle)-[:REFERENCES]->(f:Fragment)
WHERE b.has_local_copy = true
OPTIONAL MATCH (s:Scholar)-[:WROTE]->(b)
RETURN b.title                  AS title,
       collect(DISTINCT s.name) AS authors,
       b.year                   AS year,
       b.processing_level       AS level,
       b.structured_pages       AS pages_done,
       b.total_pages            AS total_pages,
       count(DISTINCT f)        AS fragments_referenced
ORDER BY fragments_referenced DESC
LIMIT 25
```
