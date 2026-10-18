# General Guidelines and Coding Standards
1. All functions should have doc-strings and type hints. Doc strings should follow the Sphinx ReadTheDocs style (eg :param some_param: This is a parameter etc..)
2. When uncertain ask clarifying questions.
3. Before making large changes form a plan and ask user for approval. 
4. Code should be modular and testable. 
5. Refrain from excessive try except blocks. These should only be used in limited circumstances.

## Project Context
See README.md for full project overview, architecture, and data sources.

## Shared Machine — READ `docs/shared_studio_runtime.md` BEFORE RUNNING ANYTHING LONG
The Mac Studio is shared PRODUCTION: api.cairogenizah.ai runs here (Docker `genizah_search-*` containers, LM Studio on :1234, cloudflared). Never stop/restart/prune other processes or containers, never eject LM Studio models, never kill cloudflared. Before long jobs check `df -h`, `~/.lmstudio/bin/lms ps`, `docker ps`. This repo owns exactly one container (Kraken on :8002) and must never bind the web app's ports (8000/8001/3000/7681/7475/9200/5601/8010). Big outputs go to the NAS (`/Volumes/home/studio_offload/`).