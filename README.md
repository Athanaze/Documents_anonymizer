Linux and Windows desktop app to load documents and anonymize them (works with names & surnames common in switzerland/europe).

A mix of llm and embeddings+cosine similarity is used to recognize the names of people, places and businesses.
Currently two passes with block size 512 and 400 (characters) are made, to get the best results out of phi-3 mini.
Quality is the priority, not speed of execution.

http://arsent.ch/s/Phi-3-mini-4k-instruct-q4.gguf

tool.py : used to anonymize/create fake id cards, driver license, etc...
