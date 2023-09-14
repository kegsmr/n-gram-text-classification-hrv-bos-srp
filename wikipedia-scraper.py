import wikiscraper as ws

ws.lang("hr")

print(ws.searchBySlug("random").getURL())