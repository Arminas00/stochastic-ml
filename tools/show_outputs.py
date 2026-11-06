"""Stampa gli output testuali di un notebook eseguito (per controllo rapido)."""
import sys

import nbformat

nb = nbformat.read(sys.argv[1], as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    for o in c.get("outputs", []):
        if o.output_type == "stream":
            print(f"[{i}]", o.text.rstrip())
        elif o.output_type in ("execute_result", "display_data"):
            if "text/plain" in o.data and "image/png" not in o.data:
                print(f"[{i}]", o.data["text/plain"])
            elif "image/png" in o.data:
                print(f"[{i}] <figura>")
        elif o.output_type == "error":
            print(f"[{i}] ERRORE", o.ename, o.evalue)
