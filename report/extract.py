from pathlib import Path
import tomllib
import re

ROOT = Path(__file__).parent.parent;

open_pattern = re.compile(r'<<(\w+)>>')
close_pattern = re.compile(r'<</(\w+)>>')

with open(ROOT / "report/config.toml", 'rb') as f:
    data = tomllib.load(f)
    snippets = data['snippets']
    sources: list[str] = snippets['sources']
    output: str = snippets['output']

    (ROOT / output).mkdir(parents=True, exist_ok=True)
    
    for source in sources:
        paths = ROOT.glob(source)
        for path in paths:
            current_snippets: dict[str, list[str]] = {}
            with open(path) as f:
                for line in f:
                    if open_match := open_pattern.search(line):
                        current_snippets[open_match.group(1)] = []
                    elif close_match := close_pattern.search(line):
                        name = close_match.group(1)
                        if name in current_snippets:
                            suffix = Path(source).suffix
                            out = ROOT / output / f"{name}{suffix}"
                            out.write_text("".join(current_snippets.pop(name)))
                    else:
                        for snippet in current_snippets.values():
                            snippet.append(line)
