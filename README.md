# Photon Emission Protocol — Python Demos

This code accompanies the paper:
Li, B., Economou, S.E. & Barnes, E. Photonic resource state generation from a minimal number of quantum emitters. npj Quantum Inf 8, 11 (2022). https://doi.org/10.1038/s41534-022-00522-6

This folder contains Jupyter notebooks that demonstrate the stabilizer-based
photon emission protocol for generating graph states.

| Notebook | Description |
|---|---|
| `demo_tree.ipynb` | Complete 3-ary trees (small and large) |
| `demo_rank1.ipynb` | Line graph and star graph |
| `demo_random.ipynb` | Random connected graph with optimal emission ordering |
| `demo_square.ipynb` | Square (cycle) graph, including a permuted ordering |

All notebooks import utilities from **`emission.py`**.

## Prerequisites

### Python packages

Install the following packages in your Python environment (e.g. `sci_env`):

```bash
pip install numpy matplotlib networkx pdf2image
```

| Package | Used for |
|---|---|
| **numpy** | Array and matrix operations (stabilizer tableaux) |
| **matplotlib** | Plotting graphs and height functions |
| **networkx** | Graph construction and layout algorithms |
| **pdf2image** | Displaying compiled circuit PDFs inline in notebooks |

> `multiprocessing`, `subprocess`, `itertools`, `os`, and `math` are part of
> the Python standard library and do not need to be installed separately.

### System dependencies

1. **LuaLaTeX** — used to compile quantum circuit diagrams (quantikz) to PDF.
   Install a TeX distribution that includes `lualatex`:
   - macOS: [MacTeX](https://www.tug.org/mactex/) or `brew install --cask mactex`
   - Linux: `sudo apt install texlive-full` (or a smaller subset that includes `lualatex` and `tikz`)
   - Windows: [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)

   The LaTeX package **quantikz** (`\usetikzlibrary{quantikz}`) must be
   available. It is included in most full TeX distributions.

2. **Poppler** — required by `pdf2image` to convert PDFs to images.
   - macOS: `brew install poppler`
   - Linux: `sudo apt install poppler-utils`
   - Windows: download from [poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases) and add the `bin/` folder to your `PATH`.

