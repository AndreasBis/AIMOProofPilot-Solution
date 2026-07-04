# Reports

These files are LaTeX fragments. From the repository root, render and inspect them locally with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp -jobname=stage_1_render "\documentclass{article}\usepackage{hyperref}\begin{document}\input{reports/stage_1.tex}\end{document}"

pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp -jobname=stage_2_render "\documentclass{article}\usepackage{hyperref}\begin{document}\input{reports/stage_2.tex}\end{document}"
```

```bash
xdg-open /tmp/stage_1_render.pdf
xdg-open /tmp/stage_2_render.pdf
```
