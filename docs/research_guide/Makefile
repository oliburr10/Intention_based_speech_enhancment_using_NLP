# Build the research guide PDF.  Requires TeX Live (pdflatex, biber, latexmk).
.PHONY: all clean pdf
all: pdf
pdf:
	latexmk
	cp build/main.pdf research_guide.pdf
clean:
	latexmk -C
	rm -rf build
