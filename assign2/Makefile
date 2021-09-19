compile: 
	black bin_class.py
	flake8 --ignore=E,W bin_class.py
	python3 bin_class.py

pdf: 
	a2ps bin_class.py -o bin_class.ps --pro=color
	a2ps Makefile -o Makefile.ps --pro=color
	ps2pdf bin_class.ps
	ps2pdf Makefile.ps
	gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=classif_AR.pdf README.pdf bin_class.pdf output.pdf Makefile.pdf

clean: 
	rm *.ps* bin_class.pdf Makefile.pdf 

	