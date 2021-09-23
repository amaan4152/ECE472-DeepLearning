compile: 
	black mnist_classify.py
	flake8 --ignore=E,W mnist_classify.py
	python3 mnist_classify.py

pdf: 
	a2ps mnist_classify.py -o mnist_classify.ps --pro=color
	a2ps Makefile -o Makefile.ps --pro=color
	ps2pdf mnist_classify.ps
	ps2pdf Makefile.ps
	gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=classif_AR.pdf output.pdf mnist_classify.pdf Makefile.pdf

clean: 
	rm *.ps* bin_class.pdf Makefile.pdf 

	