compile: 
	black cifar_class.py resnet.py darse.py
	flake8 --ignore=E,W cifar_class.py resnet.py darse.py
	python3 cifar_class.py

pdf: 
	a2ps cifar_class.py -o cifar_class.ps --pro=color
	a2ps resnet.py -o resnet.ps --pro=color
	a2ps darse.py -o darse.ps --pro=color
	a2ps Makefile -o Makefile.ps --pro=color
	ps2pdf cifar_class.ps
	ps2pdf resnet.ps
	ps2pdf darse.ps
	ps2pdf Makefile.ps
	gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=AR_CIFAR.pdf Assignment4_CIFAR.pdf cifar_class.pdf resnet.pdf darse.pdf Makefile.pdf

clean: 
	rm *.ps*

	