QUICKSTART
===================

A wavefront simulator for simulating wavefronts in a circuit context.

requirements
------------

you will require a Python installation (3.9 was used here)

`download python <https://www.python.org/downloads/>`__

Jupyter notebook is required for interactive elements

   pip install jupyter

VScode is recommended, it has builtin notebook functionality

`download VScode <https://code.visualstudio.com/download>`__

`jupyter for
VScode <https://code.visualstudio.com/docs/datascience/jupyter-notebooks>`__

For the verification module to work it is also required that LTspice is
installed and confiugred

`download
LTspice <https://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html>`__

`refer to
docs <https://wavefront-simulator.readthedocs.io/en/latest/>`__

quick install
-------------

Next download the contents of `this repositry
<https://github.com/symB10S/wavefront_simulator>`__, and unzip:

| ROOT 
| ├─── docs 
| ├─── example.ipynb 
| ├─── test.py 
| ├─── requirements.txt
| └─── wavefronts <– this is where modules are stored

The ROOT level is where ``test.py`` and ``example.ipynb`` is.

Open a terminal at this folder.

install the requirements (`a venv is always
good <https://realpython.com/python-virtual-environments-a-primer/>`__):

   pip install -r requirements.txt

Now run ``test.py`` or ``examples.ipynb`` in VScode.

For more information refer to the
`docs <https://wavefront-simulator.readthedocs.io/en/latest/>`__
