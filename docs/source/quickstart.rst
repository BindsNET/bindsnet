Quickstart
==========

Check out some example use cases for BindsNET in the :code:`examples/` folder.

For example, to run a (close) replication of the network from `this paper <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full#>`_, first navigate to the :code:`data/` folder and issue

.. code-block:: bash

	./get_MNIST.sh
	
This will download and unzip all the data file contained in the `MNIST handwritten digit dataset <http://yann.lecun.com/exdb/mnist/>`_ to the :code:`data/` directory. Then, change directory to the :code:`examples/` folder and issue

.. code-block:: bash

	python eth.py [options]


Where :code:`[options]` should be replaced with any command-line arguments you'd like to use to modify the behavior of the program.
