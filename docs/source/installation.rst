Installation
============


Pip install
-----------

Issue:

.. code-block:: bash

        pip install git+https://github.com/BindsNET/bindsnet.git


Installing from source
----------------------

On \*nix systems, issue one of the following in a shell:

.. code-block:: bash
	
	git clone https://github.com/Hananel-Hazan/bindsnet.git  # HTTPS
	git clone git@github.com:Hananel-Hazan/bindsnet.git  # SSH

Change directory into :code:`bindsnet` and issue one of the following:

.. code-block:: bash
	
	pip install . # Typical install
	pip install -e .  # Editable mode (package code can be edited without reinstall)

This will install :code:`bindsnet` and all its dependencies.


Running the tests
-----------------

If BindsNET is installed from source, install :code:`pytest` and issue the following from BindsNET's installation directory:

.. code-block:: bash

        python -m pytest test

