Installation
============
------------

To use Spots in Space, you need to have Python 3.10 or higher installed. Additionally, we recommend using conda to create a new environment in which SIS can be installed with no conflicts.

.. code-block:: console

    $ conda create -n sis python=3.10 -y
    $ conda activate sis

SIS can then by installed by cloning the repository and using pip:

.. code-block:: console

    $ git clone https://github.com/AllenInstitute/spots-in-space.git
    $ cd spots-in-space
    $ pip install ".[cellpose]"

Should you be interested in a distribution of spots-in-space without cellpose included, it can be installed as such:

.. code-block:: console

    $ git clone https://github.com/AllenInstitute/spots-in-space.git
    $ cd spots-in-space
    $ pip install .
