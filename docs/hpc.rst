sis.hpc
=======

Submitting Slurm Jobs
---------------------

.. autofunction:: sis.hpc.run_slurm_func

.. autofunction:: sis.hpc.run_slurm

SlurmJob
--------

.. autoclass:: sis.hpc.SlurmJob
    :no-members:
    :show-inheritance:
    :class-doc-from: class
   
.. rubric:: Initialization
.. automethod:: sis.hpc.SlurmJob.__init__

.. rubric:: Properties
.. autoproperty:: sis.hpc.SlurmJob.output_file
.. autoproperty:: sis.hpc.SlurmJob.output
.. autoproperty:: sis.hpc.SlurmJob.error_file
.. autoproperty:: sis.hpc.SlurmJob.error

.. rubric:: Methods
.. automethod:: sis.hpc.SlurmJob.state
.. automethod:: sis.hpc.SlurmJob.is_done
.. automethod:: sis.hpc.SlurmJob.cancel


SlurmJobArray
-------------

.. autoclass:: sis.hpc.SlurmJobArray
    :no-members:
    :show-inheritance:
    :class-doc-from: class
   
.. rubric:: Initialization
.. automethod:: sis.hpc.SlurmJobArray.__init__

.. rubric:: Properties
.. autoproperty:: sis.hpc.SlurmJobArray.output_file
.. autoproperty:: sis.hpc.SlurmJobArray.output
.. autoproperty:: sis.hpc.SlurmJobArray.error_file
.. autoproperty:: sis.hpc.SlurmJobArray.error

.. rubric:: Methods
.. autosummary::
    :toctree: generated/
    
    sis.hpc.SlurmJobArray.__iter__
    sis.hpc.SlurmJobArray.__len__
    sis.hpc.SlurmJobArray.__getitem__
    sis.hpc.SlurmJobArray.state
    sis.hpc.SlurmJobArray.is_done
    sis.hpc.SlurmJobArray.cancel
    sis.hpc.SlurmJobArray.finished_jobs
    sis.hpc.SlurmJobArray.unfinished_jobs
    sis.hpc.SlurmJobArray.wait_iter
    sis.hpc.SlurmJobArray.state_counts

JobState
--------

.. autoclass:: sis.hpc.JobState
    :no-members:
    :show-inheritance:
    :class-doc-from: class
   
.. rubric:: Initialization
.. automethod:: sis.hpc.JobState.__init__