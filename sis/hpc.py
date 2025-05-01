from __future__ import annotations
import subprocess, os, sys, re, time, pickle, atexit
import numpy as np


def run_slurm_func(run_spec, conda_env=None, **kwds):
    """Run a function on SLURM.

    Parameters
    ----------
    run_spec : tuple or dict
        Either a tuple containing (func, args, kwargs) to be called in SLURM jobs, or a dict
        where the keys are array job IDs and the values are (func, args, kwargs) tuples.
    conda_env : str
        Conda envronment from which to run *func*
    **kwds
        All other keyword arguments are passed to run_slurm().
    """
    # Pickle up the commands so it can be easily read in and run by the hpc_worker
    pkl_file = os.path.join(kwds['job_path'], kwds['job_name'] + '_func.pkl')
    pickle.dump(run_spec, open(pkl_file, 'wb'))

    assert 'command' not in kwds
    kwds['command'] = f'conda run -p {conda_env} python -m sis.hpc {pkl_file} $SLURM_ARRAY_TASK_ID'

    return run_slurm(**kwds)


def hpc_worker():
    """Invoked when running `python -m sis.hpc`
    """
    pkl_file = sys.argv[1]
    run_specs = pickle.load(open(pkl_file, 'rb'))
    if isinstance(run_specs, dict):
        array_id = int(sys.argv[2])
        print(f"Started HPC worker {array_id} from {sys.executable}")
        print(f"Call spec file: {pkl_file}")
        func, args, kwargs = run_specs[array_id]
    else:
        print(f"Started {sys.executable}")
        print(f"Call spec file: {pkl_file}")
        func, args, kwargs = run_specs
    print(f"Invoking callback: {func}")
    print("-----------------------------")
    ret = func(*args, **kwargs)
    print("-----------------------------")
    print("HPC worker complete. Callback returned:")
    print(ret)


def run_slurm(*,
    hpc_host:str,
    job_path:str,
    partition:str,
    job_name:str,
    nodes:int,
    ntasks:int=1,
    array:str|None=None,
    mincpus:int|None=1,
    gpus_per_node:int|str|None=None,
    mem:str,
    time:str,
    command:str,
    mail_user:str,
    mail_type:str='END,FAIL',
    output:str|None=None,
    error:str|None=None,
    job_file:str|None=None,
    ):
    """Call sbatch (locally or over ssh) and return a SlurmJob instance.

    Parameters
    ----------
    hpc_host : str
        Host to call sbatch from. If not 'localhost', then commands are executed
        on *hpc_host* via ssh.
    job_path : str
        Location to write slurm script file and output/error logs.
        Must be accessible from *hpc_host*.
    partition : str
        SLURM partition to allocate job
    job_name : str
        SLURM job name. This is also used to set default file names for the sbatch script,
        output log, and error log.
    nodes : int
        Number of nodes to request
    ntasks : int
        Number of tasks to launch per job.
        (Use multiple tasks when all must be scheduled to start simultaneously)
    array : str or None
        Use array="A-B" where A and B are the first and last index to start multiple,
        independent jobs. The output/error file names may include "%a" to reference the
        array index, and the command may use $SLURM_ARRAY_TASK_ID.
    mincpus : int or None
        Minimum CPUs per node
    gpus_per_node : int or str or None
        GPUs per node
    mem : str
        Maximum memory to allocate per job (e.g. '100G')
    time : str
        Maximum job time (e.g. "4:00:00")
    command : str
        Command to run per job (e.g. "srun mycommand $SLURM_ARRAY_TASK_ID")
    mail_user : str or None
        Optional email address to notify upon completion
    mail_type : str
        Events on which to send email (default='END,FAIL')
    output : str or None
        Optional location of file to store command output (default is in job_path
        with automatically chosen name)
    error : str or None
        Optional location of file to store error output (default is in job_path
        with automatically chosen name)
    job_file : str or None
        Optional location to store sbatch script (default is in job_path
        with automatically chosen name)
        
    Returns
    -------
    SlurmJob
        If no array is specified, returns a SlurmJob instance
    SlurmJobArray
        if an array is specified, returns a SlurmJobArray instance
        
    """
    if array is None:
        # Slurm replaces "%j" with the job allocation number
        filename_prefix = os.path.join(job_path, job_name + "_%j")
    else:
        # Slurm replaces "%A" with the job ID and "%a" with the array index
        filename_prefix = os.path.join(job_path, job_name + "_%A_%a")

    if output is None:
        output = filename_prefix + "_output.log"
    if error is None:
        error = filename_prefix + "_error.log"
    if job_file is None:
        job_file = os.path.join(job_path, f"{job_name}.sbatch.sh")
    if mail_user is None:
        mail_type = None

    # Take all inputs and put them into slurm script
    arglist = [
        'partition', 'job_name', 'nodes', 'ntasks', 'array', 'mincpus', 'mem',
        'gpus_per_node', 'time', 'mail_user', 'mail_type', 'output', 'error',
    ]
    args = {}
    script = "#!/bin/bash"
    for k in arglist:
        v = locals()[k]
        args[k] = v
        k = k.replace('_', '-')
        if v is not None:
            script += f"\n#SBATCH --{k}={v}"
    script += "\n" + command

    if not os.path.exists(os.path.dirname(job_file)):
        os.makedirs(os.path.dirname(job_file))

    with open(job_file, 'w', newline='\n') as fh:
        fh.write(script)

    # Run the slurm script
    sbatch_output = run(hpc_host, ['sbatch', job_file])

    # Store the results
    if args['array'] is None:
        return SlurmJob(args=args, sbatch_output=sbatch_output, job_file=job_file, host=hpc_host)
    else:
        return SlurmJobArray(args=args, sbatch_output=sbatch_output, job_file=job_file, host=hpc_host)


class SlurmJob:
    """Class representing a single SLURM job
    
    Attributes
    ----------
    args : dict
        Arguments used to create this job
    sbatch_output : str
        Output from sbatch command
    array_id : int or None
        Array index for this job (None if not an array job)
    job_array : SlurmJobArray or None
        SlurmJobArray instance if this is part of an array job, otherwise None
    job_file : str
        Location of the sbatch script file
    host : str
        Host on which the job was submitted
    start_time : time.struct_time
        Time at which the job was started
    """
    def __init__(self, args, sbatch_output, job_file, host, array_id=None, job_array=None):
        """
        Parameters
        ----------
        args : dict
            Arguments used to create this job
        sbatch_output : str
            Output from sbatch command
        job_file : str
            Location of the sbatch script file
        host : str
            Host on which the job was submitted
        array_id : int or None
            Array index for this job (None if not an array job)
        job_array : SlurmJobArray or None
            SlurmJobArray instance if this is part of an array job, otherwise None
        """
        self.args = args
        self.sbatch_output = sbatch_output
        self.array_id = array_id
        self.job_array = job_array
        self.job_file = job_file
        self.host = host
        self.start_time = time.localtime()

        # Pull out the job id
        m = re.match(r'Submitted batch job (\d+)', sbatch_output)
        self.base_job_id = m.groups()[0]

        if array_id is None:
            self.job_id = self.base_job_id
        else:
            self.job_id = self.base_job_id + '_' + str(self.array_id)

    def state(self):
        """Return the state code for this job in the form of the JobState class. 
        Which contains the state, state code, description, and whether the job is done
        
        Returns
        -------
        JobState
            An instance of JobState representing the state of this job
        """
        table = sacct(host=self.host, job_id=self.base_job_id) # Custom sacct returns a table mapping job IDs to their states
        return JobState(state=table.get(self.job_id, {'State': 'NO_INFO'})['State'])

    def is_done(self):
        """Returns True if a job is done
        
        Returns
        -------
        bool
            True if the job is done (either completed, failed, or cancelled)
        """
        return (self.state().is_done and os.path.exists(self.output_file)) or self.state().state == 'CANCELLED' # Need to consider job cancelled before output file made

    @property
    def output_file(self):
        """Returns the output file name for this job
        
        Returns
        -------
        str
            The output file name for this job, with any %j, %A, or %a replaced with the job ID, base job ID, or array ID
        """
        return self._filename_replacement(self.args['output'])

    @property
    def error_file(self):
        """Returns the error file name for this job
        
        Returns
        -------
        str
            The error file name for this job, with any %j, %A, or %a replaced with the job ID, base job ID, or array ID
        """
        return self._filename_replacement(self.args['error'])

    @property
    def output(self):
        """Returns the contents of the output file for this job
        
        Returns
        -------
        str
        """
        return open(self.output_file, 'r').read()

    @property
    def error(self):
        """Returns the contents of the error file for this job
        
        Returns
        -------
        str
        """
        return open(self.error_file, 'r').read()

    def _filename_replacement(self, fn):
        """Takes in a filename string and replaces any %j, %A, or %a with the job ID, base job ID, or array ID
        (as SLURM itself does when it creates the output/error files)
        
        Parameters
        ---------- 
        fn : str
            Filename string
        
        Returns
        -------
        fn : str
            The filename string with any %j, %A, or %a replaced with the job ID, base job ID, or array ID
        """
        fn = fn.replace(r'%j', self.job_id)
        fn = fn.replace(r'%A', self.base_job_id)
        fn = fn.replace(r'%a', str(self.array_id))
        return fn

    def cancel(self):
        """Cancels the job python subprocess to directly invoke SLURM's scancel
        """
        run(self.host, ['scancel', str(self.job_id)])


class SlurmJobArray(SlurmJob):
    """Class representing a SLURM job array with a list of SlurmJob instances
    
    Attributes
    ----------
    args : dict
        Arguments used to create this job
    sbatch_output : str
        Output from sbatch command
    job_file : str
        Location of the sbatch script file
    host : str
        Host on which the job was submitted
    job_id : str
        Job ID of the job array
    jobs : list[SlurmJob]
        List of SlurmJob instances representing each job in the array
    """
    def __init__(self, args, sbatch_output, job_file, host):
        """
        Parameters
        ----------
        args : dict
            Arguments used to create this job
        sbatch_output : str
            Output from sbatch command
        job_file : str
            Location of the sbatch script file
        host : str
            Host on which the job was submitted
        """
        self.args = args
        self.sbatch_output = sbatch_output
        self.job_file = job_file
        self.host = host

        start, _, stop = args['array'].partition('-') # pull out JobArray indices
        
        # Pull out the job id
        m = re.match(r'Submitted batch job (\d+)', sbatch_output)
        self.job_id = m.groups()[0]

        # Create a SlurmJob for each job in the array
        self.jobs = []
        for i in range(int(start), int(stop)+1):
            job = SlurmJob(args, sbatch_output, job_file, host, array_id=i, job_array=self)
            self.jobs.append(job)

    def __iter__(self):
        for job in self.jobs:
            yield job

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, item):
        return self.jobs[item]

    def state(self):
        """Return a dictionary mapping job IDs to JobStates for every job in the array
        
        Returns
        -------
        dict
            job state for every job in the array
        """
        return {job.job_id:job.state() for job in self.jobs}

    def is_done(self):
        """Returns True if all jobs in the array are done (either completed, failed, or cancelled)
        
        Returns
        -------
        bool
        """
        return all([job.is_done() for job in self.jobs])

    def cancel(self):
        """Cancels all jobs in the job array
        """
        run(self.host, ['scancel', str(self.job_id)])

    def finished_jobs(self):
        """Returns a list of all finished jobs in the array (type=list[SlurmJob])
        
        Returns
        -------
        list[SlurmJob]
            List of SlurmJob instances that are done (either completed, failed, or cancelled)
        """
        return [job for job in self.jobs if job.is_done()]

    def unfinished_jobs(self):
        """Returns a list of all unfinished jobs in the array (type=list[SlurmJob])
        
        Returns
        -------
        list[SlurmJob]
            List of SlurmJob instances that are not done
        """
        return [job for job in self.jobs if not job.is_done()]

    def wait_iter(self):
        """Iterator that returns jobs as they finish and does not finish until all jobs are done
        """
        jobs = self.jobs[:]
        while len(jobs) > 0:
            for j in jobs[:]:
                if j.is_done():
                    jobs.remove(j)
                    yield(j)
            time.sleep(3)

    def state_counts(self):
        """Returns a dictionary mapping state codes to counts of jobs in that state
        
        Returns
        -------
        dict
            e.g. {'R': 5, 'PD': 3, ...}
        """
        return dict(zip(*np.unique([s.state_code for s in self.state().values()], return_counts=True)))


class JobState:
    """Class representing the state of a SLURM job. 
    One of state_code or state must be provided and the other is derived.
    
    Attributes
    ----------
    state : str
        SLURM state (e.g. 'RUNNING')
    state_code : str
        SLURM state code (e.g. 'R' for running)
    description : str
        Description of the state (e.g. 'Job currently has an allocation.')
    is_done : bool
        True if the job is done (either completed, failed, or cancelled)
    state_codes : dict
        Dictionary containing all information about jobs. Maps state codes to full names and descriptions
    state_descriptions : dict
        Dictionary mapping job state names to their descriptions
    state_to_code : dict
        Dictionary mapping job state names to their state codes
    """
    state_codes = {
        'BF':  ('BOOT_FAIL',       'Job terminated due to launch failure, typically due to a hardware failure (e.g. unable to boot the node or block and the job can not be requeued).'),
        'CA':  ('CANCELLED',       'Job was explicitly cancelled by the user or system administrator.  The job may or may not have been initiated.'),
        'CD':  ('COMPLETED',       'Job has terminated all processes on all nodes with an exit code of zero.'),
        'CF':  ('CONFIGURING',     'Job has been allocated resources, but are waiting for them to become ready for use (e.g. booting).'),
        'CG':  ('COMPLETING',      'Job is in the process of completing. Some processes on some nodes may still be active.'),
        'DL':  ('DEADLINE',        'Job terminated on deadline.'),
        'F':   ('FAILED',          'Job terminated with non-zero exit code or other failure condition.'),
        'NF':  ('NODE_FAIL',       'Job terminated due to failure of one or more allocated nodes.'),
        'OOM': ('OUT_OF_MEMORY',   'Job experienced out of memory error.'),
        'PD':  ('PENDING',         'Job is awaiting resource allocation.'),
        'PR':  ('PREEMPTED',       'Job terminated due to preemption.'),
        'R':   ('RUNNING',         'Job currently has an allocation.'),
        'RD':  ('RESV_DEL_HOLD',   'Job is being held after requested reservation was deleted.'),
        'RF':  ('REQUEUE_FED',     'Job is being requeued by a federation.'),
        'RH':  ('REQUEUE_HOLD',    'Held job is being requeued.'),
        'RQ':  ('REQUEUED',        'Completing job is being requeued.'),
        'RS':  ('RESIZING',        'Job is about to change size.'),
        'RV':  ('REVOKED',         'Sibling was removed from cluster due to other cluster starting the job.'),
        'SI':  ('SIGNALING',       'Job is being signaled.'),
        'SE':  ('SPECIAL_EXIT',    'The job was requeued in a special state. This state can be set by users, typically in EpilogSlurmctld, if the job has terminated with a particular exit value.'),
        'SO':  ('STAGE_OUT',       'Job is staging out files.'),
        'ST':  ('STOPPED',         'Job has an allocation, but execution has been stopped with SIGSTOP signal.  CPUS have been retained by this job.'),
        'S':   ('SUSPENDED',       'Job has an allocation, but execution has been suspended and CPUs have been released for other jobs.'),
        'TO':  ('TIMEOUT',         'Job terminated upon reaching its time limit.'),
        'NO':  ('NO_INFO',         'Squeue returned no information about the requested job ID'),
    }

    state_descriptions = {v[0]: v[1] for v in state_codes.values()}
    state_to_code = {v[0]: k for k, v in state_codes.items()}
    
    def __init__(self, state_code=None, state=None):
        """
        Parameters
        ----------
        state_code : str or None
            SLURM state code (e.g. 'R' for running)
        state : str or None
            SLURM state (e.g. 'RUNNING')
        """
        # Check to make sure that state and state code don't conflict    
        if state and state_code:
            if state_code != JobState.state_to_code[state]:
                raise ValueError('state_code and state contain conflicting values')
                
        if state:
            self.state = state
            self.state_code = JobState.state_to_code[state]
            self.description = JobState.state_descriptions[state]
        elif state_code:
            self.state_code = state_code
            self.state, self.description = JobState.state_codes[state_code]
        else:
            raise ValueError('Must input one of state_code or state')
        
        self.is_done = self.state_code in ('BF', 'CA', 'CD', 'DL', 'F', 'NF', 'OOM', 'PR', 'TO', 'NO') # States that indicate the job is done

        
    def __repr__(self):
        return f"<JobState {self.state}>"
    

_last_squeue = {}
def squeue(host, job_id, cache_duration=10):
    """Uses python subprocess and SLURM's squeue to return a information about a job
    
    Parameters
    ----------
    host : str
        Host to call squeue from. If not 'localhost', then commands are executed
        on *host* via ssh.
    job_id : str
        Job ID to query
    cache_duration : int
        Duration in seconds to cache the results of the squeue call (default=10)
    
    Returns
    -------
    table : dict
        Dictionary containing information about a job
    """
    global _last_squeue
    # 10-second cache to rate limit squeue calls
    last_time, last_state = _last_squeue.get(job_id, (None, None))
    now = time.time()
    if last_time is None or now - last_time > cache_duration:
        stat = run(host, ['squeue', f'--job={job_id}']) # use subprocess to call squeue
        lines = stat.split('\n')
        cols = re.split(r'\s+', lines[0].strip()) # Get column names from first line
        table = {}
        for line in lines[1:]:
            line = line.strip()
            if line == '':
                continue
            tokens = re.split(r'\s+', line)
            tokens = tokens[:7] + [' '.join(tokens[7:])]  # NODELIST(REASON) column may contain spaces
            # populate the table with job information
            fields = {cols[i]:field for i,field in enumerate(tokens)}
            table[fields.pop('JOBID')] = fields
        _last_squeue[job_id] = (now, table)
    else:
        table = last_state
    return table


_last_sacct = {}
def sacct(host, job_id, cache_duration=10):
    """Uses python subprocess and SLURM's sacct to specifically query the state of a job.
    This is used to fill information for JobState
    
    Parameters
    ----------
    host : str
        Host to call sacct from. If not 'localhost', then commands are executed
        on *host* via ssh.
    job_id : str
        Job ID to query
    cache_duration : int
        Duration in seconds to cache the results of the squeue call (default=10)
    
    Returns
    -------
    table : dict
        Dictionary containing information about the state of a job
    """
    global _last_sacct
    # 10-second cache to rate limit sacct calls
    last_time, last_state = _last_sacct.get(job_id, (None, None))
    now = time.time()
    if last_time is None or now - last_time > cache_duration:
        # use subprocess to call sacct w/ only ID and state. -X ensures only 1 line is printed for each job
        stat = run(host, ['sacct', f'--job={job_id}', '--format=JobID%20,State%20', '-X'])
        lines = stat.split('\n')
        table = {}
        for line in lines[2:]: # First 2 lines are headers
            line = line.strip()
            if line == '':
                continue
            tokens = re.split(r'\s+', line)
            table[tokens[0]] = {'State': tokens[1]} # index rather than iterate because cancelled is formatted: '15420407_1   CANCELLED by 20416'
        _last_sacct[job_id] = (now, table)
    else:
        table = last_state
    return table


def double_mem(mem: str):
	"""This function takes a memory amount stored as a string, doubles it, and then returns the new amount in the same format
    
    SLURM recognizes 2000M and 2GB as the same amount of memory, so this function will return the doubled amount in the same format as the input
    
	Parameters
    ----------
    mem : str
        Memory amount in a string representation e.g. ('700M' or '10gb')
	
    Returns
    -------
    str
        input memory doubled in same format e.g. ('1400M' or '20gb')
	"""
	i = len(mem)
	while i > 0:
        # To isolate the number part of the memory string we keep trying to cast to int until its possible 
        # (i.e. until the string no longer contains a letter)
		try: 
			return str(int(mem[:i]) * 2) + mem[i:]
		except ValueError:
			i -= 1
			continue
	raise ValueError("Memory string was not in expected format")


def memory_to_bytes(mem: str):
    """This function takes a memory amount stored as a string (in slurm supported formats) and converts it to integer bytes
    Used to compare memory string amounts
    
	Parameters
    ----------
    mem : str
        Memory amount in a slurm-supported string representation e.g. ('700M' or '10gb')
	
    Returns
    -------
    int
        input memory as bytes of type integer e.g. (734003200 or 10737418240)
	"""
    # Regular expression pattern
    pattern = r'^(\d+)([gGmMkKtT]?[bB]?)$'
    match = re.match(pattern, str(mem))
    if match:
        amount, unit = match.groups()
        amount = int(amount)
        # Convert to bytes based on the unit
        if unit.lower() in ['g', 'gb']:
            return amount * 1024**3
        elif unit.lower() in ['m', 'mb', '']: # no unit means MB
            return amount * 1024**2
        elif unit.lower() in ['k', 'kb']:
            return amount * 1024
        elif unit.lower() in ['t', 'tb']:
            return amount * 1024**4
            
    raise ValueError(f'Invalid memory format: {mem}')


def slurm_time_to_seconds(time: str):
    """This function takes a time amount stored as a string (in slurm supported formats) and converts it to integer seconds
    Used to compare time string amounts
    
    Parameters
    ----------
    time : str
        time amount in a slurm-supported string representation e.g. ('2-13:01:45' or '10' or '10:00')
    
    Returns
    -------
    int
        input time as seconds of type integer e.g. (219705 or 600 or 600)
    """
    # All possible time formats that SLURM supports
    patterns = [
        (r'^(\d+)$', 'minutes'),
        (r'^(\d+):(\d+)$', 'minutes:seconds'),
        (r'^(\d+):(\d+):(\d+)$', 'hours:minutes:seconds'),
        (r'^(\d+)-(\d+)$', 'days-hours'),
        (r'^(\d+)-(\d+):(\d+)$', 'days-hours:minutes'),
        (r'^(\d+)-(\d+):(\d+):(\d+)$', 'days-hours:minutes:seconds'),
    ]
    # Try each pattern until one matches
    for pattern, format in patterns:
        match = re.match(pattern, time)
        if match:
            groups = match.groups()
            # Convert to integers
            groups = [int(g) for g in groups]
            # Convert to seconds based on the format
            if format == 'minutes':
                return groups[0] * 60
            elif format == 'minutes:seconds':
                return groups[0] * 60 + groups[1]
            elif format == 'hours:minutes:seconds':
                return groups[0] * 60 * 60 + groups[1] * 60 + groups[2]
            elif format == 'days-hours':
                return groups[0] * 24 * 60 * 60 + groups[1] * 60 * 60
            elif format == 'days-hours:minutes':
                return groups[0] * 24 * 60 * 60 + groups[1] * 60 * 60 + groups[2] * 60
            elif format == 'days-hours:minutes:seconds':
                return groups[0] * 24 * 60 * 60 + groups[1] * 60 * 60 + groups[2] * 60 + groups[3]
    # If no pattern matched, raise an error
    raise ValueError(f'Invalid time format: {time}')


def seconds_to_time(seconds: int):
    """This function takes seconds and converts them to a standard time format of 'days-hours:minutes:seconds' as a string
    Used in conjunction with slurm_time_to_seconds to double time strings
    
    Parameters
    ----------
    second s: int
        time amount in a seconds (1800 or 217803)
    
    Returns
    -------
    str
        input time as standardized time string e.g. ("0-00:30:00" or "2-12:30:03")
    """
    # Calculate time components in descending order of magnitude using floor division and modulo
    days = seconds // (24 * 60 * 60)
    seconds %= (24 * 60 * 60)
    hours = seconds // (60 * 60)
    seconds %= (60 * 60)
    minutes = seconds // 60
    seconds %= 60
    # Format and return the result
    return f"{days}-{hours:02}:{minutes:02}:{seconds:02}"


ssh_connections = {}

def run(host: str, cmd: list):
    """Run a command on any host. If the host is other than localhost, then the command is run over ssh.
    
    Parameters
    ----------
    host : str
        Host to call sbatch from. If not 'localhost', then commands are executed
        on *host* via ssh.
    cmd : list
        Command to run
        
    Returns
    -------
    str
        Output from the command
    """
    global ssh_connections
    assert isinstance(cmd, list)
    if host != 'localhost':
        if host not in ssh_connections:
            # if the host is not localhost & we don't have a connection open already, then we need to open an ssh connection
            conn = subprocess.Popen(['ssh', '-NM', host])
            ssh_connections[host] = conn
        cmd = ['ssh', host] + cmd
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as exc:
        print(exc.output)
        raise

def close_ssh_connections():
    """Closes all ssh connections which have been created using run()
    """
    global ssh_connections
    for proc in ssh_connections.values():
        proc.kill()
    ssh_connections = {}

atexit.register(close_ssh_connections)


if __name__ == '__main__':
    hpc_worker()
