from __future__ import annotations
import subprocess, os, sys, re, time, pickle, atexit
import numpy as np


def run_slurm_func(run_spec, conda_env=None, **kwds):
    """Run a function on SLURM.

    Parameters
    ----------
    run_spec : tuple | dict
        Either a tuple containing (func, args, kwargs) to be called in SLURM jobs, or a dict
        where the keys are array job IDs and the values are (func, args, kwargs) tuples.
    conda_env : str
        Conda envronment from which to run *func*
    **kwds
        All other keyword arguments are passed to run_slurm().
    """
    pkl_file = os.path.join(kwds['job_path'], kwds['job_name'] + '_func.pkl')
    pickle.dump(run_spec, open(pkl_file, 'wb'))

    assert 'command' not in kwds
    kwds['command'] = f'conda run -p {conda_env} python -m sawg.hpc {pkl_file} $SLURM_ARRAY_TASK_ID'

    return run_slurm(**kwds)


def hpc_worker():
    """Invoked when running `python -m sawg.hpc`
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
    array : str | None
        Use array="A-B" where A and B are the first and last index to start multiple,
        independent jobs. The output/error file names may include "%a" to reference the
        array index, and the command may use $SLURM_ARRAY_TASK_ID.
    mincpus : int | None
        Minimum CPUs per node
    gpus_per_node : int | str | None
        GPUs per node
    mem : str
        Maximum memory to allocate per job (e.g. '100G')
    time : str
        Maximum job time (e.g. "4:00:00")
    command : str
        Command to run per job (e.g. "srun mycommand $SLURM_ARRAY_TASK_ID")
    mail_user : str | None
        Optional email address to notify upon completion
    mail_type : str
        Events on which to send email (default='END,FAIL')
    output : str | None
        Optional location of file to store command output (default is in job_path
        with automatically chosen name)
    error : str | None
        Optional location of file to store error output (default is in job_path
        with automatically chosen name)
    job_file : str | None
        Optional location to store sbatch script (default is in job_path
        with automatically chosen name)
    """
    if array is None:
        filename_prefix = os.path.join(job_path, job_name + "_%j")
    else:
        filename_prefix = os.path.join(job_path, job_name + "_%A_%a")

    if output is None:
        output = filename_prefix + "_output.log"
    if error is None:
        error = filename_prefix + "_error.log"
    if job_file is None:
        job_file = os.path.join(job_path, f"{job_name}.sbatch.sh")
    if mail_user is None:
        mail_type = None

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

    sbatch_output = run(hpc_host, ['sbatch', job_file])

    if args['array'] is None:
        return SlurmJob(args=args, sbatch_output=sbatch_output, job_file=job_file, host=hpc_host)
    else:
        return SlurmJobArray(args=args, sbatch_output=sbatch_output, job_file=job_file, host=hpc_host)


class SlurmJob:
    def __init__(self, args, sbatch_output, job_file, host, array_id=None, job_array=None):
        self.args = args
        self.sbatch_output = sbatch_output
        self.array_id = array_id
        self.job_array = job_array
        self.job_file = job_file
        self.host = host
        self.start_time = time.localtime()

        m = re.match(r'Submitted batch job (\d+)', sbatch_output)
        self.base_job_id = m.groups()[0]

        if array_id is None:
            self.job_id = self.base_job_id
        else:
            self.job_id = self.base_job_id + '_' + str(self.array_id)

    def state(self):
        """Return the state code for this job, or a dictionary of state codes for array jobs.
        """
        table = squeue(host=self.host, job_id=self.base_job_id)
        return JobState(table.get(self.job_id, {'ST': 'NO'})['ST'])

    def is_done(self):
        return self.state().is_done and os.path.exists(self.output_file)

    @property
    def output_file(self):
        return self._filename_replacement(self.args['output'])

    @property
    def error_file(self):
        return self._filename_replacement(self.args['error'])

    @property
    def output(self):
        return open(self.output_file, 'r').read()

    @property
    def error(self):
        return open(self.error_file, 'r').read()

    def _filename_replacement(self, fn):
        fn = fn.replace(r'%j', self.job_id)
        fn = fn.replace(r'%A', self.base_job_id)
        fn = fn.replace(r'%a', str(self.array_id))
        return fn

    def cancel(self):
        run(self.host, ['scancel', str(self.job_id)])


class SlurmJobArray(SlurmJob):
    def __init__(self, args, sbatch_output, job_file, host):
        self.args = args
        self.sbatch_output = sbatch_output
        self.job_fie = job_file
        self.host = host

        start, _, stop = args['array'].partition('-')

        m = re.match(r'Submitted batch job (\d+)', sbatch_output)
        self.job_id = m.groups()[0]

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
        return {job.job_id:job.state() for job in self.jobs}

    def is_done(self):
        return all([job.is_done() for job in self.jobs])

    def cancel(self):
        run(self.host, ['scancel', str(self.job_id)])

    def finished_jobs(self):
        return [job for job in self.jobs if job.is_done()]

    def unfinished_jobs(self):
        return [job for job in self.jobs if not job.is_done()]

    def wait_iter(self):
        """Iterator that returns jobs as they finish
        """
        jobs = self.jobs[:]
        while len(jobs) > 0:
            for j in jobs[:]:
                if j.is_done():
                    jobs.remove(j)
                    yield(j)
            time.sleep(3)

    def state_counts(self):
        return dict(zip(*np.unique([s.state_code for s in self.state().values()], return_counts=True)))


class JobState:

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
    def __init__(self, state_code):
        self.state_code = state_code
        self.state, self.description = JobState.state_codes[state_code]
        self.is_done = state_code in ('BF', 'CA', 'CD', 'DL', 'F', 'NF', 'OOM', 'PR', 'TO', 'NO')

    def __repr__(self):
        return f"<JobState {self.state}>"

_last_squeue = {}
def squeue(host, job_id, cache_duration=10):
    global _last_squeue
    # 10-second cache to rate limit squeue calls
    last_time, last_state = _last_squeue.get(job_id, (None, None))
    now = time.time()
    if last_time is None or now - last_time > cache_duration:
        stat = run(host, ['squeue', f'--job={job_id}'])
        lines = stat.split('\n')
        cols = re.split(r'\s+', lines[0].strip())
        table = {}
        for line in lines[1:]:
            line = line.strip()
            if line == '':
                continue
            tokens = re.split(r'\s+', line)
            tokens = tokens[:7] + [' '.join(tokens[7:])]  # NODELIST(REASON) column may contain spaces
            fields = {cols[i]:field for i,field in enumerate(tokens)}
            table[fields.pop('JOBID')] = fields
        _last_squeue[job_id] = (now, table)
    else:
        table = last_state
    return table


ssh_connections = {}

def run(host:str, cmd:list):
    """Run a command on any host. If the host is other than localhost,
    then the command is run over ssh.
    """
    global ssh_connections
    assert isinstance(cmd, list)
    if host != 'localhost':
        if host not in ssh_connections:
            conn = subprocess.Popen(['ssh', '-NM', host])
            ssh_connections[host] = conn
        cmd = ['ssh', host] + cmd
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as exc:
        print(exc.output)
        raise

def close_ssh_connections():
    global ssh_connections
    for proc in ssh_connections.values():
        proc.kill()
    ssh_connections = {}

atexit.register(close_ssh_connections)



if __name__ == '__main__':
    hpc_worker()
