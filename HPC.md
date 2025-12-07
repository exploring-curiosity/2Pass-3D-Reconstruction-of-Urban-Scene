# NYU HPC Guide  
**Course**: Selected Topics in Signal Processing: Advanced Computer Vision

## Logging In

You need to be on the NYU network to log in to the cluster. If you are off campus, you can use the NYU VPN or `gw.hpc.nyu.edu`.

First, you need to log in:

```bash
ssh [NETID]@greene.hpc.nyu.edu
```

Example:

```bash
ssh sp7835@greene.hpc.nyu.edu
```

It asks you to enter the password, which is your NetID password.

Once you're on Greene, it will show `log-1`, `log-2`, or `log-3`. Greene is reserved for research purposes only. You need to go into the burst node using:

```bash
ssh burst
```

Now you will see the prompt change to `log-burst` instead of `log-1` etc.




## Understanding the Filesystem

The Greene HPC cluster has different directories optimized for various storage needs.

| Directory  | Variable   | Purpose                | Flushed After | Quota             |
|------------|------------|------------------------|---------------|-------------------|
| `/archive` | `$ARCHIVE` | Long-term storage      | No            | 2TB / 20K inodes  |
| `/home`    | `$HOME`    | Configuration files    | No            | 50GB / 30K inodes |
| `/scratch` | `$SCRATCH` | Temporary data storage | Yes (60 days) | 5TB / 1M inodes   |
| `/vast`    | `$VAST`    | Large no of files      | Yes (60 days) | 2TB / 5M inodes   |


- **Check Your Quota:**

  ```bash
  myquota
  ```


- **Recommended:** Store the data you want to keep in `/scratch/[netid]` and temporary data in `/tmp`.


## Requesting Resources

 [Official Docs](https://sites.google.com/nyu.edu/nyu-hpc/training-support/tutorials/slurm-tutorial) for the NYU HPC for the SLURM tutorial 

Below is a example command.
This requests a V100 for 4 hours:

```bash
srun --account=ece_gy_9193_001-2024fa --partition=n1s8-v100-1 --gres=gpu:1 --time=04:00:00 --pty /bin/bash
```

Other partitions are `n1s8-t4-1`, `c12m85-a100-1`. you can use `interactive` to debug, copy.  You can run `sinfo` to get a list of available partitions.


## Exiting the Current Session

Use `CTRL+D` or type `exit`.


## Checkpointing and Job Persistence

Always save checkpoints and load from them, as your jobs might die.

Note: We are running spot instances in the cloud, which may be shut down by Google. More information on spot instances can be found here:
- [Google Spot Instances](https://cloud.google.com/compute/docs/instances/spot)
- [Google Preemptible Instances](https://cloud.google.com/compute/docs/instances/preemptible)

Enable checkpoint/restart files for production runs, saving them to your `/scratch/[NetID]` folder. Jobs will be requeued automatically if nodes are shut down by GCP. Add the following directive in your Slurm script:

```bash
#SBATCH --requeue
```

## Singularity Setup

First, request a GPU using:

```bash
srun --account=ece_gy_9193_001-2024fa --partition=n1s8-v100-1 --gres=gpu:1 --time=04:00:00 --pty /bin/bash
```

Then, copy the overlay file for Singularity (this is done only once). It can hold 25GB and 500k files:

```bash
cd /scratch/[NETID]
scp greene-dtn:/scratch/work/public/overlay-fs-ext3/overlay-25GB-500K.ext3.gz .
gunzip -vvv ./overlay-25GB-500K.ext3.gz
```

Next, copy the Singularity container:

```bash
scp -rp greene-dtn:/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif .
```

Now, run Singularity with the overlay and GPU:

```bash
singularity exec --bind /scratch --nv --overlay  /scratch/[NETID]/overlay-25GB-500K.ext3:rw /scratch/[NETID]/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash
```
Filesystems can be mounted as read-write (`rw`) or read-only (`ro`) when we use it with singularity.
- read-write: use this one when setting up env (installing conda, libs, other static files)
- read-only: use this one when running your jobs. It has to be read-only since multiple processes will access the same image. It will crash if any job has already mounted it as read-write.


Once inside Singularity:

```bash
Singularity> cd /ext3/
Singularity> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
```

- **Installation Prefix:** `/ext3/miniconda3`
- **Modify `~/.bashrc`:** Yes

**Set Up Conda:**

```bash
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
```



### Installing Python Libraries

Ensure Conda environment is activated.

```bash
conda create -n newEnv python==3.9
conda activate newEnv
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Running Batch Jobs

For longer experiments or multiple jobs, use batch jobs.

### Batch Job Workflow

1. **Log In** to Greene.

2. **Log In** to burst.

3. **Submit an `sbatch` Script**.


**Write the Batch Script:**

```bash
#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --account=ece_gy_9193_001-2024fa
#SBATCH --partition=n1s8-v100-1
#SBATCH --open-mode=append
#SBATCH --export=ALL
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=myTest
#SBATCH --mail-type=END
#SBATCH --mail-user=bob.smith@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --requeue


singularity exec --bind /scratch --nv --overlay  /scratch/[netid]/overlay-25GB-500K.ext3:rw /scratch/[netid]/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate newEnv
cd /scratch/[net_id]/
python temp_file.py

```

**Submit Batch Job:**

```bash
sbatch gpu_job.slurm
```

**Check Job Status:**

```bash
squeue -u [netid]
```

To cancel a job:

```bash
scancel [JOBID]
```


### Checking Job Output

After job completion, check the output log saved under the filenames specified above in the SBATCH script
```bash
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
```
## Copying Files

To copy to burst node, you need to copy to greene first from local. Then on burst node, copy from greene
### Copying from Local to Greene:


1. First copy from local to Greene, run below command on local
```bash
scp [opt-flags] local-dest [NETID]@greene.hpc.nyu.edu:"[greene-dest]"
```

### Copying from Greene to Burst:
2. Then copy from greene to burst, run below command on burst node
```bash
scp [opt-flags] greene-dtn:"[greene-dest]" [burst-dest]
```

### Copying from Burst to Local:

1. First copy from burst to Greene, run below comand on burst:

```bash
scp [opt-flags] [burst-dest] greene-dtn:"[greene-dest]"
```

2. Then copy from Greene to local, run below command on local:

```bash
scp [opt-flags] [NETID]@greene.hpc.nyu.edu:"[greene-dest]" [local-dest]
```

### Copying Workflow:

```bash
local -> greene -> burst
burst -> greene -> local
```
You can also set up GitHub SSH keys and pull code directly from GitHub.


## General flow
1. On your local machine ssh to Greene  

2. On Greene, ssh to a Burst Node

3. On the Burst Node, request for a GPU or an Interactive node. You can copy stuff on interactive node.

4. After successfully acquiring the resource start your Singularity container

5. Once in your singularity container, activate your Conda environment and do your things.

Note: bash commands don't work inside singularity, you need to exit container by typing `exit`.


## Optional Tips/Notes

### Open OnDemand Server for Cloud Bursting

There is an **Open OnDemand (OOD)** server available for cloud bursting:  
[https://ood-burst-001.hpc.nyu.edu/](https://ood-burst-001.hpc.nyu.edu/) (NYU VPN is required when working off campus).

From this OOD server, students can:
- Launch compute nodes without logging into the Greene cluster.
- Run Jupyter notebooks.
- Open a terminal.
- Transfer data from local computers.


---

### Installing Software Without `sudo`

If you need to install software that requires `sudo`, first check if it can be installed using **Conda** (e.g., `ffmpeg`, `colmap`, etc.). You can also try `module load` for some software.  
Singularity might also have additional software available, which can be found here:  
`/scratch/work/public/singularity`

---

### SSH Configuration

If you're facing issues copying files on the burst node, create an SSH config file on your GPU or interactive node:

1. Open the config file using `vim`:

   ```bash
   vim ~/.ssh/config
   ```

2. Press `i` to enter insert mode, then copy and paste the following configuration:

   ```bash
   Host greene.hpc.nyu.edu dtn.hpc.nyu.edu greene-dtn
     StrictHostKeyChecking no
     ServerAliveInterval 60
     ForwardAgent yes
     UserKnownHostsFile /dev/null
     LogLevel ERROR
   ```

3. To exit, press `ESC`, then type `:wq` to save and quit.

You can also create a similar SSH config file on your local machine:

```bash
Host hpc
    HostName greene.hpc.nyu.edu
    User [NETID]
    StrictHostKeyChecking no
    ServerAliveInterval 60
    UserKnownHostsFile /dev/null
    LogLevel ERROR
```

---

### Using `tmux`

Use **`tmux`** to keep jobs alive when disconnecting from the session. You can look up tutorials online to learn how to use `tmux`.

---

### GPU Resource Monitoring

Jobs on the burst node will be killed if resource utilization is low for more than 30 minutes.  
Use the following command to check GPU information:

```bash
nvidia-smi
```

---

### File Indexing Limitations

The HPC has limitations when indexing a large number of files:
- `scratch` has a **2M file limit**.
- `vast` has a **5M file limit**.

If you have a large number of files, consider compressing them into a **tar archive** and extracting them as needed.

---

### Sharing Files

To share files with others on the HPC, you can use **`setfacl`**. More information can be found [here](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/hpc-storage/data-management/sharing-data-on-hpc).

