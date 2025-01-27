import os

data_name = 'simulation'
version = 'without_quoting'

### First set up directories on the cluster:
project_dir = os.getcwd()
source_dir = os.path.join(project_dir, 'data', data_name, version)
slurm_dir = os.path.join(project_dir, 'slurm', data_name, version)
out_dir = os.path.join(project_dir, 'out', data_name, version)
err_dir = os.path.join(project_dir, 'err', data_name, version)
python_dir = os.path.join(project_dir, 'python', data_name)

if not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(err_dir):
    os.mkdir(err_dir)


# For now just use the environment for testing.
# partition = 'gpu-test'
partition = 'gpu'

### A dictionary of scenarios to be explored
# Default values correspond with the classical TBIP model (with topic-specific locations) with gamma and CAVI updates.
# List only the FLAGS that you want to be changed.

# First scenario
scenarios = {}
for s in ['', '_zero', '_party', '_diverge', '_estimate']:
    scenarios[s] = {"scenario": s,
                    "reest_seed": 0,
                    "data_name": data_name,
                    "version": version,
                    "eps": 1,
                    "learningrate": 0.0001,
                    "numtopics": 25,
                    "batchsize": 512,
                    "maxsteps": 250000,
                    "printsteps": 25000,
                    }

### Creating slurm files and one file to trigger all initial jobs with session 97.
with open(os.path.join(slurm_dir, 'run_all_97.slurm'), 'w') as all_file:
    all_file.write('#! /bin/bash\n\n')
    for name in scenarios:
        flags = '  --session=97  --preinitializeparameters=True  --initialize_ideal_by_party=False'
        for key in scenarios[name]:
            flags = flags+'  --'+key+'='+str(scenarios[name][key])
        with open(os.path.join(slurm_dir, '97'+name+'.slurm'), 'w') as file:
            file.write('#!/bin/bash\n')
            file.write('#SBATCH --job-name=' + data_name + ' # short name for your job\n')
            file.write('#SBATCH --partition='+partition+'\n')

            # Other potential computational settings.
            # file.write('#SBATCH --nodes=1 # number of nodes you want to use\n')
            # file.write('#SBATCH --ntasks=2 # total number of tasks across all nodes\n')
            # file.write('#SBATCH --cpus-per-task=1 # cpu-cores per task (>1 if multi-threaded tasks)\n')
            # file.write('#SBATCH --mem=1G # total memory per node\n')
            file.write('#SBATCH --mem-per-cpu=356000 # in megabytes, default is 4GB per task\n')
            # file.write('#SBATCH --mail-user=jan.vavra@wu.ac.at\n')
            # file.write('#SBATCH --mail-type=ALL\n')

            file.write('#SBATCH -o '+os.path.join(out_dir, '%x_%j_%N.out')+'      # save stdout to file. '
                                             'The filename is defined through filename pattern\n')
            file.write('#SBATCH -e '+os.path.join(err_dir, '%x_%j_%N.err')+'      # save stderr to file. '
                                             'The filename is defined through filename pattern\n')
            file.write('#SBATCH --time=11:59:59 # total run time limit (HH:MM:SS)\n')
            file.write('\n')
            file.write('. /opt/apps/2023-04-11_lmod.bash\n')
            file.write('ml purge\n')
            file.write('ml miniconda3-4.10.3-gcc-12.2.0-ibprkvn\n')
            file.write('\n')
            file.write('cd /home/jvavra/TBIP/\n')
            # file.write('conda activate env_TBIP\n')
            file.write('conda activate tf_1_15\n')
            file.write('\n')
            file.write('python3 '+os.path.join(python_dir, '03_sess_97.py')+flags+'\n')
        # Add a line for running the batch script to the overall slurm job.
        all_file.write('sbatch --dependency=singleton '+os.path.join(slurm_dir, '97'+name+'.slurm'))
        all_file.write('\n')

### Creating slurm files and one file to trigger all the jobs that continue from previous session to the next (98-114)
with open(os.path.join(slurm_dir, 'run_all_98_114.slurm'), 'w') as all_file:
    all_file.write('#! /bin/bash\n\n')
    for name in scenarios:
        for s in range(98, 115):
            flags = '  --session=' + str(s)
            for key in scenarios[name]:
                flags = flags+'  --'+key+'='+str(scenarios[name][key])
            with open(os.path.join(slurm_dir, str(s)+name+'.slurm'), 'w') as file:
                file.write('#!/bin/bash\n')
                file.write('#SBATCH --job-name=' + data_name + ' # short name for your job\n')
                file.write('#SBATCH --partition='+partition+'\n')

                # Other potential computational settings.
                # file.write('#SBATCH --nodes=1 # number of nodes you want to use\n')
                # file.write('#SBATCH --ntasks=2 # total number of tasks across all nodes\n')
                # file.write('#SBATCH --cpus-per-task=1 # cpu-cores per task (>1 if multi-threaded tasks)\n')
                # file.write('#SBATCH --mem=1G # total memory per node\n')
                file.write('#SBATCH --mem-per-cpu=356000 # in megabytes, default is 4GB per task\n')
                # file.write('#SBATCH --mail-user=jan.vavra@wu.ac.at\n')
                # file.write('#SBATCH --mail-type=ALL\n')

                file.write('#SBATCH -o '+os.path.join(out_dir, '%x_%j_%N.out')+'      # save stdout to file. '
                                                 'The filename is defined through filename pattern\n')
                file.write('#SBATCH -e '+os.path.join(err_dir, '%x_%j_%N.err')+'      # save stderr to file. '
                                                 'The filename is defined through filename pattern\n')
                file.write('#SBATCH --time=11:59:59 # total run time limit (HH:MM:SS)\n')
                file.write('\n')
                file.write('. /opt/apps/2023-04-11_lmod.bash\n')
                file.write('ml purge\n')
                file.write('ml miniconda3-4.10.3-gcc-12.2.0-ibprkvn\n')
                file.write('\n')
                file.write('cd /home/jvavra/TBIP/\n')
                # file.write('conda activate env_TBIP\n')
                file.write('conda activate tf_1_15\n')
                file.write('\n')
                file.write('python3 '+os.path.join(python_dir, '03_variational_initialization_beta_eta.py')+flags+'\n')
            # Add a line for running the batch script to the overall slurm job.
            all_file.write('sbatch --dependency=singleton '+os.path.join(slurm_dir, str(s)+name+'.slurm'))
            all_file.write('\n')
