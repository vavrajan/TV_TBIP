#! /bin/bash

sbatch --dependency=singleton /home/jvavra/TBIP/slurm/jae_sim/without_quoting/97.slurm
sbatch --dependency=singleton /home/jvavra/TBIP/slurm/jae_sim/without_quoting/97_zero.slurm
sbatch --dependency=singleton /home/jvavra/TBIP/slurm/jae_sim/without_quoting/97_party.slurm
sbatch --dependency=singleton /home/jvavra/TBIP/slurm/jae_sim/without_quoting/97_diverge.slurm
sbatch --dependency=singleton /home/jvavra/TBIP/slurm/jae_sim/without_quoting/97_estimate.slurm
