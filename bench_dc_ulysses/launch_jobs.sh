# launch job_*.slurm in slurm_jobs
# Usage: bash launch_jobs.sh

# delete .out files in slurm_out
# for out in slurm_out/*.out; do
#     rm -vf $out
# done


for job in slurm_jobs/job_*.slurm; do
    # echo "Submitting job $job"
    sbatch $job
done