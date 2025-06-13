#!/bin/sh

#SBATCH --account=nlpgroup80

# The line below selects the group of nodes you require
#SBATCH --partition=a100

# The line below reserves 1 worker node and 2 cores
#SBATCH --nodes=1 --ntasks=4 --gres=gpu:ampere80:1

# The line below indicates the wall time your job will need, 10 hours for example.
#SBATCH --time=48:00:00

# A sensible name for your job, try to keep it short
#SBATCH --job-name="Wav2Vec2_Model"

#Modify the lines below for email alerts. Valid type values are NONE, BEGIN, END, FAIL, REQUEUE, ALL 
#SBATCH --mail-user=skscla001@myuct.ac.za
#SBATCH --mail-type=ALL

# The cluster is configured primarily for OpenMPI and PMI. Use srun to launch parallel jobs if your code is parallel aware.
# To protect the cluster from code that uses shared memory and grabs all available cores the cluster has the following 
# environment variable set by default: OMP_NUM_THREADS=1
# If you feel compelled to use OMP then uncomment the following line:
# export OMP_NUM_THREADS=$SLURM_NTASKS

# NB, for more information read https://computing.llnl.gov/linux/slurm/sbatch.html

# Use module to gain easy access to software, typing module avail lists all packages.
# Example:
# module load python/anaconda-python-3.7

# If your code is capable of running in parallel and requires a command line argument for the number of cores or threads such as -n 30 or -t 30 then you can link the reserved cores to this with the $SLURM_NTASKS variable for example -n $SLURM_NTASKS instead of -n 30

# Your science stuff goes here...

#export CUDA_VISIBLE_DEVICES=$(ncvd)

export HF_HOME="/scratch/skscla001/hf_cache"

env_name="mms"

# Creating and installing libraries in the virtual environment
echo " "
echo "---------- Step 0: Installing libraries  --------------"
echo " "

module load python/miniconda3-py3.12
source activate $env_name

echo "---------- Step 1: Running model ----------------------"

python wav2vec2_model.py \
    --model_name_or_path="facebook/wav2vec2-base-960h" \
    --learning_rate=1e-5 \
    --num_train_epochs=5 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=4 \
    --train_split_path="/content/adapter-wavlm/ASR/train.csv" \
    --eval_split_path="/content/adapter-wavlm/ASR/dev.csv" \
    --output_dir="./wav2vec2-base-960h-libri10h" \
    --overwrite_output_dir \
    --save_model
echo " "
echo "---------- Step 1: Processing complete  ----------------------"

conda deactivate
exit
