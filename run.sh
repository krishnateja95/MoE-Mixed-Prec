module use /soft/modulefiles/
module load conda
conda activate MoE_Mixed_Precision

export LMUData="/lus/grand/projects/datascience/krishnat/home_dir_code/datasets/VLMEvalKit/"

python VLMEvalKit/run.py --data MME --model molmoE-1B-0924 --mode all --work-dir ./results