# module use /soft/modulefiles/
# module load conda
# conda activate MoE_Mixed_Precision

source ~/.bashrc
conda init
conda activate MoE_Mixed_prec

export LMUData="/vast/users/schittyvenkata/home_dir_code/datasets"

python VLMEvalKit/run.py --data MME --model molmoE-1B-0924 --mode all --work-dir ./results_customs