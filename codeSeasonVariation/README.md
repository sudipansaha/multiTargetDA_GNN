Code for "Season Variation" dataset. 

**Instructions to prepare the dataset** please refer to the directory "multiSeasonDatasetPreparation". After preparing the dataset, please create two directories here, one called "sen12SeasonVariation" where four image directories (fall, spring, summer, winter) are to be placed. Additionally, create a directory called "data/sen12SeasonVariation" where four .txt files (fall.txt, spring.txt, summer.txt, winter.txt) are to be placed.

**Running the code** 

**Example for Source summer, Target spring, fall, winter**

CUDA_VISIBLE_DEVICES=0 nohup python srcSeason/main.py --seed 0 --threshold 0.85 --method 'CDAN' --encoder ResNet18 --dataset sen12SeasonVariation --data_root ./sen12SeasonVariation/ --source summer --target spring fall winter --source_iters 1000 --adapt_iters 5000 --finetune_iters 5000 --lambda_node 0.3 --num_workers 4 --output_dir sen12Season-dcgct/summer_restSeed0Threshold85/CDAN > nohupSummerRestSeed0Threshold85.out &

Note that you may need to change your GPU ID or the file name where you want to save result

Our code is based on the https://github.com/Evgeneus/Graph-Domain-Adaptaion Main differences are in main.py and utils.py
