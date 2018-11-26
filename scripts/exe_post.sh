exp_dir="./experiment/"$1
echo $exp_dir
mkdir $exp_dir
python src/main.py --cuda-use --checkpoint-dir-name $1 --mode 0 --teacher-forcing-ratio 0.83 --post-flag
