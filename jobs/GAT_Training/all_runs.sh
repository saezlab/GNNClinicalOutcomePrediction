sbatch --job-name=GAT_Training_1 -p gpu --gres=gpu:1 --mem=2g  -n 1 --time=7-00:00:00 --output=results/output_1 "1_1000.sh"
sleep 1
