for ((i=1;i<=10;i++));
do 
CUDA_VISIBLE_DEVICES=0,1 python run_test_seed.py
done
