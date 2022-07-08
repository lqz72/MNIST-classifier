@REM python train.py --data_path ./dataset --epochs 10 --batch_size 64 --learning_rate 0.001
@REM python train.py --data_path ./dataset --epochs 10 --batch_size 64 --learning_rate 0.002
@REM python train.py --data_path ./dataset --epochs 10 --batch_size 64 --learning_rate 0.004
@REM python train.py --data_path ./dataset --epochs 10 --batch_size 64 --learning_rate 0.006
@REM python train.py --data_path ./dataset --epochs 10 --batch_size 64 --learning_rate 0.008
@REM python train.py --data_path ./dataset --epochs 10 --batch_size 64 --learning_rate 0.010
@REM python train.py --data_path ./dataset --epochs 10 --batch_size 64 --learning_rate 0.020


python train.py --data_path ./dataset --epochs 10 --batch_size 64 --learning_rate 0.001 --optimizer SGD
python train.py --data_path ./dataset --epochs 10 --batch_size 64 --learning_rate 0.001 --optimizer Momentum
python train.py --data_path ./dataset --epochs 10 --batch_size 64 --learning_rate 0.001 --optimizer NAG
python train.py --data_path ./dataset --epochs 10 --batch_size 64 --learning_rate 0.001 --optimizer AdaGrad
python train.py --data_path ./dataset --epochs 10 --batch_size 64 --learning_rate 0.001 --optimizer RMSProp
python train.py --data_path ./dataset --epochs 10 --batch_size 64 --learning_rate 0.001 --optimizer Adam
python train.py --data_path ./dataset --epochs 10 --batch_size 64 --learning_rate 0.001 --optimizer NAdam






