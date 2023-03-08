# CSV training

## How to run training

```
conda activate wenet

rm -f training_logs.out
nohup bash run_step_4.sh &> training_logs.out &

tail -f training_logs.out
```

## How to run tensorboard

```
rm -f tensorboard_logs.out

nohup /usr/bin/python3 /home/yehor/.local/bin/tensorboard --logdir tensorboard/conformer/ --host 0.0.0.0 &> tensorboard_logs.out &
```
