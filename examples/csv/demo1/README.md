# CSV training

## How to run training

```
conda activate wenet

# edit local/prepare_data.sh file and set correct file paths to train/test datasets

# format for those datasets is simple:
#
# path,text
# /path/to/a.wav,hello
# /path/to/b.wav,world

# then run Bash scripts

bash run_step_0.sh
bash run_step_1.sh
bash run_step_2.sh
bash run_step_3.sh

# Run training (in background mode):

rm -f training_logs.out
nohup bash run_step_4.sh &> training_logs.out &

# Look on logs
tail -f training_logs.out
```

## How to run tensorboard to see the metrics

```
rm -f tensorboard_logs.out

# Run in background mode:

nohup /usr/bin/python3 /home/yehor/.local/bin/tensorboard --logdir tensorboard/conformer/ --host 0.0.0.0 &> tensorboard_logs.out &
```
