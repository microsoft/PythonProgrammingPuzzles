from strictfire import StrictFire as Fire  # aborts early on invalid arguments
import os
import csv
import subprocess
import shlex
import random
import numpy as np
import torch
import utils

def fine_tune(
    train_txt="../data/generated_sol_100.txt",
    output_dir = "../outputs/",
    subdir="out",
    model_path="EleutherAI/gpt-neo-2.7B",
    gpu=0,
    num_gpus=1,
    epochs=4,
    seed=0,
    ):
    """
    Fine tune the model on the puzzles in train_txt file and save the results to OUTPUT_DIR/output_subdir

    train_txt: the (possibly gzipped) file containing the text to fine tune on (default: ../data/generated_sol_100.txt)
    subdir: the subdirectory to save the results to (default "out")
    model_path: the path to the model to fine tune (default "EleutherAI/gpt-neo-2.7B")
    gpu: which GPU(s) to use, e.g.: 0,1 (default 0) 
    epochs: how many epochs to train for (default 4)
    seed: the random seed to use, not sure if this affects fine tuning (default 0)    
    """    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # create output dir if necessary
    output_path = os.path.join(output_dir, subdir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    text = utils.load_text_file(train_txt)  # decompresses if ends in .gz
    tokenizer = utils.load_tokenizer(model_path) 
    num_toks = utils.num_tokens(text, tokenizer, verbose=True)
    assert num_toks > 1024, "Not enough tokens in text to fine tune"

    # create csv    
    train_file = os.path.join(output_path, "train.csv")
    with open(train_file, mode="w", encoding="utf-8") as csv_file:
        fieldnames = ["text"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"text": text})
    
    output_path_finetuned = os.path.join(output_path, "finetuned")    

    # keep gradient_accumulation_steps at 1 bc setting it to 2 effectively doubles the batch
    # size which gets tricky when batch sizes are small (ft_tokens will no longer be accurate)
    gradient_accumulation_steps = 1
    per_device_train_batch_size = 4

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if len(cuda_visible_devices):
        print("os.environ(CUDA_VISIBLE_DEVICES)", cuda_visible_devices)
        del os.environ["CUDA_VISIBLE_DEVICES"]    
    print("os.environ(CUDA_VISIBLE_DEVICES)", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    master_port = 29600  # During training deepspeed uses a port to syncronize.  2 jobs need to set different ports to run in parallel
    if type(gpu) in [list, tuple]:
        master_port += gpu[0]
        gpu = ",".join([str(g) for g in gpu])
    else:
        master_port += gpu

    gpu_string = f'--include=localhost:{gpu}'

    if num_gpus > 1:
        gpu_string = f"--num_nodes=1 --num_gpus={num_gpus}",
    # If gpu is passed in as negative - it's the count of gpu to use - a bit of a hack
    if gpu < 0:
        num_gpus = abs(gpu)
        gpu_string = f"--num_nodes=1 --num_gpus={num_gpus}"

    print("gpu_string", gpu_string)

    cmd = " ".join(
        [
            "deepspeed",
            f"--master_port={master_port}",
            gpu_string, 
            # f'--include=localhost:{gpu}',            
            # "--num_nodes=1",
            # f"--num_gpus={num_gpus}",
            "neo_train.py",
            f"--model_name_or_path={model_path}",
            f"--train_file={train_file}",
            f"--output_dir={output_path_finetuned}",
            "--overwrite_output_dir",
            "--ignore_data_skip",
            "--deepspeed",
            "ds_config_gptneo.json",
            f"--save_strategy=no", # ATK remove checkpointing for large datasets
            # pretty sure this is just dataset cache
            "--overwrite_cache",
            # logging frequency
            "--logging_steps=5",
            "--do_train",
            "--report_to none", # turns off report_to WANDB for instance
            "--fp16",
            f"--num_train_epochs={epochs}",
            # overrides num_train_epochs if set to a positive value. This is the number of gradient steps that happen total.
            f"--per_device_train_batch_size={per_device_train_batch_size}",
            "--use_fast_tokenizer=False",
            f"--gradient_accumulation_steps={gradient_accumulation_steps}",
            "--learning_rate=5e-06",
            # linear increase from this up to learning rate, then LR schedule happens (which itself is linear decreasing until max_steps)
            "--warmup_steps=10",
        ]
    )

    utils.info(f"running command: {cmd}")
    print(f"Command to run:{cmd}") # Why is this different than what utils.info prints out, utils.info truncates it
    # exit()
    res = subprocess.run(shlex.split(cmd), check=True)
    utils.info(str(res))
        

if __name__ == "__main__":
    Fire(fine_tune)
