# import os
# import sqlite3
# from urllib.request import urlretrieve
# import trlx


# url = "https://raw.githubusercontent.com/JD-P/simulacra-aesthetic-captions/main/sac_public_2022_06_29.sqlite"
# dbpath = "sac_public_2022_06_29.sqlite"
# if __name__ == "__main__":
#   main()

# def main():
#   if not os.path.exists(dbpath):
#     print(f"fetching {dbpath}")
#     urlretrieve(url, dbpath)

#   conn = sqlite3.connect(dbpath)
#   c = conn.cursor()
#   c.execute(
#       "SELECT prompt, rating FROM ratings "
#       "JOIN images ON images.id=ratings.iid "
#       "JOIN generations ON images.gid=generations.id "
#       "WHERE rating IS NOT NULL;"
#   )


#   #open a port for profiling
#   server = xp.start_server(3294)
  
#   device = xm.xla_device()

#   prompts, ratings = tuple(map(list, zip(*c.fetchall())))
#   prompts = prompts.to(device)
#   ratings = ratings.to(device)

  
#   trainer = trlx.train(
#       "gpt2",
#       samples=prompts,
#       rewards=ratings,
#       eval_prompts=["Hatsune Miku, Red Dress"] * 64,
#   ).to(device)

#   trainer.save_pretrained('.')
#   trainer.save_pretrained('./trlx-run-tpu')
#   trainer.save_pretrained('gs://mazumdera-test-bucket/trlx/04192023/1/')


# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import json
import os
import sys
from typing import List

import torch
from datasets import load_dataset
from transformers import pipeline

import trlx

from trlx.data.default_configs import TRLConfig, default_ppo_config

#Anisha: pt/xla installations
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.debug.profiler as xp
import torch_xla.distributed.xla_backend

def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the xla process group
    torch.distributed.init_process_group("xla", rank=rank, world_size=world_size)

def main(hparams={}):
    
    #open a port for profiling
    server = xp.start_server(3294)
    
    device = xm.xla_device()
    print(f"Anisha: device = {device}")

    
    new_rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()
    setup(rank=new_rank, world_size=world_size)


    # Merge sweep config with default config if given
    config = TRLConfig.update(default_ppo_config().to_dict(), hparams)

    # if torch.cuda.is_available():
    #     device = int(os.environ.get("LOCAL_RANK", 0))
    # else:
    #     device = -1

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        # "lvwerra/gpt2-imdb",
        # "gpt2",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )
    # #Anisha: since ValueError: Pipeline with tokenizer without pad_token cannot do batching. You can try to set it with `pipe.tokenizer.pad_token_id = model.config.eos_token_id`.
    # sentiment_fn.tokenizer.pad_token = sentiment_fn.tokenizer.eos_token
    # sentiment_fn.tokenizer.padding_side = "left"

    def reward_fn(samples: List[str], **kwargs) -> List[float]:

        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return sentiments

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]


    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["I don't know much about Hungarian underground"] * 256,
        config=config,
    )

    trainer.save_pretrained('.')
    trainer.save_pretrained('./trlx-run-tpu')
    trainer.save_pretrained('gs://mazumdera-test-bucket/trlx/04192023/1/')


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
