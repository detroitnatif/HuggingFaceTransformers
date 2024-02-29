import torch

from datasets import load_dataset
from transformers import pipeline, AutoTokenizer

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler


dataset = load_dataset("imdb")
# print(dataset['train']['text'][:5])
# print(dataset['train']['label'][:5])x

def build_dataset(config, dataset_name='imdb', input_min = 2, input_max=8):
    

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token


    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x['review']) > 200, batched=False)

    input_size = LengthSampler(input_min, input_max)

    def tokenize(sample):
        sample['input_ids'] = tokenizer.encode(sample['review'])[: input_size()]
        sample['query'] = tokenizer.decode(sample["input_ids"])
        return sample
    
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
    


if __name__ == '__main__':   #START
    print("running")
    config = PPOConfig(
        model_name="lvwerra/gpt2-imdb",
        learning_rate=1.41e-5,
        log_with="wandb"
    )

    import wandb
    wandb.init()

    dataset = build_dataset(config)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)  # this is the frozen model which we'll compare to

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ppo_trainer = PPOTrainer(config,  model, ref_model, tokenizer, dataset, data_collator=collator)    
    device = ppo_trainer.accelerator.device    
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"

    sentiment_pipe = pipeline("sentiment-analysis", model='lvwerra/distilbert-imdb', device=device)

    sent_kwargs = {'return_all_scores': True, "function_to_apply": "none", "batch_size": 16}
    text = "this was really bad!!"
    print(sentiment_pipe(text, **sent_kwargs))

    text = "this was really good!!"
    print(sentiment_pipe(text, **sent_kwargs))

    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    response_generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,

    }

    for epoch, batch in enumerate(ppo_trainer.dataloader):
        
        query_tensors = batch["input_ids"]

        response_tensors = []

        for query in query_tensors:
            gen_len = output_length_sampler()
            response_generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **response_generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:]) # only take the piece generated, remove acutal prompt
        batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        texts = [q + r for q, r in zip(batch['query'], batch['response'])]

        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)  # this gives a dictionary, we want the positive, which is our rewards 

        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        ppo_trainer.log_stats(stats, batch, rewards)

    # model.save_pretrained('gpt-imbd-pos-v2', push_to_hub=False)

    # tokenizer.save_pretrained('gpt-imbd-pos-v2', push_to_hub=False)
