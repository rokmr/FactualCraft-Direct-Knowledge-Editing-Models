from transformers import GPTJForCausalLM, AutoTokenizer, AutoModel
import torch
import os
import json
import random
from tqdm import tqdm
import re

directory_path = "D:/NLP/"

#device = "cuda"
model = GPTJForCausalLM.from_pretrained(directory_path)#.to(device)
tokenizer = AutoTokenizer.from_pretrained(directory_path)



prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids#.to(device)

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)



import torch
from transformers import AutoTokenizer, AutoModel

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_sent_embeddings(sents, contriever, tok, BSZ=32):    
    all_embs = []
    for i in tqdm(range(0, len(sents), BSZ)):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to("cuda:4")
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs

def retrieve_facts(query, fact_embs, contriever, tok, k=1):
    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to("cuda:4")
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(k, largest=True)
    return knn.indices

# with open('datasets/MQuAKE-CF-3k.json', 'r') as f:
#     dataset = json.load(f)


with open('datasets/MQuAKE-CF.json', 'r') as f:
    dataset = json.load(f)


# with open('datasets/MQuAKE-CF-3k.json', 'r') as f:
#     dataset = json.load(f)

new_facts = set()
for d in dataset:
    for r in d["requested_rewrite"]:
        new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
new_facts = list(new_facts)

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(device)
contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").to(device)
tokenizer_c = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")

embs = get_sent_embeddings(new_facts, contriever, tokenizer_c)

# Run test for retrieval index
fact_ids = retrieve_facts("Who is the president of the US?", embs, contriever, tokenizer_c)
print(new_facts[fact_ids[0]])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model.to(device)



from transformers import StoppingCriteria
class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, target_sequence, prompt):
        self.target_sequence = target_sequence
        self.prompt=prompt
        # print(self.target_sequence, self.prompt )
        # self.prev = ""

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = tokenizer.decode(input_ids[0])
        generated_text = generated_text.replace(self.prompt,'')
        # print("gen ", generated_text)
        # Check if the target sequence appears in the generated text
        if self.target_sequence in generated_text:
            return True  # Stop generation

        pattern = r"Final answer:\s*(.*)\n"

        matches = re.findall(pattern, generated_text)

        if matches:
            # print(generated_text)
            return True


        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

def call_gptj(curr_prompt, stop):
    input_ids = tokenizer(curr_prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.9,
                max_new_tokens=256,
                # max_length=5000,
                stopping_criteria=MyStoppingCriteria(stop, curr_prompt))
    # print("outtt           ", out[0], out.shape)
    generated_text = tokenizer.decode(out[0])
    # print(generated_text)
    return generated_text

# read prompts
with open('prompts/MeLLo-prompt.txt', 'r') as f:
    task_prompt = f.read()
stop = "Retrieved fact:"
# stop = ["Retrieved fact:", "Final answer:"]
# print("task ", task_prompt)

# Run MeLLo on the first T (T=10) examples
T = 10

cor = 0
tot = 0

for d in tqdm(dataset[:1000]):
    tot += 1
    for q in d["questions"]:
        found_ans = False
        prompt = task_prompt + "\n\nQuestion: " + q
        # print("ppp   ",prompt)
        # print("dadadadwww")
        for i in range(4):
            # prompt the model to generate a subquestion and a tentative answer
            gen = call_gptj(prompt, stop)
            # print("gener ", gen.strip().split('\n'))
            last_sent = gen.strip().split('\n')[-1]
            # print("lasda")
            # print(last_sent)
            # print("fsjksdf")
            # if final answer is there, get the answer and exit
            if last_sent.startswith('Final answer: '):
                # print("found")
                found_ans = True
                ans = last_sent[len("Final answer: "):]
                break
            
            # otherwise, extract the generated subquestion
            if len(gen.strip().split('\n')) < 2:
                break # failed case
            subquestion = gen.strip().split('\n')[-3]
            # print("sub ", subquestion)
            # print("dadsfsd")
            if not subquestion.startswith('Subquestion: '):
                break # failed case
            subquestion = subquestion[len("Subquestion: "):]
            
            # retrieve an edited fact using the generated subquestion
            fact_ids = retrieve_facts(subquestion, embs, contriever, tokenizer_c)
            fact_sent = new_facts[fact_ids[0]]

            print(fact_sent)
  
            prompt = gen + 'Retrieved fact: ' + fact_sent + '.'
            
            
        prompt = prompt + gen
        
        if not found_ans:
            continue
        # if the answer is correct
        if ans == d["new_answer"] or ans in d["new_answer_alias"]:
            cor += 1
            break

print(f'Multi-hop acc = {cor / tot} ({cor} / {tot})')



from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

model_name_or_path = "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ"
model_basename = "model.safetensors"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

quantize_config = BaseQuantizeConfig.from_pretrained(model_name_or_path)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        use_safetensors=True,
        # model_basename=model_basename,
        device=device,
        use_triton=False,
        quantize_config=None)


# with open('datasets/MQuAKE-CF-3k.json', 'r') as f:
#     dataset = json.load(f)


with open('datasets/MQuAKE-CF.json', 'r') as f:
    dataset = json.load(f)


# with open('datasets/MQuAKE-CF-3k.json', 'r') as f:
#     dataset = json.load(f)

new_facts = set()
for d in dataset:
    for r in d["requested_rewrite"]:
        new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
new_facts = list(new_facts)

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(device)
contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").to(device)
tokenizer_c = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")

embs = get_sent_embeddings(new_facts, contriever, tokenizer_c)

# Run test for retrieval index
fact_ids = retrieve_facts("Who is the president of the US?", embs, contriever, tokenizer_c)
print(new_facts[fact_ids[0]])


def call_gptq(curr_prompt, stop):
    input_ids = tokenizer(curr_prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.9,
                max_new_tokens=256,
                # max_length=5000,
                stopping_criteria=MyStoppingCriteria(stop, curr_prompt))
    # print("outtt           ", out[0], out.shape)
    generated_text = tokenizer.decode(out[0])
    # print(generated_text)
    return generated_text

# read prompts
with open('prompts/MeLLo-prompt.txt', 'r') as f:
    task_prompt = f.read()
stop = "Retrieved fact:"

T = 10

cor = 0
tot = 0

for d in tqdm(dataset[:1500]):
    tot += 1
    for q in d["questions"]:
        found_ans = False
        prompt = task_prompt + "\n\nQuestion: " + q
        # print("ppp   ",prompt)
        # print("dadadadwww")
        for i in range(4):
            # prompt the model to generate a subquestion and a tentative answer
            gen = call_gptq(prompt, stop)
            # print("gener ", gen.strip().split('\n'))
            last_sent = gen.strip().split('\n')[-1]
            if last_sent.startswith('Final answer: '):
                # print("found")
                found_ans = True
                ans = last_sent[len("Final answer: "):]
                break
            
            # otherwise, extract the generated subquestion
            if len(gen.strip().split('\n')) < 2:
                break # failed case
            subquestion = gen.strip().split('\n')[-3]
            # print("sub ", subquestion)
            # print("dadsfsd")
            if not subquestion.startswith('Subquestion: '):
                break # failed case
            subquestion = subquestion[len("Subquestion: "):]
            
            # retrieve an edited fact using the generated subquestion
            fact_ids = retrieve_facts(subquestion, embs, contriever, tokenizer_c)
            fact_sent = new_facts[fact_ids[0]]

            print(fact_sent)
            prompt = gen + 'Retrieved fact: ' + fact_sent + '.'
            
            
        prompt = prompt + gen
        
        if not found_ans:
            continue
        # if the answer is correct
        if ans == d["new_answer"] or ans in d["new_answer_alias"]:
            cor += 1
            break

print(f'Multi-hop acc = {cor / tot} ({cor} / {tot})')