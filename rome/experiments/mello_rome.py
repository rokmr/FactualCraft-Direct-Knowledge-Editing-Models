import os
import json
import random
from tqdm import tqdm
import re

import torch
from transformers import AutoTokenizer, AutoModel
from util import nethook
from experiments.py.demo import stop_execution
from rome import ROMEHyperParams, apply_rome_to_model
from util.generate import generate_fast
from util.globals import *

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

with open('datasets/MQuAKE-CF-3k.json', 'r') as f:
    dataset = json.load(f)


# with open('datasets/MQuAKE-CF.json', 'r') as f:
#     dataset = json.load(f)


# with open('datasets/MQuAKE-T.json', 'r') as f:
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

import torch
from transformers import GPTJForCausalLM, AutoTokenizer

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", low_cpu_mem_usage=True)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


from transformers import StoppingCriteria
class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, target_sequence, prompt):
        self.target_sequence = target_sequence
        self.prompt=prompt
        # print(self.target_sequence, self.prompt )
        # self.prev = ""

    def __call__(self, input_ids, scores, **kwargs):
        # Get the generated text as a string
        generated_text = tokenizer.decode(input_ids[0])
        generated_text = generated_text.replace(self.prompt,'')
        # print("gen ", generated_text)
        # Check if the target sequence appears in the generated text
        if self.target_sequence in generated_text:
            return True  # Stop generation
        # elif '.' in generated_text:
        #     self.prev = ""
        #     return False
        # Define the regex pattern
        pattern = r"Final answer:\s*(.*)\n"

        # Search for the pattern in the text
        matches = re.findall(pattern, generated_text)

        # Check if matches is empty or not
        if matches:
            # print(generated_text)
            return True

        # self.prev = self.prev + generated_text
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

def call_edited(curr_prompt, edits):
    input_ids = tokenizer(curr_prompt, return_tensors="pt").input_ids.to(device)
    nethook.set_requires_grad(True, model)
    path = ( HPARAMS_DIR
            /ROME
            / f"{model.config._name_or_path.replace('/', '_')}.json"
    )
    hparams = ROMEHyperParams.from_json(path)
    model_, orig_weights = apply_rome_to_model(model, tokenizer, edits, hparams, return_orig_weights=True)
    generated_text = generate_fast(model_, tokenizer, curr_prompt, max_out_len=150)

    return generated_text



# read prompts
with open('prompts/MeLLo-prompt.txt', 'r') as f:
    task_prompt = f.read()
stop = "Retrieved fact:"
# stop = ["Retrieved fact:", "Final answer:"]
# print("task ", task_prompt)


cor = 0
tot = 0

firsth_count = 0
sech_count=0
thirdh_count=0
fourth_count=0
firsth_tot = 0
sech_tot = 0
thirdh_tot = 0
fourth_tot=0

for d in tqdm(dataset[:3000]):
    tot += 1
    for q in d["questions"]:
        found_ans = False
        prompt = task_prompt + "\n\nQuestion: " + q
        # print("ppp   ",prompt)
        # print("dadadadwww")
        for i in range(4):
            # prompt the model to generate a subquestion and a tentative answer
            #gen = call_gptj(prompt, stop)
            gen = call_edited(prompt, d["requested_rewrite"])
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

            if i==0:
                firsth_tot +=1
            elif i==1:
                sech_tot += 1
            elif i==2:
                thirdh_tot+=1
            else:
                fourth_tot+=1

            inter_ans = reply.strip().split('\n')[-1]

            if inter_ans == d["single_hops"][i]["answer"] or ans in d["single_hops"][i]["answer_alias"]:
                if i==0:
                    firsth_count +=1
                elif i==1:
                    sech_count += 1
                elif i==2:
                    thirdh_count+=1
                else:
                    fourth_count+=1
 
            # retrieve an edited fact using the generated subquestion
            fact_ids = retrieve_facts(subquestion, embs, contriever, tokenizer_c)
            fact_sent = new_facts[fact_ids[0]]

            print(fact_sent)
            
            # put the retrieved fact at the end of the prompt, the model self-checks if it contradicts
            # prompt = prompt + gen + 'Retrieved fact: ' + fact_sent + '.'
            prompt = gen + 'Retrieved fact: ' + fact_sent + '.'
            
            
        prompt = prompt + gen
        
        if not found_ans:
            continue
        # if the answer is correct
        if ans == d["new_answer"] or ans in d["new_answer_alias"]:
            cor += 1
            break

print(f'Multi-hop acc = {cor / tot} ({cor} / {tot})')

print("One hop acc = "+str(firsth_count/firsth_tot))
print("Second hop acc = "+str(sech_count/sech_tot))
print("Third hop acc = "+str(thirdh_count/thirdh_tot))
print("Fourth hop acc = "+str(fourth_count/fourth_tot))