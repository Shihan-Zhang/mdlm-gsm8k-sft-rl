import json, random
from datasets import load_dataset

SEED = 20261023
N_DEV = 500
ds = load_dataset('openai/gsm8k', 'main')
train, test = ds['train'], ds['test']

idx = list(range(len(train)))
random.Random(SEED).shuffle(idx)
dev_idx, rest_idx = sorted(idx[:N_DEV]), sorted(idx[N_DEV:])

def conv(i):
    r = train[i]
    return {'prompt': r['question'], 'response_ground_truth': r['answer']}

out = '/root/autodl-tmp/s-flm/data/'
json.dump([conv(i) for i in dev_idx], open(out + 'gsm8k_dev.json', 'w'), ensure_ascii=False, indent=1)
json.dump([conv(i) for i in rest_idx], open(out + 'gsm8k_train_rest.json', 'w'), ensure_ascii=False, indent=1)
json.dump({'seed': SEED, 'dev_idx': dev_idx, 'rest_idx': rest_idx}, open(out + 'gsm8k_split_indices.json', 'w'))
print('dev', len(dev_idx), 'rest', len(rest_idx), 'overlap', len(set(dev_idx) & set(rest_idx)))

repo_test = json.load(open(out + 'gsm8k_test.json'))
same_q = sum(a['prompt'].strip() == b['question'].strip() for a, b in zip(repo_test, test))
same_a = sum(a['response_ground_truth'].strip() == b['answer'].strip() for a, b in zip(repo_test, test))
print('repo test vs HF test: same questions', same_q, '/', len(test), '| same answers', same_a, '/', len(test))
train_q = set(q.strip() for q in train['question'])
print('test questions also in train:', sum(q.strip() in train_q for q in test['question']))
