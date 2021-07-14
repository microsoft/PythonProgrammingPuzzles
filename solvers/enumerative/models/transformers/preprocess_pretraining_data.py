import glob
import gzip
import json
import random
from tqdm import tqdm

for split in ['train', 'valid', 'test']:
    data_path = 'python_data/{}/*.jsonl.gz'.format(split)
    #output_file = 'python_data/{}.jsonl'.format(split)
    output_file = 'python_data/{}.txt'.format(split)

    if False:
        of = open(output_file, 'w')

        for file_name in glob.glob(data_path):
            with gzip.open(file_name, 'rb') as f:
                for line in tqdm(f, desc=file_name):
                    example = json.loads(line) 
                    code = example['code']
                    doc = example['docstring']
                    assert doc in code
                    new_c = code[:code.index(doc)] + code[code.index(doc) + len(doc):]
                    #new_c = new_c.replace('\n', '\\n')
                    #doc = doc.replace('\n', '\\n')
                    if random.random() < 0.5:
                        inp = new_c
                        out = doc
                    else:
                        inp = doc
                        out = new_c
                        
                    of.write(json.dumps(dict(inp=inp, out=out)) + '\n')

        of.close()

    if True:
        of = open(output_file, 'w')

        for file_name in glob.glob(data_path):
            with gzip.open(file_name, 'rb') as f:
                for line in tqdm(f, desc=file_name):
                    example = json.loads(line) 
                    of.write(example['original_string'].replace('\n', '\\n'))
                    of.write('\n')

        of.close()

    if False:
        out_splt = split if split != 'valid' else 'val'
        src = 'python_data/{}.source'.format(out_splt)
        tgt = 'python_data/{}.target'.format(out_splt)
        of_src = open(src, 'w')
        of_tgt = open(tgt, 'w')
        for file_name in glob.glob(data_path):
            with gzip.open(file_name, 'rb') as f:
                for line in tqdm(f, desc=file_name):
                    example = json.loads(line)
                    code = example['code']
                    doc = example['docstring']
                    assert doc in code
                    new_c = code[:code.index(doc)] + code[code.index(doc) + len(doc):]
                    new_c = new_c.replace('\n', '\\n')
                    doc = doc.replace('\n', '\\n')
                    if random.random() < 0.5:
                        of_src.write(new_c)
                        of_src.write('\n')
                        of_tgt.write(doc)
                        of_tgt.write('\n')
                    else:
                        of_src.write(doc)
                        of_src.write('\n')
                        of_tgt.write(new_c)
                        of_tgt.write('\n')
        of_src.close()
        of_tgt.close()
