from random import randint
srcs = ['de', 'en', 'en', 'zh']
tgts = ['en', 'de', 'zh', 'en']
data_dict = {}
for src, tgt in zip(srcs, tgts):
    key  = f'{src}-{tgt}'
    data_dict[key] = [(src.strip(), tgt.strip()) for src, tgt in  zip(open(f'newstest22.{src}-{tgt}.{src}').readlines(),open(f'newstest22.{src}-{tgt}.{tgt}').readlines())]
    index = 0
    total_len = len(data_dict[key])
    with open(f'newstest22.concat.{key}.{src}', 'w') as src_f, open(f'newstest22.concat.{key}.{tgt}', 'w') as tgt_f:
        while index<total_len:
            window_size = randint(3,5)
            item = data_dict[key][index:index+window_size]
            index += window_size
            src_f.write(' '.join([i[0] for i in item])+'\n')
            tgt_f.write(' '.join([i[1] for i in item])+'\n')
