import pandas as pd
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.word.context_word_embs as nav
import random
import gc
import math
from itertools import chain
import nltk

df = pd.read_excel("train.xlsx")
li = [[0, 2], [1, 3], [2, 2], [4, 2], [5, 2]]  # input info [category = x, augmentation factor = y]
col = df.columns
count = 0
for i in li:
    cat = i[0]
    aug_fac = i[1]
    for k in range(aug_fac):
        if k == 0:
            aug = nas.AbstSummAug(max_length=1024)
        elif k == 1:
            aug = naw.SynonymAug(aug_src='wordnet')
        else:
            aug = nav.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")

        df_temp = df.loc[df[col[1]] == cat]
        all_text = df_temp[col[0]].tolist()
        augmented_text = [aug.augment(text) for text in all_text]
        # augmented_text = list(chain.from_iterable(augmented_text))
        augmented_text_dict = {col[0]: augmented_text}
        df3 = pd.DataFrame(augmented_text_dict)
        df3[col[1]] = cat
        df = pd.concat([df, df3], axis=0)
        del df3
        del augmented_text_dict
        del augmented_text
        del aug
        gc.collect()
        count = count + 1
        print(count)
df.to_excel("aug_train.xlsx", index=False)
