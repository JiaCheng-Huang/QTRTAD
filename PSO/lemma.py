import pickle

import data_helper

word_candidate={}

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

train_texts, train_labels, test_texts, test_labels = data_helper.split_imdb_files()
train_text = [text.lower().split(' ') for text in train_texts]
NNS={}
NNPS={}
JJR={}
JJS={}
RBR={}
RBS={}
VBD={}
VBG={}
VBN={}
VBP={}
VBZ={}
inv_NNS={}
inv_NNPS={}
inv_JJR={}
inv_JJS={}
inv_RBR={}
inv_RBS={}
inv_VBD={}
inv_VBG={}
inv_VBN={}
inv_VBP={}
inv_VBZ={}
s_ls=['NNS','NNPS','JJR','JJS','RBR','RBS','VBD','VBG','VBN','VBP','VBZ']
s_noun=['NNS','NNPS']
s_verb=['VBD','VBG','VBN','VBP','VBZ']
s_adj=['JJR','JJS']
s_adv=['RBR','RBS']
f=open('PSO/pos_tags.pkl','rb')
all_pos_tag=pickle.load(f)
for idx in range(len(train_text)):
    print(idx)
    #text=train_text[idx]
    pos_tags = all_pos_tag[idx]
    for i in range(len(pos_tags)):
        pair=pos_tags[i]
        if pair[1] in s_ls:
            if pair[1][:2]=='NN':
                w=wnl.lemmatize(pair[0],pos='n')
            elif pair[1][:2]=='VB':
                w = wnl.lemmatize(pair[0], pos='v')
            elif pair[1][:2]=='JJ':
                w = wnl.lemmatize(pair[0], pos='a')
            else:
                w = wnl.lemmatize(pair[0], pos='r')
            eval('inv_'+pair[1])[w]=pair[0]
            eval(pair[1])[pair[0]]=w
f=open('PSO/sss_dict.pkl','wb')
pickle.dump((NNS,NNPS,JJR,JJS,RBR,RBS,VBD,VBG,VBN,VBP,VBZ,inv_NNS,inv_NNPS,inv_JJR,inv_JJS,inv_RBR,inv_RBS,inv_VBD,inv_VBG,inv_VBN,inv_VBP,inv_VBZ),f)
