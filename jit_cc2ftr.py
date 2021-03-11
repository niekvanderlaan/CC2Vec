import argparse
import pickle
import numpy as np 
from jit_padding import padding_message, clean_and_reformat_code, padding_commit_code, mapping_dict_msg, mapping_dict_code, convert_msg_to_label
from jit_cc2ftr_train import train_model
from jit_cc2ftr_extracted import extracted_cc2ftr
from jit_utils import mini_batches, save
import torch
import os, datetime
import torch
import torch.nn as nn
from tqdm import tqdm
from jit_cc2ftr_model import HierachicalRNN
import gc

def read_args():
    parser = argparse.ArgumentParser()
    # Training our model
    parser.add_argument('-project', type=str, default='openstack', help='name of the dataset')

    # Training our model
    parser.add_argument('-train', action='store_true', help='training attention model')

    parser.add_argument('-train_data', type=str, default='./data/jit/openstack_train.pkl', help='the directory of our training data')
    parser.add_argument('-test_data', type=str, default='./data/jit/openstack_test.pkl', help='the directory of our testing data')
    parser.add_argument('-dictionary_data', type=str, default='./data/jit/openstack_dict.pkl', help='the directory of our dicitonary data')

    # Predicting our data
    parser.add_argument('-predict', action='store_true', help='extracting features')
    parser.add_argument('-predict_data', type=str, help='the directory of our extracting data')
    parser.add_argument('-name', type=str, help='name of our output file')

    # Predicting our data
    parser.add_argument('-load_model', type=str, default=None, help='loading our model')

    # Number of parameters for reformatting commits
    parser.add_argument('--msg_length', type=int, default=256, help='the length of the commit message')
    parser.add_argument('--code_file', type=int, default=2, help='the number of files in commit code')
    parser.add_argument('--code_line', type=int, default=10, help='the number of LOC in each hunk of commit code')
    parser.add_argument('--code_length', type=int, default=64, help='the length of each LOC of commit code')

    # Predicting our data
    parser.add_argument('--predict', action='store_true', help='predicting testing data')

    # Number of parameters for Attention model
    parser.add_argument('-embed_size', type=int, default=64, help='the dimension of embedding vector')
    parser.add_argument('-hidden_size', type=int, default=32, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout for training PatchNet')
    parser.add_argument('-l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=50, help='the number of epochs')    
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')    

    # CUDA
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')
    return parser

class JIT_CC2ftr():

    def __init__(self, params):
        self.params = params
        self.params.cuda = (not params.no_cuda) and torch.cuda.is_available()

        if params.train is True:
            train_data = pickle.load(open(params.train_data, 'rb'))
            train_ids, train_labels, train_messages, train_codes = train_data

            test_data = pickle.load(open(params.test_data, 'rb'))
            test_ids, test_labels, test_messages, test_codes = test_data

            self.ids = train_ids + test_ids
            self.labels = list(train_labels) + list(test_labels)
            self.msgs = train_messages + test_messages
            self.codes = train_codes + test_codes

            dictionary = pickle.load(open(params.dictionary_data, 'rb'))
            self.dict_msg, self.dict_code = dictionary
        else:
            data = pickle.load(open(params.predict_data, 'rb'))
            self.ids, self.labels, self.msgs, self.codes = data

            dictionary = pickle.load(open(params.dictionary_data, 'rb'))
            self.dict_msg, self.dict_code = dictionary

    def apply_padding(self):
        params = self.params
        self.msgs = padding_message(data=self.msgs, max_length=params.msg_length)
        self.added_code, self.removed_code = clean_and_reformat_code(self.codes)
        self.added_code = padding_commit_code(data=self.added_code, max_file=params.code_file, max_line=params.code_line,
                                         max_length=params.code_length)
        self.removed_code = padding_commit_code(data=self.removed_code, max_file=params.code_file, max_line=params.code_line,
                                           max_length=params.code_length)

    def apply_mapping(self):
        self.msgs = mapping_dict_msg(pad_msg=self.msgs, dict_msg=self.dict_msg)
        self.added_code = mapping_dict_code(pad_code=self.added_code, dict_code=self.dict_code)
        self.removed_code = mapping_dict_code(pad_code=self.removed_code, dict_code=self.dict_code)
        self.msgs = convert_msg_to_label(pad_msg=self.msgs, dict_msg=self.dict_msg)

    def _clean_unused_variables(self):
        del self.msgs
        del self.added_code
        del self.removed_code
        gc.collect()

    def train(self):
        params = self.params
        batches = mini_batches(X_added_code=self.added_code, X_removed_code=self.removed_code, Y=self.msgs,
                               mini_batch_size=params.batch_size)

        msg_labels_shape = self.msgs.shape

        self._clean_unused_variables()

        params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        params.vocab_code = len(self.dict_code)

        if len(msg_labels_shape) == 1:
            params.class_num = 1
        else:
            params.class_num = msg_labels_shape[1]

        # Device configuration
        params.device = torch.device('cuda' if params.cuda else 'cpu')
        model = HierachicalRNN(args=params)
        if params.cuda:
            model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
        criterion = nn.BCEWithLogitsLoss()

        batches = batches[:400]
        params.num_epochs = 10
        for epoch in range(1, params.num_epochs + 1):
            total_loss = 0
            for i, (batch) in enumerate(tqdm(batches)):
                # reset the hidden state of hierarchical attention model
                state_word = model.init_hidden_word()
                state_sent = model.init_hidden_sent()
                state_hunk = model.init_hidden_hunk()

                pad_added_code, pad_removed_code, labels = batch
                labels = torch.FloatTensor(labels)
                optimizer.zero_grad()
                predict = model.forward(pad_added_code, pad_removed_code, state_hunk, state_sent, state_word)
                loss = criterion(predict, labels)
                loss.backward()
                total_loss += loss.detach()
                optimizer.step()

            print('Training: Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))
            save(model, params.save_dir, 'epoch', epoch)

    def predict(self):
        params = self.params
        batches = mini_batches(X_added_code=self.added_code, X_removed_code=self.removed_code, Y=self.msgs,
                               mini_batch_size=params.batch_size, shuffled=False)

        msg_labels_shape = self.msgs.shape
        params.vocab_code = len(self.dict_code)
        if len(msg_labels_shape) == 1:
            params.class_num = 1
        else:
            params.class_num = msg_labels_shape[1]

        model = HierachicalRNN(args=params)
        model.load_state_dict(torch.load(params.load_model))
        if params.cuda:
            model = model.cuda()

        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        commit_ftrs = list()
        with torch.no_grad():
            for i, (batch) in enumerate(tqdm(batches)):
                state_word = model.init_hidden_word()
                state_sent = model.init_hidden_sent()
                state_hunk = model.init_hidden_hunk()

                pad_added_code, pad_removed_code, labels = batch
                labels = torch.FloatTensor(labels)
                commit_ftr = model.forward_commit_embeds_diff(pad_added_code, pad_removed_code, state_hunk, state_sent,
                                                              state_word)
                commit_ftrs.append(commit_ftr)
            commit_ftrs = torch.cat(commit_ftrs).cpu().detach().numpy()
        pickle.dump(commit_ftrs, open(params.name, 'wb'))

if __name__ == '__main__':
    params = read_args().parse_args()
    cc2ftr = JIT_CC2ftr(params)

    cc2ftr.apply_padding()
    cc2ftr.apply_mapping()
    if params.train is True:
        cc2ftr.train()
    
    elif params.predict is True:
        data = pickle.load(open(params.predict_data, 'rb'))
        ids, labels, msgs, codes = data 

        dictionary = pickle.load(open(params.dictionary_data, 'rb'))   
        dict_msg, dict_code = dictionary  

        pad_msg = padding_message(data=msgs, max_length=params.msg_length)
        added_code, removed_code = clean_and_reformat_code(codes)
        pad_added_code = padding_commit_code(data=added_code, max_file=params.code_file, max_line=params.code_line, max_length=params.code_length)
        pad_removed_code = padding_commit_code(data=removed_code, max_file=params.code_file, max_line=params.code_line, max_length=params.code_length)

        pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dict_msg)
        pad_added_code = mapping_dict_code(pad_code=pad_added_code, dict_code=dict_code)
        pad_removed_code = mapping_dict_code(pad_code=pad_removed_code, dict_code=dict_code)
        pad_msg_labels = convert_msg_to_label(pad_msg=pad_msg, dict_msg=dict_msg)
        
        data = (pad_added_code, pad_removed_code, pad_msg_labels, dict_msg, dict_code)
        params.batch_size = 1
        extracted_cc2ftr(data=data, params=params)
        print('--------------------------------------------------------------------------------')
        print('--------------------------Finish the extracting process-------------------------')
        print('--------------------------------------------------------------------------------')
        exit()
