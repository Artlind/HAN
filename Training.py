"""
Training
"""
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.optim as optim
import pandas as pd
from HAN import HAN
from tqdm import tqdm
from Global_embedd import global_dict

# =============================================================================
# Hyperparameters
# =============================================================================

embedding_dim = 128
EPOCH = 15
batch_size = 16
LEARNING_RATE = 0.01

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('Initialisation of cuda')
    torch.cuda.init()
else:
    print('Mode CPU')
    DEVICE = torch.device('cpu')

path = "path_to_groundtruth.npy"
# =============================================================================
# Préparation des chemins
# =============================================================================

dict_id_number = np.load(path).item()

number_cat = max(dict_id_number.values())+1

# =============================================================================
# Répartition du dataset
# =============================================================================

eval_rate = 0.24
test_rate = 0.16
sample = None
softmax = nn.Softmax(dim=1).to(DEVICE)

# =============================================================================
# Deep learning
# =============================================================================


def learn(train_data, test_data, eval_data, dict_id_number, embedd_dict):
    """
    train_data, test_data, eval_data : dataframes with columns {url content cat}
    dict_id_number : dictionnary {id(url) : category} for groundtruth
    embedd_dict : dictionnary giving number to every word
    """
    print()
    print("Training wih ", len(train_data), " data")
    lr = LEARNING_RATE
    model = HAN(embedding_dim, len(embedd_dict), batch_size,
                number_cat, DEVICE, embedd_dict).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss().to(DEVICE)
    learning_process = False
    epoch_static = 0
    losses_train = []
    train_data_used = [[train_data['id'][i],
                        train_data['content'][i]] for i in range(len(train_data))]
    dataloader = torch.utils.data.DataLoader(train_data_used,
                                             batch_size,
                                             shuffle=True,
                                             drop_last=True)
    for epoch in tqdm(range(EPOCH)):
        if learning_process and (epoch-epoch_static)%3 == 1:
            lr = lr/2.0
            optimizer = optim.Adam(model.parameters(), lr=lr)
        for ids, contents in dataloader:
            model.zero_grad()
            v = model(list(contents))
            targets = torch.tensor([dict_id_number[id] for id in ids]).to(DEVICE)
            loss = loss_function(v, targets)
            loss.backward()
            optimizer.step()
        model.eval()
        pres, loss_train = eval(model, train_data, dict_id_number)
        acc, loss_test = eval(model, test_data, dict_id_number)
        losses_train.append(float(loss_train))
        if not learning_process:
            """After 3 epochs we decrease LR"""
            if epoch >= 3:
                if losses_train[-1] > losses_train[-2] and not learning_process:
                    learning_process = True
                    epoch_static = epoch
                    print()
                    print("Starting to decrease LR")
        model.train()
        print()
        print("Epoch number : ", epoch+1, "Accuracy on train {}%".format(round(100*pres, 2)))
        print("Accuracy : ", round(100*float(acc), 2), "%")
        print("Loss train : ", float(loss_train))
        print("Loss test : ", float(loss_test))
        print("_______________")
        torch.save(model.state_dict(), 'modeles/current_model')
    eval_acc = eval(model, eval_data, dict_id_number)[0]
    print()
    print("Accuracy on evaluation", round(100*float(eval_acc), 2), "%")
    torch.save({'model' : model.state_dict(), 'optimizer' : optimizer.state_dict()},
               '/Users/arthur/local/HAN/tweets/modeles/modele_{}data_{}%acc'.format(len(train_data),
                                                                                    round(100*float(eval_acc), 2)))
    return print("Finito")

# =============================================================================
# Eval
# =============================================================================

def eval(model, data, dict_id_number):
    """
    data  : datframe {ligne id(Nan) url content cat(number)}
    Output : tuple (acc, loss)
    Accuracy is the average success probability
    """
    loss_function = nn.CrossEntropyLoss().to(DEVICE)
    n = len(data)
    acc = 0.0
    loss = 0.0
    ids, contents = data['id'], data['content']
    with torch.no_grad():
        for i in range(n):
            model.zero_grad()
            out = model([contents[i]]).to(DEVICE)
            target = torch.tensor([dict_id_number[ids[i]]]).to(DEVICE)
            probs = softmax(out).to(DEVICE)
            acc += float(probs[0][int(target)])
            loss += float(loss_function(out, target))
    return (acc/n, loss/n)

def discret_eval(model, eval_data):
    """
    Compute the discrete accuracy counting the number of success divided by number of data
    """
    with torch.no_grad():
        good = 0.0
        for i in tqdm(range(len(eval_data))):
            id = eval_data["id"][i]
            content = eval_data["content"][i]
            true_cat = dict_id_number[id]
            cat_model = model.cat(content)
            if cat_model == true_cat:
                good += 1
            if i%(len(eval_data)//20) == 0 and i != 0:
                print(str(round(100*good/i, 2))+'%')
    acc = round(100*(good/len(eval_data)), 2)
    return "Accuracy discret = {}%".format(acc)


# =============================================================================
# Main
# =============================================================================

all_data = pd.read_pickle("dataset")
def main():
    """
    Creating data and training
    """
    n_all = len(all_data)
    n_eval = int(eval_rate*n_all)
    n_test = int(test_rate*n_eval)
    train_data = all_data[n_eval:].reset_index(drop=True)
    eval_data = all_data[:n_eval]
    test_data = eval_data[:n_test]
    embedd_dict = global_dict(all_data)
    print()
    print("Embedding dict created with {} words".format(max(embedd_dict.values())))
    return learn(train_data, test_data, eval_data, dict_id_number, embedd_dict)


if __name__ == '__main__':
    main()
