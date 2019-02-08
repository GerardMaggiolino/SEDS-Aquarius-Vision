import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from cnn import LanderCNN
from dataloader import create_loaders
from torchvision import transforms 
from datetime import datetime


def main(): 
    ''' 
    Calls training or classifying functions.
    '''
    train()


def train(): 
    '''
    Runs entire training procedure from start. Saves model.
    '''
    # Hyperparam
    alpha = 0.001

    epoch = 50
    batch = 16
    early_stop_ep = 8
    split = [0.7, 0.2, 0.1]
    seed = None
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # Model 
    model = LanderCNN()
    optim = torch.optim.Adam(model.parameters(), lr=alpha)
    crit = nn.BCEWithLogitsLoss()

    # Early stop 
    best_ep = 0
    t = datetime.now()
    model_save_name = f'models/{t.month}{t.day}{t.hour}{t.minute}-model.pt'
    info_file_name = 'models/model_info.txt'

    # Data
    train_set, val_set, test_set = create_loaders(split, transform, batch, seed)

    # Seed
    if seed is not None:
        np.random.seed(seed) 
        torch.manual_seed(seed)
        random.seed(seed)

    # Graphing 
    val_loss = []
    train_loss = []
    counter = 0

    # Epochs
    for ep in range(epoch): 
        # Minibatches
        val_loss.append(0)
        train_loss.append(0)
        counter = 0
        for batch_img, batch_label in train_set:     
            optim.zero_grad()
            logits = model.forward(batch_img)
            loss = crit(logits, batch_label)
            train_loss[ep] += loss.item()
            counter += 1

            loss.backward()
            optim.step()
        train_loss[ep] /= counter
        print(f'Epoch {ep}\nTrain loss:\t{train_loss[ep]}')

        # Validate
        with torch.no_grad():
            counter = 0
            for batch_img, batch_label in val_set: 
                logits = model.forward(batch_img)
                loss = crit(logits, batch_label)
                val_loss[ep] += loss.item()
                counter += 1
            val_loss[ep] /= counter
        print(f'Validation loss:\t{val_loss[ep]}')

        # Early stop 
        if val_loss[ep] <= val_loss[best_ep]: 
            torch.save(model.state_dict(), model_save_name)
            best_ep = ep 
        elif best_ep < ep - early_stop_ep: 
            model.load_state_dict(torch.load(model_save_name))
            print(f'Early stop on epoch {ep}, reloading from {best_ep}.')
            break
    print(f'Saved model on epoch {best_ep}.')


    # Run testing 
    correct, total = test(test_set, model)
    print(f'Total Correct: {correct}')
    print(f'Total Tested: {total}')
    cat_acc = [round(i / j * 100, 2) for i, j in zip(correct, total)]
    print(f'Percent Accuracy: {cat_acc}')
    print(f'Percent Accuracy: {round(sum(correct) / sum(total) * 100, 2)}')

    # Save model information 
    with open(info_file_name, 'a') as f: 
        f.write(model_save_name)
        f.write(f'\nC: {correct}\nT: {total}\n')
        f.write(f'{cat_acc}\n')
        f.write(f'{round(sum(correct) / sum(total) * 100, 3)}\n\n')


    # Display 
    eps = [i for i in range(len(train_loss))]
    plt.plot(eps, train_loss, label="Training Loss")
    plt.plot(eps, val_loss, label="Validation Loss")
    plt.legend()
    plt.show()


def test(data, model): 
    '''
    Tests data with model.

    Parameters
    ----------
    data : torch.utils.data.DataLoader
    model : torch.nn.Module

    Returns
    -------
    list, list 
    Correct, Total over each category. 
    '''
    # Test accuracy 
    model.eval()
    correct = [0, 0, 0, 0, 0]
    total = correct.copy()

    with torch.no_grad():
        # Over batches
        for batch_img, batch_label in data: 
            activation = torch.sigmoid(model.forward(batch_img))
            # Over examples
            for act, label in zip(activation, batch_label):
                target_label = model_label = 0
                # Find model label 
                for neuron in act: 
                    if neuron > 0.5: 
                        model_label += 1
                    else: 
                        break
                # Find target label 
                for ind in label: 
                    if ind == 1: 
                        target_label += 1
                    else: 
                        break
                correct[target_label] += int(target_label == model_label)
                total[target_label] += 1

    # Correct for categories with zero examples
    for ind in range(len(total)): 
        if total[ind] == 0: 
            total[ind] == float('nan')
            correct[ind] == float('nan')

    return correct, total 


if __name__ == '__main__': 
    main()
