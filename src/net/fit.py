from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import sys
from net.fit_functions import *
from functools import partial


def test(model, loss_function, loader, device, metric_fn):
    model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    
    running_loss = []
    running_acc = []
    
    with torch.no_grad():

        for i, data in enumerate(loader):
            
            images, labels=data
        
            images = images.to(device)
            labels = labels.to(device)
            
            
            pred = model(images)
            loss = loss_function(pred, labels)
            score = metric_fn(pred, labels).mean()
            
            
        
            running_loss.append(loss.item())
            running_acc.append(score.item())
            
        
       
        
    
    
    

        
    
    model.train()
    return np.mean(running_loss), np.mean(running_acc)


def train(model, loss_function, optimizer, loader, device, epoch, metric_fn):
    model.train()
    loss_avg = 0.
    score_count = 0.
    total = 0.
    
    progress_bar = tqdm(loader)
    for i, data in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))
        
        
        images, labels=data
        images = images.to(device)
        labels = labels.to(device)
        
        model.zero_grad()
        
        pred= model(images)
            
        loss = loss_function(pred, labels)
        score = metric_fn(pred, labels)
        total += labels.size(0)
        score_count += score.sum()
        accuracy = score_count.double() / total
        

        
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        
        loss_avg +=loss.item()
       
        

        progress_bar.set_postfix(
            loss='%.3f' % (loss_avg / (i + 1)),
            score='%.3f' % accuracy)
    return loss_avg / (i + 1), accuracy


def perform_training(epochs, model, train_loader, test_loader, optimizer, loss_function,
                    device, checkpoint, metric_fn, csv_logger):
    
    train_loss = []
    train_acc  = []
    test_loss  = []
    test_acc   = []
    model = model.to(device)
    
    
    for epoch in range(epochs):
        loss_tra, score_tra = train(model, loss_function, optimizer, train_loader, device, epoch, metric_fn)
        loss_tra, score_tra  = test(model, loss_function,train_loader, device, metric_fn)
        train_loss.append(loss_tra)
        train_acc.append(score_tra)
        
        loss_test, score_test  = test(model, loss_function, test_loader, device, metric_fn)
        test_loss.append(loss_test)
        test_acc.append(score_test)
        
        tqdm.write('test_loss: %.3f, test_score: %.4f' % (loss_test, score_test))
        states = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict()
                }

        row = {'epoch': str(epoch), 'train_loss': str(loss_tra), 'test_loss': str(loss_test),
                'train_acc': str(score_tra), 'test_acc': str(score_test)}
        
        csv_logger.writerow(row)
        if checkpoint.early_stopping(score_test, states) and (epoch+1)>=int(epochs*0.5):
            tqdm.write("Early stopping with {:.3f} best score, the model did not improve after {} iterations".format(
                    checkpoint.best, checkpoint.num_bad_epochs))
            break

        
    csv_logger.close()
    
    return train_loss, train_acc,  test_loss, test_acc



def get_prediction(model, dataloader, checkpoint, device, num_class):
    

    model=checkpoint.load_saved_model(model)
    model.eval()
    num_elements = dataloader.len if hasattr(dataloader, 'len') else len(dataloader.dataset)
    batch_size   = dataloader.batchsize if hasattr(dataloader, 'len') else dataloader.batch_size
    num_batches = len(dataloader)

    predictions = torch.zeros(num_elements, num_class)
    correct_labels = torch.zeros(num_elements, num_class)

    values = range(num_batches)
    
    with tqdm(total=len(values), file=sys.stdout) as pbar:
    
        with torch.no_grad():
            for i, data in enumerate(dataloader):
            
                start = i*batch_size
                end = start + batch_size

                if i == num_batches - 1:
                    end = num_elements
                
                
                inputs, labels = data
               
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                out = model(inputs)
                    
                
                pred = torch.softmax(out, 1)
                prob, pred = torch.max(pred.data, 1)
                
                predictions[start:end] = pred.unsqueeze(1)
                correct_labels[start:end] = labels.unsqueeze(1).long()
            
                pbar.set_description('processed: %d' % (1 + i))
                pbar.update(1)
            pbar.close()

            

    predictions = predictions.cpu().numpy()
    correct_labels = correct_labels.cpu().numpy()
    assert(num_elements == len(predictions))
    
    
    return predictions, correct_labels
