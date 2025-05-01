import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import pandas as pd

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, 
                num_epochs=25, save_dir='weights'):
    """
    Train the model.
    
    Args:
        model (nn.Module): Model to train
        dataloaders (dict): Dictionary containing train and val dataloaders
        criterion (nn.Module): Loss function
        optimizer (torch.optim): Optimizer
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        device (torch.device): Device to train on
        num_epochs (int): Number of epochs to train for
        save_dir (str): Directory to save model weights
    
    Returns:
        dict: Dictionary containing training history and best model weights
    """
    # Create directory to save weights
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize variables
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'train_fitzpatrick_acc': {i: [] for i in range(1, 7)},
        'val_fitzpatrick_acc': {i: [] for i in range(1, 7)}
    }
    
    # Start training
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Track accuracy by Fitzpatrick skin type
            fitzpatrick_correct = {i: 0 for i in range(1, 7)}
            fitzpatrick_total = {i: 0 for i in range(1, 7)}
            
            # Iterate over data
            for batch in tqdm(dataloaders[phase], desc=f'{phase} batch'):
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)
                fitzpatrick_scales = batch['fitzpatrick_scale'].to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Track accuracy by Fitzpatrick skin type
                for i in range(1, 7):
                    mask = (fitzpatrick_scales == i)
                    if torch.sum(mask) > 0:
                        fitzpatrick_correct[i] += torch.sum((preds == labels.data) & mask).item()
                        fitzpatrick_total[i] += torch.sum(mask).item()
            
            # Calculate epoch statistics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # Calculate accuracy by Fitzpatrick skin type
            fitzpatrick_acc = {}
            for i in range(1, 7):
                if fitzpatrick_total[i] > 0:
                    fitzpatrick_acc[i] = fitzpatrick_correct[i] / fitzpatrick_total[i]
                else:
                    fitzpatrick_acc[i] = 0.0
            
            # Update history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            for i in range(1, 7):
                history[f'{phase}_fitzpatrick_acc'][i].append(fitzpatrick_acc[i])
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase} Fitzpatrick Accuracy: {fitzpatrick_acc}')
            
            # Deep copy the model if best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()
                
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_acc,
                }, os.path.join(save_dir, 'best_model.pth'))
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(history['val_loss'][-1])
            else:
                scheduler.step()
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': epoch_acc,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        print()
    
    # Calculate training time
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return {
        'model': model,
        'history': history,
        'best_acc': best_acc
    }

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): DataLoader for evaluation
        criterion (nn.Module): Loss function
        device (torch.device): Device to evaluate on
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    
    # Track predictions and ground truth for further analysis
    all_preds = []
    all_labels = []
    all_fitzpatrick = []
    
    # Track accuracy by Fitzpatrick skin type
    fitzpatrick_correct = {i: 0 for i in range(1, 7)}
    fitzpatrick_total = {i: 0 for i in range(1, 7)}
    
    # Iterate over data
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluation'):
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            fitzpatrick_scales = batch['fitzpatrick_scale'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Track predictions and ground truth
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_fitzpatrick.extend(fitzpatrick_scales.cpu().numpy())
            
            # Track accuracy by Fitzpatrick skin type
            for i in range(1, 7):
                mask = (fitzpatrick_scales == i)
                if torch.sum(mask) > 0:
                    fitzpatrick_correct[i] += torch.sum((preds == labels.data) & mask).item()
                    fitzpatrick_total[i] += torch.sum(mask).item()
    
    # Calculate evaluation metrics
    loss = running_loss / len(dataloader.dataset)
    acc = running_corrects.double() / len(dataloader.dataset)
    
    # Calculate accuracy by Fitzpatrick skin type
    fitzpatrick_acc = {}
    for i in range(1, 7):
        if fitzpatrick_total[i] > 0:
            fitzpatrick_acc[i] = fitzpatrick_correct[i] / fitzpatrick_total[i]
        else:
            fitzpatrick_acc[i] = 0.0
    
    print(f'Evaluation Loss: {loss:.4f} Acc: {acc:.4f}')
    print(f'Fitzpatrick Accuracy: {fitzpatrick_acc}')
    
    return {
        'loss': loss,
        'accuracy': acc.item(),
        'fitzpatrick_accuracy': fitzpatrick_acc,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'fitzpatrick_scales': np.array(all_fitzpatrick)
    }