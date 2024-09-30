import os
import torch
import numpy as np
from tqdm import tqdm

from .utils import get_scores, print_scores, save_scores, timeit, save_model, get_save_model_path


def run_epoch(model, dataloader, optimizer, criterion, device, args=None):
    model.train()

    losses = []
    actual_labels = []
    predicted_labels = []

    for i, (audio, label) in enumerate(dataloader):

        audio = audio.to(device).squeeze(1)
        label = label.to(device)
 

        logits = model(audio)
        loss = criterion(logits, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

        actual_labels.extend(label.cpu().numpy())
        predicted_labels.extend(logits.argmax(axis=1).cpu().numpy())

    avg_loss = sum(losses) / len(losses)

    return avg_loss, actual_labels, predicted_labels


@timeit
def run_evaluation(model, dataloader, criterion, device):
    model.eval()

    losses = []
    actual_labels = []
    predicted_labels = []
    
    print("\n\nEvaluating the model ...")
    with torch.no_grad():
        for i, (audio, label) in enumerate(dataloader):
        # for i, (audio, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            print(f"Batch {i+1}/{len(dataloader)}")

            audio = audio.to(device).squeeze(1)
            label = label.to(device)
            
            logits = model(audio)
            loss = criterion(logits, label)

            losses.append(loss.item())

            actual_labels.extend(label.cpu().numpy())
            predicted_labels.extend(logits.argmax(axis=1).cpu().numpy())

    avg_loss = sum(losses) / len(losses)

    return avg_loss, actual_labels, predicted_labels


@timeit
def run_training(model, train_dataloader, test_dataloader, optimizer, criterion, device, epochs=50, args=None):
    
    for epoch in tqdm(range(epochs), total=epochs):

        train_loss, actual_labels, predicted_labels = run_epoch(model, train_dataloader, optimizer, criterion, device, args=args)

        if (epoch+1)%5 == 0:
            accuracy, f1_score, precision, recall =  get_scores(actual_labels, predicted_labels, args.classnames)
            print(f"\n\n-------------------------------\nTrain Evaluation (Epoch {epoch + 1}/{epochs})\n-------------------------------\n")
            print_scores(accuracy, f1_score, precision, recall, train_loss) 
            

        if (epoch+1)%args.freq_test_model == 0:
            test_loss, actual_labels, predicted_labels = run_evaluation(model, test_dataloader, criterion, device)
            accuracy, f1_score, precision, recall =  get_scores(actual_labels, predicted_labels, args.classnames)
            print(f"\n\n-------------------------------\nTest Evaluation\n-------------------------------\n")
            print_scores(accuracy, f1_score, precision, recall, test_loss)

            if (epoch == epochs-1) and args.do_logging:
                print("\n\nFinal Evaluation")
                print("Saving Results ...")
                save_scores(args.seed, epoch, accuracy, f1_score, precision, recall, test_loss, args.json_file_path)
                print("Results Saved\n\n")
    

    if args.save_model:
        save_model_path = get_save_model_path(args)
        save_model(args, model, save_model_path)
        print(f"Model saved to {save_model_path}")
        