import glob
import os
import random
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from inception_utils import *
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, roc_curve)
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import models, transforms

#SETUP SEED FOR TRAINING REPLICATION
setup_seed(0)

#VERIFY CUDA AVAILABLE AND SELECTED 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/10243,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/10243,1), 'GB')
# Verify the number of available GPUs
num_gpus = torch.cuda.device_count()
print('Number of available GPUs:', num_gpus)
# Loading and processing data
if device.type == 'cuda':
    num_workers = 8
else:
    num_workers = 0
## Load all datapoints
data = []

for filename in glob.glob('/gpfs/scratch/jdl624/IHC_labels/*.csv'):
    data_frame = pd.read_csv(filename)
    case_labels = data_frame['filenames'].apply(lambda x: x.split('/')[0])
    positive_points_dir = f'/gpfs/scratch/jdl624/positive_points/{case_labels[0]}'
    img_names = data_frame['filenames'].apply(lambda x: x.split('/')[-1])
    WSI_names = glob.glob(f'/gpfs/data/razavianlab/data/hist/nyu/IHC_tiles/{case_labels[0]}/*')
    for name in WSI_names:
        wsi_name = os.path.basename(name)
        image_files = glob.glob(f'/gpfs/data/razavianlab/data/hist/nyu/IHC_tiles/{case_labels[0]}/{wsi_name}/*.jpg')
        positive_points_files = glob.glob(f'{positive_points_dir}{case_labels[0]}/*-points.tsv')
        filtered_files = [file for file, case, img_name in zip(image_files, case_labels, img_names) if f"{wsi_name}-points.tsv" not in os.listdir(positive_points_dir)]
        data.extend([(file, label) for file, label, case, img_name in zip(filtered_files, data_frame['labels'], case_labels, img_names)])

data_all = pd.DataFrame(data, columns=['FILENAME', 'LABEL'])
data_all['LABEL'] = data_all['LABEL'].replace(['Negative', 'Positive'], [0, 1])

# Create list of cases that the outer loop will iterate through
image_cases = set(np.unique([x.split('/')[-3] for x in data_all['FILENAME']]))

# Preparing for model training

#$ Load and initialize model and feature extractor
#$ Define hyperparameters
lr = 3e-5
batch_size = 256
weight_decay = 0.001
num_epochs = 15

# Perform nested cross-validation
outer_folds = 5
inner_folds = 3
outer_fold_accuracies = []
outer_fold_auc_scores = []
outer_models = []

for fold in range(outer_folds):
    fold = fold+1
    print(f"Outer Fold {fold}:")
    
    # Randomly select test and validation cases
    outerloop_cases = image_cases
    test_case = random.sample(outerloop_cases, 1)[0]
    outerloop_cases.discard(test_case)
    print(f"Test Case: {test_case}")
    
    test_data = data_all[data_all['FILENAME'].str.startswith(f'/gpfs/data/razavianlab/data/hist/nyu/IHC_tiles/{test_case}/')].reset_index(drop=True)
    ## Create test DataLoader
    test_data = ImageDataset(test_data, transform=
                                       transforms.Compose([transforms.ToPILImage(),
                                                           transforms.ToTensor(),
                                                           transforms.Resize(299)]))
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers = num_workers)
    
    print("Test samples:", len(test_data))
    
    inner_models = []
    
    for inner_fold in range(inner_folds):
        inner_fold = inner_fold+1
        
        validation_case = random.sample(outerloop_cases, 1)[0]
        print(f"Validation Case: {validation_case}")
        
        # Set the training, validation, and test data using pandas operations
        train_data = data_all.loc[~data_all['FILENAME'].str.startswith((f'/gpfs/data/razavianlab/data/hist/nyu/IHC_tiles/{test_case}/', f'/gpfs/data/razavianlab/data/hist/nyu/IHC_tiles/{validation_case}/'))].reset_index(drop=True)

        validation_data = data_all[data_all['FILENAME'].str.startswith(
            f'/gpfs/data/razavianlab/data/hist/nyu/IHC_tiles/{validation_case}/')].reset_index(drop=True)


        # Print the number of samples in each set for verification (optional)
        print("Train samples:", len(train_data))
        print("Validation samples:", len(validation_data))

        ## Create train DataLoader
        training_data = ImageDataset(train_data, transform=transforms.Compose([transforms.ToPILImage(),
                                                                               transforms.ToTensor(),
                                                                               transforms.Resize(299), 
                                                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
                                                                               transforms.RandomRotation(15),
                                                                               transforms.RandomHorizontalFlip()]))
        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers = num_workers)

        ## Create validation Dataloader
        validation_data = ImageDataset(validation_data, transform=
                                       transforms.Compose([transforms.ToPILImage(),
                                                           transforms.ToTensor(),
                                                           transforms.Resize(299)]))
        valid_loader = DataLoader(validation_data, batch_size=batch_size, num_workers = num_workers)

        #LOAD AND INITIALIZE MODEL
        model = models.inception_v3(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = DataParallel(model)
        model.to(device)

        ## Define the optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        since = time.time()
        acc_dict = {'train': [], 'validation': []}
        loss_dict = {'train': [], 'validation': []}
        auc_dict = {'train':[], 'validation':[]}
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_targets = []
            train_predictions = []
            train_proba = []

            for data in train_loader:
                optimizer.zero_grad()
                image = data[0].to(device)
                label = data[1].to(device)

                output = model(image.float())
                if hasattr(output, 'logits'):
                    output = output.logits
                loss = criterion(output, label)

                train_loss += loss.item() * image.size(0)
                _, predicted_labels = torch.max(output, 1)
                p = f.softmax(output, dim=1)
                train_proba.extend(p[:,1].cpu().detach().cpu().numpy())
                train_predictions.extend(predicted_labels.cpu().tolist())
                train_targets.extend(label.cpu().tolist())
                num_imgs = image.size()[0]
                loss.backward()
                optimizer.step()

            train_loss /= len(train_loader.dataset)
            train_accuracy = accuracy_score(train_targets, train_predictions)
            train_auc = roc_auc_score(train_targets, train_proba)

            acc_dict['train'].append(train_accuracy)
            loss_dict['train'].append(train_loss)
            auc_dict['train'].append(train_auc)

            # Evaluate on the validation set
            model.eval()
            valid_loss = 0.0
            valid_predictions = []
            valid_targets = []
            valid_proba = []

            with torch.no_grad():
                for data in valid_loader:
                    optimizer.zero_grad()
                    image = data[0].to(device)
                    label = data[1].to(device)

                    output = model(image.float())
                    if hasattr(output, 'logits'):
                        output = output.logits
                    loss = criterion(output, label)
                    valid_loss += loss.item() * image.size(0)
                    _, predicted_labels = torch.max(output, 1)
                    p = f.softmax(output, dim=1)
                    valid_proba.extend(p[:,1].detach().cpu().numpy())
                    valid_predictions.extend(predicted_labels.cpu().tolist())
                    valid_targets.extend(label.cpu().tolist())

                valid_loss /= len(valid_loader.dataset)
                valid_accuracy = accuracy_score(valid_targets, valid_predictions)
                valid_auc = roc_auc_score(valid_targets, valid_proba)

                scheduler.step(valid_loss)

                acc_dict['validation'].append(valid_accuracy)
                loss_dict['validation'].append(valid_loss)
                auc_dict['validation'].append(valid_auc)

                print(f"Epoch[{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Train Accuracy: {train_accuracy:.4f} Train AUC: {train_auc:.4f}")
                print(f"Epoch [{epoch+1}/{num_epochs}] Valid Loss: {valid_loss:.4f} Valid Accuracy: {valid_accuracy:.4f} Valid AUC: {valid_auc:.4f}")
                
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        #SAVE RESULTS IN CSVS
        loss_df = pd.DataFrame.from_dict(loss_dict, orient='index')
        acc_df = pd.DataFrame.from_dict(acc_dict, orient='index')
        auc_df = pd.DataFrame.from_dict(auc_dict, orient='index')
        loss_df.to_csv(f'/gpfs/scratch/jdl624/EGFR_Mutation_Results/Inception_V3/Inception_V3_train&valid_loss_df_NCV_parallel_outerloop{fold}_innerfold{inner_fold}.csv', index=False)
        acc_df.to_csv(f'/gpfs/scratch/jdl624/EGFR_Mutation_Results/Inception_V3/Inception_V3_train&valid_acc_df_NCV_parallel_outerloop{fold}_innerfold{inner_fold}.csv', index=False)
        auc_df.to_csv(f'/gpfs/scratch/jdl624/EGFR_Mutation_Results/Inception_V3/Inception_V3_train&valid_auc_df_NCV_parallel_outerloop{fold}_innerfold{inner_fold}.csv', index=False)

        #PLOTTING LOSS AND ACCURACY
        fig, axs = plt.subplots(1,3,figsize=(16,8))
        axs[0].plot(loss_dict['train'], color='blue', label='train loss')
        axs[0].plot(loss_dict['validation'], color='red', label='validataion loss')
        axs[1].plot(acc_dict['train'], color = 'blue', label='train accuracy')
        axs[1].plot(acc_dict['validation'], color = 'red', label='valid accuracy')
        axs[2].plot(auc_dict['train'], color = 'blue', label='train AUC')
        axs[2].plot(auc_dict['validation'], color = 'red', label='valid AUC')
        axs[0].set_title("Epoch Loss")
        axs[1].set_title("Epoch Accuracy ")
        axs[2].set_title("Epoch AUC ")
        axs[0].legend(loc='center left')
        axs[1].legend(loc='center left')
        axs[2].legend(loc='center left')
        plt.savefig(f"/gpfs/scratch/jdl624/EGFR_Mutation_Results/Inception_V3/Inception_V3_TRAIN_METRICS_NCV_parallel_outerloop{fold}_innerfold{inner_fold}.png")
        plt.close()
        
        #SAVE VERSION OF INNER TRAINED MODEL
        inner_models.append((model.state_dict(), valid_auc))
       
        print(f'Training metrics saved for outerfold{fold}_innerfold{inner_fold}')

    # Evaluate on the test set
    
    best_inner_model = max(inner_models, key=lambda x: x[1])  # Select the model with the highest validation AUC
    model_state, best_val_metrics = best_inner_model
    
    model = models.inception_v3(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = DataParallel(model)
    model.to(device)
    model.load_state_dict(model_state)
    
    model.eval()
    test_loss = 0.0
    test_predictions = []
    test_targets = []
    test_proba = []
    
    with torch.no_grad():
        for data in test_loader:
            image = data[0].to(device)
            label = data[1].to(device)

            output = model(image.float())
            if hasattr(output, 'logits'):
                output = output.logits
            loss = criterion(output, label)
            _, predicted_labels = torch.max(output, 1)
            p = f.softmax(output, dim=1)
            
            test_proba.extend(p[:,1].cpu().detach().cpu().numpy())
            test_predictions.extend(predicted_labels.cpu().tolist())
            test_targets.extend(label.cpu().tolist())

    test_accuracy = accuracy_score(test_targets, test_predictions)
    outer_fold_accuracies.append(test_accuracy)
    test_loss += loss.item() * image.size(0)
    test_auc = roc_auc_score(test_targets, test_proba)
    
    test_results = {'loss': test_loss, 'accuracy': test_accuracy, 'auc': test_auc}
    test_results = pd.DataFrame.from_dict(test_results, orient='index')
    test_results.to_csv(f'/gpfs/scratch/jdl624/EGFR_Mutation_Results/Inception_V3/InceptionV3_testresults_outerloop{fold}.csv', index=True)
                                
    fpr, tpr, _ = roc_curve(test_targets,  test_proba)
    outer_fold_auc_scores.append(test_auc)

    plt.plot(fpr,tpr, label=f'Loop {fold}, Test Case {test_case} AUC=' + str(test_auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot([0,1],[0,1],"k--")
    plt.legend(loc=4)
    plt.savefig(f'/gpfs/scratch/jdl624/EGFR_Mutation_Results/Inception_V3/Inception_V3_ROC_CURVE_parallel_outerloop_{fold}.png')
    plt.show()

    #CLASSIFICATION REPORT CREATE AND SAVE
    
        #CREATE AND SAVE CONFUSION MATRIX
    def plot_confusion_matrix(cm, classes,
                              normalize=True,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f"/gpfs/scratch/jdl624/EGFR_Mutation_Results/Inception_V3/Inception_V3_CONFUSION_MATRIX_parallel_outerloop{fold}.png")
        plt.close()

    def cm(y_true, y_pred):
        cnf_matrix = confusion_matrix(test_targets, test_predictions)
        np.set_printoptions(precision=2)
        plt.figure()
        class_names = ['0: Negative for Mutation','1: Positive for Mutation']
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix')
    
    class_names = ['0: Negative for Mutation','1: Positive for Mutation']
    cm(test_targets, test_predictions)
    classificationreport = classification_report(test_targets, test_predictions, target_names = class_names, output_dict=True)
    print(classificationreport)
    df = pd.DataFrame(classificationreport).transpose()
    df.to_csv(f'/gpfs/scratch/jdl624/EGFR_Mutation_Results/Inception_V3/Inception_V3_classification_report_parallel_outerloop{fold}.csv', index=True)
    print(f"Test Accuracy: {test_accuracy:.4f}\n")

    outer_models.append((model.state_dict(), test_auc))

# Print the average accuracy across outer folds
average_accuracy = sum(outer_fold_accuracies) / len(outer_fold_accuracies)
stdev_accuracy = statistics.stdev(outer_fold_accuracies)
average_auc_score = sum(outer_fold_auc_scores) / len(outer_fold_auc_scores)
stdev_auc_score = statistics.stdev(outer_fold_auc_scores)
print(f"Average Accuracy: {average_accuracy}", f"Average AUC: {average_auc_score}")
print(f"StDev Accuracy: {stdev_accuracy}", f"StDev AUC: {stdev_auc_score}")   

best_model = max(outer_models, key=lambda x: x[1])  # Select the model with the highest testing AUC
model_state, best_test_metrics = best_model
best_acc, best_auc = best_test_metrics
torch.save(model_state, '/gpfs/scratch/jdl624/EGFR_Mutation_Results/Inception_V3/InceptionV3_Models/best_egfr_inceptionv3.pt')
print(f'Complete, congratulations! The best model accuracy is {best_acc} and best AUC is {best_auc}. Best model saved as best_egfr_inceptionv3.pt')


                                                        