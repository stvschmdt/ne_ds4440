import torch
from sklearn.metrics import (
                    classification_report, 
            accuracy_score, mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score)



# Handles forward and backward propagation for one epoch 
def train_one_epoch(model, X_train, y_train, criterion, optimizer):
    model.train()  # set model to training mode
    optimizer.zero_grad()

    # forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # backward pass and optimization
    loss.backward()
    optimizer.step()

    return loss.item()

# Evaluates a MLP model for a classification task
def classification_evaluate_model(model, X_test, y_test, criterion):
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        outputs = model(X_test)
        loss = criterion(outputs, y_test)
        # convert probabilities to binary predictions
        preds = (outputs >= 0.5).float()
        # calculate metrics
        accuracy = accuracy_score(y_test.numpy(), preds.numpy())
        precision = precision_score(y_test.numpy(), preds.numpy())
        recall = recall_score(y_test.numpy(), preds.numpy())
        f1 = f1_score(y_test.numpy(), preds.numpy())
    return loss.item(), accuracy, precision, recall, f1

# Trains a MLP classification model
def classification_train_model(model, X_train, y_train, X_test, y_test, epochs, criterion, optimizer):
    for epoch in range(1, epochs + 1):
        # train the model for one epoch
        train_loss = train_one_epoch(model, X_train, y_train, criterion, optimizer)

        # print weights and biases every epoch
        weights = model.fc.weight.data
        biases = model.fc.bias.data
        print(f"Epoch {epoch}/{epochs}".center(50, "-"))
        print(f"  Weights: {weights}")
        print(f"  Biases: {biases}")
        print(f"  Training Loss: {train_loss:.4f}\n")
        if epoch % 5 == 0:
            test_loss, accuracy, precision, recall, f1 = classification_evaluate_model(model, X_test, y_test, criterion)

            print(f"  Test Loss       : {test_loss:.4f}\n")

        # print evaluation metrics
            print(f"  Evaluation Metrics".center(50, "-"))
            print(f"  Test Loss       : {test_loss:.4f}")
            print(f"  Accuracy        : {accuracy:.4f}")
            print(f"  Precision       : {precision:.4f}")
            print(f"  Recall          : {recall:.4f}")
            print(f"  F1 Score        : {f1:.4f}\n")
            print("-" * 50)

# Evaluates a MLP model for a regression task
def regression_evaluate_model(model, X_test, y_test, criterion):
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        outputs = model(X_test)

        loss = criterion(outputs, y_test)

        preds = outputs.detach().numpy()
        actuals = y_test.numpy()

        mse = mean_squared_error(actuals, preds)
        mae = mean_absolute_error(actuals, preds)
        r2 = r2_score(actuals, preds)

    return loss.item(), mse, mae, r2

# Trains a MLP regression model
def regression_train_model(model, X_train, y_train, X_test, y_test, epochs, criterion, optimizer):
    for epoch in range(1, epochs + 1):
        # train the model for one epoch
        train_loss = train_one_epoch(model, X_train, y_train, criterion, optimizer)

        # print weights and biases every epoch
        weights = model.fc.weight.data
        biases = model.fc.bias.data


        print(f"Epoch {epoch}/{epochs}".center(50, "-"))
        print(f"  Weights: {weights}")
        print(f"  Biases: {biases}")
        print(f"  Training Loss: {train_loss:.4f}")

        # evaluate and print metrics every 5 epochs
        if epoch % 5 == 0:
            test_loss, mse, mae, r2 = regression_evaluate_model(
                model, X_test, y_test, criterion
            )
            print(f"  Testing Loss: {test_loss:.4f}")
            # print evaluation metrics
            print(f"  Evaluation Metrics".center(50, "-"))
            print(f"  Test Loss       : {test_loss:.4f}")
            print(f"  MSE        : {mse:.4f}")
            print(f"  MAE       : {mae:.4f}")
            print(f"  r2          : {r2:.4f}")
            print("-" * 50)