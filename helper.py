import torch
from sklearn.metrics import (
                    classification_report, 
            accuracy_score, mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt


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
    train_losses = []
    test_losses = []
    for epoch in range(1, epochs + 1):
        # train the model for one epoch
        train_loss = train_one_epoch(model, X_train, y_train, criterion, optimizer)
        train_losses.append(train_loss)
        # print weights and biases every epoch
        weights = model.fc.weight.data
        biases = model.fc.bias.data
        print(f"Epoch {epoch}/{epochs}".center(50, "-"))
        print(f"  Weights: {weights}")
        print(f"  Biases: {biases}")
        print(f"  Training Loss: {train_loss:.4f}\n")
        if epoch % 5 == 0:
            test_loss, accuracy, precision, recall, f1 = classification_evaluate_model(model, X_test, y_test, criterion)
            test_losses.append(test_loss)
            print(f"  Test Loss       : {test_loss:.4f}\n")

        # print evaluation metrics
            print(f"  Evaluation Metrics".center(50, "-"))
            print(f"  Test Loss       : {test_loss:.4f}")
            print(f"  Accuracy        : {accuracy:.4f}")
            print(f"  Precision       : {precision:.4f}")
            print(f"  Recall          : {recall:.4f}")
            print(f"  F1 Score        : {f1:.4f}\n")
            print("-" * 50)
    return train_losses, test_losses

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
    train_losses = []
    test_losses = []
    for epoch in range(1, epochs + 1):
        # train the model for one epoch
        train_loss = train_one_epoch(model, X_train, y_train, criterion, optimizer)
        train_losses.append(train_loss)
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
            test_losses.append(test_loss)
            print(f"  Testing Loss: {test_loss:.4f}")
            # print evaluation metrics
            print(f"  Evaluation Metrics".center(50, "-"))
            print(f"  Test Loss       : {test_loss:.4f}")
            print(f"  MSE        : {mse:.4f}")
            print(f"  MAE       : {mae:.4f}")
            print(f"  r2          : {r2:.4f}")
            print("-" * 50)
    return train_losses, test_losses

# Evaluastes the mode and finds correct and incorrect predictions of Fashion MNIST images
def evaluate_and_show_examples(model, test_loader, device='cuda'):
    """
    Evaluate model and show one correct and one incorrect prediction.
    """
    class_labels = {
        0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
        5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"
    }
    
    model.eval()
    correct_example = None
    incorrect_example = None
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Find one correct and one incorrect example
            for i in range(len(labels)):
                if correct_example is None and predicted[i] == labels[i]:
                    correct_example = (images[i].cpu(), labels[i].item(), predicted[i].item())
                if incorrect_example is None and predicted[i] != labels[i]:
                    incorrect_example = (images[i].cpu(), labels[i].item(), predicted[i].item())
                if correct_example and incorrect_example:
                    break
            if correct_example and incorrect_example:
                break
    
    # Plot the examples
    plt.figure(figsize=(10, 5))
    
    # Plot correct prediction
    plt.subplot(1, 2, 1)
    image, true_label, pred_label = correct_example
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(f'Correct Prediction\nTrue: {class_labels[true_label]}')
    
    # Plot incorrect prediction
    plt.subplot(1, 2, 2)
    image, true_label, pred_label = incorrect_example
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(f'Incorrect Prediction\nTrue: {class_labels[true_label]}\nPredicted: {class_labels[pred_label]}')
    
    plt.show()

# Analyzes predictions of FashionMNIST MLP
def analyze_predictions(X_test, y_test, y_pred):
    # Convert df_analyse to DataFrame if it's a NumPy array or PyTorch tensor
    df_analyse = X_test.copy()

    df_analyse['true_label'] = y_test
    df_analyse['predicted_label'] = y_pred

    # Identify correct and incorrect predictions
    correct_preds = df_analyse[df_analyse['true_label'] == df_analyse['predicted_label']]
    incorrect_preds = df_analyse[df_analyse['true_label'] != df_analyse['predicted_label']]

    # Sample examples for correct and incorrect predictions
    correct_examples = correct_preds.sample(n=min(3, len(correct_preds)), random_state=random_state)
    incorrect_examples = incorrect_preds.sample(n=min(3, len(incorrect_preds)), random_state=random_state)

    print("\nCorrect Predictions:")
    display(correct_examples)

    print("\nIncorrect Predictions:")
    display(incorrect_examples)

    print("-"*40)

    return [correct_examples, incorrect_examples]
# Helps visualize correct and incorrect examples of FashionMNIST MLP 
def visualize_predictions(correct_examples, incorrect_examples):
    """
    Visualize the predictions made by the model.
    - First row: Correct predictions.
    - Second row: Incorrect predictions.
    """

    # Combine correct and incorrect examples into a single DataFrame.
    all_samples = pd.concat([correct_examples, incorrect_examples]).reset_index(drop=True)

    # Check if there are samples to display.
    if all_samples.empty:
        print("No samples to display.")
        return

    # Set up a grid for displaying images (2 rows, 3 columns -> up to 6 images).
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # Flatten the grid of axes into a list for easier access.
    axes_flat = axes.ravel()

    # Loop through the first 6 examples in the DataFrame.
    for i in range(min(6, len(all_samples))):  # Make sure we don't try to plot more than we have.
        row = all_samples.iloc[i]  # Get the current row.

        # Extract the image pixel values and reshape them into a 28x28 grid.
        pixel_values = row.iloc[:784].values  # Assume first 784 columns are pixel values.
        img_2d = pixel_values.reshape(28, 28)

        # Get the true label and predicted label.
        true_lbl = row['true_label']
        pred_lbl = row['predicted_label']

        # Display the image and add labels.
        ax = axes_flat[i]  # Get the current subplot.
        ax.imshow(img_2d, cmap='gray')  # Display the image in grayscale.
        ax.set_title(
            f"True: {true_lbl} ({fashion_mnist_classes[true_lbl]})\n"
            f"Predicted: {pred_lbl} ({fashion_mnist_classes[pred_lbl]})"
        )  # Show the true and predicted labels.
        ax.axis('off')  # Hide the axes for a cleaner look.

    # Turn off any unused plots if there are fewer than 6 images.
    for j in range(len(all_samples), 6):
        axes_flat[j].axis('off')

    # Adjust layout and display the plot.
    plt.tight_layout()
    plt.show()

# Helper function to run inference on a single row of data
def run_inference_on_row(row, label, model):
    # Convert the row into a tensor
    row_tensor = torch.tensor(row.values, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        logits = model(row_tensor)  # Raw logits
        probabilities = torch.softmax(logits, dim=1)  # Apply softmax to convert logits to probabilities

    # Get predicted label
    predicted_label = torch.argmax(probabilities).item()

    # Print details
    print("Logits:", logits.numpy())
    print("Softmax Probabilities:", probabilities.numpy())
    print("Correct Label:", label)
    print("Predicted Label:", predicted_label)

    return predicted_label
