import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for i, loss in enumerate(fold_losses):
    plt.plot(loss, label=f'Fold {i+1}')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
for i, acc in enumerate(fold_accuracies):
    plt.plot(acc, label=f'Fold {i+1}')
plt.title('Training Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
