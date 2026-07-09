import torchvision
import torchvision.datasets as datasets

print("Downloading CIFAR-10 training dataset...")
train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=None
)
print(f"Training dataset size: {len(train_dataset)}")

print("\nDownloading CIFAR-10 test dataset...")
test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=None
)
print(f"Test dataset size: {len(test_dataset)}")

print("\nCIFAR-10 dataset downloaded successfully!")