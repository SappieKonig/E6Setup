#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import threading
import random
import numpy as np
from tqdm import tqdm

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def load_cifar10(batch_size=128):
    print("📦 Loading CIFAR-10 dataset...")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"✅ Loaded CIFAR-10: {len(trainset)} training, {len(testset)} test images")
    return trainloader, testloader

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_on_device(device_id, results):
    """Train model on a specific GPU device"""
    print(f"🚀 Starting training on GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
    
    device = torch.device(f'cuda:{device_id}')
    torch.cuda.set_device(device)
    
    # Set identical seed for all GPUs
    set_seed(42)
    
    # Load data
    trainloader, testloader = load_cifar10(batch_size=128)
    
    # Create model
    model = SimpleCNN(num_classes=10).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    epochs = 3  # Shorter for parallel testing
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(trainloader, desc=f'GPU {device_id} Epoch {epoch+1}/{epochs}', 
                         position=device_id, leave=True, ncols=80)
        
        for i, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 100 == 99:
                acc = 100. * correct / total
                train_pbar.set_postfix({'Loss': f'{running_loss/100:.3f}', 'Acc': f'{acc:.2f}%'})
                running_loss = 0.0
        
        scheduler.step()
        
        # Quick test evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f'GPU {device_id} Epoch {epoch+1}/{epochs} - Test Accuracy: {test_acc:.2f}%')
        
    elapsed_time = time.time() - start_time
    final_acc = test_acc
    
    # Save model with device ID
    # torch.save(model.state_dict(), f'cifar10_model_gpu_{device_id}.pth')
    
    results[device_id] = {
        'accuracy': final_acc,
        'time': elapsed_time,
        'device_name': torch.cuda.get_device_name(device_id)
    }
    
    print(f"✅ GPU {device_id} completed! Accuracy: {final_acc:.2f}%, Time: {elapsed_time:.1f}s")

def main():
    print("🎯 E6Setup CIFAR-10 Multi-GPU Independent Training")
    print("=" * 60)
    
    num_gpus = torch.cuda.device_count()
    print(f"🚀 Found {num_gpus} CUDA GPU(s):")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print(f"\n🔥 Running independent training on all {num_gpus} GPUs simultaneously...")
    print("Each GPU will train its own model copy for 3 epochs")
    print("🌱 Using identical seed (42) for reproducibility test")
    
    # Results dictionary shared between threads
    results = {}
    threads = []
    
    # Start training on each GPU in separate threads
    for gpu_id in range(num_gpus):
        thread = threading.Thread(target=train_on_device, args=(gpu_id, results))
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Report results
    print("\n" + "=" * 60)
    print("🎉 All GPU training completed!")
    print("\nResults:")
    for gpu_id in sorted(results.keys()):
        result = results[gpu_id]
        print(f"  GPU {gpu_id} ({result['device_name']}): {result['accuracy']:.2f}% accuracy in {result['time']:.1f}s")
    
    if results:
        avg_acc = sum(r['accuracy'] for r in results.values()) / len(results)
        total_time = max(r['time'] for r in results.values())  # Total time is the longest GPU
        print(f"\n📊 Average accuracy: {avg_acc:.2f}%")
        print(f"⏱️ Total training time: {total_time:.1f}s (parallel execution)")

if __name__ == '__main__':
    main()