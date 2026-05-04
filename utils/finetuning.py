import torch
from torch.utils.data import DataLoader

def train_one_epoch(model_wrapper, optimizer, data_loader, device, epoch, print_freq=10):
    model = model_wrapper.model
    model.train()
    total_loss = 0

    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        if i % print_freq == 0:
            print(f"Epoch: [{epoch}]  [{i}/{len(data_loader)}]  loss: {losses.item():.4f}")
            
    return total_loss / len(data_loader)

def finetune_model(model_wrapper, dataset, num_epochs=3, batch_size=4):
    device = model_wrapper.device
    model = model_wrapper.model
    
    # Simple dataloader
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Using SGD as standard
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    print(f"Starting finetuning for {model_wrapper.__class__.__name__} for {num_epochs} epochs")
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model_wrapper, optimizer, data_loader, device, epoch)
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")
    
    print("Finetuning completed.")
