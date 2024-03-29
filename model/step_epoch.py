import torch
import wandb

def train(
    model: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    loss_function: torch.nn.functional,
    epoch_nb_epoch: list,
    log_interval: float,
):
    print(f"\n{'-'*10}Epoch: {epoch_nb_epoch[0]}/{epoch_nb_epoch[1]}{'-'*10}")
    log_interval_value = int(log_interval * len(train_loader))\
        if int(log_interval * len(train_loader)) else 1
    
    for i_batch, (data) in enumerate(train_loader):
        model.train()
        data = *[x.to(device) for x in data[:-1]], data[-1].to(device)
        output = model(*data[:-1])
        loss = loss_function(output, data[-1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #torch.cuda.empty_cache()
        #torch.cuda.synchronize()
        if (i_batch+1) % log_interval_value == 0\
            or i_batch+1 == len(train_loader):
            #or i_batch+1 == 1\
            eval(
                model,
                device,
                train_loader,
                valid_loader,
                i_batch,
                scheduler,
                loss_function,
                epoch_nb_epoch,
            )
        else:
            print("{}Batch: {}/{} | Train Batch (Accuracy/Loss): {:.4f}/{:.4f}{}".format(
                ' '*5,
                i_batch+1,
                len(train_loader),
                torch.mean((output == data[-1]).float()).item(),
                loss_function(output, data[-1], reduction='mean').item(),
                ' '*10
            ))

def get_model_info(
    model: torch.nn.Module,
    device: torch.device,
    loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.functional,
):
    accuracy, loss = [], []
    for i_batch, (data) in enumerate(loader):
        print(f"{' '*5}RUNNING EVALUATION: {i_batch}/{len(loader)}{' '*20}", end="\r")
        data = *[x.to(device) for x in data[:-1]], data[-1].to(device)
        output = model(*data[:-1])
        
        accuracy.append(
            torch.mean(
                (output == data[-1]).float()
            ).item()
        )
        loss.append(
            loss_function(output, data[-1], reduction='mean').item()
        )
        
    return sum(accuracy)/float(len(accuracy)), sum(loss)/float(len(loss))
     
def eval(
    model: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    i_batch: int,
    scheduler: torch.optim.lr_scheduler,
    loss_function: torch.nn.functional,
    epoch_nb_epoch: list,
):
    model.eval()
    with torch.no_grad():
        train_accuracy, train_loss = get_model_info(
            model,
            device,
            train_loader,
            loss_function,
        )
        valid_accuracy, valid_loss = get_model_info(
            model,
            device,
            valid_loader,
            loss_function,
        ) if valid_loader else (None, None)
    scheduler.step(valid_accuracy) if scheduler else None
    
    wandb.log({
        "epoch": epoch_nb_epoch[0] + (i_batch+1)/len(train_loader),
        "train_accuracy": train_accuracy,
        "train_loss": train_loss,
        "valid_accuracy": valid_accuracy,
        "valid_loss": valid_loss,
    }) if wandb.run else None
    
    print("{}Batch: {}/{} | Train (Accuracy/Loss): {:.4f}/{:.4f} | Valid (Accuracy/Loss): {:.4f}/{:.4f}{}".format(
        ' '*5,
        i_batch+1, len(train_loader),
        train_accuracy, train_loss,
        valid_accuracy if valid_loader else 0, valid_loss if valid_loader else 0,
        " "*10,
    ))