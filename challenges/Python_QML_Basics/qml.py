import time
import os
import copy

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Pennylane
import pennylane as qml
from pennylane import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# Need to define quantum stuff outside the main function
n_qubits = 4                # Number of qubits
q_depth = 6                 # Depth of the quantum circuit (number of variational layers)
q_delta = 0.01              # Initial spread of random quantum weights

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    train_model(rank,world_size)

def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])


dev = qml.device("lightning.kokkos", wires=n_qubits)
@qml.qnode(dev, interface="torch")
def quantum_net(q_input_features, q_weights_flat):
    """
    The variational quantum circuit.
    """

    # Reshape weights
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(q_input_features)

    # Sequence of trainable variational layers
    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k])

    # Expectation values in the Z basis
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    return tuple(exp_vals)


class DressedQuantumNet(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self,device):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()
        self.pre_net = nn.Linear(512, n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth * n_qubits))
        self.post_net = nn.Linear(n_qubits, 2)

        self.device=device

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, n_qubits)
        q_out = q_out.to(self.device)
        for elem in q_in:
            q_out_elem = torch.hstack(quantum_net(elem, self.q_params)).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        # return the two-dimensional prediction from the postprocessing layer
        return self.post_net(q_out)



def train_model(rank, world_size): #model, criterion, optimizer, scheduler, num_epochs):
    torch.cuda.set_device(0) #assuming 1 gpu per MPI rank on Odo
    device = torch.cuda.current_device()
    print(f"Rank {rank} is using device {torch.cuda.current_device()}")

    step = 0.0004               # Learning rate
    batch_size = 4              # Number of samples for each training step
    num_epochs = 30              # Number of training epochs
    gamma_lr_scheduler = 0.1    # Learning rate reduction applied every 10 epochs.
    start_time = time.time()    # Start of the computation timer

    model = torchvision.models.resnet18(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    # Notice that model_hybrid.fc is the last layer of ResNet18
    model.fc = DressedQuantumNet(device=device)

    model = model.to(device)
    ddp_model = DDP(model, device_ids=[device])

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.fc.parameters(), lr=step)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma_lr_scheduler)

    data_transforms = {
        "train": transforms.Compose(
            [
                # transforms.RandomResizedCrop(224),     # uncomment for data augmentation
                # transforms.RandomHorizontalFlip(),     # uncomment for data augmentation
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # Normalize input channels using mean values and standard deviations of ImageNet.
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    data_dir = '/gpfs/wolf2/olcf/stf007/world-shared/9b8/hymenoptera_data'

    image_datasets = {
        x if x == "train" else "validation": datasets.ImageFolder(
            os.path.join(data_dir, x), data_transforms[x]
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "validation"]}
    class_names = image_datasets["train"].classes

    #splits up data across the devices (using a sampler forces us to use shuffle=False in the dataloader below)
    train_sampler = {
        x: torch.utils.data.distributed.DistributedSampler(image_datasets[x],num_replicas=world_size,rank=rank)
        for x in ["train", "validation"]
    }

    # Initialize dataloader
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, sampler=train_sampler[x])
        for x in ["train", "validation"]
    }

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10000.0  # Large arbitrary number
    best_acc_train = 0.0
    best_loss_train = 10000.0  # Large arbitrary number

    if (rank==0):
        print("Training started:")

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ["train", "validation"]:
            if phase == "train":
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_preds = 0

            # Iterate over data.
            dataset_sizes_local = len(dataloaders[phase])
            it = 0
            for inputs, labels in dataloaders[phase]:
                since_batch = time.time()
                batch_size_ = len(inputs)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # Track/compute gradient and make an optimization step only when training
                with torch.set_grad_enabled(phase == "train"):
                    outputs = ddp_model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Print iteration results
                running_loss += loss.item() * batch_size_
                batch_corrects = torch.sum(preds == labels.data).item()
                running_corrects += batch_corrects
                running_preds += len(preds)

                if (rank==0):
                    print(
                        "Phase: {} Epoch: {}/{} Iter: {}/{} Batch time: {:.4f}".format(
                            phase,
                            epoch + 1,
                            num_epochs,
                            it + 1,
                            dataset_sizes_local,
                            time.time() - since_batch,
                        ),
                        end="\r",
                        flush=True,
                    )
                it += 1

            # Print epoch results
            #epoch_loss = running_loss / running_preds
            #epoch_acc = running_corrects / running_preds

            acc_tensor = torch.tensor([running_loss,running_corrects,running_preds])
            acc_tensor = acc_tensor.to(device)
            dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)

            epoch_loss = acc_tensor[0] / acc_tensor[2]
            epoch_acc = acc_tensor[1] / acc_tensor[2]

            if(rank==0):
                print(
                    "Phase: {} Epoch: {}/{} Ave. Loss: {:.4f} Ave. Acc: {:.4f}       ".format(
                        "train" if phase == "train" else "validation  ",
                        epoch + 1,
                        num_epochs,
                        epoch_loss,
                        epoch_acc,
                    )
                )

            # Check if this is the best model wrt previous epochs
            if phase == "validation" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "validation" and epoch_loss < best_loss:
                best_loss = epoch_loss
            if phase == "train" and epoch_acc > best_acc_train:
                best_acc_train = epoch_acc
            if phase == "train" and epoch_loss < best_loss_train:
                best_loss_train = epoch_loss
      
            # Update learning rate
            if phase == "train":
                scheduler.step()

    # Print final results
    model.load_state_dict(best_model_wts)
    time_elapsed = time.time() - since

    # Sync ranks for final printing
    torch.distributed.barrier()
    time.sleep(5)

    if (rank==0):
        print(f"\n Hi from Rank {rank}")
        print(
            "\nTraining completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
        )
        print("Best average loss: {:.4f} | Best average accuracy: {:.4f}".format(best_loss, best_acc))

    return


##############################################################################
# We are ready to perform the actual training process.

if __name__ == "__main__":
    n_gpus_total = torch.cuda.device_count()

    print(f'Total GPUs on the system: {n_gpus_total}')

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    world_rank = rank = comm.Get_rank()
    backend = None
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(world_rank)
    os.environ['LOCAL_RANK'] = "0"

    master_addr = os.environ["MASTER_ADDR"]
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_SOCKET_IFNAME'] = 'hsn0'
    print(f'Total GPUs being used this run: {world_size}')
    setup(rank, world_size)
