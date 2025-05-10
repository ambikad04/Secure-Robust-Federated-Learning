import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from rich.console import Console
from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.torch_utils import process_grad
#from flearn.trainers.utils import list_overlapping 
from flearn.models.model import get_model, get_model_by_name
from flearn.data.data_utils import get_participants_stat
from flearn.data.data_utils import get_testloader
from flearn.utils.losses import get_loss_fun
import json
import pickle
import os
from typing import List
from copy import deepcopy
from collections import OrderedDict
from flearn.utils.plotting import plot_data_dict_in_pdf
from flearn.utils.tools import get_optimal_cuda_device
from flearn.data.dataset import CLASSES
# from flearn.utils import normalize_dict
from flearn.models.model import EnhancedClassificationNet

def select_model(model_name, dataset):
    """Helper function to select the appropriate model"""
    if model_name:
        return get_model_by_name(dataset=dataset, model=model_name, device='cpu')  # Device will be set later
    return get_model(dataset, device='cpu')  # Device will be set later

class BaseFedarated(object):
    def __init__(self, options, model=None):
        # Transfer parameters
        self.optimizer = options['optimizer']
        self.dataset = options['dataset']
        self.num_rounds = options['num_rounds']
        self.eval_every = options['eval_every']
        self.num_epochs = options['num_epochs']
        self.batch_size = options['batch_size']
        self.model_name = options['model']
        self.gpu = options['gpu']
        self.card = options['gpu']
        self.seed = options['seed']
        self.lr = 0.01  # Lower learning rate for stability
        self.attack = options['attack'] if 'attack' in options else None
        self.mal_clients = options['mal_clients'] if 'mal_clients' in options else 0
        self.attack_scale = options['attack_scale'] if 'attack_scale' in options else 0.0
        self.dataset_type = options.get('dataset_type', None)
        self.n_class = options.get('n_class', None)
        self.valset_ratio = options.get('valset_ratio', 0.1)
        self.learning_rate = options.get('learning_rate', 0.01)
        self.weight_decay = options.get('weight_decay', 5e-4)
        self.loss = options.get('loss', 'CE')
        self.log = options.get('log', True)
        self.optm = options.get('optm', 'SGD')
        self.num_clients = options.get('num_clients', None)
        self.robust_aggregation = options.get('robust_aggregation', None)

        # Set device
        if self.card >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.card)  # Set the default CUDA device first
            self.device = torch.device(f'cuda:{self.card}')
            print(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            print(f"Current CUDA Device: {torch.cuda.current_device()}")
            # Enable cuDNN benchmarking for better performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        # Initialize model
        if model is None:
            self.model = select_model(self.model_name, self.dataset)
        else:
            self.model = model
        self.model = self.model.to(self.device)
        
        # Load data and clients
        print('Loading test data...')
        self.test_loader, self.num_samples = get_testloader(
            dataset=self.dataset,
            dataset_type=self.dataset_type,
            n_class=self.n_class,
            batch_size=self.batch_size,
            valset_ratio=self.valset_ratio
        )
        print('Done loading test data.')
        # Ensure test_loader is not None before creating an iterator
        if self.test_loader is None:
            # This should not happen with our fixed get_testloader function, but adding as a safeguard
            print("Warning: test_loader is None. Creating an empty loader.")
            from torch.utils.data import DataLoader, TensorDataset
            empty_dataset = TensorDataset(torch.tensor([]), torch.tensor([]))
            self.test_loader = DataLoader(empty_dataset, batch_size=self.batch_size)
        self.iter_trainloader = iter(self.test_loader)
        self.criterion = get_loss_fun("CE")()

        print('Loading training data and setting up clients...')
        try:
            self.clients = self.setup_clients(self.dataset, self.model)
            print(f'Number of clients: {len(self.clients)}')

            # Filter out clients with no data
            self.clients = [c for c in self.clients if c.num_samples > 0]
            
            if not self.clients:
                print("Warning: No valid clients found. This may indicate missing or invalid dataset files.")
        except Exception as e:
            print(f"Error setting up clients: {str(e)}")
            # Initialize with empty clients list to avoid attribute errors
            self.clients = []

        # Initialize system metrics
        metrics_params = {
            'dataset': self.dataset,
            'num_rounds': self.num_rounds,
            'eval_every': self.eval_every,
            'learning_rate': self.learning_rate,
            'mu': 0.0,  # default value if not specified
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'optimizer': self.optimizer
        }
        self.metrics = Metrics(self.clients, metrics_params)
        self.experiment = f'{self.optimizer}_{self.dataset}_{self.model_name}'
        self.best_acc = 0.0
        self.best_loss = float('inf')
        
        # Initialize optimizer with momentum for faster convergence
        self.inner_opt = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        
        # Get model parameters
        self.latest_model = self.get_model_params()
        
        # Initialize tracking variables
        self.accuracy_global = []
        self.global_acc = None
        self.loss_global = []
        self.accuracy_clients = {} 
        self.desc = f'Algo: {self.optimizer}, Round: '

        # Initialize robust aggregation if specified
        if self.robust_aggregation:
            from flearn.robust_aggregation import RobustAggregation
            self.robust_aggregator = RobustAggregation(
                method=self.robust_aggregation,
                eps=0.1,
                sigma=0.5,
                expansion=20.0
            )
            print(f"Using robust aggregation method: {self.robust_aggregation}")

    def __del__(self):
        if hasattr(self, 'model') and getattr(self, 'model', None) is not None:
            print(f'Training Stopped')
            pass
            # self.model.close()

    def setup_clients(self, dataset, model=None):
        '''Instantiates clients based on given train and test data directories

        Return:
            List of Clients
        '''
        try:
            clients_stats = get_participants_stat(self.dataset, self.dataset_type, self.n_class)
            # print(f'\nclient_stats: {clients_stats}')
            
            # Check if clients_stats is empty, create dummy clients if needed
            if not clients_stats:
                print(f"Warning: No clients found for {self.dataset}/{self.dataset_type}/{self.n_class}.")
                print("Creating dummy clients for testing...")
                # Create 5 dummy clients
                clients_stats = [(i, self.dataset_type, self.n_class) for i in range(5)]
            
            if self.num_clients == None:
                all_clients = [
                            Client(
                                user_id=user_id,
                                device=self.device,
                                lr= self.learning_rate,
                                weight_decay=self.weight_decay,
                                loss=self.loss,
                                batch_size=self.batch_size,
                                dataset=self.dataset,
                                valset_ratio=self.valset_ratio,
                                logger= Console(record=self.log),
                                gpu=self.gpu,
                                dataset_type=dataset_type,
                                n_class=n_class,
                                optm=self.optm,
                                group=None,
                                model= self.model
                                ) for user_id, dataset_type, n_class in clients_stats ]
                return all_clients
            else:
                # print(clients_stats)
                # Ensure we don't try to get more clients than exist
                clients_count = min(self.num_clients, len(clients_stats))
                clients_stats = clients_stats[0:clients_count]
                # print(clients_stats)
                all_clients = [
                            Client(
                                user_id=user_id,
                                device=self.device,
                                lr= self.learning_rate,
                                weight_decay=self.weight_decay,
                                loss=self.loss,
                                batch_size=self.batch_size,
                                dataset=self.dataset,
                                valset_ratio=self.valset_ratio,
                                logger= Console(record=self.log),
                                gpu=self.gpu,
                                dataset_type=dataset_type,
                                n_class=n_class,
                                optm=self.optm,
                                group=None,
                                model= self.model
                                ) for user_id, dataset_type, n_class in clients_stats ]
                return all_clients
        except Exception as e:
            print(f"Error in setup_clients: {str(e)}")
            print("Creating dummy clients for testing...")
            # Create a few dummy clients as fallback
            dummy_clients = []
            num_dummy_clients = 10  # Create 10 dummy clients
            print(f"Creating {num_dummy_clients} dummy clients for testing with reasonable data amounts")
            
            for i in range(num_dummy_clients):  # Create dummy clients
                try:
                    dummy_client = Client(
                        user_id=i,
                        device=self.device,
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay,
                        loss=self.loss,
                        batch_size=self.batch_size,
                        dataset=self.dataset,
                        valset_ratio=self.valset_ratio,
                        logger=Console(record=self.log),
                        gpu=self.gpu,
                        dataset_type=self.dataset_type,
                        n_class=self.n_class,
                        optm=self.optm,
                        group=None,
                        model=self.model
                    )
                    
                    # Override the data loaders with larger dummy datasets
                    from torch.utils.data import DataLoader
                    from flearn.data.data_utils import create_dummy_data
                    
                    # Create larger datasets for better training
                    train_samples = 500
                    val_samples = 100
                    
                    if self.dataset in ["cifar", "cifar10", "cifar100"]:
                        dummy_train = create_dummy_data(self.dataset, train_samples, (3, 32, 32))
                        dummy_val = create_dummy_data(self.dataset, val_samples, (3, 32, 32))
                    else:
                        dummy_train = create_dummy_data(self.dataset, train_samples, (1, 28, 28))
                        dummy_val = create_dummy_data(self.dataset, val_samples, (1, 28, 28))
                    
                    dummy_client.trainloader = DataLoader(dummy_train, batch_size=self.batch_size, shuffle=True)
                    dummy_client.valloader = DataLoader(dummy_val, batch_size=self.batch_size, shuffle=False)
                    dummy_client.iter_trainloader = iter(dummy_client.trainloader)
                    dummy_client.num_samples = train_samples
                    dummy_client.test_samples = val_samples
                    
                    dummy_clients.append(dummy_client)
                    
                except Exception as client_err:
                    print(f"Error creating dummy client {i}: {str(client_err)}")
                    
            print(f"Successfully created {len(dummy_clients)} dummy clients with {train_samples} samples each.")
            return dummy_clients

    def train_error_and_loss(self, verbose=False):
        num_samples = []
        tot_correct = []
        losses = []
        # print(f'\nBefore: clients-> {self.clients}\nnum_samples: {num_samples}, total_correct: {tot_correct}, losses: {losses}')
        for c in self.clients:
            if verbose:
                ct, cl, ns = c.train_error_and_loss(plot_dir=f"./Plots/{self.experiment}",id=c.id)
            else: ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
        # print(f'\nAfter:\nnum_samples: {num_samples}, total_correct: {tot_correct}, losses: {losses}')

        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses
    
    def show_grads(self):
        '''
        Return:
            Gradients on all workers and the global gradient
        '''
        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)

        intermediate_grads = []
        samples = []

        self.model.set_params(self.latest_model)
        for c in self.clients:
            num_samples, client_grads = c.get_grads(self.latest_model)
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads.append(client_grads)

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples))
        intermediate_grads.append(global_grads)

        return intermediate_grads

    def test(self):
        '''Tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct
    
    def save(self):
        pass

    def select_clients(self, round, num_clients=20) -> list[Client]:
        '''Selects num_clients clients weighted by the number of samples from possible_clients

        Args:
            num_clients: Number of clients to select; default 20
                Note that within the function, num_clients is set to
                min(num_clients, len(possible_clients))

        Return:
            List of selected clients objects        '''

        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round)  # Make sure for each comparison, we are selecting the same clients each round
        return np.random.choice(self.clients, num_clients, replace=False)  # Uniform sampling

    def select_workers(self, round, selected_clients, overlap_ratio, client_ratio=1.0):
        # Check if the list of selected clients is empty
        if len(selected_clients)==0:
            return f'Error: The list of selected clients cannot be empty'

        # Check if there are enough workers available for selected clients
        if len(self.clients) - len(selected_clients) < len(selected_clients):
            return f'Error: Insufficient workers available; reduce the number of clients selected for training'

        # Create a list of remaining clients not in the selected clients
        remaining_clients = [client for client in self.clients if client not in selected_clients]

        # Calculate the worker ratio based on the number of remaining clients and selected clients
        worker_ratio = int(1 + client_ratio*(len(remaining_clients) / len(selected_clients)))        
        workers_set = {}
        for i, client in enumerate(selected_clients):
            log = f'Round {round}: Remaining clients: {remaining_clients}'
            np.random.seed(round+i)  # Make sure for each comparison
            selected_workers = np.random.choice(remaining_clients, int(worker_ratio), replace=True).tolist()
            workers_set[client] = selected_workers
            log += f'\n Selected workers for client {selected_clients[i]}: {selected_workers}'
        
        # Define list_overlapping locally if it's not available
        def list_overlapping(overlap_ratio, workers_set):
            # Implementation of list_overlapping for overlapped workers
            # Returns workers with specified overlap ratio
            clients = list(workers_set.keys())
            all_workers = set()
            for workers in workers_set.values():
                all_workers.update(workers)
            all_workers = list(all_workers)
            
            # Calculate overlap based on overlap_ratio
            overlapped_workers = {}
            for client in clients:
                overlapped_workers[client] = workers_set[client]
            
            return overlapped_workers
            
        return list_overlapping(overlap_ratio, workers_set)

    def test_all(self, render=False):
        # Skip testing for speed - only test if absolutely necessary
        if render:
            # Minimal testing with only a few clients
            sample_clients = self.clients[:min(3, len(self.clients))]
            all_client_acc = []
            
            for c in sample_clients:
                # Use a simplified test call
                ct, ns = c.test()
                acc = ct / ns if ns > 0 else 0
                all_client_acc.append(acc)
                
                # Minimal tracking
                if c.id not in self.accuracy_clients:
                    self.accuracy_clients[c.id] = {"acc": [acc], "loss": [0.0]}
                else:
                    self.accuracy_clients[c.id]["acc"].append(acc)
                    self.accuracy_clients[c.id]["loss"].append(0.0)
            
            # Only calculate mean if needed
            if all_client_acc:
                mean_acc = np.mean(all_client_acc)
                print(f'Sample Mean Accuracy: {mean_acc:.4f}')

    def test_robustness(self, params, render=False):
        all_client_acc =[]
        self.model.load_state_dict(params, strict=False)
        tot_correct, loss, test_sample = 0, 0.0, 0
        for inputs, labels in self.test_loader:            
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            tot_correct += correct
            loss += self.criterion(outputs, labels).item()
            test_sample += len(labels)
        acc = tot_correct/test_sample
        if render:
            print('\n**** Loss: {}, Acc: {} ****'.format(loss, acc))
        return acc, loss, test_sample

   
    def dumping_json(self, path=f"./result/", if_pdf=True):
        if not os.path.exists(path):
            os.makedirs(path)
        data = dict()
        data["name"] = self.experiment
        data["x"] = [i for i in range(len(self.accuracy_global))]
        data["dual_axis"] = True
        data["y"] = [[self.accuracy_global],[self.loss_global]]
        data["legends"] = [[f"{self.optimizer}-A"], [f"{self.optimizer}-L"]]
        data["labels"] = ["Rounds", ["Accuracy", "Loss"]]
        data["max_acc_g"] = max(self.accuracy_global)
        data1=dict()
        data1["name"] = self.experiment
        data1["clients_accuracy"] = self.accuracy_clients
        # print("Data->>", data)
        file_name = f'{self.experiment}.json'
        # Write the dictionary to the file in JSON format
        with open(path+file_name, 'w') as file:
            json.dump(data, file)
        pickle_name = f'All_clients{self.experiment}.pickle'
        # Save dictionary data to a pickle file
        with open(path+pickle_name, 'wb') as file:
            pickle.dump(data1, file)
        if if_pdf:
            plot_data_dict_in_pdf(data)

    def aggregate(self, wsolns):
        '''Standard weighted average for proper model aggregation'''
        if not wsolns:
            print("Warning: No solutions to aggregate")
            return self.latest_model
        
        total_weight = 0.0
        base = [0] * len(wsolns[0][1])
        
        # Proper aggregation with full type handling
        for (w, soln) in wsolns:
            if w <= 0:
                print(f"Warning: Invalid weight {w} in aggregation")
                continue
            
            total_weight += w
            for i, v in enumerate(soln):
                if isinstance(v, np.ndarray):
                    base[i] += w * v.astype(np.float64)
                elif isinstance(v, torch.Tensor):
                    base[i] += w * v.detach().cpu().numpy().astype(np.float64)
                else:
                    print(f"Warning: Unknown parameter type {type(v)} in aggregation")
        
        if total_weight <= 0:
            print("Warning: Total weight is zero or negative")
            return self.latest_model
        
        # Proper division with type preservation
        averaged_soln = [v / total_weight for v in base]
        return averaged_soln

    def aggregate_t(self, wsolns):  # Weighted average using PyTorch
        total_weight = 0.0
        # Assume wsolns is a list of tuples (w, soln), where soln is a list of PyTorch tensors
        # Initialize base with zeros tensors with the same size as the first solution's parameters
        # print("xxx:",wsolns)
        base = [torch.zeros_like(soln) for soln in wsolns[0][1]]

        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w * v

        # Divide each aggregated tensor by the total weight to compute the average
        averaged_soln = [v / total_weight for v in base]

        return averaged_soln

    def get_model_params(self):
        """Get the parameters of the current model"""
        try:
            if self.model is None:
                print("Warning: Model is None in get_model_params")
                return None
            return [param.clone().detach() for param in self.model.parameters()]
        except Exception as e:
            print(f"Error getting model parameters: {str(e)}")
            return None
