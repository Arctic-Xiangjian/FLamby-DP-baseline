import time
from typing import List
import collections
import torch
from tqdm import tqdm
import copy
from flamby.strategies.utilsbatch import DataLoaderWithMemory, _Model
from flamby.utils import evaluate_each_model_on_tests
from flamby.utils import evaluate_model_on_tests
from flamby.datasets.fed_ixi import metric

class FedAvg:
    """Federated Averaging Strategy class.

    The Federated Averaging strategy is the most simple centralized FL strategy.
    Each client first trains his version of a global model locally on its data,
    the states of the model of each client are then weighted-averaged and returned
    to each client for further training.

    References
    ----------
    - https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    training_dataloaders : List
        The list of training dataloaders from multiple training centers.
    model : torch.nn.Module
        An initialized torch model.
    loss : torch.nn.modules.loss._Loss
        The loss to minimize between the predictions of the model and the
        ground truth.
    optimizer_class : torch.optim.Optimizer
        The class of the torch model optimizer to use at each step.
    learning_rate : float
        The learning rate to be given to the optimizer_class.
    num_updates : int
        The number of updates to do on each client at each round.
    nrounds : int
        The number of communication rounds to do.
    dp_target_epsilon: float
        The target epsilon for (epsilon, delta)-differential
        private guarantee. Defaults to None.
    dp_target_delta: float
        The target delta for (epsilon, delta)-differential private
        guarantee. Defaults to None.
    dp_max_grad_norm: float
        The maximum L2 norm of per-sample gradients; used to enforce
        differential privacy. Defaults to None.
    log: bool, optional
        Whether or not to store logs in tensorboard. Defaults to False.
    log_period: int, optional
        If log is True then log the loss every log_period batch updates.
        Defauts to 100.
    bits_counting_function : Union[callable, None], optional
        A function making sure exchanges respect the rules, this function
        can be obtained by decorating check_exchange_compliance in
        flamby.utils. Should have the signature List[Tensor] -> int.
        Defaults to None.
    logdir: str, optional
        Where logs are stored. Defaults to ./runs.
    log_basename: str, optional
        The basename of the created log_file. Defaults to fed_avg.
    """

    def __init__(
        self,
        training_dataloaders: List,
        test_dataloader_center: List,
        test_dataloaders: List,
        model: torch.nn.Module,
        loss: torch.nn.modules.loss._Loss,
        optimizer_class: torch.optim.Optimizer,
        learning_rate: float,
        num_updates: int,
        nrounds: int,
        dp_target_epsilon: float = None,
        dp_target_delta: float = None,
        dp_max_grad_norm: float = None,
        log: bool = False,
        log_period: int = 100,
        bits_counting_function: callable = None,
        logdir: str = "./runs",
        log_basename: str = "fed_avg",
        seed=None,
    ):
        """
        Cf class docstring
        """
        self._seed = seed if seed is not None else int(time.time())

        self.training_dataloaders_with_memory = [
            DataLoaderWithMemory(e) for e in training_dataloaders
        ]
        self.training_sizes = [len(e) for e in self.training_dataloaders_with_memory]
        self.total_number_of_samples = sum(self.training_sizes)
        self.test_dataloader_center=test_dataloader_center
        self.test_dataloaders=test_dataloaders
        self.dp_target_epsilon = dp_target_epsilon
        self.dp_target_delta = dp_target_delta
        self.dp_max_grad_norm = dp_max_grad_norm

        self.log = log
        self.log_period = log_period
        self.log_basename = log_basename
        self.logdir = logdir
        self.global_acc_center=collections.OrderedDict({'Average':[],'client_test_0_local':[],'client_test_1_local':[],'client_test_2_local':[]})
        self.local_acc=collections.OrderedDict({'client_test_0_local':[],'client_test_1_local':[],'client_test_2_local':[],'client_test_0_average':[],'client_test_1_average':[],'client_test_2_average':[]})
        self.models_list = [
            _Model(
                model=model,
                optimizer_class=optimizer_class,
                lr=learning_rate,
                train_dl=_train_dl,
                dp_target_epsilon=self.dp_target_epsilon,
                dp_target_delta=self.dp_target_delta,
                dp_max_grad_norm=self.dp_max_grad_norm,
                loss=loss,
                nrounds=nrounds,
                log=self.log,
                client_id=i,
                log_period=self.log_period,
                log_basename=self.log_basename,
                logdir=self.logdir,
                seed=self._seed,
            )
            for i, _train_dl in enumerate(training_dataloaders)
        ]
        self.nrounds = nrounds
        self.num_updates = num_updates

        self.num_clients = len(self.training_sizes)
        self.bits_counting_function = bits_counting_function

    def _local_optimization(self, _model: _Model, dataloader_with_memory):
        """Carry out the local optimization step.

        Parameters
        ----------
        _model: _Model
            The model on the local device used by the optimization step.
        dataloader_with_memory : dataloaderwithmemory
            A dataloader that can be called infinitely using its get_samples()
            method.
        """
        _model._local_train(dataloader_with_memory, self.num_updates)

    def perform_round(self):
        """Does a single federated averaging round. The following steps will be
        performed:

        - each model will be trained locally for num_updates batches.
        - the parameter updates will be collected and averaged. Averages will be
          weighted by the number of samples in each client
        - the averaged updates willl be used to update the local model
        """
        #GLOBAL_MODEL=copy.deepcopy(self.models_list[0])
    
        global_acc_center=evaluate_model_on_tests(self.models_list[0].model,self.test_dataloader_center,metric)
        self.global_acc_center['Average'].append(global_acc_center['client_test_0'])
        local_acc_gobal_model=evaluate_model_on_tests(self.models_list[0].model,self.test_dataloaders,metric)
        self.local_acc['client_test_0_average'].append(local_acc_gobal_model['client_test_0'])
        self.local_acc['client_test_1_average'].append(local_acc_gobal_model['client_test_1'])
        self.local_acc['client_test_2_average'].append(local_acc_gobal_model['client_test_2'])
        for _model, dataloader_with_memory, acc_keys in zip(
            self.models_list, self.training_dataloaders_with_memory,list(self.global_acc_center.keys())[1:]
        ):
            # Local Optimization
            self._local_optimization(_model, dataloader_with_memory)
            global_acc_local=evaluate_model_on_tests(_model.model,self.test_dataloader_center,metric)
            self.global_acc_center[acc_keys].append(global_acc_local['client_test_0'])
        local_acc_local_model=evaluate_each_model_on_tests([self.models_list[0].model,self.models_list[1].model,self.models_list[2].model],self.test_dataloaders,metric)
        self.local_acc['client_test_0_local'].append(local_acc_local_model['client_test_0'])
        self.local_acc['client_test_1_local'].append(local_acc_local_model['client_test_1'])
        self.local_acc['client_test_2_local'].append(local_acc_local_model['client_test_2'])

        fed_state_dict=collections.OrderedDict()
        # Aggregation step
        if self.nrounds!=0:
            weight_keys=self.models_list[0].model.state_dict().keys()
            for key in weight_keys:
                key_avg=0
                for _model,train_size in zip(self.models_list,self.training_sizes):
                    key_avg=key_avg+_model.model.state_dict()[key]*train_size/float(self.total_number_of_samples)
                fed_state_dict[key]=key_avg
            for _model in self.models_list:
                _model.model.load_state_dict(fed_state_dict)
        


    def run(self):
        """This method performs self.nrounds rounds of averaging
        and returns the list of models.
        """
        if self.nrounds!=0:
            for _ in tqdm(range(self.nrounds)):
                self.perform_round()
            
        else:
            self.perform_round()
        return self.local_acc,self.global_acc_center,[m.model for m in self.models_list]
