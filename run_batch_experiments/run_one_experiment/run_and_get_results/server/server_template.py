import random
import torch


from torch.utils.data.sampler import SubsetRandomSampler


# For measuring communication overhead
class Server:
    # Initialization function
    def __init__(
        self,
        server_modules, config, logger, clients
    ):
        # server_modules is a dictionary, config stores hyperparameters, accessible via config.get("param")
        # Initialize your server modules here, store server_modules["param"] as self.variable, also store config as self.config
        return
    
    # Select several clients for updating


    def arrange_server_data_to_client():
        server_data = {}  # server_data is a dictionary
        # Can use hyperparameters from self.config

        # Prepare the data you want to send to each client here

        return server_data
    def merge_data(received_data_list):
        merged_data = {}

        # Define how to merge all client data here
        # Can use hyperparameters from self.config

        return merged_data
    def process(merged_data):
        
        # Define how to process the merged data here
        # Can use hyperparameters from self.config

        return
    

    def select_clients(self):
        return (
            self.clients if self.join_ratio == 1.0
            else random.sample(self.clients, int(round(len(self.clients) * self.join_ratio)))
        )

    def evaluate(self):
        self.global_model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for x, target in self.test_loader:
                x, target = x.to(self.device), target.to(self.device, dtype=torch.int64)
                pred = self.global_model(x)
                _, pred_label = torch.max(pred.data, 1)
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()
        return correct / float(total)