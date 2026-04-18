class Client:
    def __init__(
            self, client_modules, config, logger, i,
        ):
        # client_modules is a dictionary, config stores hyperparameters, which can be extracted using config.get("param")
        # Initialize your server modules here, store server_modules["param"] as self.variable, and also store config as self.config
        
        return
        
    def receive_data_from_server(self, server_data):
        
        # Specify here how to handle the received server_data
        # Hyperparameters from self.config can be used
        return 

    def process(self):

        # Specify the client's execution process here
        # Hyperparameters from self.config can be used
       
        return 

    def send_data_to_server(self, data_sent):
        
        # Data to be transmitted should be represented as a dictionary
        
       
        # Hyperparameters from self.config can be used
        
        return data_sent