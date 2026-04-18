import random
import torch


from torch.utils.data.sampler import SubsetRandomSampler


#用于测量通讯开销的
class Server:
    #初始化函数
    def __init__(
        self,
        server_modules,config,logger,clients
    ):
       #server_modules是一个字典,config中存储了超参数，可以用config.get("param")提取
       #在这里填写你要初始化的server模块,把server_modules["param"]存为self.变量，也把config存为self.config
       return
    
    #选择若干个客户端进行更新


    def arrange_server_data_to_client():
        server_data={}#server_data是一个字典
        #可以用到self.config中的超参数

        #在这里填写你要准备给每个客户端的数据

        return server_data
    def merge_data(received_data_list):
        merged_data={}

        #在这里填写你合并的所有client的data的方式
        #可以用到self.config中的超参数

        return merged_data
    def process(merged_data):
        
        #在这里填写你对合并后的data的处理方式
        #可以用到self.config中的超参数

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

