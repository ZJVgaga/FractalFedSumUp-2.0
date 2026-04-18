

class Client:
    def __init__(
            self,client_modules,config,logger,i,
        ):
         #client_modules是一个字典,config中存储了超参数，可以用config.get("param")提取
       #在这里填写你要初始化的server模块,把server_modules["param"]存为self.变量，也把config存为self.config
        
        return
        
    def receive_data_from_server(self,server_data):
        
        #在这里填写接收到的server_data怎么处理
         #可以用到self.config中的超参数
        return 

    def process(self):

        #在这里填写client的运行过程
          #可以用到self.config中的超参数
       
        return 

    def send_data_to_server(self,data_sent):
        
        #要传输的数据用字典表示
        
       
         #可以用到self.config中的超参数
        
        return  data_sent