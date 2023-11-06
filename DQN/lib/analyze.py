import time


class analyze():
    def __init__(self,net) -> None:
        self.net = net
         
    
    
    
    def GetAllParameters(self):
        """
        至於 analyze 功能，在調用 analyze(net).Getparameters() 時，
        它會使用 self.net.named_parameters() 遍歷 net 中的所有參數
        。由於 net 物件包含整個 DQNConv1DLarge 模型，這包括所有 Sequential 塊中的所有參數
        。analyze 類別不需要了解模型的內部結構；它只是使用 PyTorch 的 nn.Module 提供的介面來訪問參數。
        
                
        現在所運行的模型可以調參數總量 {'conv': 86016, 'fc_val': 2753537, 'fc_adv': 2754563} 
        """
        numels =[]
        for p in self.net.parameters():            
            numels.append(p.numel())

        names =[]
        for p in self.net.named_parameters():
            names.append(p[0])
            
        
        count_dict={}
        for key,value in dict(zip(names,numels)).items():
            if 'conv' in key:
                if 'conv' in count_dict:
                    count_dict['conv'] += value
                else:
                    count_dict['conv'] = value
            elif 'fc_val' in key:
                if 'fc_val' in count_dict:
                    count_dict['fc_val'] += value
                else:
                    count_dict['fc_val'] = value
            elif 'fc_adv' in key:
                if 'fc_adv' in count_dict:
                    count_dict['fc_adv'] += value
                else:
                    count_dict['fc_adv'] = value
           
        return count_dict