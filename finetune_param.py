import torch

class TrainAbleParam:
    # hard code for the adapter
    def __init__(self):
        self.adapter_weight = [
            'query_tokens','intermediate_query','output_query'
       ]
        self.lm_weight_fc1 = [
            'language_model','fc1'
       ]
        self.lm_weight_fc2 = [
            'language_model','fc2'
       ]
        self.ca_output = [
            'crossattention','output','dense','bias'
       ]
        self.not_in_lm_weight_layers = [
            #6,25
            4,27
       ]
        
        
    def set_trainable_weights(self, model):
        change_params = []
        for name, param in model.named_parameters():
            param.requires_grad = False

        for trainable_ in self.adapter_weight:
            for name, param in model.named_parameters():
                if trainable_ in name.split('.'):
                    print(name, trainable_)
                    print('T')
                    param.requires_grad = True
                    change_params.append(name)
        
        for name, param in model.named_parameters():
                name_split = name.split('.')
                if set(self.lm_weight_fc1).issubset(set(name_split)) or set(self.lm_weight_fc2).issubset(set(name_split)):
                    if int(name_split[4]) < self.not_in_lm_weight_layers[0] or int(name_split[4]) > self.not_in_lm_weight_layers[1]:
                        print(name)
                        print('T')
                        param.requires_grad = True
                        change_params.append(name)  
                # if set(self.ca_output).issubset(set(name_split)):
                #     print(name)
                #     print('T')
                #     param.requires_grad = True
                #     change_params.append(name)
                    
        return change_params

