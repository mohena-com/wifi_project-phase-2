from config_reader import ConfigReader

from DS_WifiCSIDataset import WifiCSIDataset
from DL_CSILSTMNet import CSILSTMNet
from DL_DenseNet1D import DenseNet1D
from DL_EfficientNet1DLSTM import EfficientNet1DLSTM
from DL_MobileNetV3 import MobileNetV3_1D_LSTM



cr = ConfigReader("csi_id_config.properties")

list_batch_size = cr.get_int_list("batch_size")
list_learning_rate = cr.get_float_list("learning_rate")
list_optimizer = cr.get_list("optimizer", type_func=str)
list_weight_decay = cr.get_float_list("weight_decay")   
list_epochs = cr.get_int_list("epochs")

print(f"list_batch_size: {list_batch_size}")
print(f"list_learning_rate: {list_learning_rate}")  
print(f"list_optimizer: {list_optimizer}")
print(f"list_weight_decay: {list_weight_decay}")
print(f"list_epochs: {list_epochs}")


var_model_definitions = {
    "CSILSTMNet": (
        CSILSTMNet,
        {
            'csi_input_size': [99],
            'meta_input_size': [12],
            'window_size': [128],
            'num_classes': [31],
            'lr': list_learning_rate,
            'batch_size': list_batch_size,
            'optimizer': list_optimizer,
            'weight_decay': list_weight_decay,
            'epochs': list_epochs,
        },
    ),
    "DenseNet1D": (
        DenseNet1D,
        {
            'csi_channels': [99],
            'meta_feature_dim': [12],
            'num_classes': [31],
            'lr': list_learning_rate,
            'batch_size': list_batch_size,
            'optimizer': list_optimizer,
            'weight_decay': list_weight_decay,
            'epochs': list_epochs,
        },
    ),
    "EfficientNet1DLSTM": (
        EfficientNet1DLSTM,
        {
            'in_channels': [99],
            'meta_seq_len': [128],
            'meta_feature_dim': [12],
            'num_classes': [31],
            'lr': list_learning_rate,
            'batch_size': list_batch_size,
            'optimizer': list_optimizer,
            'weight_decay': list_weight_decay,
            'epochs': list_epochs,
        },
    ),
    "MobileNetV3_1D_LSTM": (
        MobileNetV3_1D_LSTM,
        {
            'csi_channels': [99],
            'meta_feature_dim': [12],
            'num_classes': [31],
            'lr': list_learning_rate,
            'batch_size': list_batch_size,
            'optimizer': list_optimizer,
            'weight_decay': list_weight_decay,
            'epochs': list_epochs,
        },
    ),
}

model_defs = var_model_definitions  
model_list = cr.get("training_model_list")
print( f"model_list from config: {model_list}")
for model_name, (model_class, param_grid) in model_defs.items():     
    if model_name not in model_list: 
        print(f"=== Not Matched Model: {model_name} ====")   
        continue        
    else:
        print(f"=== Matched Model: {model_name} ====")      
        print(f"Class: {model_class}")
        print(f"Param Grid: {param_grid}")