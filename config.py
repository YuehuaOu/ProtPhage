import torch


class Config:
    def __init__(self):
        # hyper parameters
        self.binary_train_data = 'data/split_time/binary/binary_train.csv'
        self.binary_test_data = 'data/split_time/binary/binary_test.csv'
        self.binary_val_data = 'data/split_time/binary/binary_valid.csv'

        self.train_muti_data = "data/split_time/multi_class/muticlass_train.csv"
        self.val_muti_data = "data/split_time/multi_class/muticlass_val.csv"
        self.test_muti_data = "data/split_time/multi_class/muticlass_test.csv"

        self.embedding_file = 'embedding/embedding.xlsx'

        self.case_study_data = 'data/case_study/case_data.csv'
        self.embedding_file_case_study = 'embedding/embedding_for_case_study.xlsx'

        self.binary_model = 'model/split_time/binary/model.pth'
        self.muti_model = 'model/split_time/multi_class/model.pth'


        self.seed = 42
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
        self.epochs = 20
        self.batch_size = 64

config = Config()