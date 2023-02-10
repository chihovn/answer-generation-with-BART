import argparse


class  BasicArguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--is_main', type=bool, default=True, help='whether the generator function is training or not')
        self.parser.add_argument('--device', type=str, default='cuda:0', help='gpu or cpu')
        self.parser.add_argument('--seed', type=int, default=0, help='random seed for initialization')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='all things are saved here')
        self.parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/experiment', help='models are saved here')
        self.parser.add_argument('--checkpoint_exist', type=bool, default=False, help='whether checkpoint folder is exits or not')
        self.parser.add_argument('--logger', type=bool, default=True, help='whether to log information out or not')
    
    def parse(self):
        opt = self.parser.parse_args()
        return opt

class DataTrainingArguments():
    '''
    These are as the name suggests arguments needed for the dataset. Such as the directory name where your files are stored etc. You'll need these to load/process the dataset.
    '''
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--train_data', type=str, default=None, help='path of train data')
        self.parser.add_argument('--eval_data', type=str, default=None, help='path of eval data')

    def parse(self):
        opt = self.parser.parse_args()
        return opt


class ModelArguments():
    '''
    These are the arguments for the model that you want to use such as the model_name_or_path, tokenizer_name etc. You'll need these to load the model and tokenizer
    '''
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--model_name', type=str, default='facebook/bart', help='name of model')
        self.parser.add_argument('--model_size', type=str, default='base', help='size of model')
        self.parser.add_argument('--model_path', type=str, default=None, help='path for retraining')



    def parse(self):
        opt = self.parser.parse_args()
        return opt

class TrainingArguments():
    '''
    These are basicaly the training hyperparameters such as learning rate, batch size, weight decay, gradient accumulation steps etc
    '''
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        # self.parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint in the encoder')
        self.parser.add_argument("--batch_size", default=1, type=int, help="Batch size for training.")
        self.parser.add_argument('--num_epochs', type=int, default=1, help='number of training epochs')
        self.parser.add_argument("--max_length", default=1024, type=int)
        # self.parser.add_argument('--save_freq', type=int, default=5000, help='save model every <save_freq> steps during training')
        self.parser.add_argument('-- print_freq', type=int, default=1000, help='print intermdiate results of evaluation every <eval_print_freq> steps')
        self.parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
        self.parser.add_argument('--backward_freq', type=int, default=16, help='learning rate')

    def parse(self):
        opt = self.parser.parse_args()
        return opt