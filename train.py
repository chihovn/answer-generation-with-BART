import torch

from src.utils import init_checkpoint_folder, log_gpu_utilization, get_basic_parser, get_data_parser, get_model_parser, get_training_parser, get_logger
from src.trainer import Trainer, prepare_dataset, prepare_training_stuff


def main(basic_args):
    # set seed
    torch.manual_seed(basic_args.seed)
    init_checkpoint_folder(basic_args)

    if basic_args.logger:
        logger = get_logger(basic_args.is_main, basic_args.checkpoint_path/'run.log')

    if basic_args.logger and basic_args.device == 'cuda:0':
        log_gpu_utilization(logger)

    data_args = get_data_parser()

    train_dataset = prepare_dataset(logger, basic_args, data_args, data_type='train')

    if data_args.eval_data != None:
        eval_dataset = prepare_dataset(logger, basic_args, data_args, data_type="eval")
    else:
        eval_dataset =  None
    
    if basic_args.logger and basic_args.device == 'cuda:0':
        log_gpu_utilization(logger)

    model_args = get_model_parser()
    
    tokenizer, model = prepare_training_stuff(logger, basic_args, model_args)

    if basic_args.logger and basic_args.device == 'cuda:0':
        log_gpu_utilization(logger)
    
    training_args = get_training_parser()

    trainer = Trainer(
            model=model, 
            tokenizer=tokenizer, 
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset, 
            training_args=training_args,
            basic_args=basic_args,
            model_args=model_args,
            logger=logger)

    if basic_args.logger:
        logger.info('=============Start training=============')
    
    trainer.train()

    if basic_args.logger and basic_args.device == 'cuda:0':
        log_gpu_utilization(logger)


if __name__ == '__main__':
    basic_args = get_basic_parser()
    main(basic_args)