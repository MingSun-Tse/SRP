import torch
import torch.nn as nn
import copy, time

import utility
import data
import model
import loss
from option import args
from torchsummaryX import summary
from smilelogging import Logger

# Use smilelogging
logger = Logger(args) 

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args, logger)

# Select different trainers corresponding to different methods
if args.method in ['']:
    from trainer import Trainer
elif args.method in ['SRP']:
    from trainer import Trainer
    from pruner import pruner_dict

def get_model_complexity(model, savepath):
    import builtins, functools
    _model = copy.deepcopy(model)
    _model.to('cpu') # Use cpu mode
    _model.n_GPUs = 1

    height_lr = 1280 // args.scale[0]
    width_lr  = 720  // args.scale[0]
    dummy_input = torch.zeros((1, 3, height_lr, width_lr))

    flops_f = open(checkpoint.get_path(savepath), 'w+')
    original_print = builtins.print # Temporarily change the print fn
    builtins.print = functools.partial(print, file=flops_f, flush=True)
    summary(_model, dummy_input, {'idx_scale': args.scale[0]})
    builtins.print = original_print # Switch back to previous print fn

def parse_model_complexity(f):
    params, flops = 0, 0
    factor = {'k':1e3, 'M': 1e6, 'G': 1e9, 'T':1e12}
    for line in open(f):
        if line.startswith('Total params'):
            muliplier = factor[line.split()[-1][-1]]
            params = float(line.split()[-1][:-1]) * muliplier / 1e6 # E.g., Total params             1.372318M
        if line.startswith('Mult-Adds'):
            muliplier = factor[line.split()[-1][-1]]
            flops = float(line.split()[-1][:-1]) * muliplier / 1e9 # E.g., Mult-Adds             316.2488832G
    return params, flops

def main():
    global model, checkpoint
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            
            # Different methods require different model settings
            if args.method in ['']: # Original setting
                _model = model.Model(args, checkpoint)
            
            elif args.method in ['SRP']:
                _model = model.Model(args, checkpoint)
                class passer: pass
                passer.ckp = checkpoint
                passer.loss = loss.Loss(args, checkpoint) if not args.test_only else None
                passer.loader = loader
                pruner = pruner_dict[args.method].Pruner(_model, args, logger=logger, passer=passer)
                
                # Get Params/FLOPs before pruning
                get_model_complexity(_model, 'model_complexity_before_prune.txt')

                _model = pruner.prune()

                # Get Params/FLOPs after pruning
                get_model_complexity(_model, 'model_complexity_after_prune.txt')
                params_before, flops_before = parse_model_complexity(checkpoint.get_path('model_complexity_before_prune.txt'))
                params_after, flops_after = parse_model_complexity(checkpoint.get_path('model_complexity_after_prune.txt'))
                print(f'Params_before {params_before}M FLOPs_before {flops_before}G | Params_after {params_after}M FLOPs_after {flops_after}G | Compression {params_before/params_after:.4f}x Speedup {flops_before/flops_after:.4f}x')
                time.sleep(3)
                print('-' * 30 + ' [Pruned. Start finetuning] ' + '-' * 30)
                
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            
            if args.greg_mode in ['all']: 
                checkpoint.save(t, epoch=0, is_best=False)
                print(f'==> Regularizing all wn_scale parameters done. Checkpoint saved. Exit!')
                exit(0)
            
            # Get model complexity
            if args.method == '':
                get_model_complexity(_model, 'model_complexity.txt')
                params, flops = parse_model_complexity(checkpoint.get_path('model_complexity.txt'))
                print(f'Params {params}M FLOPs {flops}G')
            
            # Print model layer-wise structure
            from layer import LayerStruct
            LayerStruct(_model.model, (nn.Conv2d, nn.Linear))

            while not t.terminate():
                lrate = t.train()
                t.test(lrate)

            checkpoint.done()

if __name__ == '__main__':
    main()
