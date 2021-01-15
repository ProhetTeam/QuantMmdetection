import argparse

import torch
from mmcv import Config
import time
import os.path as osp
from mmdet.models import build_detector
from thirdparty.mtransformer import build_mtransformer
from mmdet.utils import collect_env, get_root_logger
try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')
try:
    import torchprof
except ImportError:
    raise ImportError('Please install torchprof: pip install torchprof')
try:
    import pytorch_memlab
except ImportError:
    raise ImportError("Please install pytorch_memlab: pip install pytorch_memlab")

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    parser.add_argument(
        '--not-quant',
        action='store_true',
        help='train or inference.')
    args = parser.parse_args()
    return args


def main():
    """ 1. This Part is for Compuation Profiler """
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.get('work_dir', osp.splitext(osp.basename(args.config))[0]), f'Performance-Profiler {timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

   
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    logger.info(f'\n{split_line} \nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    logger.info(f'!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')
    

    """ 2. This Part is for Memory Profiler """
    if hasattr(cfg, "quant_transformer") and (not args.not_quant):
        model_transformer = build_mtransformer(cfg.quant_transformer)
        model = model_transformer(model, logger= logger)
    
    batch_input = torch.ones(()).new_empty(
                (1, *input_shape),
                dtype=next(model.parameters()).dtype,
                device=next(model.parameters()).device)

    with torch.autograd.profiler.profile(use_cuda = torch.cuda.is_available() ,profile_memory=True, record_shapes = True) as prof:
        model(batch_input)
    logger.info(f'\nPytorch Evaluate :\n {prof}')
    prof.export_chrome_trace('profiles')


    """3. This is layer by layer analysis"""
    with torchprof.Profile(model, use_cuda=torch.cuda.is_available(), profile_memory=True) as prof:
        model(batch_input)
    logger.info(f'Layer by Layer Evaluate NO small OP : \n{prof.display(show_events = False)}')
    trace, event_lists_dict = prof.raw()
    logger.info(f"The trace is: \n{trace[8]}")
    # Trace(path=('AlexNet', 'features', '0'), leaf=True, module=Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)))

    logger.info(f"\n{event_lists_dict[trace[8].path][0]}")
    logger.info(f'Layer by Layer Evaluate with small OP :\n{prof.display(show_events=True)}')



if __name__ == '__main__':
    main()
