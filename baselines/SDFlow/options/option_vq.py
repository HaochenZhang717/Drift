import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='The first stage of SDflow training',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader  
    parser.add_argument('--dataname', type=str, default='elect', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=96, help='training motion length')
    parser.add_argument('--datasets-dir', '--datasets_dir', dest='datasets_dir', type=str, default=None, help='root directory for data_provider datasets')
    parser.add_argument('--rel-path', '--rel_path', dest='rel_path', type=str, default=None, help='relative path for data_provider datasets')
    parser.add_argument('--rel-path-train', '--rel_path_train', dest='rel_path_train', type=str, default=None, help='train split relative path')
    parser.add_argument('--rel-path-valid', '--rel_path_valid', dest='rel_path_valid', type=str, default=None, help='validation split relative path')
    parser.add_argument('--rel-path-test', '--rel_path_test', dest='rel_path_test', type=str, default=None, help='test split relative path')
    parser.add_argument('--stride', type=int, default=1, help='sliding window stride for data_provider datasets')
    parser.add_argument('--window-stride', '--window_stride', dest='window_stride', type=int, default=None, help='alias for sliding window stride')
    parser.add_argument('--ts-stride', '--ts_stride', dest='ts_stride', type=int, default=None, help='alias for sliding window stride')
    parser.add_argument('--input-emb-width', '--input_emb_width', dest='input_emb_width', type=int, default=None, help='number of input channels; inferred when omitted')
    parser.add_argument('--column', type=str, default='glucose', help='value column for GlucoseSliding')

    ## optimization
    parser.add_argument('--total-iter', default=200000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[50000, 400000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")

    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")

    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset', 'lfq', 'ema_reset_sim'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for VQ')
    parser.add_argument("--resume-gpt", type=str, default=None, help='resume pth for GPT')

    ## output directory 
    parser.add_argument('--out-dir', type=str, default='./output_vqfinal/', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=2000, type=int, help='evaluation frequency')
    parser.add_argument('--skip-eval', action='store_true', help='skip expensive metric evaluation and save final checkpoint')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')


    parser.add_argument('--recon', type=float, default=1.0, help='recon loss in standard VQ')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'mps'], help='training device')


    
    return parser.parse_args()
