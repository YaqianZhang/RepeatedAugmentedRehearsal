from utils.utils import boolean_string


def parse_scr(parser):
    ####################SupContrast######################
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--examine_train', type=boolean_string, default=False,
                        )
    parser.add_argument('--no_aug', type=boolean_string, default=False,
                        )
    parser.add_argument('--single_aug', type=boolean_string, default=False,
                        )
    parser.add_argument('--aug_type', default="")
    parser.add_argument('--softmaxhead_lr', type=float, default=0.1)

    parser.add_argument('--buffer_tracker', type=boolean_string, default=False,
                        help='Keep track of buffer with a dictionary')
    parser.add_argument('--warmup', type=int, default=4,
                        help='warmup of buffer before retrieve')
    parser.add_argument('--head', type=str, default='mlp',
                        help='projection head')

    parser.add_argument('--use_softmaxloss', type=boolean_string, default=False)
    parser.add_argument('--softmax_nlayers', type=int, default=1, help="softmax head for scr")
    parser.add_argument('--softmax_nsize', type=int, default=1024, help="softmax head size for scr")
    parser.add_argument('--softmax_membatch', type=int, default=100, help="softmax mem batchsize for scr")
    parser.add_argument('--softmax_dropout', type=boolean_string, default=False,
                        help="whether to use dropout in softmax head")
    parser.add_argument('--softmax_type', type=str, default='None', choices=['None', 'seperate', 'meta'])


    return parser