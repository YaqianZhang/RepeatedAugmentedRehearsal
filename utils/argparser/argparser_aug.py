
from utils.utils import boolean_string


def parse_aug(parser):

    ########################## RAR #####################
    parser.add_argument("--adjust_aug_flag",default=False,type=boolean_string)
    parser.add_argument("--randaug",default=False)
    parser.add_argument("--randaug_type",default="static",choices=["dynamic","static"])
    parser.add_argument("--aug_target",default="both",choices=["mem","incoming","both","none"])
    parser.add_argument("--scraug",default=False)
    parser.add_argument("--randaug_N", default=0,type=int)
    parser.add_argument("--randaug_N_mem", default=0,type=int)
    parser.add_argument("--randaug_N_incoming", default=0,type=int)
    parser.add_argument("--randaug_M", default=1,type=int)
    parser.add_argument("--aug_start",default=0,type=int)
    # parser.add_argument("--do_cutmix", dest="do_cutmix", default=False, type=boolean_string)
    # parser.add_argument("--cutmix_prob", default=0.5, type=float)
    # parser.add_argument("--cutmix_batch", default=10, type=int)
    # parser.add_argument("--cutmix_type", default="random", choices=["most_confused","train_mem","random","cross_task","mixed"])

    return parser
