from utils.utils import boolean_string


def parse_der(parser):

    parser.add_argument("--DER_alpha",default=0.3,type=float)
    

    return parser