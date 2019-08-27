from .cpc_model import CPCSelfSupervised   
import pytorch_lightning as ptl   
from test_tube import HyperOptArgumentParser

def main(params):
    model = CPCSelfSupervised()
    
    trainer = Trainer()
    trainer.fit(model)


if __name__ == '__main__':
    # get args
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parser = HyperOptArgumentParser(strategy=strategy, add_help=False)
    parser = CPCSelfSupervised.add_model_specific_args(parent_parser, root_dir)
    args = parse_args()   

    main(args)   
