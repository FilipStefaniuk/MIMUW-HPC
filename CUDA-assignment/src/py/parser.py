import argparse

def parse():

    parser = argparse.ArgumentParser(description="CUDA neural network")

    parser.add_argument('--training_data', required=True, metavar='PATH', help="training data path")
    parser.add_argument('--epsilon', default=0.1, type=float, metavar='EPS', help="treshold error value")
    parser.add_argument('--learning_rate', default=0.1, type=float, metavar='RATE', help="learning rate for backpropagation")
    parser.add_argument('--epochs', default=2, type=int, help="maximum number of epochs")
    parser.add_argument('--random', metavar='[true/false]', choices=['true', 'false'], default='true',
                                    help="if true, initial weights are set randomly; if false, initial weights are set to 1")
    
    return parser.parse_args()