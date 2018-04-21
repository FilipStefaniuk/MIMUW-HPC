#!/bin/env python3

from src.py import loader, parser, nn

if __name__ == '__main__':
    args = parser.parse()
    data_X, data_Y = loader.load_data(args.training_data)
    nn.fit(data_X, data_Y, **vars(args))
