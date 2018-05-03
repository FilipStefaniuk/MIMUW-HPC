#!/bin/env python3

from src.py import loader, parser, nn

if __name__ == '__main__':
    args = parser.parse()
    data = loader.load_data(args.training_data)
    nn.fit(data, **vars(args))
