#!/bin/env python3

from src.py import loader, parser, nn

if __name__ == '__main__':

    args = parser.parse()
    data = loader.load_data(args.training_data)
    result = nn.fit(data, **vars(args))

    with open('results.txt', 'w') as results_file:
        results_file.write(str(result))