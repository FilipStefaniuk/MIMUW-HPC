from PIL import Image
import os
import csv

def load_data(dir):

    # data_X = []
    # data_Y = []

    data = []

    for root, _, files in os.walk(dir):
        
        metadata = []
        images = {}
        
        for file in files:

            if file.endswith('.csv'):
                with open(os.path.join(root, file)) as f:
                    reader = csv.DictReader(f, delimiter=';')
                    for line in reader:
                        metadata.append(line)

            if file.endswith('.ppm'):
                img = Image.open(os.path.join(root, file))
                img = img.resize((64, 64), Image.BILINEAR).convert('L')
                images[file] = list(map(int, img.tobytes()))
            
        for row in metadata:
            data.append((images[row['Filename']], int(row['ClassId'])))
            
    
    return data