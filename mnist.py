# -*- coding: UTF-8 -*-
##--------------------------------------
# Autor: Èric Mor
# Institut Salvador Vilaseca
# Curs 2017-2018
# 
# Aquest arxiu requereix les següents llibreries:
#  - theano
#  - numpy
#  - matplotlib
#
# Aquest arxiu serveix per a entrenar un model de xarxa neuronal per a aprengui a reconèixer digits del 0 al 9,
# utilitzant la base de dades de MNIST.
# Hi ha dos models diferents que es poden entrenar:
# - Una xarxa neuronal totalment connectada (arriba fins a una precisió del 94% aprox)
# - Una xarxa neuronal convolucional (arriba fins a una precisió del 98% o més)
#
# Més informació al github del projecte: https://github.com/emordo/Xarxes-neuronals
#
##--------------------------------------

import network_gpu as nn
import struct
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt

def load_data_set(images_path, labels_path):
    """
    Carrega les imatges i les seves etiquetes corresponents.
    """
    
    images = []
    
    with open(images_path, "r+b") as f:
        f.read(4)  # magic
        
        images_count = struct.unpack(">i", f.read(4))[0]
        width = struct.unpack(">i", f.read(4))[0]
        height = struct.unpack(">i", f.read(4))[0]
        
        
        for _ in range(images_count):
            image_data = struct.unpack(">" + str(width * height) + "B", f.read(width * height))
            
            images.append(np.array(image_data, dtype=np.float32).reshape(1, width, height) / 255.0)
        
    with open(labels_path, "r+b") as f:
        f.read(4)  # magic
        
        labels_count = struct.unpack(">i", f.read(4))[0]
        
        labels = struct.unpack(">" + str(labels_count) + "B", f.read(labels_count))
        
    return images, labels

def make_category_labels(labels):
    """
    Conerteix uns dígits (per exemple, 3) en deu neurones de sortida (per exemple, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]). 
    """

    real_labels = []
    for label in labels:
        result = np.zeros(10)
        result[label] = 1
        real_labels.append(result)
        
    return np.array(real_labels, dtype=np.float32).reshape(len(labels), 10)

def check_accuracy(model, test_images, test_labels, validation_count):
    """
    Comprova la precisió d'un model de xarxa neuronal amb uns quants exemples de validació.
    Torna el valor en tant per 1.
    """
    correct = 0
    
    result = model.feedforward(test_images)
    
    for i in range(validation_count):
        if result[i].argmax() == test_labels[i]:
            correct += 1
            
    return correct / validation_count


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Experiment de MNIST amb xarxes neuronals.')
    parser.add_argument('-d', help='Carpeta on estan les dades de MNIST.',        dest='data_dir',          type=str,   default='')
    parser.add_argument('-epochs', help="Nombre de repeticions.",                 dest='num_epochs',        type=int, default=12)
    parser.add_argument('-batch', help="Mida del batch.",                 dest='mini_batch_size',        type=int, default=25)
    parser.add_argument('-lr', help="Taxa d'aprenentatge.",                       dest='learning_rate',         type=float, default=3.0)
    parser.add_argument('-model', help="0: totalment connectat. 1: convolucional",                 dest='model',        type=int, default=0)
    parser.add_argument('-save', help="Guardar còpies de la xarxa durant l'entrenament?", dest='save', type=bool, default=False)
    args = parser.parse_args()
    
    if args.model == 0:
        model = nn.NeuralNetwork()
        model.add_layer(nn.Flatten())
        model.add_layer(nn.FullyConnected('SIGMOID', 100))
        model.add_layer(nn.FullyConnected('SIGMOID', 10))
        model.build((1, 28, 28))

    else:
        model = nn.NeuralNetwork()
        model.add_layer(nn.Convolutional('RELU', 32, 3, initializer=nn.glorot_initializer))
        model.add_layer(nn.Convolutional('RELU', 64, 3, initializer=nn.glorot_initializer))
        model.add_layer(nn.Pooling(extent=2, stride=2))
        model.add_layer(nn.Dropout(0.25))
        model.add_layer(nn.Flatten())
        model.add_layer(nn.FullyConnected('RELU', 128, initializer=nn.glorot_initializer))
        model.add_layer(nn.Dropout(0.5))
        model.add_layer(nn.FullyConnected('SIGMOID', 10, initializer=nn.glorot_initializer))
        model.build((1, 28, 28))
    
    print("Carregant dades....")    
    images, labels = load_data_set("%strain-images.idx3-ubyte" % args.data_dir, "%strain-labels.idx1-ubyte" % args.data_dir)
    test_images, test_labels = load_data_set("%st10k-images.idx3-ubyte" % args.data_dir, "%st10k-labels.idx1-ubyte" % args.data_dir)
    
    # Per a poder entrenar necessitem expressar les etiquetes com a grups de 10 neurones.
    category_labels = make_category_labels(labels)

    # En aquestes llistes guardarem informació per a fer gràfiques després
    epochs = []
    accuracies = []
    times = []
    
    # El nombre d'exemples que s'utilitzen per comprovar la precisió
    validation_count = 300
    # Cada quans batches comprovem la precisió
    batches = 20
    
    mini_batch_size = args.mini_batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    
    num_examples = len(images)
    
    for e in range(num_epochs):
        print()
        print("# Entrenant epoch %d..." % e)
        
        for i in range(0, num_examples, mini_batch_size*batches):
            
            start_time = time.clock()
            model.train(images, category_labels, mini_batch_size, learning_rate, mini_batch_size*batches, i)
            end_time = time.clock()
            
            accuracy = check_accuracy(model, test_images, test_labels, validation_count)
            
            epochs.append(e + i/num_examples)
            accuracies.append(accuracy)
            times.append(end_time - start_time)
            
            print("#-- Entrenats %d/%d  -- Precisió: %f%%  -- Temps: %f s" % (i, num_examples, accuracy*100, end_time - start_time))
            
        # Cada epoch comprovem la precisió amb tots els exemples de validació.
        accuracy = check_accuracy(model, test_images, test_labels, 10000)
        print("Precisió de l'epoch %d: %f%%" % (e, accuracy))
        
        if args.save:
            model.save("backup-%d.nn" % e)
    
    if args.save:
        # Guardem el model ja entrenat.
        model.save("trained_network.nn")
    
    # Dibuixem una gràfica de la precisió
    plt.plot(epochs, accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Precisió')
    plt.show()
