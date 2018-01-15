# Xarxes neuronals

Aquest projecte és un intent de programar xarxes neuronals artificials (un tipus de intel·ligència artificial) desde 0. El vaig realitzar com a part del treball de recerca del curs 2017-2018 a l'Institut Salvador Vilaseca.

Tot el codi està programat en python 3.6.3. El codi està escrit en anglès, però l'he documentat en català. Hi ha tres arxius:
 - *network.py*: Aquest és la xarxa més bàsica, amb totes les operacions explicades al treball. És més aviat per demostrar com es podia programar totes les formules, però es millor utilitzar la xarxa a l'arxiu network_gpu.py
 
 - *network_gpu.py:* Aquesta xarxa és més completa, té més optimitzacions (dropout, mètodes d'inicialització, funció softmax) i és pot executar utilitzant la tarjeta gràfica, de manera que va molt més ràpid. S'ha d'utilitzar aquest arxiu fins i tot si l'ordinador no té suport per executar-ho a la tarjeta gràfica.
 
 - *mnist.py*: En aquest arxiu hi ha dos models de xarxes neuronals que s'entrenen per reconèixer digits de la MNIST Database.
 
![MNIST Database](https://i1.wp.com/www.parallelr.com/wp-content/uploads/2016/02/mnist.jpg)
 
Es requereixen les següents llibreries:
  - *scipy* (només si s'utilitza l'arxiu *network.py*)
  - *theano* (només si s'utilitza l'arxiu *network_gpu.py* o *mnist.py*)
  - *numpy*
  - *matplotlib* (només si s'utilitza l'arxiu *mnist.py*)
  
Aquí hi ha un exemple de com es poden crear xarxes neuronals amb aquests arxius:
```python
import network as nn

xarxa = nn.NeuralNetwork()
xarxa.add_layer(nn.Convolutional('RELU', 64, 3))
xarxa.add_layer(nn.Flatten())
xarxa.add_layer(nn.FullyConnected('SIGMOID', 100))
xarxa.add_layer(nn.FullyConnected('SIGMOID', 10))
xarxa.build((1, 28, 28))
```

## MNIST Database
L'arxiu *mnist.py* es pot executar per a entrenar una xarxa per a que reconegui digits escrits de la MNIST Database. Fa falta dscarregar i descomprimir els 4 arxius que hi ha a http://yann.lecun.com/exdb/mnist/; a l'hora de descomprimirlos, s'han de deixar a la mateixa carpeta on està els arxius de codi.

Per a executar-lo, s'ha d'obrir una "ventana de comandos" (CMD) a la carpeta on estan els arxius del codi, i executar el següent:
`python mnist.py`
 Al cap d'un minut o així haurà acabat de construir el model i començarà a entrenar la xarxa.
 
 Hi ha diversos paràmetres que es poden modificar a l'hora d'executar-lo per provar diferents models i entrenaments:
  - *model*: Si el valor és 0, s'utilitzarà un model simple de xarxa totalment connectada. Si és 1, s'utilitzarà una xarxa convolucional (que pot arribar a més precisió).
  - *epochs*: El nombre de repeticions que es fan a l'entrenament, 12 per defecte.
  - *batch*: La mida del 'batch' (exemples que es processen alhora), 25 per defecte.
  - *lr*: La taxa d'aprenentatge, 3.0 per defecte.
  - *save*: Si és 'True', es guardaran copies a la carpeta de la xarxa neuronal a mesura que s'entrena.
  - *d*: Serveix per a especificar la carpeta estan les dades d'entrenament; si no s'especifica s'agafaren de la carpeta on està el codi.
  
 Per exemple, si volem 30 repeticions amb taxa d'aprenentatge 0.2:
 `python mnist.py -epochs 30 -lr 0.2`

