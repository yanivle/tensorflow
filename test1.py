import tensorflow as tf
from tensorflow import keras

import math
import numpy as np
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

WIDTH = 30
HEIGHT = 30
N_HIDDEN = 128
N_TRAINING = 100000
N_TESTING = 1000
NOISE_P = 0

np.set_printoptions(linewidth=150)


def CirclePredicateFromCenter(centerx, centery, x, y, radius):
    return (x-centerx)**2+(y-centery)**2 <= radius**2


def SquarePredicateFromCenter(centerx, centery, x, y, radius):
    return max(abs(x-centerx), abs(y-centery)) <= radius


def RandomShape(predicate_from_center, noise_p):
    img = np.zeros((WIDTH, HEIGHT))
    centerx = random.randint(5, WIDTH - 1 - 5)
    centery = random.randint(5, HEIGHT - 1 - 5)
    radius = random.randint(5, 15)
    for x in range(WIDTH):
        for y in range(HEIGHT):
            prob = noise_p
            if predicate_from_center(centerx, centery, x, y, radius):
                prob = 1.0 - noise_p
            if random.random() < prob:
                img[x, y] = 1
    return img


def RandomCircle(noise_p=0.2): return RandomShape(CirclePredicateFromCenter, noise_p)


def RandomSquare(noise_p=0.2): return RandomShape(SquarePredicateFromCenter, noise_p)


def create_dataset(n, noise_p=0.2, low_noise=False):
    imgs = []
    labels = []
    for i in range(n):
        real_noise = noise_p
        if low_noise:
            real_noise = random.random() * noise_p
        if random.getrandbits(1):
            img = RandomCircle(real_noise)
            label = 0
        else:
            img = RandomSquare(real_noise)
            label = 1
        imgs.append(img)
        labels.append(label)
    return np.stack(imgs, axis=0), np.stack(labels, axis=0)


def plot_data(data, title, fname=None):
    viridis = cm.get_cmap('viridis', 256)

    fig, ax = plt.subplots()
    ax.pcolormesh(data, cmap=viridis)
    plt.title(title)
    plt.axis('equal')
    if fname:
        plt.savefig(fname)
    else:
        plt.show()
    plt.close(fig)


plot_data(RandomCircle(), 'Circle')
plot_data(RandomCircle(), 'Circle')
plot_data(RandomCircle(), 'Circle')
plot_data(RandomSquare(), 'Square')
plot_data(RandomSquare(), 'Square')
plot_data(RandomSquare(), 'Square')

print('Creating dataset...')
# train_images, train_labels = create_dataset(N_TRAINING, noise_p=0.2, low_noise=True)
train_images, train_labels = create_dataset(N_TRAINING, noise_p=NOISE_P, low_noise=True)

print('Training...')

class_names = ['circle', 'square']

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(WIDTH, HEIGHT)),
    keras.layers.Dense(N_HIDDEN, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

first_layer = model.layers[1]
weights, biases = first_layer.get_weights()
for i in range(N_HIDDEN):
    weights_for_neuron = weights[:, i]
    img_to_plot = weights_for_neuron.reshape(WIDTH, HEIGHT)
    img_to_plot -= np.min(img_to_plot)
    img_to_plot /= np.max(img_to_plot)
    plot_data(img_to_plot, f'weights for neuron {i}', f'/tmp/neuron{i}.png')

second_layer = model.layers[2]
weights, biases = second_layer.get_weights()
for i in range(2):
    weights_for_neuron = weights[:, i]
    print(f'neuron {i}: {class_names[i]}')
    print('Max weights idx:')
    print(weights_for_neuron.argsort()[-10:][::-1])
    print('Min weights idx:')
    print(weights_for_neuron.argsort()[:10][::-1])

test_images, test_labels = create_dataset(N_TESTING, NOISE_P)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# for i in range(10):
#     noise = 0.1 * i
#     print('noise:', noise)
#     test_images, test_labels = create_dataset(1000, noise)
#     test_loss, test_acc = model.evaluate(test_images, test_labels)
#     print('Test accuracy:', test_acc)

# for i in range(10):
#     # noise = float(input("Prob? "))
#     noise = 0.1
#     if random.getrandbits(1):
#         img = RandomCircle(noise)
#         label = 0
#     else:
#         img = RandomSquare(noise)
#         label = 1
#     print()
#     print(img)
#     predictions = model.predict(np.stack([img], axis=0))
#     predicted_class = np.argmax(predictions)
#     print(class_names[label])
#     print(class_names[predicted_class])
