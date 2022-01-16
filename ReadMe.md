# [Project 10](Project10)

### Task

Use a generative model for the Yale faces dataset which also encodes the facial expression in some way. Evaluate the model, as well as its ability to change the facial expression for a given image by changing the according label in the model (e.g. CGAN or infoGAN).

### Our Solution

- Train a Conditional GAN (Generator and Discriminator) and generate in interval steps example images: [main.py](Project10/main.py)


# [Project 11](Project11)

### Task

Devise a classifier for the German traffic sign data set (you might use an existing network). Try to attack the model using several different attacks. Include, among others, a few pixel attack. Evaluate the success rate of different attack methods and discuss in how far the attacks go unnoticed by a human.

_[Download dataset instructions](Project11/get_dataset.txt)_

### Our Solution

- Train a ResNet18 model: [model.py](Project11/model.py)
- Evaluate this model: [eval_model.py](Project11/eval_model.py)
- Evaluate different attack methods on this model: [attack_evaluation.py](Project11/attack_evaluation.py)
