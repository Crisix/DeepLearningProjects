"""This script evaluates different adversarial attacks with different arguments.

It logs the results in the folder named 'evaluation_results' and creates for every run a new folder inside it.
The log file is result.log. Every different attack config generates a new folder, which contains the successful adversarial examples.

Adding a new evaluation: Create an eval_definition tuple and put it inside the list of the evaluate_attack calls at the end of the script.
"""

import datetime
import logging
import os

import torch
import torchattacks

from collections import namedtuple
from Project11.load_test import load_test_data
from model import MyModel
from utils import label_map
from torchvision import transforms
from torchvision.utils import save_image

eval_definition = namedtuple('eval_definition', 'attack options number_runs')

# put different arguments in a list, that the script should evaluate
fgsm_attack = eval_definition(torchattacks.FGSM, {'eps': [0.0035, 0.007, 0.015, 0.03, 0.06, 0.12, 0.24, 0.5]}, 8)
# basic_iterative_method_attack = eval_definition(torchattacks.BIM, {'eps': 4 / 255, 'alpha': 1 / 255, 'steps': 0}, 1)
## basic_iterative_method_attack_eps = eval_definition(torchattacks.BIM, {'eps': [4 / 255, 25 / 255,50 / 255,12 / 255,100 / 255,2 / 255], 'alpha': 1 / 255, 'steps': 0}, 6)
## basic_iterative_method_attack_alpha = eval_definition(torchattacks.BIM, {'eps': 4 / 255, 'alpha': [1 / 255, 4 / 255,12 / 255,25 / 255,50 / 255], 'steps': 0}, 5)
basic_iterative_method_attack_eps = eval_definition(torchattacks.BIM, {'eps': [0.0035, 0.007, 0.015, 0.03, 0.06, 0.12, 0.24, 0.5], 'alpha': 0.0035, 'steps': 0}, 8)
# one_pixel_attack = eval_definition(torchattacks.OnePixel, {'pixels': 10, 'steps': 75, 'popsize': 100, 'inf_batch': 32}, 1)
one_pixel_attack_steps = eval_definition(torchattacks.OnePixel, {'pixels': 10, 'steps': [35, 75], 'popsize': 100, 'inf_batch': 32}, 2)
# deep_fool_attack = eval_definition(torchattacks.DeepFool, {'steps': 50, 'overshoot': 0.02}, 1)
deep_fool_attack_overshoot = eval_definition(torchattacks.DeepFool, {'steps': 50, 'overshoot': [0.02, 0.1, 0.2, 0.5, 0.005]}, 5)
deep_fool_attack_steps = eval_definition(torchattacks.DeepFool, {'steps': [25, 50, 75], 'overshoot': 0.02}, 3)
# auto_attack = eval_definition(torchattacks.AutoAttack, {'norm': 'Linf', 'eps': .05, 'version': 'standard', 'n_classes': 43, 'seed': None, 'verbose': False}, 1)
auto_attack_eps = eval_definition(torchattacks.AutoAttack, {'norm': 'Linf', 'eps': [.05, 0.1, 0.2, 0.5, 0.01], 'version': 'standard', 'n_classes': 43, 'seed': None, 'verbose': False}, 5)


def eval_attack(eval_model, attack_class, options, eval_data, model_predictions, folder):
    """
    Executes an attack with a single run config. Writes successful adversarial examples inside the run folder (depends on config).

    :param eval_model: lightning model
    :param attack_class: the class reference to the requested attack
    :param options: dict of arguments for the attack
    :param eval_data: the images
    :param model_predictions:  the labels
    :param folder: str, run folder, where to put the folder of this run and inside that the adversarial examples.
    :return:
    """
    eval_folder = f"{folder}/{attack_class.__name__}({','.join([f'{opt_name}={opt_value}' for opt_name, opt_value in options.items()])})"
    options['model'] = eval_model
    attack_model = attack_class(**options)
    adversarial_examples = attack_model(eval_data, model_predictions)
    adversarial_predictions = eval_model(adversarial_examples).argmax(dim=1)
    successful_adversarial = torch.where((model_predictions != adversarial_predictions) == True)[0]
    if not os.path.isdir(eval_folder):
        os.mkdir(eval_folder)
    for idx in successful_adversarial:
        save_image(adversarial_examples[idx],
                   f"{eval_folder}/IMG-{idx}--FROM--{label_map[str(model_predictions[idx].item())]}--TO--{label_map[str(adversarial_predictions[idx].item())]}.png")

    return successful_adversarial.shape[0] / model_predictions.shape[0]


def get_variable_options(evaluation_options):
    """
    Transforms evaluation options to a dict with the "multiple/variable" (not fix) options as values and their names as keys.

    :param evaluation_options: dict of options, list values indicate multiple evaluations (2. argument of eval_definition)
    :return: filtered evaluation_options if the value is an instance of list
    """
    return {opt_name: opt_values for opt_name, opt_values in evaluation_options.items() if isinstance(opt_values, list)}


def get_options(evaluation_options, number_runs):
    """
    Generator of options from multiple options.

    :param evaluation_options: dict of options, list values indicate multiple evaluations (2. argument of eval_definition)
    :param number_runs: indicates how many options are generated (e.g. == len of the multiple eval option)
    :return: Generator, yields single attack parameters as dict
    """
    variable_option_names = get_variable_options(evaluation_options)
    for run in range(number_runs):
        options = {}
        for opt_name, opt_values in sorted(evaluation_options.items()):
            if opt_name in variable_option_names:
                options[opt_name] = opt_values[run % len(opt_values)]
            else:
                options[opt_name] = opt_values
        yield options


def evaluate_attacks(eval_model, attack_variants, data, model_predictions):
    """
    Executes all attack variations on the passed model with the passed data.
    Logs the results inside a log file in the evaluation_results folder and the specific run folder.
    Inside that folder the function stores  also the successful adversarial examples

    :param eval_model: lightning model
    :param attack_variants: list of eval definition tuples
    :param data: torch tensor stacked images
    :param model_predictions:  stacked predictions
    :return: None
    """
    eval_time = datetime.datetime.now()
    attack_names = [eval_def.attack.__name__ for eval_def in attack_variants]
    folder_name = f"evaluation_results/{eval_time:%Y-%m-%d_%H-%M}_{'_'.join(attack_names)}"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    logging.basicConfig(filename=f"{folder_name}/results.log", encoding='utf-8', level=logging.INFO, format="%(message)s")

    logging.info(f"Evaluation: {folder_name}")
    for attack, evaluation_options, number_runs in attack_variants:
        logging.info(f"Start {attack.__name__} evaluation on {data.shape[0]} images:")
        variable_options = get_variable_options(evaluation_options)
        fixed_options = {opt_name: opt_value for opt_name, opt_value in evaluation_options.items() if opt_name not in variable_options}
        logging.info(f"- fixed options: {fixed_options}")
        logging.info(f"- variable options: {variable_options}")
        for run_options in get_options(evaluation_options, number_runs):
            options_description = str(run_options)
            result = eval_attack(eval_model, attack, run_options, data, model_predictions, folder_name)
            logging.info(f"- - {options_description}: {result}")

        logging.info(f"end {attack.__name__} evaluation ({datetime.datetime.now():%H:%M})\n")
    logging.info(f"finished")


if __name__ == '__main__':
    data_transforms = transforms.Compose([
        transforms.Resize([112, 112]),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ToTensor()
    ])
    # out trained resnet18
    model = MyModel.load_from_checkpoint("model_checkpoint.ckpt")
    test_data = load_test_data(transform=data_transforms)
    # first 100 images
    images, _ = zip(*test_data.image_labels[:100])
    images = torch.vstack([x.unsqueeze(0) for x in images])
    predicted_labels = model(images).argmax(dim=1)

    # change here the attack definition
    evaluate_attacks(model,
                     [
                         fgsm_attack,
                         basic_iterative_method_attack_eps,
                         deep_fool_attack_steps,
                         deep_fool_attack_overshoot,
                         auto_attack_eps,
                         one_pixel_attack_steps,
                     ],
                     images,
                     predicted_labels)
