import datetime
import logging
import os
from collections import namedtuple

import torch
import torchattacks

from Project11.load_test import load_test_data
from model import MyModel
from utils import label_map
from torchvision import transforms
from torchvision.utils import save_image


eval_definition = namedtuple('eval_definition', 'attack options number_runs')


# put different arguments in a list, that the script should evaluate
fgsm_attack = eval_definition(torchattacks.FGSM, {'eps': [0.0035, 0.007, 0.015, 0.03, 0.06, 0.12, 0.24, 0.5]}, 8)
one_pixel_attack = eval_definition(torchattacks.OnePixel, {'pixels': 10, 'steps': 75, 'popsize': 100, 'inf_batch': 32}, 1)
deep_fool_attack = eval_definition(torchattacks.DeepFool, {'steps': 50, 'overshoot': 0.02}, 1)
auto_attack = eval_definition(torchattacks.AutoAttack, {'norm': 'Linf', 'eps': .05, 'version': 'standard', 'n_classes': 43, 'seed': None, 'verbose': False}, 1)


def eval_attack(eval_model, attack_class, options, eval_data, model_predictions, folder):
    eval_folder = f"{folder}/{attack_class.__name__}({','.join([f'{opt_name}={opt_value}' for opt_name, opt_value in options.items()])})"
    options['model'] = eval_model
    attack_model = attack_class(**options)
    adversarial_examples = attack_model(eval_data, model_predictions)
    adversarial_predictions = eval_model(adversarial_examples).argmax(dim=1)
    successful_adversarial = torch.where((model_predictions != adversarial_predictions) == True)[0]
    if not os.path.isdir(eval_folder):
        os.mkdir(eval_folder)
    for idx in successful_adversarial:
        save_image(adversarial_examples[idx], f"{eval_folder}/IMG-{idx}--FROM--{label_map[str(model_predictions[idx].item())]}--TO--{label_map[str(adversarial_predictions[idx].item())]}.png")

    return successful_adversarial.shape[0] / model_predictions.shape[0]


def get_variable_options(evaluation_options):
    return {opt_name: opt_values for opt_name, opt_values in evaluation_options.items() if isinstance(opt_values, list)}


def get_options(evaluation_options, number_runs):
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

        logging.info(f"end {attack.__name__} evaluation ({datetime.datetime.now():%H-%M})\n")
    logging.info(f"finished")


if __name__ == '__main__':
    data_transforms = transforms.Compose([
        transforms.Resize([112, 112]),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ToTensor()
    ])
    model = MyModel.load_from_checkpoint("model_checkpoint.ckpt")
    test_data = load_test_data(transform=data_transforms)
    images, _ = zip(*test_data.image_labels[:10])
    images = torch.vstack([x.unsqueeze(0) for x in images])
    predicted_labels = model(images).argmax(dim=1)

    # change here the attack definition
    evaluate_attacks(model, [auto_attack], images, predicted_labels)
