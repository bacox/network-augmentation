import json
import time

import torch

from conf import settings
from utils import get_test_dataloader
from vgg_variant import construct_vgg_variant
import requests


def load_weights_from_server(model_path: str):
    server_address = 'http://0.0.0.0:5000'

    model_url = f'{server_address}/models/{model_path}'
    print(f'Loading model from {model_url}')
    state_dict = torch.hub.load_state_dict_from_url(model_url, map_location='cpu')

    return state_dict

def test_model(conv: int, fcl: int, model_path: str, args):
    # load dataset
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        # settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args['b'],
    )
    weights = load_weights_from_server(model_path)
    net, arch_name = construct_vgg_variant(conv_variant=conv, fcl_variant=fcl, batch_norm=True, progress=True,
                                           pretrained=False)
    # net.load_state_dict(torch.load(args.weights))
    net.load_state_dict(weights)

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            if args['gpu']:
                image = image.cuda()
                label = label.cuda()

            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            # compute top 5
            correct_5 += correct[:, :5].sum()

            # compute top1
            correct_1 += correct[:, :1].sum()

    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))


def next_model(server: str):
    url = f'{server}/next'
    try:
        resp = requests.get(url=url)
        data = resp.json()
        if data['result']:
            return True, data
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        print('Is the model serving server running?')
        raise SystemExit(e)
    return False, {}

def save_result_to_server(server: str, result_data: json):
    res = requests.post(f'{server}/save-result', json=result_data)
    if res.ok:
        print(res.json())

def process_loop(server: str):
    # Check for new model to inference
    # - If so --> Do inference
    # Else timeout
    has_model, data = next_model(server)
    timeout_duration = 10 # seconds
    if has_model:
        print(data)
        model_path = data['model_url']
        model_params = json.loads(data['model_params'])
        conv = model_params['conv']
        fcl = model_params['fcl']
        args = {
            'b': 128,
            'gpu': False
        }
        test_model(conv=conv, fcl=fcl, model_path=model_path, args=args)

        # Save result to server
        save_result_to_server(server, result_data={'result': 'hello world'})
    else:
        print('No models to inference --> timeout')
        time.sleep(timeout_duration)


def run(server: str):
    print('Stop loop by CTRL+C')
    try:
        while True:
            process_loop(server)
    except KeyboardInterrupt as ke:
            print('Stopping reconnecting loop by action of user')


if __name__ == '__main__':
    print('Inference client')

    conv = 2
    fcl = 2
    server = 'http://0.0.0.0:5000'

    run(server=server)
    print('Stopping server')
    # url = f'{server}/next'
    #
    # import requests
    #
    # res = requests.post(f'{server}/save-result', json={"mytext": "lalala"})
    # if res.ok:
    #     print(res.json())
    #
    # print('Intermediate')
    #
    #
    # try:
    #     resp = requests.get(url=url)
    #     data = resp.json()
    # except requests.exceptions.RequestException as e:  # This is the correct syntax
    #     print('Is the model serving server running?')
    #     raise SystemExit(e)
    #
    # if data['result']:
    #     model_path = data['model_url']
    #     conv = data['model_params']['conv']
    #     fcl = data['model_params']['fcl']
    #     args = {
    #         'b': 128,
    #         'gpu': False
    #     }
    #     test_model(conv=conv, fcl=fcl, model_path=model_path, args=args)
    # else:
    #     print('No model to inference!')
    #     print(data)
    # # model_name = 'vgg_c2_f2-1-regular.pth'
    # # model_path = f'vgg_c2_f2/Thursday_10_December_2020_16h_02m_34s/{model_name}'
    # # args = {
    # #     'b': 128,
    # #     'gpu': False
    # # }
    # # test_model(conv=conv, fcl=fcl, model_path=model_path, args=args)
