"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from .options.test_options import TestOptions
import json
from .data import create_dataset
from .models import create_model
from .util.visualizer import save_images
from .util import html_util
import ntpath

def test(data_dir, class_A, class_B, img_size, checkpoints_dir, vgg_checkpoint):

    netG = 'resnet_9blocks'
    data_root = os.path.join(data_dir, 'cycle_gan', class_A + '_' + class_B)
    results_dir = os.path.join(data_root, 'results')

    # workaround for cycle_gan convention
    name = os.path.basename(checkpoints_dir)
    checkpoints_dir = os.path.dirname(checkpoints_dir)

    args = [
        '--model', 'test',
        '--no_dropout',
        '--results_dir', results_dir,
        '--dataroot', data_root,
        '--checkpoints_dir', checkpoints_dir,
        '--name', name,
        '--model_suffix', '_A',
        '--num_test', '500',
        '--aux_net', 'vgg2d',
        '--aux_checkpoint', vgg_checkpoint,
        '--aux_input_size', str(img_size),
        '--aux_output_classes', '6',
        '--aux_downsample_factors', '2,2x2,2x2,2x2,2',
        '--aux_input_nc', '1',
        '--num_threads', '1',
        '--verbose',
        '--netG', netG,
        '--load_size', str(img_size),
        '--crop_size', str(img_size),
    ]

    opt = TestOptions().parse(args)

    test_loop(opt)

def test_loop(opt):

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html_util.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        aux_infos = model.get_current_aux_infos()
        aux_infos = dict(aux_infos)
        aux_infos_json = {v: list(float(k) for k in aux_infos[v][0]) for v in aux_infos.keys()}
        
        short_path = ntpath.basename(img_path[0])
        name = os.path.splitext(short_path)[0]
        json.dump(aux_infos_json, open(webpage.get_image_dir() + "/{}_aux.json".format(name), "w+"))

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    test_loop(opt)
