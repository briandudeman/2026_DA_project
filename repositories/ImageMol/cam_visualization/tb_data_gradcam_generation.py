import sys
sys.path.append("../")
import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from cam_visualization.GradCAM import GradCAM
from model.cnn_model_utils import load_model


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_image(img_path, normalize, img_transformer):
    img = Image.open(img_path).convert('RGB')
    img_trans = img_transformer(img)
    img_show = img_trans.numpy().transpose(1, 2, 0)
    return img_show, normalize(img_trans)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch GradCAM')
    parser.add_argument('--image_model', type=str, default="ResNet18", help='e.g. ResNet18, ResNet34')
    parser.add_argument('--resume', required=True, type=str, metavar='PATH')
    parser.add_argument('--img_directory', type=str, required=True, help='path to image directory')
    parser.add_argument('--gradcam_save_directory', type=str, required=True, help='path to saved gradcam directory')
    parser.add_argument('--thresh', type=float, default=0.0, help='thresh of gradcam')
    args = parser.parse_args()

    # 1. initialize model
    model = load_model(args.image_model, imageSize=224, num_classes=1)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            
            full_checkpoint = torch.load(args.resume)
            try:
                model.load_state_dict(full_checkpoint)
            except:
                checkpoint = full_checkpoint["model_state_dict"]
                model.load_state_dict(checkpoint)
            epoch = full_checkpoint['epoch']
            print("=> loading completed")
            print("resume model info: epoch: {}".format(epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    #model = model.cuda()
    model.eval()

    img_directory = os.fsencode(args.img_directory)
    
    for img in os.listdir(img_directory):

        img_index = os.fsdecode(img).replace(".png", "")
        print("img_index:", img_index)
        # 2. initialize data
        img_cuda = torch.FloatTensor(1, 3, 224, 224)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_transformer = [transforms.CenterCrop(224), transforms.ToTensor()]
        img_show, image = get_image(os.path.join(img_directory, img), normalize, transforms.Compose(img_transformer))
        img_cuda.copy_(torch.unsqueeze(image, 0))

        # 3. run gradcam
        gradcam_obj = GradCAM(img=(img_show, img_cuda), model=model, gradcam_path=os.path.join(args.gradcam_save_directory, f"{img_index}.png"), thresh=args.thresh)
        heatmap = gradcam_obj()
        print("execute completed.")

