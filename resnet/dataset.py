#setting up dataloaders for validaton and testing, here you can find a dataloader that returns the paths of files as well
import torch
from torchvision import datasets, transforms
import PIL
from os.path import expanduser


home = expanduser("~")
#Boston
data_root = home+'/data/streetContext/streetImages/Boston/modelRelated/data/resnetPOI/'
#SF
#data_root = home+'/data/streetContext/streetImages/SF/modelRelated/data/resnetPOI/'

train_root = data_root + 'train/'
val_root = data_root + 'test/'
test_root = data_root + ''


base_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(40, resample=PIL.Image.BILINEAR),
    transforms.ToTensor()
    ])

validationTransform = transforms.Compose([transforms.ToTensor()])




def get_data_loaders(batch_size):
    #
    train_dataset = datasets.ImageFolder(root=train_root, transform=base_transform)
    val_dataset = datasets.ImageFolder(root=val_root, transform=base_transform)
    test_dataset = datasets.ImageFolder(root=test_root, transform=base_transform)
    weights = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    valweights = make_weights_for_balanced_classes(val_dataset.imgs, len(val_dataset.classes))
    valweights = torch.DoubleTensor(valweights)
    valsampler = torch.utils.data.sampler.WeightedRandomSampler(valweights, len(valweights))

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, sampler = sampler, num_workers=16,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,  sampler = valsampler, num_workers=16,pin_memory=True)
    return (train_loader, val_loader)

def get_val_test_loaders(batch_size):
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return (val_loader, test_loader)



def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/(1+float(count[i]))
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight




class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + tuple([path])
        return tuple_with_path

