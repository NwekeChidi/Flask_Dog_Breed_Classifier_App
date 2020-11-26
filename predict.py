# load required libraries
import torch
from torchvision import transforms
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Predict:

###  Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

    # list of class names by index, i.e. a name can be accessed like class_names[0]
    def __init__(self):
        self.class_names = ['Affenpinscher', 'Afghan hound', 'Airedale terrier', 'Akita', 'Alaskan malamute',
                        'American eskimo dog', 'American foxhound', 'American staffordshire terrier',
                        'American water spaniel', 'Anatolian shepherd dog', 'Australian cattle dog',
                        'Australian shepherd', 'Australian terrier', 'Basenji', 'Basset hound', 'Beagle',
                        'Bearded collie', 'Beauceron', 'Bedlington terrier', 'Belgian malinois', 'Belgian sheepdog',
                        'Belgian tervuren', 'Bernese mountain dog', 'Bichon frise', 'Black and tan coonhound',
                        'Black russian terrier', 'Bloodhound', 'Bluetick coonhound', 'Border collie',
                        'Border terrier', 'Borzoi', 'Boston terrier', 'Bouvier des flandres', 'Boxer',
                        'Boykin spaniel', 'Briard', 'Brittany', 'Brussels griffon', 'Bull terrier',
                        'Bulldog', 'Bullmastiff', 'Cairn terrier', 'Canaan dog', 'Cane corso', 'Cardigan welsh corgi',
                        'Cavalier king charles spaniel', 'Chesapeake bay retriever', 'Chihuahua', 'Chinese crested',
                        'Chinese shar-pei', 'Chow chow', 'Clumber spaniel', 'Cocker spaniel', 'Collie',
                        'Curly-coated retriever', 'Dachshund', 'Dalmatian', 'Dandie dinmont terrier', 
                        'Doberman pinscher', 'Dogue de bordeaux', 'English cocker spaniel', 'English setter',
                        'English springer spaniel', 'English toy spaniel', 'Entlebucher mountain dog',
                        'Field spaniel', 'Finnish spitz', 'Flat-coated retriever', 'French bulldog', 
                        'German pinscher', 'German shepherd dog', 'German shorthaired pointer', 'German wirehaired pointer',
                        'Giant schnauzer', 'Glen of imaal terrier', 'Golden retriever', 'Gordon setter',
                        'Great dane', 'Great pyrenees', 'Greater swiss mountain dog', 'Greyhound', 'Havanese',
                        'Ibizan hound', 'Icelandic sheepdog', 'Irish red and white setter', 'Irish setter', 
                        'Irish terrier', 'Irish water spaniel', 'Irish wolfhound', 'Italian greyhound', 'Japanese chin',
                        'Keeshond', 'Kerry blue terrier', 'Komondor', 'Kuvasz', 'Labrador retriever',
                        'Lakeland terrier', 'Leonberger', 'Lhasa apso', 'Lowchen', 'Maltese', 'Manchester terrier',
                        'Mastiff', 'Miniature schnauzer', 'Neapolitan mastiff', 'Newfoundland', 'Norfolk terrier',
                        'Norwegian buhund', 'Norwegian elkhound', 'Norwegian lundehund', 'Norwich terrier',
                        'Nova scotia duck tolling retriever', 'Old english sheepdog', 'Otterhound', 'Papillon',
                        'Parson russell terrier', 'Pekingese', 'Pembroke welsh corgi', 'Petit basset griffon vendeen',
                        'Pharaoh hound', 'Plott', 'Pointer', 'Pomeranian', 'Poodle', 'Portuguese water dog',
                        'Saint bernard', 'Silky terrier', 'Smooth fox terrier', 'Tibetan mastiff',
                        'Welsh springer spaniel', 'Wirehaired pointing griffon', 'Xoloitzcuintli', 'Yorkshire terrier']

    def predict_breed(self, img_path, model, use_cuda=False):
        ## load the image and return the predicted breed
        # Load in the image and convert to rgb
        img = Image.open(img_path).convert('RGB')
        
        # transforming image
        transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
        img = transform(img)
        
        # move model to cuda if available
        if use_cuda:
            model = model.cuda()
            img = img.cuda()
        
        # Unsqueezing to add artificial dimension
        img = img.unsqueeze(0)
        
        # getting predictions, set model to evaluation
        model.eval()


        ### -------------------------------- ###
        ## Adding Functionality for dog mutts
        ### -------------------------------- ###
        class_names = self.class_names
        pred = model(img)

        # getting the probabilites
        ps = torch.exp(pred)

        # sorting and getting prediction index
        ps, idx = torch.sort(ps, descending=True)

        # detaching and unpacking lists
        ps = ps.detach().cpu().numpy()
        idx = idx.detach().cpu().numpy()
        ps, idx = ps[0], idx[0]

        # setting thresholds
        thresh_1, thresh_2 = 0.65, 0.3
        thresh_3, thresh_4 = 0.45, 0.15

        # for two breed mutts
        if ps[0] < thresh_1 and ps[1] > thresh_2:
            mutt_1 = class_names[idx[0]]
            mutt_2 = class_names[idx[1]]
            return "Wow!... You're a Mutt!\nYour ancestors might be: " + str(mutt_1) + " + "\
            + str(mutt_2)

        # for three breed mutts
        elif ps[0] < thresh_3 and ps[1] > thresh_4 and ps[2] > thresh_4:
            mutt_1 = class_names[idx[0]]
            mutt_2 = class_names[idx[1]]
            mutt_3 = class_names[idx[2]]
            return "Wow!... You're a Mutt!\nYour ancestors might be: " + str(mutt_1) + " + "\
            + str(mutt_2) + " + " + str(mutt_3) 

        else:
            predicted_breed = class_names[idx[0]]                            
            return predicted_breed