# %%
# change the flickr8k dataset to the format of the huggingface dataset
# %%
import os
def load_text(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    text = text.split('\n')
    # close the file
    file.close()
    return text
text_folder = "/data/scratch/dengm/controlnet/controlnet-demosaicing/flickr8k"
train_imgs = load_text(os.path.join(text_folder, "Flickr_8k.trainImages.txt"))
test_imgs = load_text(os.path.join(text_folder, "Flickr_8k.testImages.txt"))
dev_imgs = load_text(os.path.join(text_folder, "Flickr_8k.devImages.txt"))
train_imgs = [x for x in train_imgs if x != '']
test_imgs = [x for x in test_imgs if x != '']
dev_imgs = [x for x in dev_imgs if x != '']

# %%
print(len(train_imgs))
print(train_imgs[0])
# %%
import string
def load_descriptions(doc):
    mapping = dict()
    for line in doc:
        tokens = line.split()
        if len(tokens) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in mapping.keys():
            mapping[image_id] = []
        mapping[image_id].append(image_desc)
    return mapping
def clean_description(desc_dict):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in desc_dict.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word)>1]
            desc_list[i] =  ' '.join(desc)
descriptions = load_text(os.path.join(text_folder, "Flickr8k.token.txt"))
desc_dict = load_descriptions(descriptions)
clean_description(desc_dict)
# %%
img_folder = "/data/scratch/dengm/controlnet/controlnet-demosaicing/flickr8k/Flicker8k_Dataset"
import cv2 
import matplotlib.pyplot as plt
cnt = 0
for name in train_imgs:
    cnt += 1
    if cnt == 5:
        break
    prefix = name.split('.')[0]
    img = cv2.imread(os.path.join(img_folder, name))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    print(desc_dict[prefix])
# %%
# create huggingface dataset with column names ['image', 'caption', 'condition']
# change the image column to the corresponding image
def load_image(filename):
    img = cv2.imread(os.path.join(img_folder, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
def center_crop(img):
    h, w, c = img.shape
    if h > w:
        img = img[(h-w)//2:(h-w)//2+w, :, :]
    else:
        img = img[:, (w-h)//2:(w-h)//2+h, :]
    return img
def resize(img):
    img = cv2.resize(center_crop(img), (256, 256), interpolation=cv2.INTER_NEAREST)
    return img
def pixelate(img):
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
    return img

img1 = load_image(train_imgs[4])
print(img1.shape)

plt.imshow(resize(img1))
plt.show()

plt.imshow(pixelate(resize(img1)))
plt.show()
plt.imshow(img1)
plt.show()
import PIL
im = PIL.Image.fromarray(img1)
plt.imshow(resize(img1))
# %%
def dataset_entry(filename):
    prefix = filename.split('.')[0]
    img = resize(load_image(filename))
    pix = pixelate(img)
    return {"image": PIL.Image.fromarray(img), "caption" : desc_dict[prefix][0], "condition": PIL.Image.fromarray(pix)}

from datasets import Dataset
image_list = Dataset.from_dict({"image": train_imgs})
train_dataset = image_list.map(lambda x: dataset_entry(x["image"]), num_proc=16)
# save
train_dataset.save_to_disk("/data/scratch/dengm/controlnet/controlnet-demosaicing/flickr8k/train_dataset")
