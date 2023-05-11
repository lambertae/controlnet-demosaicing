# %%
# change the flickr8k dataset to the format of the huggingface dataset
# %%
# wget ""
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
# sort desc_dict[a] by the length of the sentence
for key, desc_list in desc_dict.items():
    desc_list.sort(key=lambda x: len(x), reverse=True)
print(desc_dict['1000268201_693b08cb0e'])

# %%
img_folder = "/data/scratch/dengm/controlnet/controlnet-demosaicing/flickr8k/Flicker8k_Dataset"
import cv2 
import matplotlib.pyplot as plt
cnt = 0
for name in test_imgs:
    cnt += 1
    if cnt == 1:
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
file = train_imgs[0]
# plot img with pixelate(resize(img))
fig, ax = plt.subplots(1, 2)
img1 = load_image(file)
ax[0].imshow(resize(img1))
ax[1].imshow(pixelate(resize(img1)))
caption = desc_dict[file.split('.')[0]][0]
# show caption in plt
fig.text(0.5, 0.05, caption, ha='center')
plt.show()

# %%
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
train_dataset = Dataset.from_dict({"img": train_imgs}).map(lambda x: dataset_entry(x["img"]), num_proc=16)
split_datasets = train_dataset.train_test_split(test_size=0.1)
print(split_datasets.column_names)
split_datasets.save_to_disk("/data/scratch/dengm/controlnet/controlnet-demosaicing/flickr8k/new_dataset")
# %%
!huggingface-cli login
# %%

train_dataset.push_to_hub("flickr8k")

# %%
val_img = pixelate(resize(load_image(dev_imgs[0])))
plt.imshow(val_img)
plt.show()
val_cap = desc_dict[dev_imgs[0].split('.')[0]][0]
# save val image
import PIL
im = PIL.Image.fromarray(val_img)
im.save("/data/scratch/dengm/controlnet/controlnet-demosaicing/flickr8k/val_img.png")
# %%

train_img = pixelate(resize(load_image(train_imgs[0])))
plt.imshow(train_img)
plt.show()
train_cap = desc_dict[train_imgs[0].split('.')[0]][0]
# save val image
import PIL
im = PIL.Image.fromarray(train_img)
im.save("/data/scratch/dengm/controlnet/controlnet-demosaicing/flickr8k/train_img.png")
print(train_cap)
# %%
