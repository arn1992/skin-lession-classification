import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from skimage.io import imread
from PIL import Image
from sklearn.model_selection import train_test_split

base_skin_dir = "D:/AI in medicine/final_project_mobilenet/skin-cancer-mnist-ham10000"
skin_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv')) # load in the data
print(skin_df.head())

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

lesion_type_dict = {
    'nv': 'Melanocytic_nevi',
    'mel': 'melanoma',
    'bkl': 'Benign_keratosis-like_lesions',
    'bcc': 'Basal_cell_carcinoma',
    'akiec': 'Actinic_keratoses',
    'vasc': 'Vascular_lesions',
    'df': 'Dermatofibroma'
}

lesion_danger = {
    'nv': 0, # 0 for benign
    'mel': 1, # 1 for malignant
    'bkl': 0, # 0 for benign
    'bcc': 1, # 1 for malignant
    'akiec': 1, # 1 for malignant
    'vasc': 0,
    'df': 0
}

skin_df["path"] = skin_df["image_id"].map(imageid_path_dict.get) # map image_id to the path of that image
skin_df["path"] = skin_df["image_id"].map(imageid_path_dict.get) # map image_id to the path of that image
skin_df["cell_type"] = skin_df["dx"].map(lesion_type_dict.get) # map dx to type of lesion
print(skin_df.head())

skin_df["Malignant"] = skin_df["dx"].map(lesion_danger.get)

print(skin_df.head())

skin_df["cell_type_idx"] = pd.Categorical(skin_df["cell_type"]).codes # give each cell type a category id
print(skin_df.sample(3))


skin_df["Malignant"].value_counts().plot(kind="bar", title="Benign vs Malignant")
plt.show()

fig, ax1 = plt.subplots(1,1)
skin_df["cell_type"].value_counts().plot(kind="bar", ax=ax1, title="Counts for each type of Lesions") # plot a graph counting the number of each cell type
plt.show()
# let's see where lesions are mostly located
skin_df["localization"].value_counts().plot(kind='bar', title="Location of Lesions")
plt.show()
skin_df["dx_type"].value_counts().plot(kind='bar', title="Treatment received")
plt.show()

skin_df["age"].hist(bins=50)
plt.show()

skin_df[skin_df["Malignant"] == 1]["age"].hist(bins=40)
plt.show()
skin_df["sex"].value_counts().plot(kind="bar", title="Male vs Female")
plt.show()
skin_df[skin_df["Malignant"] == 1]["sex"].value_counts().plot(kind="bar", title="Male vs Female. Malignant Cases")
plt.show()
skin_df["image"] = skin_df["path"].map(imread) # read the image to array values
print(skin_df.iloc[0]["image"]) # here is a sample

# let's see what is the shape of each value in the image column
print(skin_df["image"].map(lambda x: x.shape).value_counts())

# let's have a look at the image data

n_samples = 5 # choose 5 samples for each cell type
fig, m_axs = plt.subplots(7, n_samples, figsize=(4*n_samples, 3 * 7))

for n_axs, (type_name, type_rows) in zip(m_axs, skin_df.sort_values(["cell_type"]).groupby("cell_type")):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=0).iterrows()):
        c_ax.imshow(c_row["image"])
        c_ax.axis("off")
fig.savefig("category_samples.png", dpi=300)

# create a pandas dataframe to store mean value of Red, Blue and Green for each picture
rgb_info_df = skin_df.apply(lambda x: pd.Series({'{}_mean'.format(k): v for k, v
                                                 in zip(["Red", "Blue", "Green"],
                                                        np.mean(x["image"], (0, 1)))}), 1)


gray_col_vec = rgb_info_df.apply(lambda x: np.mean(x), 1) # take the mean value across columns of rgb_info_df
for c_col in rgb_info_df.columns:
    rgb_info_df[c_col] = rgb_info_df[c_col]/gray_col_vec
rgb_info_df["Gray_mean"] = gray_col_vec
rgb_info_df.sample(3)

for c_col in rgb_info_df.columns:
    skin_df[c_col] = rgb_info_df[c_col].values


# let's draw a plot showing the distribution of different cell types over colors!
sns.pairplot(skin_df[["Red_mean", "Green_mean", "Blue_mean", "Gray_mean", "cell_type"]],
             hue="cell_type", plot_kws = {"alpha": 0.5})
plt.show()



reshaped_image = skin_df["path"].map(lambda x: np.asarray(Image.open(x).resize((64,64), resample=Image.LANCZOS).\
                                                          convert("RGB")).ravel())

out_vec = np.stack(reshaped_image, 0)
out_df = pd.DataFrame(out_vec)

out_df["label"] = skin_df["cell_type_idx"]

print(out_df.head())

out_path = "D:/AI in medicine/final_project_mobilenet/hmnist_64_64_RBG.csv"
out_df.to_csv(out_path, index=False)


img = Image.open(skin_df["path"][0])
print(img.size)

print(skin_df["cell_type"].unique())

print(skin_df["path"][0])
'''
for index in skin_df.index.values.tolist():
    path = skin_df.iloc[index]["path"]
    cell_type_idx = skin_df.iloc[index]["cell_type"]
    img_id = skin_df.iloc[index]["image_id"]
    newpath = 'D:/AI in medicine/final_project_mobilenet/skin_lesion_types/{}/{}.jpg'.format(cell_type_idx,img_id)
    img = Image.open(path)
    img = img.resize((299, 299), resample=Image.LANCZOS)
    img.save(newpath, 'JPEG')
'''
reshaped_image = skin_df["path"].map(lambda x: np.asarray(Image.open(x).resize((256,192), resample=Image.LANCZOS).\
                                                          convert("RGB")))

out_vec = np.stack(reshaped_image, 0)

print(out_vec.shape)

out_vec = out_vec.astype("float32")
out_vec /= 255

labels = skin_df["cell_type_idx"].values


X_train, X_test, y_train, y_test = train_test_split(out_vec, labels, test_size=0.2,random_state=0)

np.save("test1.npy", X_test)
np.save("test1_labels.npy", y_test)
np.save("train1.npy", X_train)
np.save("train1_labels.npy", y_train)










