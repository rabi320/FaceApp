import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randint as ri
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from collections import Counter
#from timeit import default_timer as timer
import time

#title
st.markdown("""
# Face App: try to outsmart The computer!
The rules are simple:

1. You are given a black and white low quality image(changes randomly every 60 seconds) .
2. Predict the Ethincity (white, black, asian or male) - 3 points
3. Prdict the gender - 2 points.
4. PrTedict the age group (child (1-18), young adult (19-35), adult (36-50), middle aged (51-70) or elder (71-120)) - 5 points.
5. The higher score wins!

so who's vision is better? 
""")
st.markdown('---')

samples = pd.read_pickle("Data/my_samples.csv")
#samples = samples.iloc[:,1:]
#st.write(samples.pixels[1])

@st.cache()
def model_loader(model_path = None, label = None):
    """
    Args:
    >model_path (str)- the path to current model.
    >label (str)-the label to predict. 
    """
    model = models.wide_resnet50_2()
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
                          nn.Linear(num_features, 256),
                          nn.ReLU(),
                          nn.Dropout(0.4),
                          nn.Linear(256, len(samples[label].unique().tolist()),                   
                          nn.LogSoftmax(dim=1)))


    model.load_state_dict(torch.load(model_path))
    return model.eval()

#get all models in one dict
models = {f"model{i}" : model_loader(f"Data/model{i+1}.pt",samples.columns[i]) for i in range(3)}
width = st.sidebar.slider("plot width", 0.1, 25., 3.)
height = st.sidebar.slider("plot height", 0.1, 25., 1.)

@st.cache(ttl=60, show_spinner=True)
def rand_num():
    return ri(0,samples.shape[0]-1)

rand_image = rand_num()

def plot_image(num):
    """
    this function plots a random image.
    the title of the image is the index of the image and the features.
    
    Args:
    > num (int): image number. 
    """

    #labels = dict(zip(samples.columns.tolist()[:-1],samples.loc[num].tolist()[:-1]))
    #plt.title(f"sample #{num+1}- {list(labels.keys())[0]}: {ethnicities[list(labels.values())[0]]}, {list(labels.keys())[1]}: {genders[list(labels.values())[1]]}, {list(labels.keys())[2]}: {AgeGroups[list(labels.values())[2]]}")
    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)
    ax.imshow(samples["pixels"][num].reshape(48,48),cmap = 'gray')
    ax.axis('off')
    return fig



st.write("image #", rand_image)
st.pyplot(plot_image(rand_image))


ethnicities = dict(zip([i for i in range(len(samples.ethnicity.unique().tolist()))], ["white","african","asian","indian"]))
genders = dict(zip([i for i in range(len(samples.gender.unique().tolist()))], ["male","female"]))
AgeGroups = dict(zip([i for i in range(len(samples.AgeGroup.unique().tolist()))], ["child","young adult","adult","middle aged","elder"]))
    
real_ethnicity = ethnicities[samples["ethnicity"][rand_image]]
real_gender = genders[samples["gender"][rand_image]]
real_AgeGroup = AgeGroups[samples["AgeGroup"][rand_image]]


my_ethnicity = st.selectbox('ethnicity: ', list(ethnicities.values()))
my_gender = st.selectbox('gender: ', list(genders.values()))
my_AgeGroup = st.selectbox('AgeGroup: ', list(AgeGroups.values()))
    
@st.cache(ttl=60, show_spinner=True)
def my_score(): 
    score = 0
    if my_ethnicity==real_ethnicity:
        score+=3
    if my_gender==real_gender:
        score+=2
    if my_AgeGroup==real_AgeGroup:
        score+=5
    return score    

  

@st.cache(ttl=60, show_spinner=True)
def predictor(model = None, my_transforms = None, num = None):
    """
    predict with a defined model.
    
    Args:
    >model (Pytorch model): the current model.
    >my_transforms (Pytorch transforms): the transformation the data is taking before the model phase.
    > num (int): image number.
    """
    input_image = my_transforms(np.repeat(samples.pixels.iloc[num].reshape(48, 48)[...,np.newaxis], 3, -1)).unsqueeze(0)
    output = model(input_image)
    _, pred = torch.max(output, dim=1)
    return int(pred)

@st.cache(ttl=60, show_spinner=True)
def profile(num):
    """
    create a prediction profile for a specified image from sample of new data.
    3 models for 3 different attributes are initiated for prediction. 
    
    Args:
    > num (int): image number.
    """
    #plot_image(num)
    ethnicity = ethnicities[predictor(models["model0"],transforms.ToTensor(),num = num)]
    gender = genders[predictor(models["model1"],transforms.ToTensor(),num = num)]
    AgeGroup = AgeGroups[predictor(models["model2"],transforms.ToTensor(),num = num)]
    return ethnicity,gender,AgeGroup



ethnicity,gender,AgeGroup = profile(rand_image)

@st.cache(ttl=60, show_spinner=True)
def computer_score(): 
    score = 0
    if ethnicity==real_ethnicity:
        score+=3
    if gender==real_gender:
        score+=2
    if AgeGroup==real_AgeGroup:
        score+=5
    return score
    

#if st.button('your score:'):
if st.button('submit results!'):
    
    st.write("")
    st.markdown('__checking your scores...__')
    time.sleep(2)
    st.markdown('__Done!__')
    
    if my_ethnicity==real_ethnicity:
        st.write('ethnicity: ', my_ethnicity, " ✔")
        
    else:
        st.write('ethnicity: ', my_ethnicity, " ✖")
    
    if my_gender==real_gender:
        st.write('gender: ', my_gender, " ✔")        
    else:
        st.write('gender: ', my_gender, " ✖")

    if my_AgeGroup==real_AgeGroup:
        st.write('AgeGroup: ', my_AgeGroup, " ✔")
            
    else:
        st.write('AgeGroup: ', my_AgeGroup, " ✖")    
    st.write('my score: ', my_score())  

    st.write("")
    st.markdown('__checking computer scores...__')
    time.sleep(2)
    st.markdown('__Done!__')
    
#if st.button('prediction:'):
    if ethnicity==real_ethnicity:
        st.write('ethnicity: ', ethnicity, " ✔")
    else:
        st.write('ethnicity: ', ethnicity, " ✖")
    
    if gender==real_gender:
        st.write('gender: ', gender, " ✔")

    else:
        st.write('gender: ', gender, " ✖")
    if AgeGroup==real_AgeGroup:
        st.write('AgeGroup: ', AgeGroup, " ✔")

    else:
        st.write('AgeGroup: ', AgeGroup, " ✖")
        
    st.write('computer score: ', computer_score())

    st.markdown('__Answers:__')


#if st.button('reality:'):
    st.write('ethnicity : ', real_ethnicity)
    st.write('gender: ', real_gender)
    st.write('AgeGroup: ', real_AgeGroup)
    
    st.write("")
    st.markdown('__checking scores...__')
    time.sleep(3)
    
#if st.button("declare winner!"):
    if computer_score()>my_score():
        st.write("# computer wins!")
    elif computer_score()==my_score():
        st.write("# tie!")
    else:
        time.sleep(3)
        st.balloons()
        st.markdown('# you win!')