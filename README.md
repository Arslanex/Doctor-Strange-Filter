![Banne](https://i.pinimg.com/originals/3f/35/90/3f3590a3809163db554425361295f121.jpg)

# Doctor Strange Filter with Python

This project is a test of my skills in computer vision. The project was created using Mediapipe and OpenCV libraries and Python program language. As the name suggests, I tried to replicate the magic circles that appear on the palms of the popular MARVEL hero Dr. Strange when he casts a spell.

To summarize the working logic of the project, the script starts by detecting hand gestures from your device's camera and then calculates the midpoint of your palm and the openness of your palm with these points.  If your hand is open enough, a mask (the filter itself) is created using the previously given images and pasted onto the calculated midpoint.  

Of course, this was a very simple explanation. There are many things to know about perspective, rotating image objects and creating transparent objects during this project. I hope this project will help you to understand these topics :)

## Folder Structure

```
.
├── Models                    # Models to be used when creating the filter
|   ├── Inner Circles         # Circle models to be found on the inner side 
|   └── Outer Circles         # Circle models to be found on the outer side 
├── functions.py              
├── main.py                    
├── LICENSE
├── requirements.txt                   
└── README.md
```

## Setup

1. Clone this repostiory : `git clone https://github.com/Arslanex/Instagram-Mapping`
2. Instal requirements : `pip install -r requirements.txt`
3. Run main.py script : `python main.py`

## Screenshots and Videos (will loaded)

<p align="center">
  <img src="https://user-images.githubusercontent.com/44752389/221366707-a1b93627-f246-428a-8397-864493b49d43.jpg" width="250" />
  <img src="https://user-images.githubusercontent.com/44752389/221366732-49876999-5b7c-4af1-9264-9656ae03f62c.jpg" width="250" /> 
  <img src="https://user-images.githubusercontent.com/44752389/221366735-d25b821a-c143-4cf5-bf80-eb7da7644f1c.jpg" width="250" />
</p>

<p align="center">

[video](https://user-images.githubusercontent.com/44752389/221367008-c1dc0215-e486-43b8-ac88-f0c4d5c822c3)

</p>

***
<h3 align="center"> Enes ARSLAN </h3>
<p align="center">
<a href="https://www.instagram.com/_enes.arslan_/?next=%2F">
<img src="https://img.shields.io/badge/Instagram-000000?style=for-the-badge&logo=instagram&logoColor=white"/>
<a href="https://www.linkedin.com/in/enes-arslan-/">
<img src="https://img.shields.io/badge/LinkedIn-000000?style=for-the-badge&logo=linkedin&logoColor=white"/>
<a href="https://github.com/Arslanex">
<img src="https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=white"/ >
</p>
