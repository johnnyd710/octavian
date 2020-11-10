# the octavian project

python3 app/datadownloader.py -- ...
python3 

or pull docker container (docker pull johndimatteo/octavian)
and run with ```docker run -d --name octavian -p 80:80```

go to localhost/v1/octavian/docs

## short preamble on the nickname
Gaius Octavius, Julis Caesar's adopted son (later known as Octavian, and then Augustus Caesar), when Caesar was killed.
Octavius was only 18 at the time, nevertheless he fought and defeated Mark Antony to become the Rome's first Emperor.
He brought Egypt under Rome's control,  

Why the nickname? 'Oct' reminds me of optometry, probably because of OCT scans. 

## problem statement
Given an image of the retina (any size, it will be normalized to 224x244 pixels anyway), determine if the image has signs of Age Macular Degeneration (AMD). If so, respond `amd`. If not, respond `no amd`.

## purpose
Machine learning models can be deployed quickly using Python, Docker, Fastapi, and Microsoft Azure.

## method
EfficientNet, pretrained weights, 1000 classes => 2, AMD and non-AMD.
Save and commit checkpoint.pt file to repo.

## results
trained 5 epochs on the dataset from ... , achieving 98% accuracy, 

## todos
- get confidence with prediction