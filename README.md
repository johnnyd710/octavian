# the octavian project

Get the sample data by running ```python3 app/datadownloader.py --read links.txt --unzip --dest ../data```

Then train the EfficientNet with ``` python3 app/model.py --epochs 10 --restore 0 ```

And serve with ``` python3 app/main.py ```

Go to ``` localhost/v1/octavian/docs ```

## docker
Alternatively, you can pull docker container (```docker pull johndimatteo/octavian```)
and run with ```docker run -d --name octavian -p 80:80```

## why Octavian
I name every one of my projects after a Roman Emperor.
Gaius Octavius was Julis Caesar's adopted son (later known as Octavian, and then Augustus Caesar). When Caesar was killed,
Octavius was only 18, but nevertheless he fought and defeated Mark Antony to become the Rome's first Emperor. 

'Oct' reminds me of optometry, probably because of OCT scans. 

## problem statement
Given an image of the retina (any size, it will be normalized to 224x244 pixels anyway), determine if the image has signs of Age Macular Degeneration (AMD). If so, respond `amd` with a confidence score. If not, respond `no amd` with a confidence score.

## purpose
To show how machine learning models can be deployed quickly using Python, Docker, Fastapi, and Microsoft Azure.

## method
EfficientNet, transfer learning using pretrained weights. Remove the last layer meant for 1000 classes to a simple 2 class layer: AMD and non-AMD.
Save and commit checkpoint.pt file to repo.
[See Google's EfficientNet paper here.](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)
