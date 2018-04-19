# sketchShape
This code implements the tasks for sketch-based 3D shape trieval

## Method
The methods are divided into sevaral steps:
* **Multiview renderding**. The shape are rendered on 12 different views [1]. <img src="./figure/mvcnn.png" alt="multiview rendering" width="350">
* **Finetune**.  The sketches and rendered shape images are finetuned with **AlexNet** as single image classification task. The finetuned features for both shrec13 and shrec14 could be downloaded as follows:
    ```
    chmod +x download.sh
    ./download.sh
    ```
* The shape features passed through a linear combination to form one global representation. Contrastive loss are employed for both within-domain pair and across-domain pair. 

## Training
### Training on shrec13
```
chmod u+x train13.sh
./train13.sh
```

### Training on shrec14
```
chmod u+x train14.sh
./train14.sh
```
