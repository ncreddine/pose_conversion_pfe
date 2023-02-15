## UE Projet de Fin d'Étude

Ceci represente notre livrable pour notre Projet de Fin d'Étude : __Modélisation par apprentissage du comportement verbale et non verbal d'un agent virtuel__ , plus spécifiquement de la tache  de __"Conversion de pose 2D à 3D"__.

Membres du goupe :
  - Begum Bekiroglu
  - Yasmine Djabri
  - Nacereddine LADDAOUI

---
## Intro
The main subject of this repository is to build a neural network model that predicts the upper body 3D pose from 2D pose. This work is designed to fit for ISIR's  [virtual agent](https://github.com/isir/greta).

Our work is structured as so :
- Data preparation
- Training part
- Convert the 3D pose into BVH, to feed them into the virtual agent.


## Demo 
The provided  `demo.py` script compares between 3D extracted skeleton ground truth and the 2D to 3D conversion model (the 2D skeleton we need for this task is also the 3D ground truth skeleton without the depth information). To evaluate the model efficiently, the demo script takes in consideration input from the webcam and from the PATS dataset itself (cool right!), theses different tasks could be specified using arguments passed to the script.

### webcam


```
./demo.py --model path/to/model.tflite
          --num_threads 6 (default 6)
          --webcam
```

### PATS

```
./demo.py --model path/to/model.tflite
          --num_threads 6 (default 6)
          --from_json path/to/sorted.json
          --set (train/dev/test)
          --speaker (from 22 speakers)
          --video (video title of the speaker)
          --interval (annotated intervals of the video)
```

### Argcomplete

To perform a demo on PATS Dataset, the script first need's to download the video using the provided links in the `sorted.json` and then select the right interval where pose is annotated. All these parameters must be provided by the user manually , first, selecting the dataset (train, test, dev)_set_, the _speaker_, _video_ and then the _interval_. To make user's experience more ergonomic, the demo script comes with an argument completer module, provided by the `argcomplete` python library. This module dynamically offers suggestions to find out which argument you're looking for. This is very useful when selecting an interval from the JSON file.

To use the argcomplete on the termianal, follow the installation guide [here](https://github.com/kislyuk/argcomplete#installation) :

>  **Note**
>  
>  For my env I had to install it using `root` :
>  ```
>  sudo pip3 install argcomplete
>  sudo activate-global-python-argcomplete --dest=/etc/bash_completion.d/
>  ```
>  after that copying the following line to the `.bashrc`
>  ```
>  eval "$(register-python-argcomplete path/to/demo.py)"
>  ```
>  or, execute the following command in the repository folder :
>  ```
>  echo "eval \"\$(register-python-argcomplete \$(realpath ./demo.py))\"" >> ~/.bashrc
>  ```
>  And last :
>  ```
>  source ~/.bashrc
>  ```



To use it, first select the path to the `sorted.json` file using **`--from_json`** argument, after that select of the _set_ using the **`--set`**, here the program proposes 3 choices : train, test and dev (By pressing `<TAB>`) . After the set parameter, comes the **`--speaker`** argument, here the program searches through all available speaker in that particular (By pressing `<TAB>` again)... The  program does the same thing for the video and for the interval's argument, respectively, with **`--video`** and **`--interval`**

```
./demo.py --model models/linear_model.tflite --from_json data/sorted.json --set test --speaker oliver --video Marketing_to_Doctors_-_Last_Week_Tonight_with_John_Oliver_HBO-YQZ2UeOTO3I.mkv --interval 101148

```

## Results

<p align="center">
   <img src="./demo_.gif"  width="80%" height="80%">
</p>

## Integration to Greta


In order to adapt the positions found to the virtual agent Greta, we have first converted the poses that we have as numpy files to .bvh files using numpy-to-bvh.py module. Then, to transform our poses so that they can be interpretable by Greta the module "fix_bvh_for_GRETA.py" in the following git can be used: https://github.com/Michele1996/PFE-OpenPose-to-VAE-to-BVH.
