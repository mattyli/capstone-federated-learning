# Research
- what is differential privacy?
    - https://privacytools.seas.harvard.edu/differential-privacy
    - framework that provides methods to quantify privacy
- model architectures that we can leverage
    - this paper details lung segmentation from CXR
        - https://www.nature.com/articles/s41598-022-12743-y
        - they use leaky ReLU and and UNet (what is UNet?)
    - https://www.nature.com/articles/s41598-023-49337-1
        - I think this one uses CT scans instead
    - UNet - https://arxiv.org/pdf/1505.04597
        - classification -> label per image, localization -> label per pixel in image
        - Fully connected CNN
        - for localization rather than classification compared to other CV tasks
        - upsampling component has a large number of feature channels to allow the network to propagate context
            - sounds a little bit like the inception blocks in resnet?
        - no Fully Connected (FC) layers
        - use weighted loss and lots of augmentations (because the study dataset was small)
        - contracting and expanding path, after each downsampling step, double the number of feature channels (wider?)
        - they use a batch size of 1
            - need to be mindful of this
        - high momentum (0.99) 
            - pixel wise softmax over the final feature map combined with the cross entropy loss function (?)
        - weights initialized from gaussian (standard normal) distribution with deviation (sigma) sqrt(2/N) where N is the nmber of incoming nodes of one neuron
        - UNet originally looked at cell samples on slides, the only augmentations they looked at were shift, rotation, gray value. 

- what is the difference between transmitting the weights directly and the gradients?
    - how will this affect performance and convergance?
    - should we try both approaches?

- distributing the data non-IID.
# Conclusions
- UNet seems to be a very common architecture for this (segmentation) task
