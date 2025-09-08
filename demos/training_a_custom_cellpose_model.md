# How to train a custom cellpose model

1. Pick the regions and sections that you will use for your training images. You want to pick regions which cover a variety of sections, z-planes, and cellular densities. You should use 6 training images at the very least, but you will likely need more to accurately represent the diversity of regions you will be segmenting.
2. Generate the training image for each of the regions. These are RGB tiffs, with the first channel being the total mRNA “stain”, the second being DAPI, and the third being blank. This can be done using the provided jupyter notebook: `generating_training_images.ipynb`
3. Make sure all the images are in the same directory
4. Install cellpose gui
    ```
    > python -m pip install cellpose[gui]
    ```
5. Run cellpose gui
    ```
    > python -m cellpose --gpu_device mps --use_gpu
    ```
6. Open the first of your training images
7. In the segmentation pane:

    ![Cellpose GUI](./support_files/cellpose_gui.png)

    <ol type="a">
        <li>Set the cell diameter to 
            <math display="inline">
                <mfrac>
                <mn>um</mn>
                <mn>um/px</mn>
                </mfrac>
            </math>
            . In this example that is
                <math display="inline">
                    <mfrac>
                    <mn>10um</mn>
                    <mn>0.108um/px</mn>
                    </mfrac>
                    <mo>=</mo>
                    <mn>92.59</mn>
                </math></li>
        <li>Set <i>chan to segment</i> to <i>2: green</i></li>
        <li>Set <i>chan2 (optional)</i> to <i>1: red</i></li>
        <li>Click the cyto model and wait for the results</li>
    </ol>
8. Fix results you think are mis-segmented
    <ol type="a">
        <li><kbd>ctrl</kbd>/<kbd>⌘</kbd> click to delete a segmentation</li>
        <li>Right click to start outlining a cell and trace until you reach the starting point</li>
        <li>You can use <kbd>R</kbd> and <kbd>G</kbd> to toggle DAPI and Cyto stain visibility</li>
        <li>You can use <kbd>X</kbd> and <kbd>Z</kbd> to toggle mask and outline visibility</li>
    </ol>
9. Once you have fixed all segmentations, click the *Models* option in the menu bar and go down to *Train new model with image+masks in folder*
    <ol type="a">
        <li>Make sure that the settings are acceptable and click <i>OK</i></li>
        <li>The model will start training. you can view the output in the terminal</li>
    </ol>
10. Once the custom model finishes training, it is automatically applied to the next image in the folder. Repeat steps 8 and 9 until you have no more training images or are satisfied the the results of the segmentation.
11. Now for any future segmentation, you can use the outputted model file instead of any of the default models.